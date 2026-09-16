from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .data import SequenceScaler, TrialSet, flatten_trials
from .models import PhysioFuseDG


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 64
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 80
    patience: int = 12
    lr: float = 1e-3
    contrastive_weight: float = 0.15
    contrastive_temperature: float = 0.2
    modality_dropout: float = 0.15


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    has_both_classes = len(np.unique(y)) == 2
    return {
        "accuracy": accuracy_score(y, prediction),
        # A held-out subject can have only high or only low trials. Metrics
        # requiring both classes are undefined and must not be forced to zero.
        "balanced_accuracy": balanced_accuracy_score(y, prediction) if has_both_classes else np.nan,
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0) if has_both_classes else np.nan,
        "mcc": matthews_corrcoef(y, prediction) if has_both_classes else np.nan,
        "auroc": roc_auc_score(y, probability) if has_both_classes else np.nan,
        "has_both_test_classes": has_both_classes,
    }


def _cross_modal_infonce(eeg_embedding: torch.Tensor, pps_embedding: torch.Tensor, temperature: float) -> torch.Tensor:
    """Align only synchronized trial representations; all other batch members are negatives."""
    eeg_embedding = nn.functional.normalize(eeg_embedding, dim=1)
    pps_embedding = nn.functional.normalize(pps_embedding, dim=1)
    target = torch.arange(eeg_embedding.size(0), device=eeg_embedding.device)
    similarity = eeg_embedding @ pps_embedding.T / temperature
    return (nn.functional.cross_entropy(similarity, target) + nn.functional.cross_entropy(similarity.T, target)) / 2


def _fit_one(
    mode: str, train: tuple[np.ndarray, np.ndarray, np.ndarray], valid: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: TrainConfig, seed: int, device: str,
) -> PhysioFuseDG:
    seed_everything(seed)
    model = PhysioFuseDG(mode, config.hidden, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    counts = np.bincount(train[2].astype(int), minlength=2)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = DataLoader(TensorDataset(*[torch.tensor(x) for x in train]), batch_size=config.batch_size, shuffle=True)
    ve, vp, vy = [torch.tensor(x, device=device) for x in valid]
    best, best_state, stale = float("inf"), None, 0
    for _ in range(config.epochs):
        model.train()
        for eeg, pps, y in train_loader:
            eeg, pps, y = eeg.to(device), pps.to(device), y.long().to(device)
            if mode == "physiofuse":
                mask = torch.rand(eeg.size(0), device=device) < config.modality_dropout
                eeg[mask] = 0
                pps[~mask & (torch.rand(pps.size(0), device=device) < config.modality_dropout)] = 0
            logits, _, he, hp = model(eeg, pps)
            contrastive = _cross_modal_infonce(he, hp, config.contrastive_temperature) if mode == "physiofuse" else 0
            loss = criterion(logits, y) + config.contrastive_weight * contrastive
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, _, he, hp = model(ve, vp)
            contrastive = _cross_modal_infonce(he, hp, config.contrastive_temperature) if mode == "physiofuse" else 0
            loss = criterion(logits, vy.long()) + config.contrastive_weight * contrastive
        if loss.item() < best:
            best, stale = loss.item(), 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    return model.cpu()


def run_eppvr_loso(data: TrialSet, methods: list[str], tasks: list[str], config: TrainConfig, seeds: list[int], device: str,
                   max_subjects: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if max_subjects is not None:
        data = data.select_subjects(np.arange(min(max_subjects, data.n_subjects)))
    all_subjects = np.arange(data.n_subjects)
    if len(all_subjects) < 3:
        raise ValueError("LOSO needs at least three subjects: training, validation, and test.")
    total = data.n_subjects * len(methods) * len(tasks) * len(seeds)
    progress = tqdm(total=total, desc="EPPVR LOSO", unit="run", dynamic_ncols=True)
    try:
        for task in tasks:
            for held_out in all_subjects:
                remaining = all_subjects[all_subjects != held_out]
                n_validation = min(3, len(remaining) - 1)
                validation_subjects = remaining[-n_validation:]
                train_subjects = remaining[:-n_validation]
                train_e, train_p, train_y = flatten_trials(data, task, train_subjects)
                valid_e, valid_p, valid_y = flatten_trials(data, task, validation_subjects)
                test_e, test_p, test_y = flatten_trials(data, task, np.array([held_out]))
                eeg_scaler, pps_scaler = SequenceScaler().fit(train_e), SequenceScaler().fit(train_p)
                train_e, valid_e, test_e = eeg_scaler.transform(train_e), eeg_scaler.transform(valid_e), eeg_scaler.transform(test_e)
                train_p, valid_p, test_p = pps_scaler.transform(train_p), pps_scaler.transform(valid_p), pps_scaler.transform(test_p)
                for method in methods:
                    for seed in seeds:
                        if method == "late":
                            eeg_model = _fit_one("eeg", (train_e, train_p, train_y), (valid_e, valid_p, valid_y), config, seed + held_out, device)
                            pps_model = _fit_one("pps", (train_e, train_p, train_y), (valid_e, valid_p, valid_y), config, seed + 1000 + held_out, device)
                            with torch.no_grad():
                                eeg_logits, _, _, _ = eeg_model.to(device)(torch.tensor(test_e, device=device), torch.tensor(test_p, device=device))
                                pps_logits, _, _, _ = pps_model.to(device)(torch.tensor(test_e, device=device), torch.tensor(test_p, device=device))
                                prob = (torch.softmax(eeg_logits, 1)[:, 1] + torch.softmax(pps_logits, 1)[:, 1]).div(2).cpu().numpy()
                        else:
                            model = _fit_one(method, (train_e, train_p, train_y), (valid_e, valid_p, valid_y), config, seed + held_out, device)
                            model.to(device).eval()
                            with torch.no_grad():
                                logits, _, _, _ = model(torch.tensor(test_e, device=device), torch.tensor(test_p, device=device))
                                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                        rows.append({"dataset": data.name, "task": task, "method": method, "held_out_subject": held_out + 1,
                                     "seed": seed, **_metrics(test_y, prob)})
                        progress.update(1)
    finally:
        progress.close()
    return pd.DataFrame(rows)


def write_results(results: pd.DataFrame, output_root: str | Path, run_name: str = "eppvr_loso") -> tuple[Path, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    per_fold, summary = root / f"{run_name}_per_subject.csv", root / f"{run_name}_summary.csv"
    results.to_csv(per_fold, index=False)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "mcc", "auroc"]
    results.groupby(["dataset", "task", "method"])[metrics].agg(["mean", "std", "count"]).to_csv(summary)
    return per_fold, summary
