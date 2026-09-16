from __future__ import annotations

import argparse
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal, stats
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


FS = 100
BASELINE_SAMPLES = 1000
WINDOW = 200
STEP = 100
MODALITIES = {
    "FP1": 0,
    "FP2": 1,
    "EDA": 2,
    "PPG": 3,
    "SKT": 4,
    "EOG8_candidate": 8,
    "EOG9_candidate": 9,
}
MODES = ("stimulus_only", "baseline_referenced", "phase_aware", "baseline_conditioned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU temporal gate for baseline-conditioned EPPVR modeling.")
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\AA发表论文的数据\dataset\EPPVR"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/temporal_baseline_gate"))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--run-prefix", default="")
    return parser.parse_args()


def butter(values: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    nyquist = FS / 2
    if low is None:
        sos = signal.butter(4, high / nyquist, btype="lowpass", output="sos")
    elif high is None:
        sos = signal.butter(4, low / nyquist, btype="highpass", output="sos")
    else:
        sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, values)


def modality_curve(raw: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, raw.size - WINDOW + 1, STEP)
    times = (starts + WINDOW / 2 - BASELINE_SAMPLES) / FS
    if modality in {"FP1", "FP2"}:
        filtered = butter(raw, 4, 45)
        values = [np.log(np.mean(filtered[s : s + WINDOW] ** 2) + 1e-12) for s in starts]
    elif modality.startswith("EOG"):
        filtered = butter(raw, 0.1, 15)
        values = [np.log(np.mean(filtered[s : s + WINDOW] ** 2) + 1e-12) for s in starts]
    elif modality == "PPG":
        filtered = butter(raw, 0.5, 5)
        values = [np.log(np.std(filtered[s : s + WINDOW]) + 1e-12) for s in starts]
    elif modality == "EDA":
        filtered = butter(raw, None, 2)
        values = [np.mean(filtered[s : s + WINDOW]) for s in starts]
    elif modality == "SKT":
        filtered = butter(raw, None, 0.5)
        values = [np.mean(filtered[s : s + WINDOW]) for s in starts]
    else:
        raise ValueError(modality)
    return times, np.asarray(values)


def build_cache(subject_files: list[Path], cache_path: Path) -> dict[str, np.ndarray]:
    raw_rows, ref_rows, descriptor_rows, label_rows, subject_rows, trial_rows = [], [], [], [], [], []
    for subject_index, path in enumerate(tqdm(subject_files, desc="Building temporal cache", unit="subject")):
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        data = np.asarray(payload["data"], dtype=np.float64)
        labels = np.asarray(payload["label"], dtype=np.float64)
        for trial in range(data.shape[0]):
            raw_modalities, ref_modalities, descriptors = [], [], []
            for modality, channel in MODALITIES.items():
                times, values = modality_curve(data[trial, channel], modality)
                baseline = values[times < 0]
                center = float(np.median(baseline))
                mad = float(np.median(np.abs(baseline - center)))
                scale = max(1.4826 * mad, float(np.std(baseline)), 1e-8)
                stimulus = times >= 2
                raw_modalities.append(values[stimulus])
                ref_modalities.append(np.clip((values[stimulus] - center) / scale, -10, 10))
                descriptors.extend((center, np.log(scale)))
            raw_rows.append(np.stack(raw_modalities, axis=1))
            ref_rows.append(np.stack(ref_modalities, axis=1))
            descriptor_rows.append(descriptors)
            label_rows.append((int(labels[trial, 1] >= 5), int(labels[trial, 0] >= 5)))
            subject_rows.append(subject_index)
            trial_rows.append(trial + 1)
    payload = {
        "raw": np.asarray(raw_rows, dtype=np.float32),
        "referenced": np.asarray(ref_rows, dtype=np.float32),
        "descriptors": np.asarray(descriptor_rows, dtype=np.float32),
        "labels": np.asarray(label_rows, dtype=np.int64),
        "subjects": np.asarray(subject_rows, dtype=np.int64),
        "trials": np.asarray(trial_rows, dtype=np.int64),
    }
    np.savez_compressed(cache_path, **payload)
    return payload


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


class TemporalGate(nn.Module):
    def __init__(self, mode: str, modalities: int, descriptor_dim: int, hidden: int) -> None:
        super().__init__()
        self.mode = mode
        self.phase_embedding = nn.Embedding(3, modalities) if mode in {"phase_aware", "baseline_conditioned"} else None
        self.conditioner = (
            nn.Sequential(nn.Linear(descriptor_dim, hidden), nn.ReLU(), nn.Linear(hidden, modalities), nn.Sigmoid())
            if mode == "baseline_conditioned"
            else None
        )
        self.encoder = nn.GRU(modalities, hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(0.25), nn.Linear(hidden * 2, 2))

    def forward(self, sequence: torch.Tensor, descriptors: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        if self.conditioner is not None:
            sequence = sequence * (0.5 + self.conditioner(descriptors).unsqueeze(1))
        if self.phase_embedding is not None:
            sequence = sequence + self.phase_embedding(phases).unsqueeze(0)
        states, _ = self.encoder(sequence)
        weights = torch.softmax(self.attention(states).squeeze(-1), dim=1)
        pooled = (states * weights.unsqueeze(-1)).sum(1)
        return self.classifier(pooled)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    axes = tuple(range(train.ndim - 1))
    mean = train.mean(axis=axes, keepdims=True)
    std = np.maximum(train.std(axis=axes, keepdims=True), 1e-6)
    return tuple(((values - mean) / std).astype(np.float32) for values in (train, *others))


def scores(y: np.ndarray, probability: np.ndarray) -> dict[str, float | bool]:
    prediction = (probability >= 0.5).astype(int)
    both = np.unique(y).size == 2
    return {
        "balanced_accuracy": balanced_accuracy_score(y, prediction) if both else np.nan,
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0) if both else np.nan,
        "mcc": matthews_corrcoef(y, prediction) if both else np.nan,
        "auroc": roc_auc_score(y, probability) if both else np.nan,
        "has_both_test_classes": both,
    }


def fit_one(
    mode: str,
    train_sequence: np.ndarray,
    train_descriptors: np.ndarray,
    train_y: np.ndarray,
    valid_sequence: np.ndarray,
    valid_descriptors: np.ndarray,
    valid_y: np.ndarray,
    phases: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[TemporalGate, int, float]:
    set_seed(seed)
    model = TemporalGate(mode, train_sequence.shape[-1], train_descriptors.shape[-1], args.hidden).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    counts = np.bincount(train_y, minlength=2)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=args.device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_sequence), torch.from_numpy(train_descriptors), torch.from_numpy(train_y)
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_x = torch.from_numpy(valid_sequence).to(args.device)
    valid_d = torch.from_numpy(valid_descriptors).to(args.device)
    valid_y_tensor = torch.from_numpy(valid_y).to(args.device)
    phase_tensor = torch.from_numpy(phases).to(args.device)
    best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for sequence, descriptors, labels in loader:
            sequence, descriptors, labels = sequence.to(args.device), descriptors.to(args.device), labels.to(args.device)
            loss = criterion(model(sequence, descriptors, phase_tensor), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(valid_x, valid_d, phase_tensor), valid_y_tensor).item()
        if valid_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = valid_loss, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def run_loso(payload: dict[str, np.ndarray], args: argparse.Namespace) -> pd.DataFrame:
    subjects = np.unique(payload["subjects"])
    phases = np.where(np.arange(payload["raw"].shape[1]) < 8, 0, np.where(np.arange(payload["raw"].shape[1]) < 28, 1, 2)).astype(np.int64)
    rows = []
    progress = tqdm(total=len(subjects) * len(args.tasks) * len(args.modes), desc="GPU temporal LOSO", unit="model")
    for task_index, task in enumerate(("arousal", "valence")):
        if task not in args.tasks:
            continue
        for held_out in subjects:
            remaining = subjects[subjects != held_out]
            rng = np.random.default_rng(args.seed + int(held_out))
            valid_subjects = rng.choice(remaining, size=min(3, len(remaining) - 1), replace=False)
            train_subjects = remaining[~np.isin(remaining, valid_subjects)]
            train_mask = np.isin(payload["subjects"], train_subjects)
            valid_mask = np.isin(payload["subjects"], valid_subjects)
            test_mask = payload["subjects"] == held_out
            raw_train, raw_valid, raw_test = standardize(
                payload["raw"][train_mask], payload["raw"][valid_mask], payload["raw"][test_mask]
            )
            ref_train, ref_valid, ref_test = standardize(
                payload["referenced"][train_mask], payload["referenced"][valid_mask], payload["referenced"][test_mask]
            )
            descriptor_train, descriptor_valid, descriptor_test = standardize(
                payload["descriptors"][train_mask],
                payload["descriptors"][valid_mask],
                payload["descriptors"][test_mask],
            )
            y_train = payload["labels"][train_mask, task_index]
            y_valid = payload["labels"][valid_mask, task_index]
            y_test = payload["labels"][test_mask, task_index]
            for mode in args.modes:
                sequences = (raw_train, raw_valid, raw_test) if mode == "stimulus_only" else (ref_train, ref_valid, ref_test)
                model, best_epoch, valid_loss = fit_one(
                    mode,
                    sequences[0], descriptor_train, y_train,
                    sequences[1], descriptor_valid, y_valid,
                    phases, args, args.seed + int(held_out) * 17 + task_index * 1000,
                )
                model.eval()
                with torch.no_grad():
                    logits = model(
                        torch.from_numpy(sequences[2]).to(args.device),
                        torch.from_numpy(descriptor_test).to(args.device),
                        torch.from_numpy(phases).to(args.device),
                    )
                    probability = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                rows.append(
                    {
                        "task": task,
                        "held_out_subject": int(held_out) + 1,
                        "mode": mode,
                        "validation_subjects": json.dumps((valid_subjects + 1).tolist()),
                        "best_epoch": best_epoch,
                        "validation_loss": valid_loss,
                        **scores(y_test, probability),
                    }
                )
                progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def paired_tests(results: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("baseline_referenced", "stimulus_only"),
        ("phase_aware", "baseline_referenced"),
        ("baseline_conditioned", "phase_aware"),
    ]
    rows = []
    for task, group in results.groupby("task"):
        wide = group.pivot(index="held_out_subject", columns="mode", values="balanced_accuracy")
        for left, right in comparisons:
            if left not in wide or right not in wide:
                continue
            pair = wide[[left, right]].dropna()
            delta = (pair[left] - pair[right]).to_numpy()
            statistic, p_value = stats.wilcoxon(delta, zero_method="zsplit", mode="approx") if np.any(delta != 0) else (np.nan, np.nan)
            rows.append(
                {
                    "task": task,
                    "left": left,
                    "right": right,
                    "n_subjects": len(delta),
                    "mean_delta_balanced_accuracy": float(np.mean(delta)),
                    "median_delta_balanced_accuracy": float(np.median(delta)),
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_value,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    subject_files = sorted(args.data_root.glob("s*.dat"))
    if args.max_subjects is not None:
        subject_files = subject_files[: args.max_subjects]
    cache_path = args.output_root / ("temporal_cache.npz" if args.max_subjects is None else f"smoke_cache_{len(subject_files)}.npz")
    payload = load_cache(cache_path) if cache_path.exists() else build_cache(subject_files, cache_path)
    results = run_loso(payload, args)
    results.to_csv(args.output_root / f"{args.run_prefix}per_subject_results.csv", index=False)
    summary = results.groupby(["task", "mode"])[["balanced_accuracy", "macro_f1", "mcc", "auroc"]].agg(["mean", "std", "count"])
    summary.to_csv(args.output_root / f"{args.run_prefix}summary.csv")
    paired_tests(results).to_csv(args.output_root / f"{args.run_prefix}paired_tests.csv", index=False)
    print(summary.round(4))
    print(f"Saved temporal gate results to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
