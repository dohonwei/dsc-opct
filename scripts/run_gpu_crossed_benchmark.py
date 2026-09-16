from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.autograd import Function
from tqdm.auto import tqdm

from stimulus_identity_contract import (
    CROSSED_TRIAL_DATASETS,
    require_axis_contract,
)


DEFAULT_DATASETS = CROSSED_TRIAL_DATASETS


@dataclass(frozen=True)
class TrainConfig:
    hidden_dim: int
    embedding_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    adversary_weight: float
    group_dro_eta: float
    clip_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU crossed subject-stimulus emotion benchmark.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "conditional_stimulus_adversary"],
        choices=["mlp", "stimulus_adversary", "conditional_stimulus_adversary", "group_dro"],
    )
    parser.add_argument("--axes", nargs="+", default=["dual"], choices=["subject", "stimulus", "dual"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260813)
    parser.add_argument("--feature-prefix", default="stimulus__")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--adversary-weight", type=float, default=0.2)
    parser.add_argument("--group-dro-eta", type=float, default=0.05)
    parser.add_argument("--clip-value", type=float, default=10.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/gpu_crossed_benchmark_crossed_valid"),
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * gradient, None


class CrossedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        n_stimuli: int,
        model_name: str,
        adversary_weight: float,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.adversary_weight = adversary_weight
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.emotion_head = nn.Linear(embedding_dim, 1)
        adversary_input = embedding_dim + (1 if model_name == "conditional_stimulus_adversary" else 0)
        self.stimulus_head = nn.Sequential(
            nn.Linear(adversary_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_stimuli),
        )

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        embedding = self.encoder(features)
        emotion_logits = self.emotion_head(embedding).squeeze(1)
        if self.model_name not in {"stimulus_adversary", "conditional_stimulus_adversary"}:
            return emotion_logits, None, embedding
        adversary_input = embedding
        if self.model_name == "conditional_stimulus_adversary":
            if labels is None:
                labels = torch.sigmoid(emotion_logits).detach()
            adversary_input = torch.cat([embedding, labels.float().reshape(-1, 1)], dim=1)
        reversed_embedding = GradientReverse.apply(adversary_input, self.adversary_weight)
        return emotion_logits, self.stimulus_head(reversed_embedding), embedding


def shuffled_fold_map(values: np.ndarray, folds: int, seed: int) -> dict[object, int]:
    unique = np.unique(values)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return {value: index % folds for index, value in enumerate(unique)}


def make_outer_splits(frame: pd.DataFrame, axis: str, folds: int, split_seed: int):
    subject_map = shuffled_fold_map(frame.subject_id.to_numpy(), folds, split_seed)
    stimulus_map = shuffled_fold_map(frame.trial_id.to_numpy(), folds, split_seed + 1)
    subject_fold = frame.subject_id.map(subject_map).to_numpy(int)
    stimulus_fold = frame.trial_id.map(stimulus_map).to_numpy(int)
    if axis == "subject":
        combinations = [(fold, -1) for fold in range(folds)]
    elif axis == "stimulus":
        combinations = [(-1, fold) for fold in range(folds)]
    else:
        combinations = [(subject, stimulus) for subject in range(folds) for stimulus in range(folds)]
    for subject_test, stimulus_test in combinations:
        subject_valid = (subject_test + 1) % folds if subject_test >= 0 else -1
        stimulus_valid = (stimulus_test + 1) % folds if stimulus_test >= 0 else -1
        if axis == "subject":
            test = subject_fold == subject_test
            valid = subject_fold == subject_valid
            train = (subject_fold != subject_test) & (subject_fold != subject_valid)
        elif axis == "stimulus":
            test = stimulus_fold == stimulus_test
            valid = stimulus_fold == stimulus_valid
            train = (stimulus_fold != stimulus_test) & (stimulus_fold != stimulus_valid)
        else:
            test = (subject_fold == subject_test) & (stimulus_fold == stimulus_test)
            valid = (subject_fold == subject_valid) & (stimulus_fold == stimulus_valid)
            train = (
                (subject_fold != subject_test)
                & (subject_fold != subject_valid)
                & (stimulus_fold != stimulus_test)
                & (stimulus_fold != stimulus_valid)
            )
        if not (np.any(train) and np.any(valid) and np.any(test)):
            raise ValueError(f"Empty {axis} split for subject_fold={subject_test}, stimulus_fold={stimulus_test}")
        fold_name = f"s{subject_test}_t{stimulus_test}"
        yield fold_name, np.flatnonzero(train), np.flatnonzero(valid), np.flatnonzero(test)


def prepare_features(
    values: np.ndarray, train_indices: np.ndarray, valid_indices: np.ndarray, test_indices: np.ndarray, clip: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25, 75))
    train = imputer.fit_transform(values[train_indices])
    valid = imputer.transform(values[valid_indices])
    test = imputer.transform(values[test_indices])
    train = scaler.fit_transform(train)
    valid = scaler.transform(valid)
    test = scaler.transform(test)
    return tuple(np.clip(array, -clip, clip).astype(np.float32) for array in (train, valid, test))


def balanced_accuracy(y_true: np.ndarray, probability: np.ndarray) -> float:
    prediction = probability >= 0.5
    recalls = [np.mean(prediction[y_true == label] == label) for label in np.unique(y_true)]
    return float(np.mean(recalls))


def emotion_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = labels.sum().clamp_min(1.0)
    negative = (1.0 - labels).sum().clamp_min(1.0)
    positive_weight = negative / positive
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight)


def train_fold(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    stimulus_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    n_stimuli: int,
    config: TrainConfig,
    device: torch.device,
    seed: int,
) -> tuple[CrossedMLP, int, float]:
    set_seed(seed)
    model = CrossedMLP(
        input_dim=x_train.shape[1],
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        n_stimuli=n_stimuli,
        model_name=model_name,
        adversary_weight=config.adversary_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_x = torch.as_tensor(x_train, device=device)
    train_y = torch.as_tensor(y_train.astype(np.float32), device=device)
    train_stimulus = torch.as_tensor(stimulus_train.astype(np.int64), device=device)
    valid_x = torch.as_tensor(x_valid, device=device)
    valid_y = torch.as_tensor(y_valid.astype(np.float32), device=device)
    group_weights = torch.ones(n_stimuli, device=device) / n_stimuli
    best_loss = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, stimulus_logits, _ = model(train_x, train_y)
        if model_name == "group_dro":
            sample_losses = nn.functional.binary_cross_entropy_with_logits(logits, train_y, reduction="none")
            active_groups = torch.unique(train_stimulus)
            losses = torch.stack([sample_losses[train_stimulus == group].mean() for group in active_groups])
            with torch.no_grad():
                group_weights[active_groups] *= torch.exp(config.group_dro_eta * losses.detach())
                group_weights /= group_weights.sum().clamp_min(1e-12)
            loss = torch.sum(group_weights[active_groups] * losses)
        else:
            loss = emotion_loss(logits, train_y)
        if stimulus_logits is not None:
            loss = loss + nn.functional.cross_entropy(stimulus_logits, train_stimulus)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_logits, _, _ = model(valid_x)
            valid_loss = float(emotion_loss(valid_logits, valid_y).cpu())
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def predict(model: CrossedMLP, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(torch.as_tensor(values, device=device))
    return torch.sigmoid(logits).cpu().numpy()


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "task", "axis", "model", "seed"]
    for values, group in predictions.groupby(keys):
        subject_metrics = []
        for _, subject in group.groupby("subject_id"):
            truth = subject.target.to_numpy(int)
            probability = subject.probability.to_numpy(float)
            if np.unique(truth).size < 2:
                continue
            subject_metrics.append(
                {
                    "balanced_accuracy": balanced_accuracy(truth, probability),
                    "macro_f1": f1_score(truth, probability >= 0.5, average="macro", zero_division=0),
                    "auroc": roc_auc_score(truth, probability),
                }
            )
        subject_frame = pd.DataFrame(subject_metrics)
        truth = group.target.to_numpy(int)
        probability = group.probability.to_numpy(float)
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "pooled_balanced_accuracy": balanced_accuracy(truth, probability),
                "pooled_macro_f1": f1_score(truth, probability >= 0.5, average="macro", zero_division=0),
                "pooled_auroc": roc_auc_score(truth, probability),
                "subject_mean_balanced_accuracy": subject_frame.balanced_accuracy.mean(),
                "subject_mean_macro_f1": subject_frame.macro_f1.mean(),
                "subject_mean_auroc": subject_frame.auroc.mean(),
                "n_evaluable_subjects": len(subject_frame),
            }
        )
    return pd.DataFrame(rows)


def paired_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for keys, group in predictions.groupby(["dataset", "task", "axis", "model", "seed", "subject_id"]):
        truth = group.target.to_numpy(int)
        if np.unique(truth).size < 2:
            continue
        scores.append({
            "dataset": keys[0], "task": keys[1], "axis": keys[2], "model": keys[3],
            "seed": keys[4], "subject_id": keys[5],
            "balanced_accuracy": balanced_accuracy(truth, group.probability.to_numpy(float)),
        })
    score_frame = pd.DataFrame(scores)
    averaged = score_frame.groupby(
        ["dataset", "task", "axis", "model", "subject_id"], as_index=False
    ).balanced_accuracy.mean()
    rows = []
    for keys, group in averaged.groupby(["dataset", "task", "axis"]):
        wide = group.pivot(index="subject_id", columns="model", values="balanced_accuracy")
        if "mlp" not in wide:
            continue
        for model in sorted(set(wide.columns) - {"mlp"}):
            pair = wide[[model, "mlp"]].dropna()
            delta = pair[model] - pair["mlp"]
            statistic, p_value = stats.wilcoxon(delta, zero_method="zsplit", mode="approx")
            rows.append({
                "dataset": keys[0], "task": keys[1], "axis": keys[2], "model": model,
                "reference": "mlp", "n_subjects": len(pair),
                "mean_balanced_accuracy_delta": delta.mean(), "p_raw": p_value,
                "wilcoxon_statistic": statistic,
            })
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["p_holm"] = np.nan
    for _, indices in output.groupby(["dataset", "task", "axis"]).groups.items():
        indices = np.asarray(indices)
        p_values = output.loc[indices, "p_raw"].to_numpy(float)
        order = np.argsort(p_values)
        adjusted = np.empty_like(p_values)
        running = 0.0
        for rank, position in enumerate(order):
            running = max(running, (len(p_values) - rank) * p_values[position])
            adjusted[position] = min(running, 1.0)
        output.loc[indices, "p_holm"] = adjusted
    return output


def main() -> None:
    args = parse_args()
    require_axis_contract(args.datasets, args.axes, context="GPU crossed benchmark")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU is required. Use --allow-cpu only for debugging.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        adversary_weight=args.adversary_weight,
        group_dro_eta=args.group_dro_eta,
        clip_value=args.clip_value,
    )
    dataset_frames = []
    for specification in args.datasets:
        dataset, path = specification.split("=", 1)
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        dataset_frames.append((dataset, frame))
    folds_per_axis = {"subject": args.folds, "stimulus": args.folds, "dual": args.folds**2}
    total = sum(folds_per_axis[axis] for axis in args.axes)
    total *= len(dataset_frames) * len(args.tasks) * len(args.models) * len(args.seeds)
    progress = tqdm(total=total, desc=f"GPU crossed benchmark ({device})", unit="train", dynamic_ncols=True)
    rows = []
    for dataset_index, (dataset, frame) in enumerate(dataset_frames):
        columns = [column for column in frame if column.startswith(args.feature_prefix)]
        if not columns:
            raise ValueError(f"No {args.feature_prefix!r} features found for {dataset}")
        values = frame[columns].to_numpy(float)
        stimulus_values = frame.trial_id.to_numpy(int)
        stimulus_classes = {value: index for index, value in enumerate(sorted(np.unique(stimulus_values)))}
        encoded_stimuli = np.array([stimulus_classes[value] for value in stimulus_values], dtype=int)
        for task_index, task in enumerate(args.tasks):
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
            for axis in args.axes:
                splits = list(make_outer_splits(frame, axis, args.folds, args.split_seed))
                for model_index, model_name in enumerate(args.models):
                    for seed in args.seeds:
                        for fold_index, (fold, train_indices, valid_indices, test_indices) in enumerate(splits):
                            x_train, x_valid, x_test = prepare_features(
                                values, train_indices, valid_indices, test_indices, config.clip_value
                            )
                            fold_seed = (
                                seed + dataset_index * 100_000 + task_index * 10_000
                                + fold_index
                            )
                            model, best_epoch, valid_loss = train_fold(
                                model_name,
                                x_train,
                                labels[train_indices],
                                encoded_stimuli[train_indices],
                                x_valid,
                                labels[valid_indices],
                                len(stimulus_classes),
                                config,
                                device,
                                fold_seed,
                            )
                            probabilities = predict(model, x_test, device)
                            for index, probability in zip(test_indices, probabilities, strict=True):
                                rows.append({
                                    "dataset": dataset, "task": task, "axis": axis, "model": model_name,
                                    "seed": seed, "fold": fold, "subject_id": str(frame.iloc[index].subject_id),
                                    "trial_id": int(frame.iloc[index].trial_id), "target": int(labels[index]),
                                    "probability": float(probability), "best_epoch": best_epoch,
                                    "validation_loss": valid_loss,
                                })
                            progress.update(1)
    progress.close()
    predictions = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    summary = summarize(predictions)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    paired_tests(predictions).to_csv(args.output_root / "paired_tests.csv", index=False)
    manifest = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "datasets": args.datasets,
        "tasks": args.tasks,
        "models": args.models,
        "axes": args.axes,
        "seeds": args.seeds,
        "folds": args.folds,
        "split_seed": args.split_seed,
        "feature_prefix": args.feature_prefix,
        "config": asdict(config),
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
