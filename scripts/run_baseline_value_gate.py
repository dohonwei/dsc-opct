from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


FS = 100
BASELINE_SAMPLES = 1000
WINDOW = 200
STEP = 100
CHANNELS = {
    "FP1": (0,),
    "FP2": (1,),
    "EDA": (2,),
    "PPG": (3,),
    "SKT": (4,),
    "EOG8_candidate": (8,),
    "EOG9_candidate": (9,),
}
PHASES = {"early": (2.0, 10.0), "middle": (10.0, 30.0), "late": (30.0, 60.0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict LOSO gate for EPPVR baseline value.")
    parser.add_argument("--data-root", type=Path, default=Path(r"E:\AA发表论文的数据\dataset\EPPVR"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/baseline_value_gate"))
    parser.add_argument("--models", nargs="+", default=["logistic", "ridge", "extra_trees"])
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--result-prefix", default="", help="Prefix result files without replacing cached features.")
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


def window_feature(raw: np.ndarray, modality: str) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, raw.size - WINDOW + 1, STEP)
    times = (starts + WINDOW / 2 - BASELINE_SAMPLES) / FS
    if modality in {"FP1", "FP2"}:
        filtered = butter(raw, 4.0, 45.0)
        values = np.array([np.log(np.mean(filtered[s : s + WINDOW] ** 2) + 1e-12) for s in starts])
    elif modality.startswith("EOG"):
        filtered = butter(raw, 0.1, 15.0)
        values = np.array([np.log(np.mean(filtered[s : s + WINDOW] ** 2) + 1e-12) for s in starts])
    elif modality == "PPG":
        filtered = butter(raw, 0.5, 5.0)
        values = np.array([np.log(np.std(filtered[s : s + WINDOW]) + 1e-12) for s in starts])
    elif modality == "EDA":
        filtered = butter(raw, None, 2.0)
        values = np.array([np.mean(filtered[s : s + WINDOW]) for s in starts])
    elif modality == "SKT":
        filtered = butter(raw, None, 0.5)
        values = np.array([np.mean(filtered[s : s + WINDOW]) for s in starts])
    else:
        raise ValueError(modality)
    return times, values


def robust_reference(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return center, max(1.4826 * mad, float(np.std(values)), 1e-8)


def summaries(values: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.arange(values.size, dtype=float)
    slope = stats.linregress(x, values).slope if values.size > 1 else 0.0
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.1)),
        f"{prefix}_q90": float(np.quantile(values, 0.9)),
        f"{prefix}_slope": float(slope),
    }


def extract_features(subject_files: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for subject_file in tqdm(subject_files, desc="Extracting baseline gate features", unit="subject"):
        with subject_file.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        data = np.asarray(payload["data"], dtype=np.float64)
        labels = np.asarray(payload["label"], dtype=np.float64)
        for trial in range(data.shape[0]):
            row: dict[str, float | int | str] = {
                "subject_id": subject_file.stem,
                "trial_id": trial + 1,
                "valence_score": float(labels[trial, 0]),
                "arousal_score": float(labels[trial, 1]),
                "valence": int(labels[trial, 0] >= 5),
                "arousal": int(labels[trial, 1] >= 5),
            }
            for modality, indices in CHANNELS.items():
                times, raw_feature = window_feature(data[trial, indices[0]], modality)
                baseline = raw_feature[times < 0]
                center, scale = robust_reference(baseline)
                referenced = np.clip((raw_feature - center) / scale, -10, 10)
                stimulus = times >= 2
                row.update(summaries(raw_feature[stimulus], f"stimulus__{modality}"))
                row.update(summaries(referenced[stimulus], f"baseline__{modality}"))
                for phase, (start, end) in PHASES.items():
                    mask = (times >= start) & (times < end)
                    row.update(summaries(referenced[mask], f"phase__{modality}__{phase}"))
                onset_mask = times >= 2
                onset_times, onset_values = times[onset_mask], referenced[onset_mask]
                sustained = np.convolve((np.abs(onset_values) >= 2).astype(int), np.ones(3, dtype=int), mode="valid")
                hits = np.flatnonzero(sustained == 3)
                row[f"timing__{modality}__onset"] = float(onset_times[hits[0]]) if len(hits) else np.nan
                peak = int(np.argmax(np.abs(onset_values)))
                row[f"timing__{modality}__peak_time"] = float(onset_times[peak])
                row[f"timing__{modality}__peak_abs"] = float(abs(onset_values[peak]))
            rows.append(row)
    return pd.DataFrame(rows)


def representations(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "stimulus_only": [c for c in frame if c.startswith("stimulus__")],
        "baseline_referenced": [c for c in frame if c.startswith("baseline__")],
        "phase_aware": [c for c in frame if c.startswith("phase__")],
        "phase_plus_timing": [c for c in frame if c.startswith(("phase__", "timing__"))],
    }


def model_space(name: str, seed: int) -> tuple[Pipeline, dict[str, list[object]]]:
    if name == "logistic":
        estimator = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    elif name == "ridge":
        estimator = RidgeClassifier(class_weight="balanced")
        grid = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=1
        )
        grid = {"model__max_depth": [3, 6, None], "model__min_samples_leaf": [1, 3, 6]}
    else:
        raise ValueError(f"Unknown model: {name}")
    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", estimator)]
    )
    return pipeline, grid


def probabilities(model: Pipeline, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(values)[:, 1]
    scores = model.decision_function(values)
    return 1 / (1 + np.exp(-np.clip(scores, -30, 30)))


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | bool]:
    prediction = (probability >= 0.5).astype(int)
    both = np.unique(y).size == 2
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if both else np.nan,
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)) if both else np.nan,
        "mcc": float(matthews_corrcoef(y, prediction)) if both else np.nan,
        "auroc": float(roc_auc_score(y, probability)) if both else np.nan,
        "has_both_test_classes": both,
    }


def run_loso(frame: pd.DataFrame, model_names: list[str], tasks: list[str], seed: int, n_jobs: int) -> pd.DataFrame:
    feature_sets = representations(frame)
    subjects = frame.subject_id.unique()
    rows: list[dict[str, object]] = []
    total = len(subjects) * len(feature_sets) * len(model_names) * len(tasks)
    progress = tqdm(total=total, desc="Strict nested LOSO gate", unit="fit", dynamic_ncols=True)
    for task in tasks:
        for held_out in subjects:
            train_mask = frame.subject_id != held_out
            test_mask = ~train_mask
            groups = frame.loc[train_mask, "subject_id"].to_numpy()
            inner_cv = GroupKFold(n_splits=3)
            for representation, columns in feature_sets.items():
                train_x = frame.loc[train_mask, columns].to_numpy()
                test_x = frame.loc[test_mask, columns].to_numpy()
                train_y = frame.loc[train_mask, task].to_numpy()
                test_y = frame.loc[test_mask, task].to_numpy()
                for model_name in model_names:
                    pipeline, grid = model_space(model_name, seed)
                    search = GridSearchCV(
                        pipeline,
                        grid,
                        scoring="balanced_accuracy",
                        cv=inner_cv,
                        n_jobs=n_jobs,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(train_x, train_y, groups=groups)
                    probability = probabilities(search.best_estimator_, test_x)
                    rows.append(
                        {
                            "task": task,
                            "held_out_subject": held_out,
                            "representation": representation,
                            "model": model_name,
                            "n_features": len(columns),
                            "inner_best_balanced_accuracy": search.best_score_,
                            "best_params": str(search.best_params_),
                            **metrics(test_y, probability),
                        }
                    )
                    progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("baseline_referenced", "stimulus_only"),
        ("phase_aware", "baseline_referenced"),
        ("phase_plus_timing", "phase_aware"),
    ]
    for (task, model), group in results.groupby(["task", "model"]):
        for left, right in comparisons:
            wide = group.pivot(index="held_out_subject", columns="representation", values="balanced_accuracy")
            pair = wide[[left, right]].dropna()
            delta = (pair[left] - pair[right]).to_numpy()
            if len(delta) >= 8 and np.any(delta != 0):
                statistic, p_value = stats.wilcoxon(delta, zero_method="zsplit", mode="approx")
            else:
                statistic, p_value = np.nan, np.nan
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "left": left,
                    "right": right,
                    "n_subjects": len(delta),
                    "mean_delta_balanced_accuracy": float(np.mean(delta)) if len(delta) else np.nan,
                    "median_delta_balanced_accuracy": float(np.median(delta)) if len(delta) else np.nan,
                    "wilcoxon_statistic": statistic,
                    "p_raw": p_value,
                }
            )
    output = pd.DataFrame(rows)
    output["p_holm_within_task"] = np.nan
    for task, index in output.groupby("task").groups.items():
        p = output.loc[index, "p_raw"].to_numpy(float)
        finite = np.isfinite(p)
        if finite.any():
            order = np.argsort(p[finite])
            adjusted = np.empty(finite.sum())
            running = 0.0
            for rank, position in enumerate(order):
                running = max(running, (finite.sum() - rank) * p[finite][position])
                adjusted[position] = min(running, 1.0)
            output.loc[np.asarray(index)[finite], "p_holm_within_task"] = adjusted
    return output


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    subject_files = sorted(args.data_root.glob("s*.dat"))
    if args.max_subjects is not None:
        subject_files = subject_files[: args.max_subjects]
    if len(subject_files) < 4:
        raise ValueError("At least four subjects are required for nested LOSO.")
    feature_path = args.output_root / "trial_features.csv"
    if feature_path.exists() and args.max_subjects is None:
        frame = pd.read_csv(feature_path)
    else:
        frame = extract_features(subject_files)
        if args.max_subjects is None:
            frame.to_csv(feature_path, index=False)
    results = run_loso(frame, args.models, args.tasks, args.seed, args.n_jobs)
    results.to_csv(args.output_root / f"{args.result_prefix}per_subject_results.csv", index=False)
    summary = results.groupby(["task", "representation", "model"])[
        ["balanced_accuracy", "macro_f1", "mcc", "auroc"]
    ].agg(["mean", "std", "count"])
    summary.to_csv(args.output_root / f"{args.result_prefix}summary.csv")
    paired_deltas(results).to_csv(
        args.output_root / f"{args.result_prefix}paired_representation_deltas.csv", index=False
    )
    print(summary.round(4))
    print(f"Saved baseline value gate to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
