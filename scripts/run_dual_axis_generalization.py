from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


REPRESENTATIONS = {
    "stimulus_only": "stimulus__",
    "baseline_referenced": "baseline__",
    "phase_aware": "phase__",
}
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subject, video, and dual-axis EPPVR generalization audit.")
    parser.add_argument(
        "--feature-file", type=Path, default=Path("outputs/baseline_value_gate/trial_features.csv")
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dual_axis_generalization"))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--representations", nargs="+", default=list(REPRESENTATIONS), choices=list(REPRESENTATIONS))
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="Use a prespecified Ridge alpha without inner tuning, suitable for external replication.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def inner_splits(subjects: np.ndarray, trials: np.ndarray, axis: str) -> list[tuple[np.ndarray, np.ndarray]]:
    if axis == "subject":
        return list(GroupKFold(3).split(subjects, groups=subjects))
    if axis == "video":
        return list(GroupKFold(3).split(trials, groups=trials))
    if axis != "dual":
        raise ValueError(axis)
    unique_subjects = np.unique(subjects)
    unique_trials = np.unique(trials)
    subject_fold = {value: index % 3 for index, value in enumerate(unique_subjects)}
    trial_fold = {value: index % 3 for index, value in enumerate(unique_trials)}
    splits = []
    for fold in range(3):
        validation = np.asarray(
            [subject_fold[s] == fold and trial_fold[t] == fold for s, t in zip(subjects, trials, strict=True)]
        )
        training = np.asarray(
            [subject_fold[s] != fold and trial_fold[t] != fold for s, t in zip(subjects, trials, strict=True)]
        )
        splits.append((np.flatnonzero(training), np.flatnonzero(validation)))
    return splits


def select_alpha(
    x: np.ndarray, y: np.ndarray, subjects: np.ndarray, trials: np.ndarray, axis: str
) -> float:
    losses = {alpha: [] for alpha in ALPHAS}
    for train_index, valid_index in inner_splits(subjects, trials, axis):
        for alpha in ALPHAS:
            prediction = make_ridge(alpha).fit(x[train_index], y[train_index]).predict(x[valid_index])
            losses[alpha].append(mean_absolute_error(y[valid_index], prediction))
    return min(ALPHAS, key=lambda alpha: (np.mean(losses[alpha]), alpha))


def outer_masks(frame: pd.DataFrame, axis: str):
    subjects = frame.subject_id.unique()
    trials = frame.trial_id.unique()
    if axis == "subject":
        for subject in subjects:
            test = frame.subject_id == subject
            yield str(subject), ~test, test
    elif axis == "video":
        for trial in trials:
            test = frame.trial_id == trial
            yield str(trial), ~test, test
    elif axis == "dual":
        observed_pairs = frame[["subject_id", "trial_id"]].drop_duplicates()
        for subject, trial in observed_pairs.itertuples(index=False, name=None):
            test = (frame.subject_id == subject) & (frame.trial_id == trial)
            train = (frame.subject_id != subject) & (frame.trial_id != trial)
            yield f"{subject}|{trial}", train, test
    else:
        raise ValueError(axis)


def concordance(y: np.ndarray, prediction: np.ndarray) -> float:
    covariance = np.mean((y - y.mean()) * (prediction - prediction.mean()))
    denominator = y.var() + prediction.var() + (y.mean() - prediction.mean()) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def run_regression(
    frame: pd.DataFrame,
    tasks: list[str],
    representations: list[str],
    fixed_alpha: float | None = None,
) -> pd.DataFrame:
    rows = []
    folds_per_axis = {
        "subject": frame.subject_id.nunique(),
        "video": frame.trial_id.nunique(),
        "dual": len(frame[["subject_id", "trial_id"]].drop_duplicates()),
    }
    total = sum(folds_per_axis.values()) * len(tasks) * len(representations)
    progress = tqdm(total=total, desc="Dual-axis generalization", unit="fit", dynamic_ncols=True)
    for task in tasks:
        target_column = f"{task}_score"
        for representation in representations:
            columns = [column for column in frame if column.startswith(REPRESENTATIONS[representation])]
            for axis in ("subject", "video", "dual"):
                for fold, train_mask, test_mask in outer_masks(frame, axis):
                    train = frame.loc[train_mask]
                    test = frame.loc[test_mask]
                    x_train = train[columns].to_numpy(float)
                    x_test = test[columns].to_numpy(float)
                    y_train = train[target_column].to_numpy(float)
                    alpha = fixed_alpha if fixed_alpha is not None else select_alpha(
                        x_train,
                        y_train,
                        train.subject_id.to_numpy(),
                        train.trial_id.to_numpy(),
                        axis,
                    )
                    physiology = make_ridge(alpha).fit(x_train, y_train).predict(x_test)
                    global_mean = np.full(len(test), y_train.mean())
                    if axis == "subject":
                        known_prior = train.groupby("trial_id")[target_column].mean()
                        prior = test.trial_id.map(known_prior).to_numpy(float)
                        prior_name = "seen_video_prior"
                    elif axis == "video":
                        known_prior = train.groupby("subject_id")[target_column].mean()
                        prior = test.subject_id.map(known_prior).to_numpy(float)
                        prior_name = "seen_subject_prior"
                    else:
                        prior = global_mean
                        prior_name = "no_available_prior"
                    for model_name, prediction in (
                        ("global_mean", global_mean),
                        (prior_name, prior),
                        ("physiology", physiology),
                    ):
                        for row_index, truth, estimate in zip(test.index, test[target_column], prediction, strict=True):
                            rows.append(
                                {
                                    "task": task,
                                    "representation": representation,
                                    "axis": axis,
                                    "fold": fold,
                                    "subject_id": frame.loc[row_index, "subject_id"],
                                    "trial_id": int(frame.loc[row_index, "trial_id"]),
                                    "model": model_name,
                                    "alpha": alpha if model_name == "physiology" else np.nan,
                                    "target": float(truth),
                                    "prediction": float(np.clip(estimate, 1, 9)),
                                }
                            )
                    progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["task", "representation", "axis", "model"]):
        task, representation, axis, model = keys
        subject_metrics = []
        for _, subject in group.groupby("subject_id"):
            rho = (
                stats.spearmanr(subject.target, subject.prediction).statistic
                if np.ptp(subject.prediction.to_numpy()) > 1e-10
                else np.nan
            )
            subject_metrics.append(
                {
                    "mae": mean_absolute_error(subject.target, subject.prediction),
                    "spearman": rho if np.isfinite(rho) else np.nan,
                    "ccc": concordance(subject.target.to_numpy(), subject.prediction.to_numpy()),
                }
            )
        subject_metrics = pd.DataFrame(subject_metrics)
        rows.append(
            {
                "task": task,
                "representation": representation,
                "axis": axis,
                "model": model,
                "pooled_mae": mean_absolute_error(group.target, group.prediction),
                "subject_mean_mae": subject_metrics.mae.mean(),
                "subject_mean_spearman": subject_metrics.spearman.mean(),
                "subject_mean_ccc": subject_metrics.ccc.mean(),
                "n_predictions": len(group),
            }
        )
    return pd.DataFrame(rows)


def paired_subject_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, representation, axis), group in predictions.groupby(["task", "representation", "axis"]):
        errors = (
            group.assign(abs_error=lambda x: np.abs(x.target - x.prediction))
            .groupby(["subject_id", "model"], as_index=False).abs_error.mean()
            .pivot(index="subject_id", columns="model", values="abs_error")
        )
        baseline = "seen_video_prior" if axis == "subject" else "seen_subject_prior" if axis == "video" else "global_mean"
        comparisons = list(dict.fromkeys((("physiology", "global_mean"), ("physiology", baseline))))
        for left, right in comparisons:
            pair = errors[[left, right]].dropna()
            improvement = pair[right] - pair[left]
            statistic, p_value = stats.wilcoxon(improvement, zero_method="zsplit", mode="approx")
            rows.append(
                {
                    "task": task,
                    "representation": representation,
                    "axis": axis,
                    "left": left,
                    "right": right,
                    "n_subjects": len(pair),
                    "mean_mae_improvement": improvement.mean(),
                    "p_raw": p_value,
                    "wilcoxon_statistic": statistic,
                }
            )
    output = pd.DataFrame(rows)
    output["p_holm_within_task"] = np.nan
    for _, indices in output.groupby("task").groups.items():
        indices = np.asarray(indices)
        p = output.loc[indices, "p_raw"].to_numpy(float)
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        running = 0.0
        for rank, position in enumerate(order):
            running = max(running, (len(p) - rank) * p[position])
            adjusted[position] = min(running, 1.0)
        output.loc[indices, "p_holm_within_task"] = adjusted
    return output


def video_identity_audit(frame: pd.DataFrame, representations: list[str], n_jobs: int) -> pd.DataFrame:
    rows = []
    subjects = frame.subject_id.unique()
    progress = tqdm(total=len(subjects) * len(representations), desc="Video identity LOSO", unit="fit")
    for representation in representations:
        columns = [column for column in frame if column.startswith(REPRESENTATIONS[representation])]
        for held_out in subjects:
            train = frame.subject_id != held_out
            test = ~train
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")),
                ]
            )
            model.fit(frame.loc[train, columns], frame.loc[train, "trial_id"])
            prediction = model.predict(frame.loc[test, columns])
            truth = frame.loc[test, "trial_id"].to_numpy()
            recalls = [np.mean(prediction[truth == label] == label) for label in np.unique(truth)]
            rows.append(
                {
                    "representation": representation,
                    "held_out_subject": held_out,
                    "accuracy": accuracy_score(frame.loc[test, "trial_id"], prediction),
                    "balanced_accuracy": float(np.mean(recalls)),
                    "chance": 1 / frame.trial_id.nunique(),
                }
            )
            progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.feature_file)
    predictions = run_regression(frame, args.tasks, args.representations, args.fixed_alpha)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    summarize_predictions(predictions).to_csv(args.output_root / "summary.csv", index=False)
    paired_subject_tests(predictions).to_csv(args.output_root / "paired_tests.csv", index=False)
    video_audit = video_identity_audit(frame, args.representations, args.n_jobs)
    video_audit.to_csv(args.output_root / "video_identity_loso.csv", index=False)
    print(summarize_predictions(predictions).to_string(index=False))
    print("\nVideo identity balanced accuracy:")
    print(video_audit.groupby("representation").balanced_accuracy.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
