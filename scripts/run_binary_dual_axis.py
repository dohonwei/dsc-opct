from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


CS = (0.01, 0.1, 1.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary subject/video/dual-axis affect generalization.")
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stimulus_shortcut_audit"))
    parser.add_argument("--feature-prefix", default="stimulus__")
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument(
        "--fixed-c",
        type=float,
        default=None,
        help="Use a prespecified C without inner tuning, suitable for external replication.",
    )
    return parser.parse_args()


def make_model(c: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=c, max_iter=3000, class_weight="balanced")),
        ]
    )


def inner_splits(subjects: np.ndarray, trials: np.ndarray, axis: str):
    if axis == "subject":
        return list(GroupKFold(3).split(subjects, groups=subjects))
    if axis == "video":
        return list(GroupKFold(3).split(trials, groups=trials))
    unique_subjects, unique_trials = np.unique(subjects), np.unique(trials)
    subject_fold = {value: index % 3 for index, value in enumerate(unique_subjects)}
    trial_fold = {value: index % 3 for index, value in enumerate(unique_trials)}
    splits = []
    for fold in range(3):
        valid = np.array(
            [subject_fold[s] == fold and trial_fold[t] == fold for s, t in zip(subjects, trials, strict=True)]
        )
        train = np.array(
            [subject_fold[s] != fold and trial_fold[t] != fold for s, t in zip(subjects, trials, strict=True)]
        )
        splits.append((np.flatnonzero(train), np.flatnonzero(valid)))
    return splits


def select_c(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, trials: np.ndarray, axis: str) -> float:
    scores = {c: [] for c in CS}
    for train, valid in inner_splits(subjects, trials, axis):
        for c in CS:
            model = make_model(c).fit(x[train], y[train])
            prediction = model.predict(x[valid])
            scores[c].append(balanced_accuracy_score(y[valid], prediction))
    return max(CS, key=lambda c: (np.mean(scores[c]), -c))


def outer_masks(frame: pd.DataFrame, axis: str):
    if axis == "subject":
        for subject in frame.subject_id.unique():
            test = frame.subject_id == subject
            yield str(subject), ~test, test
    elif axis == "video":
        for trial in frame.trial_id.unique():
            test = frame.trial_id == trial
            yield str(trial), ~test, test
    else:
        observed_pairs = frame[["subject_id", "trial_id"]].drop_duplicates()
        for subject, trial in observed_pairs.itertuples(index=False, name=None):
            test = (frame.subject_id == subject) & (frame.trial_id == trial)
            train = (frame.subject_id != subject) & (frame.trial_id != trial)
            yield f"{subject}|{trial}", train, test


def prior_probability(
    train: pd.DataFrame, test: pd.DataFrame, target: str, axis: str
) -> tuple[str, np.ndarray]:
    if axis == "subject":
        mapping = train.groupby("trial_id")[target].mean()
        return "seen_video_prior", test.trial_id.map(mapping).to_numpy(float)
    if axis == "video":
        mapping = train.groupby("subject_id")[target].mean()
        return "seen_subject_prior", test.subject_id.map(mapping).to_numpy(float)
    return "no_available_prior", np.full(len(test), train[target].mean())


def run(
    frame: pd.DataFrame,
    dataset: str,
    tasks: list[str],
    prefix: str,
    fixed_c: float | None,
) -> pd.DataFrame:
    columns = [column for column in frame if column.startswith(prefix)]
    rows = []
    total = len(tasks) * (
        frame.subject_id.nunique()
        + frame.trial_id.nunique()
        + len(frame[["subject_id", "trial_id"]].drop_duplicates())
    )
    progress = tqdm(total=total, desc=f"{dataset} binary dual-axis", unit="fit", dynamic_ncols=True)
    for task in tasks:
        target = f"{task}_binary"
        frame[target] = (frame[f"{task}_score"] >= 5).astype(int)
        for axis in ("subject", "video", "dual"):
            for fold, train_mask, test_mask in outer_masks(frame, axis):
                train, test = frame.loc[train_mask], frame.loc[test_mask]
                x_train, x_test = train[columns].to_numpy(float), test[columns].to_numpy(float)
                y_train = train[target].to_numpy(int)
                c = fixed_c if fixed_c is not None else select_c(
                    x_train, y_train, train.subject_id.to_numpy(), train.trial_id.to_numpy(), axis
                )
                physiology = make_model(c).fit(x_train, y_train).predict_proba(x_test)[:, 1]
                global_probability = np.full(len(test), y_train.mean())
                prior_name, prior = prior_probability(train, test, target, axis)
                for model_name, probability in (
                    ("global_prevalence", global_probability),
                    (prior_name, prior),
                    ("physiology", physiology),
                ):
                    for index, truth, estimate in zip(test.index, test[target], probability, strict=True):
                        rows.append(
                            {
                                "dataset": dataset,
                                "task": task,
                                "axis": axis,
                                "fold": fold,
                                "subject_id": frame.loc[index, "subject_id"],
                                "trial_id": int(frame.loc[index, "trial_id"]),
                                "model": model_name,
                                "target": int(truth),
                                "probability": float(estimate),
                                "C": c if model_name == "physiology" else np.nan,
                            }
                        )
                progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["dataset", "task", "axis", "model"]):
        per_subject = []
        for _, subject in group.groupby("subject_id"):
            if subject.target.nunique() < 2:
                continue
            label = (subject.probability >= 0.5).astype(int)
            per_subject.append(
                {
                    "balanced_accuracy": balanced_accuracy_score(subject.target, label),
                    "macro_f1": f1_score(subject.target, label, average="macro", zero_division=0),
                    "auroc": roc_auc_score(subject.target, subject.probability),
                }
            )
        metrics = pd.DataFrame(per_subject)
        pooled_label = (group.probability >= 0.5).astype(int)
        rows.append(
            {
                "dataset": keys[0],
                "task": keys[1],
                "axis": keys[2],
                "model": keys[3],
                "pooled_balanced_accuracy": balanced_accuracy_score(group.target, pooled_label),
                "pooled_macro_f1": f1_score(group.target, pooled_label, average="macro", zero_division=0),
                "pooled_auroc": roc_auc_score(group.target, group.probability),
                "subject_mean_balanced_accuracy": metrics.balanced_accuracy.mean(),
                "subject_mean_macro_f1": metrics.macro_f1.mean(),
                "subject_mean_auroc": metrics.auroc.mean(),
                "n_evaluable_subjects": len(metrics),
            }
        )
    return pd.DataFrame(rows)


def paired_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, task, axis), group in predictions.groupby(["dataset", "task", "axis"]):
        subject_scores = []
        for (subject_id, model), subject in group.groupby(["subject_id", "model"]):
            if subject.target.nunique() < 2:
                continue
            score = balanced_accuracy_score(subject.target, subject.probability >= 0.5)
            subject_scores.append({"subject_id": subject_id, "model": model, "score": score})
        wide = pd.DataFrame(subject_scores).pivot(index="subject_id", columns="model", values="score")
        prior = "seen_video_prior" if axis == "subject" else "seen_subject_prior" if axis == "video" else "global_prevalence"
        for left, right in dict.fromkeys((("physiology", "global_prevalence"), ("physiology", prior))):
            pair = wide[[left, right]].dropna()
            delta = pair[left] - pair[right]
            statistic, p_value = stats.wilcoxon(delta, zero_method="zsplit", mode="approx")
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "axis": axis,
                    "left": left,
                    "right": right,
                    "n_subjects": len(pair),
                    "mean_balanced_accuracy_delta": delta.mean(),
                    "p_raw": p_value,
                    "wilcoxon_statistic": statistic,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.feature_file)
    predictions = run(frame, args.dataset, args.tasks, args.feature_prefix, args.fixed_c)
    slug = args.dataset.lower()
    predictions.to_csv(args.output_root / f"{slug}_binary_predictions.csv", index=False)
    summary = summarize(predictions)
    summary.to_csv(args.output_root / f"{slug}_binary_summary.csv", index=False)
    paired_tests(predictions).to_csv(args.output_root / f"{slug}_binary_paired_tests.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
