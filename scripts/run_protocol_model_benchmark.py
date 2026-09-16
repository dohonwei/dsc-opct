from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

from stimulus_identity_contract import CROSSED_TRIAL_DATASETS, require_crossed_datasets


DEFAULT_DATASETS = CROSSED_TRIAL_DATASETS
PROTOCOLS = ("random", "subject", "stimulus", "dual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-model protocol sensitivity benchmark.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear_logistic", "rbf_svm", "extra_trees", "hist_gradient_boosting"],
        choices=["linear_logistic", "rbf_svm", "extra_trees", "hist_gradient_boosting"],
    )
    parser.add_argument("--feature-prefix", default="stimulus__")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/protocol_model_benchmark_crossed_valid"),
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=None,
        help="Analyze an existing prediction CSV without fitting models.",
    )
    return parser.parse_args()


def balanced_accuracy(y_true: np.ndarray, probability: np.ndarray, weights: np.ndarray | None = None) -> float:
    prediction = probability >= 0.5
    recalls = []
    for label in (0, 1):
        mask = y_true == label
        if not np.any(mask):
            return np.nan
        if weights is None:
            recalls.append(np.mean(prediction[mask] == label))
        else:
            label_weights = weights[mask]
            denominator = label_weights.sum()
            if denominator <= 0:
                return np.nan
            recalls.append(np.sum(label_weights * (prediction[mask] == label)) / denominator)
    return float(np.mean(recalls))


def clip_scaled_features(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -10.0, 10.0)


def make_models(seed: int, n_jobs: int) -> dict[str, Pipeline]:
    preprocess = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clip", FunctionTransformer(clip_scaled_features)),
    ]
    return {
        "linear_logistic": Pipeline(
            preprocess
            + [("model", LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=seed))]
        ),
        "rbf_svm": Pipeline(
            preprocess
            + [("model", SVC(C=1.0, gamma="scale", class_weight="balanced", probability=False, random_state=seed))]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=10,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def shuffled_fold_map(values: np.ndarray, folds: int, seed: int) -> dict[object, int]:
    unique = np.unique(values)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return {value: index % folds for index, value in enumerate(unique)}


def make_splits(frame: pd.DataFrame, labels: np.ndarray, protocol: str, folds: int, seed: int):
    if protocol == "random":
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(frame, labels)):
            yield str(fold), train, test
        return
    subject_map = shuffled_fold_map(frame.subject_id.to_numpy(), folds, seed)
    stimulus_map = shuffled_fold_map(frame.trial_id.to_numpy(), folds, seed + 1)
    subject_fold = frame.subject_id.map(subject_map).to_numpy(int)
    stimulus_fold = frame.trial_id.map(stimulus_map).to_numpy(int)
    if protocol == "subject":
        for fold in range(folds):
            test = subject_fold == fold
            yield str(fold), np.flatnonzero(~test), np.flatnonzero(test)
    elif protocol == "stimulus":
        for fold in range(folds):
            test = stimulus_fold == fold
            yield str(fold), np.flatnonzero(~test), np.flatnonzero(test)
    else:
        for subject_test in range(folds):
            for stimulus_test in range(folds):
                test = (subject_fold == subject_test) & (stimulus_fold == stimulus_test)
                train = (subject_fold != subject_test) & (stimulus_fold != stimulus_test)
                if not np.any(test) or not np.any(train):
                    raise ValueError(
                        f"Empty dual split for subject_fold={subject_test}, stimulus_fold={stimulus_test}"
                    )
                yield (
                    f"s{subject_test}_t{stimulus_test}",
                    np.flatnonzero(train),
                    np.flatnonzero(test),
                )


def model_probability(model: Pipeline, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    decision = model.decision_function(features)
    return expit(np.asarray(decision, dtype=float))


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    specifications = []
    for item in args.datasets:
        dataset, path = item.split("=", 1)
        specifications.append((dataset, Path(path)))
    models = {
        name: model for name, model in make_models(args.seed, args.n_jobs).items() if name in args.models
    }
    fold_counts = {"random": args.folds, "subject": args.folds, "stimulus": args.folds, "dual": args.folds**2}
    total = len(specifications) * len(args.tasks) * len(models) * sum(fold_counts[p] for p in args.protocols)
    progress = tqdm(total=total, desc="Protocol-model benchmark", unit="fit", dynamic_ncols=True)
    rows = []
    for dataset, path in specifications:
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["row_id"] = np.arange(len(frame))
        feature_columns = [column for column in frame if column.startswith(args.feature_prefix)]
        if not feature_columns:
            raise ValueError(f"No {args.feature_prefix!r} columns for {dataset}")
        features = frame[feature_columns].to_numpy(float)
        for task in args.tasks:
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
            for protocol in args.protocols:
                for fold, train, test in make_splits(frame, labels, protocol, args.folds, args.seed):
                    if np.unique(labels[train]).size < 2:
                        raise ValueError(f"Single-class training set: {dataset}/{task}/{protocol}/{fold}")
                    global_probability = np.full(len(test), labels[train].mean())
                    stimulus_prior = pd.Series(labels[train]).groupby(frame.iloc[train].trial_id.to_numpy()).mean()
                    mapped_prior = frame.iloc[test].trial_id.map(stimulus_prior).to_numpy(float)
                    prior_available = np.isfinite(mapped_prior)
                    mapped_prior[~prior_available] = labels[train].mean()
                    for baseline_name, probability in (
                        ("global_prevalence", global_probability),
                        ("stimulus_prior", mapped_prior),
                    ):
                        for index, estimate, available in zip(test, probability, prior_available, strict=True):
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "task": task,
                                    "protocol": protocol,
                                    "model": baseline_name,
                                    "fold": fold,
                                    "row_id": int(frame.iloc[index].row_id),
                                    "subject_id": str(frame.iloc[index].subject_id),
                                    "trial_id": int(frame.iloc[index].trial_id),
                                    "target": int(labels[index]),
                                    "probability": float(estimate),
                                    "stimulus_prior_available": bool(available),
                                }
                            )
                    for model_name, template in models.items():
                        model = clone(template).fit(features[train], labels[train])
                        probability = model_probability(model, features[test])
                        for index, estimate in zip(test, probability, strict=True):
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "task": task,
                                    "protocol": protocol,
                                    "model": model_name,
                                    "fold": fold,
                                    "row_id": int(frame.iloc[index].row_id),
                                    "subject_id": str(frame.iloc[index].subject_id),
                                    "trial_id": int(frame.iloc[index].trial_id),
                                    "target": int(labels[index]),
                                    "probability": float(estimate),
                                    "stimulus_prior_available": False,
                                }
                            )
                        progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["dataset", "task", "protocol", "model"]):
        truth = group.target.to_numpy(int)
        probability = group.probability.to_numpy(float)
        per_subject = []
        for _, subject in group.groupby("subject_id"):
            subject_truth = subject.target.to_numpy(int)
            subject_probability = subject.probability.to_numpy(float)
            if np.unique(subject_truth).size < 2:
                continue
            per_subject.append(
                {
                    "balanced_accuracy": balanced_accuracy(subject_truth, subject_probability),
                    "macro_f1": f1_score(
                        subject_truth, subject_probability >= 0.5, average="macro", zero_division=0
                    ),
                    "auroc": roc_auc_score(subject_truth, subject_probability),
                }
            )
        subject_frame = pd.DataFrame(per_subject)
        rows.append(
            {
                "dataset": keys[0],
                "task": keys[1],
                "protocol": keys[2],
                "model": keys[3],
                "pooled_balanced_accuracy": balanced_accuracy(truth, probability),
                "pooled_macro_f1": f1_score(truth, probability >= 0.5, average="macro", zero_division=0),
                "pooled_auroc": roc_auc_score(truth, probability),
                "subject_mean_balanced_accuracy": subject_frame.balanced_accuracy.mean(),
                "subject_mean_macro_f1": subject_frame.macro_f1.mean(),
                "subject_mean_auroc": subject_frame.auroc.mean(),
                "n_evaluable_subjects": len(subject_frame),
                "stimulus_prior_coverage": group.stimulus_prior_available.mean() if keys[3] == "stimulus_prior" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def two_way_cluster_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    merged = left.merge(
        right,
        on=["row_id", "subject_id", "trial_id", "target"],
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    truth = merged.target.to_numpy(int)
    left_probability = merged.probability_left.to_numpy(float)
    right_probability = merged.probability_right.to_numpy(float)
    observed = balanced_accuracy(truth, left_probability) - balanced_accuracy(truth, right_probability)
    subject_codes, subjects = pd.factorize(merged.subject_id, sort=True)
    stimulus_codes, stimuli = pd.factorize(merged.trial_id, sort=True)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=float)
    valid = 0
    for _ in range(repetitions):
        subject_frequency = np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects))
        stimulus_frequency = np.bincount(rng.integers(0, len(stimuli), len(stimuli)), minlength=len(stimuli))
        weights = subject_frequency[subject_codes] * stimulus_frequency[stimulus_codes]
        left_score = balanced_accuracy(truth, left_probability, weights)
        right_score = balanced_accuracy(truth, right_probability, weights)
        if np.isfinite(left_score) and np.isfinite(right_score):
            deltas[valid] = left_score - right_score
            valid += 1
    if valid < repetitions * 0.95:
        raise RuntimeError(f"Only {valid}/{repetitions} valid two-way bootstrap replicates")
    deltas = deltas[:valid]
    return {
        "balanced_accuracy_delta": observed,
        "ci_low": np.quantile(deltas, 0.025),
        "ci_high": np.quantile(deltas, 0.975),
        "p_bootstrap": 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)),
        "bootstrap_valid": valid,
    }


def protocol_contrasts(predictions: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows = []
    models = sorted(set(predictions.model) - {"global_prevalence", "stimulus_prior"})
    comparisons = (("random", "dual"), ("subject", "dual"), ("stimulus", "dual"))
    groups = list(predictions.groupby(["dataset", "task"]))
    progress = tqdm(
        total=len(groups) * len(models) * len(comparisons),
        desc="Two-way cluster bootstrap",
        unit="contrast",
        dynamic_ncols=True,
    )
    for group_index, ((dataset, task), group) in enumerate(groups):
        for model_index, model in enumerate(models):
            model_group = group.loc[group.model == model]
            for comparison_index, (left_protocol, right_protocol) in enumerate(comparisons):
                result = two_way_cluster_bootstrap(
                    model_group.loc[model_group.protocol == left_protocol],
                    model_group.loc[model_group.protocol == right_protocol],
                    repetitions,
                    seed + group_index * 1000 + model_index * 10 + comparison_index,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "model": model,
                        "left_protocol": left_protocol,
                        "right_protocol": right_protocol,
                        **result,
                    }
                )
                progress.update(1)
    progress.close()
    output = pd.DataFrame(rows)
    output["p_holm_global"] = np.nan
    p_values = output.p_bootstrap.to_numpy(float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[position])
        adjusted[position] = min(running, 1.0)
    output["p_holm_global"] = adjusted
    return output


def panel_protocol_contrasts(predictions: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    candidate_models = sorted(set(predictions.model) - {"global_prevalence", "stimulus_prior"})
    comparisons = (("random", "dual"), ("subject", "dual"), ("stimulus", "dual"))
    rows = []
    groups = list(predictions.groupby(["dataset", "task"]))
    progress = tqdm(
        total=len(groups) * len(comparisons),
        desc="Panel two-way bootstrap",
        unit="contrast",
        dynamic_ncols=True,
    )
    for group_index, ((dataset, task), group) in enumerate(groups):
        reference = group.loc[(group.protocol == "dual") & group.model.isin(candidate_models)]
        for comparison_index, (left_protocol, right_protocol) in enumerate(comparisons):
            left = group.loc[(group.protocol == left_protocol) & group.model.isin(candidate_models)]
            merged = left.merge(
                reference,
                on=["model", "row_id", "subject_id", "trial_id", "target"],
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            subject_codes, subjects = pd.factorize(merged.subject_id, sort=True)
            stimulus_codes, stimuli = pd.factorize(merged.trial_id, sort=True)
            truth = merged.target.to_numpy(int)
            model_indices = {
                model: np.flatnonzero(merged.model.to_numpy() == model) for model in candidate_models
            }

            def panel_delta(weights: np.ndarray | None = None) -> tuple[float, np.ndarray]:
                deltas = []
                for model in candidate_models:
                    indices = model_indices[model]
                    model_weights = None if weights is None else weights[indices]
                    left_score = balanced_accuracy(
                        truth[indices], merged.probability_left.to_numpy(float)[indices], model_weights
                    )
                    right_score = balanced_accuracy(
                        truth[indices], merged.probability_right.to_numpy(float)[indices], model_weights
                    )
                    deltas.append(left_score - right_score)
                values = np.asarray(deltas, dtype=float)
                return float(np.mean(values)), values

            observed, model_deltas = panel_delta()
            rng = np.random.default_rng(seed + group_index * 100 + comparison_index)
            bootstrap = np.empty(repetitions, dtype=float)
            valid = 0
            for _ in range(repetitions):
                subject_frequency = np.bincount(
                    rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)
                )
                stimulus_frequency = np.bincount(
                    rng.integers(0, len(stimuli), len(stimuli)), minlength=len(stimuli)
                )
                weights = subject_frequency[subject_codes] * stimulus_frequency[stimulus_codes]
                estimate, _ = panel_delta(weights)
                if np.isfinite(estimate):
                    bootstrap[valid] = estimate
                    valid += 1
            if valid < repetitions * 0.95:
                raise RuntimeError(f"Only {valid}/{repetitions} valid panel bootstrap replicates")
            bootstrap = bootstrap[:valid]
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "left_protocol": left_protocol,
                    "right_protocol": right_protocol,
                    "n_models": len(candidate_models),
                    "mean_balanced_accuracy_delta": observed,
                    "min_model_delta": float(np.min(model_deltas)),
                    "max_model_delta": float(np.max(model_deltas)),
                    "ci_low": np.quantile(bootstrap, 0.025),
                    "ci_high": np.quantile(bootstrap, 0.975),
                    "p_bootstrap": 2 * min(np.mean(bootstrap <= 0), np.mean(bootstrap >= 0)),
                    "bootstrap_valid": valid,
                }
            )
            progress.update(1)
    progress.close()
    output = pd.DataFrame(rows)
    p_values = output.p_bootstrap.to_numpy(float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[position])
        adjusted[position] = min(running, 1.0)
    output["p_holm_global"] = adjusted
    return output


def rank_stability(summary: pd.DataFrame) -> pd.DataFrame:
    candidate = summary.loc[~summary.model.isin(["global_prevalence", "stimulus_prior"])].copy()
    rows = []
    for (dataset, task), group in candidate.groupby(["dataset", "task"]):
        table = group.pivot(index="model", columns="protocol", values="subject_mean_balanced_accuracy")
        dual_rank = table["dual"].rank(ascending=False, method="average")
        for protocol in ("random", "subject", "stimulus"):
            protocol_rank = table[protocol].rank(ascending=False, method="average")
            tau, p_value = stats.kendalltau(protocol_rank, dual_rank)
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "left_protocol": protocol,
                    "right_protocol": "dual",
                    "kendall_tau": tau,
                    "p_value": p_value,
                    "left_winner": table[protocol].idxmax(),
                    "dual_winner": table["dual"].idxmax(),
                    "winner_changed": table[protocol].idxmax() != table["dual"].idxmax(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    admitted_datasets = require_crossed_datasets(args.datasets, context="Protocol sensitivity benchmark")
    missing = set(PROTOCOLS) - set(args.protocols)
    if missing:
        raise ValueError(f"Full protocol audit requires all protocols; missing {sorted(missing)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.prediction_file, low_memory=False) if args.prediction_file else run_benchmark(args)
    observed_datasets = set(predictions.dataset.unique())
    if observed_datasets != set(admitted_datasets):
        raise ValueError(
            f"Prediction datasets {sorted(observed_datasets)} do not match admitted datasets "
            f"{sorted(admitted_datasets)}"
        )
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    summary = summarize(predictions)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    contrasts = protocol_contrasts(predictions, args.bootstrap_repetitions, args.seed)
    contrasts.to_csv(args.output_root / "protocol_contrasts.csv", index=False)
    panel_contrasts = panel_protocol_contrasts(predictions, args.bootstrap_repetitions, args.seed)
    panel_contrasts.to_csv(args.output_root / "panel_protocol_contrasts.csv", index=False)
    ranks = rank_stability(summary)
    ranks.to_csv(args.output_root / "rank_stability.csv", index=False)
    manifest = {
        "datasets": args.datasets,
        "tasks": args.tasks,
        "protocols": args.protocols,
        "feature_prefix": args.feature_prefix,
        "folds": args.folds,
        "seed": args.seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "models": args.models,
        "prediction_file": str(args.prediction_file) if args.prediction_file else None,
        "stimulus_identity_contract": {
            "admitted_datasets": admitted_datasets,
            "eppvr_excluded": True,
        },
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nProtocol contrasts:")
    print(contrasts.to_string(index=False))
    print("\nPanel protocol contrasts:")
    print(panel_contrasts.to_string(index=False))
    print("\nRank stability:")
    print(ranks.to_string(index=False))


if __name__ == "__main__":
    main()
