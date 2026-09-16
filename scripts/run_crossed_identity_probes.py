from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

import run_protocol_model_benchmark as benchmark
from run_multiseed_protocol_sensitivity import holm_adjust
from stimulus_identity_contract import (
    CROSSED_UNIFIED_DATASETS,
    require_crossed_datasets,
    stable_analysis_seed,
)


DEFAULT_DATASETS = CROSSED_UNIFIED_DATASETS
DEFAULT_SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)
AXES = ("stimulus_across_subjects", "subject_across_stimuli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-axis subject and stimulus identity probes.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--axes", nargs="+", default=list(AXES), choices=list(AXES))
    parser.add_argument("--models", nargs="+", default=["linear", "rbf"], choices=["linear", "rbf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--feature-prefix", default="stimulus__")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/crossed_identity_probes_crossed_valid"),
    )
    parser.add_argument(
        "--reuse-predictions",
        type=Path,
        default=None,
        help="Reuse a completed row-level prediction CSV after contract validation.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def make_models(seed: int) -> dict[str, Pipeline]:
    preprocess = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clip", FunctionTransformer(benchmark.clip_scaled_features)),
    ]
    return {
        "linear": Pipeline(
            preprocess
            + [
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=seed,
                    ),
                )
            ]
        ),
        "rbf": Pipeline(
            preprocess
            + [
                (
                    "model",
                    SVC(
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                )
            ]
        ),
    }


def axis_contract(axis: str) -> tuple[str, str]:
    if axis == "stimulus_across_subjects":
        return "trial_id", "subject_id"
    return "subject_id", "trial_id"


def balanced_multiclass_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    labels = np.unique(truth)
    return float(np.mean([np.mean(prediction[truth == label] == label) for label in labels]))


def run_probes(args: argparse.Namespace) -> pd.DataFrame:
    specifications = [item.split("=", 1) for item in args.datasets]
    total = len(specifications) * len(args.axes) * len(args.models) * len(args.seeds) * args.folds
    progress = tqdm(total=total, desc="Crossed identity probes", unit="fit", dynamic_ncols=True)
    rows = []
    for dataset, path in specifications:
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["row_id"] = np.arange(len(frame))
        feature_columns = [column for column in frame if column.startswith(args.feature_prefix)]
        if not feature_columns:
            raise ValueError(f"No {args.feature_prefix!r} columns for {dataset}")
        features = frame[feature_columns].to_numpy(float)
        for split_seed in args.seeds:
            templates = {name: model for name, model in make_models(split_seed).items() if name in args.models}
            for axis in args.axes:
                target_column, group_column = axis_contract(axis)
                targets = frame[target_column].astype(str).to_numpy()
                group_map = benchmark.shuffled_fold_map(frame[group_column].to_numpy(), args.folds, split_seed)
                group_folds = frame[group_column].map(group_map).to_numpy(int)
                for fold in range(args.folds):
                    test = group_folds == fold
                    train = ~test
                    missing_train_classes = set(targets[test]) - set(targets[train])
                    if missing_train_classes:
                        raise ValueError(
                            f"{dataset}/{axis}/seed={split_seed}/fold={fold} has unseen probe classes: "
                            f"{sorted(missing_train_classes)}"
                        )
                    for model_name, template in templates.items():
                        model = clone(template).fit(features[train], targets[train])
                        prediction = model.predict(features[test])
                        for index, estimate in zip(np.flatnonzero(test), prediction, strict=True):
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "axis": axis,
                                    "model": model_name,
                                    "split_seed": split_seed,
                                    "fold": fold,
                                    "row_id": int(frame.iloc[index].row_id),
                                    "subject_id": str(frame.iloc[index].subject_id),
                                    "trial_id": int(frame.iloc[index].trial_id),
                                    "target_identity": targets[index],
                                    "predicted_identity": str(estimate),
                                }
                            )
                        progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def validate_predictions(predictions: pd.DataFrame, args: argparse.Namespace) -> None:
    key = ["dataset", "axis", "model", "split_seed", "row_id"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate identity-probe OOF predictions")
    dataset_counts = {
        dataset: len(pd.read_csv(path, usecols=["subject_id"]))
        for dataset, path in (item.split("=", 1) for item in args.datasets)
    }
    groups = predictions.groupby(["dataset", "axis", "model", "split_seed"]).size()
    expected_groups = len(args.datasets) * len(args.axes) * len(args.models) * len(args.seeds)
    if len(groups) != expected_groups:
        raise ValueError(f"Observed {len(groups)} probe groups, expected {expected_groups}")
    for (dataset, _, _, _), count in groups.items():
        if count != dataset_counts[dataset]:
            raise ValueError(f"Incomplete identity-probe OOF predictions for {dataset}: {count}")
    for axis, group in predictions.groupby("axis"):
        held_out_identity = "subject_id" if axis == "stimulus_across_subjects" else "trial_id"
        leakage = group.groupby(["dataset", "model", "split_seed", "fold"])[held_out_identity].apply(
            lambda values: values.nunique()
        )
        if (leakage <= 0).any():
            raise ValueError(f"Invalid held-out groups for {axis}")


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for keys, group in predictions.groupby(["dataset", "axis", "model", "split_seed"]):
        truth = group.target_identity.to_numpy()
        prediction = group.predicted_identity.to_numpy()
        n_classes = len(np.unique(truth))
        rows.append(
            {
                "dataset": keys[0],
                "axis": keys[1],
                "model": keys[2],
                "split_seed": keys[3],
                "n_classes": n_classes,
                "chance": 1 / n_classes,
                "accuracy": accuracy_score(truth, prediction),
                "balanced_accuracy": balanced_multiclass_accuracy(truth, prediction),
            }
        )
    per_seed = pd.DataFrame(rows)
    aggregate = (
        per_seed.groupby(["dataset", "axis", "model"])
        .agg(
            n_classes=("n_classes", "first"),
            chance=("chance", "first"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            split_seed_sd=("balanced_accuracy", "std"),
            split_seed_min=("balanced_accuracy", "min"),
            split_seed_max=("balanced_accuracy", "max"),
            mean_accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )
    aggregate["chance_ratio"] = aggregate.mean_balanced_accuracy / aggregate.chance
    aggregate["above_chance"] = aggregate.mean_balanced_accuracy - aggregate.chance
    return per_seed, aggregate


def bootstrap_identity_classes(
    predictions: pd.DataFrame,
    repetitions: int = 5000,
    seed: int = 20260813,
) -> pd.DataFrame:
    rows = []
    groups = list(predictions.groupby(["dataset", "axis", "model"], sort=True))
    progress = tqdm(total=len(groups), desc="Identity-class bootstrap", unit="probe", dynamic_ncols=True)
    for probe_key, group in groups:
        dataset, axis, model = probe_key
        seeds = np.sort(group.split_seed.unique())
        identities = np.sort(group.target_identity.unique())
        seed_identity_recalls = np.empty((len(seeds), len(identities)), dtype=float)
        for seed_index, split_seed in enumerate(seeds):
            seed_group = group.loc[group.split_seed == split_seed]
            for identity_index, identity in enumerate(identities):
                identity_group = seed_group.loc[seed_group.target_identity == identity]
                seed_identity_recalls[seed_index, identity_index] = np.mean(
                    identity_group.predicted_identity == identity_group.target_identity
                )
        observed = float(seed_identity_recalls.mean())
        chance = 1 / len(identities)
        rng = np.random.default_rng(stable_analysis_seed(seed, dataset, axis, model))
        bootstrap = np.empty(repetitions, dtype=float)
        for repetition in range(repetitions):
            seed_sample = rng.integers(0, len(seeds), len(seeds))
            identity_sample = rng.integers(0, len(identities), len(identities))
            bootstrap[repetition] = seed_identity_recalls[np.ix_(seed_sample, identity_sample)].mean()
        centered = bootstrap - chance
        lower = (np.sum(centered <= 0) + 1) / (repetitions + 1)
        upper = (np.sum(centered >= 0) + 1) / (repetitions + 1)
        rows.append(
            {
                "dataset": dataset,
                "axis": axis,
                "model": model,
                "n_classes": len(identities),
                "chance": chance,
                "mean_balanced_accuracy": observed,
                "ci_low": np.quantile(bootstrap, 0.025),
                "ci_high": np.quantile(bootstrap, 0.975),
                "p_bootstrap": min(1.0, 2 * min(lower, upper)),
            }
        )
        progress.update(1)
    progress.close()
    output = pd.DataFrame(rows)
    output["p_holm_global"] = holm_adjust(output.p_bootstrap.to_numpy(float))
    return output


def main() -> None:
    args = parse_args()
    admitted_datasets = require_crossed_datasets(args.datasets, context="Cross-axis identity probes")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.reuse_predictions is None:
        predictions = run_probes(args)
    else:
        predictions = pd.read_csv(args.reuse_predictions, low_memory=False)
        predictions = predictions.loc[
            predictions.dataset.isin(admitted_datasets)
            & predictions.axis.isin(args.axes)
            & predictions.model.isin(args.models)
            & predictions.split_seed.isin(args.seeds)
        ].copy()
    validate_predictions(predictions, args)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    per_seed, summary = summarize(predictions)
    per_seed.to_csv(args.output_root / "per_seed_summary.csv", index=False)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    inference = bootstrap_identity_classes(predictions)
    inference.to_csv(args.output_root / "identity_class_bootstrap.csv", index=False)
    manifest = {
        "datasets": args.datasets,
        "axes": args.axes,
        "models": args.models,
        "split_seeds": args.seeds,
        "folds": args.folds,
        "feature_prefix": args.feature_prefix,
        "reused_predictions": str(args.reuse_predictions) if args.reuse_predictions else None,
        "bootstrap_seed_scheme": "sha256(base_seed|dataset|axis|model)",
        "stimulus_identity_contract": {
            "admitted_datasets": admitted_datasets,
            "eppvr_excluded": True,
        },
        "stimulus_probe_grouping": "held-out subjects",
        "subject_probe_grouping": "held-out stimuli",
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nIdentity-class bootstrap:")
    print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
