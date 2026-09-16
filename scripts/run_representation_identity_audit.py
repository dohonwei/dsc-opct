from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm

import run_crossed_identity_probes as probes
import run_protocol_model_benchmark as benchmark
from stimulus_identity_contract import require_crossed_datasets


REPRESENTATIONS = {
    "all": ("stimulus__",),
    "absolute_power": ("stimulus__log_power_",),
    "relative_power": ("stimulus__relative_power_",),
    "log_difference": ("stimulus__log_difference_",),
    "normalized_asymmetry": ("stimulus__normalized_asymmetry_",),
    "scale_robust": ("stimulus__relative_power_", "stimulus__normalized_asymmetry_"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identity decodability across frontal feature representations.")
    parser.add_argument("--datasets", nargs="+", default=list(probes.DEFAULT_DATASETS))
    parser.add_argument("--representations", nargs="+", default=list(REPRESENTATIONS), choices=list(REPRESENTATIONS))
    parser.add_argument("--axes", nargs="+", default=list(probes.AXES), choices=list(probes.AXES))
    parser.add_argument("--models", nargs="+", default=["linear", "rbf"], choices=["linear", "rbf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(probes.DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/representation_identity_audit_crossed_valid"),
    )
    parser.add_argument(
        "--reuse-predictions",
        type=Path,
        default=None,
        help="Reuse a completed row-level prediction CSV after contract validation.",
    )
    return parser.parse_args()


def columns_for(frame: pd.DataFrame, representation: str) -> list[str]:
    prefixes = REPRESENTATIONS[representation]
    columns = [column for column in frame if any(column.startswith(prefix) for prefix in prefixes)]
    if not columns:
        raise ValueError(f"No features found for representation {representation}")
    if representation == "all" and len(columns) not in {72, 90}:
        raise ValueError(
            f"Expected 72 canonical or 90 SEED-IV all features, observed {len(columns)}"
        )
    return columns


def run(args: argparse.Namespace) -> pd.DataFrame:
    specifications = [item.split("=", 1) for item in args.datasets]
    total = (
        len(specifications)
        * len(args.representations)
        * len(args.axes)
        * len(args.models)
        * len(args.seeds)
        * args.folds
    )
    progress = tqdm(total=total, desc="Representation identity audit", unit="fit", dynamic_ncols=True)
    rows = []
    for dataset, path in specifications:
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["row_id"] = np.arange(len(frame))
        for representation in args.representations:
            feature_columns = columns_for(frame, representation)
            features = frame[feature_columns].to_numpy(float)
            for split_seed in args.seeds:
                templates = {
                    name: model for name, model in probes.make_models(split_seed).items() if name in args.models
                }
                for axis in args.axes:
                    target_column, group_column = probes.axis_contract(axis)
                    targets = frame[target_column].astype(str).to_numpy()
                    group_map = benchmark.shuffled_fold_map(
                        frame[group_column].to_numpy(), args.folds, split_seed
                    )
                    group_folds = frame[group_column].map(group_map).to_numpy(int)
                    for fold in range(args.folds):
                        test = group_folds == fold
                        train = ~test
                        missing = set(targets[test]) - set(targets[train])
                        if missing:
                            raise ValueError(
                                f"{dataset}/{representation}/{axis}/seed={split_seed}/fold={fold} "
                                f"has unseen classes {sorted(missing)}"
                            )
                        for model_name, template in templates.items():
                            model = clone(template).fit(features[train], targets[train])
                            prediction = model.predict(features[test])
                            for index, estimate in zip(np.flatnonzero(test), prediction, strict=True):
                                rows.append(
                                    {
                                        "dataset": dataset,
                                        "representation": representation,
                                        "n_features": len(feature_columns),
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


def validate(predictions: pd.DataFrame, args: argparse.Namespace) -> None:
    key = ["dataset", "representation", "axis", "model", "split_seed", "row_id"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate representation-probe OOF rows")
    dataset_counts = {
        dataset: len(pd.read_csv(path, usecols=["subject_id"]))
        for dataset, path in (item.split("=", 1) for item in args.datasets)
    }
    groups = predictions.groupby(["dataset", "representation", "axis", "model", "split_seed"]).size()
    expected = (
        len(args.datasets)
        * len(args.representations)
        * len(args.axes)
        * len(args.models)
        * len(args.seeds)
    )
    if len(groups) != expected:
        raise ValueError(f"Observed {len(groups)} groups, expected {expected}")
    for (dataset, _, _, _, _), count in groups.items():
        if count != dataset_counts[dataset]:
            raise ValueError(f"Incomplete OOF probe group for {dataset}: {count}")


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["dataset", "representation", "n_features", "axis", "model", "split_seed"]
    ):
        truth = group.target_identity.to_numpy()
        prediction = group.predicted_identity.to_numpy()
        n_classes = len(np.unique(truth))
        rows.append(
            {
                "dataset": keys[0],
                "representation": keys[1],
                "n_features": keys[2],
                "axis": keys[3],
                "model": keys[4],
                "split_seed": keys[5],
                "n_classes": n_classes,
                "chance": 1 / n_classes,
                "balanced_accuracy": probes.balanced_multiclass_accuracy(truth, prediction),
            }
        )
    return pd.DataFrame(rows)


def relative_to_all(per_seed: pd.DataFrame) -> pd.DataFrame:
    all_scores = per_seed.loc[per_seed.representation == "all"].rename(
        columns={"balanced_accuracy": "all_balanced_accuracy"}
    )[
        ["dataset", "axis", "model", "split_seed", "all_balanced_accuracy"]
    ]
    merged = per_seed.merge(
        all_scores,
        on=["dataset", "axis", "model", "split_seed"],
        validate="many_to_one",
    )
    merged["identity_bacc_delta_vs_all"] = merged.balanced_accuracy - merged.all_balanced_accuracy
    return (
        merged.groupby(["dataset", "representation", "n_features", "axis", "model"])
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_identity_bacc_delta_vs_all=("identity_bacc_delta_vs_all", "mean"),
            split_seed_sd=("balanced_accuracy", "std"),
            split_seed_min=("balanced_accuracy", "min"),
            split_seed_max=("balanced_accuracy", "max"),
            chance=("chance", "first"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    admitted_datasets = require_crossed_datasets(
        args.datasets, context="Representation identity audit"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.reuse_predictions is None:
        predictions = run(args)
    else:
        predictions = pd.read_csv(args.reuse_predictions, low_memory=False)
        predictions = predictions.loc[
            predictions.dataset.isin(admitted_datasets)
            & predictions.representation.isin(args.representations)
            & predictions.axis.isin(args.axes)
            & predictions.model.isin(args.models)
            & predictions.split_seed.isin(args.seeds)
        ].copy()
    validate(predictions, args)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    per_seed = summarize(predictions)
    per_seed.to_csv(args.output_root / "per_seed_summary.csv", index=False)
    summary = relative_to_all(per_seed)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    manifest = {
        "datasets": args.datasets,
        "representations": {name: REPRESENTATIONS[name] for name in args.representations},
        "axes": args.axes,
        "models": args.models,
        "split_seeds": args.seeds,
        "folds": args.folds,
        "reused_predictions": str(args.reuse_predictions) if args.reuse_predictions else None,
        "stimulus_identity_contract": {
            "admitted_datasets": admitted_datasets,
            "eppvr_excluded": True,
        },
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
