from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm

import run_protocol_model_benchmark as benchmark
import run_representation_identity_audit as identity_audit
from stimulus_identity_contract import require_crossed_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint-unseen emotion gate for identity-reduced representations.")
    parser.add_argument("--datasets", nargs="+", default=list(identity_audit.probes.DEFAULT_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument(
        "--representations",
        nargs="+",
        default=list(identity_audit.REPRESENTATIONS),
        choices=list(identity_audit.REPRESENTATIONS),
    )
    parser.add_argument("--models", nargs="+", default=["linear_logistic", "rbf_svm"], choices=["linear_logistic", "rbf_svm"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(identity_audit.probes.DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/representation_emotion_gate_crossed_valid"),
    )
    parser.add_argument(
        "--identity-root",
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


def run(args: argparse.Namespace) -> pd.DataFrame:
    specifications = [item.split("=", 1) for item in args.datasets]
    total = (
        len(specifications)
        * len(args.tasks)
        * len(args.representations)
        * len(args.models)
        * len(args.seeds)
        * args.folds**2
    )
    progress = tqdm(total=total, desc="Representation emotion gate", unit="fit", dynamic_ncols=True)
    rows = []
    for dataset, path in specifications:
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["row_id"] = np.arange(len(frame))
        for split_seed in args.seeds:
            templates = {
                name: model
                for name, model in benchmark.make_models(split_seed, args.n_jobs).items()
                if name in args.models
            }
            subject_map = benchmark.shuffled_fold_map(frame.subject_id.to_numpy(), args.folds, split_seed)
            stimulus_map = benchmark.shuffled_fold_map(frame.trial_id.to_numpy(), args.folds, split_seed + 1)
            subject_fold = frame.subject_id.map(subject_map).to_numpy(int)
            stimulus_fold = frame.trial_id.map(stimulus_map).to_numpy(int)
            for task in args.tasks:
                labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
                for representation in args.representations:
                    feature_columns = identity_audit.columns_for(frame, representation)
                    features = frame[feature_columns].to_numpy(float)
                    for subject_test in range(args.folds):
                        for stimulus_test in range(args.folds):
                            test = (subject_fold == subject_test) & (stimulus_fold == stimulus_test)
                            train = (subject_fold != subject_test) & (stimulus_fold != stimulus_test)
                            train_indices = np.flatnonzero(train)
                            test_indices = np.flatnonzero(test)
                            if len(test_indices) == 0 or np.unique(labels[train_indices]).size < 2:
                                raise ValueError(
                                    f"Invalid dual fold: {dataset}/{task}/{split_seed}/"
                                    f"s{subject_test}_t{stimulus_test}"
                                )
                            for model_name, template in templates.items():
                                model = clone(template).fit(features[train_indices], labels[train_indices])
                                probability = benchmark.model_probability(model, features[test_indices])
                                for index, estimate in zip(test_indices, probability, strict=True):
                                    rows.append(
                                        {
                                            "dataset": dataset,
                                            "task": task,
                                            "representation": representation,
                                            "n_features": len(feature_columns),
                                            "model": model_name,
                                            "split_seed": split_seed,
                                            "fold": f"s{subject_test}_t{stimulus_test}",
                                            "row_id": int(frame.iloc[index].row_id),
                                            "subject_id": str(frame.iloc[index].subject_id),
                                            "trial_id": int(frame.iloc[index].trial_id),
                                            "target": int(labels[index]),
                                            "probability": float(estimate),
                                        }
                                    )
                                progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def validate(predictions: pd.DataFrame, args: argparse.Namespace) -> None:
    key = ["dataset", "task", "representation", "model", "split_seed", "row_id"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate representation-emotion OOF rows")
    if not np.isfinite(predictions.probability.to_numpy(float)).all():
        raise ValueError("Non-finite representation-emotion probability")
    dataset_counts = {
        dataset: len(pd.read_csv(path, usecols=["subject_id"]))
        for dataset, path in (item.split("=", 1) for item in args.datasets)
    }
    groups = predictions.groupby(["dataset", "task", "representation", "model", "split_seed"]).size()
    expected = (
        len(args.datasets)
        * len(args.tasks)
        * len(args.representations)
        * len(args.models)
        * len(args.seeds)
    )
    if len(groups) != expected:
        raise ValueError(f"Observed {len(groups)} groups, expected {expected}")
    for (dataset, _, _, _, _), count in groups.items():
        if count != dataset_counts[dataset]:
            raise ValueError(f"Incomplete joint-unseen OOF group for {dataset}: {count}")


def per_seed_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["dataset", "task", "representation", "n_features", "model", "split_seed"]
    ):
        rows.append(
            {
                "dataset": keys[0],
                "task": keys[1],
                "representation": keys[2],
                "n_features": keys[3],
                "model": keys[4],
                "split_seed": keys[5],
                "balanced_accuracy": benchmark.balanced_accuracy(
                    group.target.to_numpy(int), group.probability.to_numpy(float)
                ),
            }
        )
    output = pd.DataFrame(rows)
    all_scores = output.loc[output.representation == "all"].rename(
        columns={"balanced_accuracy": "all_balanced_accuracy"}
    )[["dataset", "task", "model", "split_seed", "all_balanced_accuracy"]]
    output = output.merge(
        all_scores,
        on=["dataset", "task", "model", "split_seed"],
        validate="many_to_one",
    )
    output["balanced_accuracy_delta_vs_all"] = output.balanced_accuracy - output.all_balanced_accuracy
    return output


def gate_summary(per_seed: pd.DataFrame, identity_root: Path) -> pd.DataFrame:
    emotion = (
        per_seed.groupby(["representation", "n_features"])
        .agg(
            emotion_bacc_delta_vs_all=("balanced_accuracy_delta_vs_all", "mean"),
            emotion_delta_min=("balanced_accuracy_delta_vs_all", "min"),
            emotion_delta_max=("balanced_accuracy_delta_vs_all", "max"),
            emotion_positive_fraction=("balanced_accuracy_delta_vs_all", lambda values: np.mean(values > 0)),
        )
        .reset_index()
    )
    identity = pd.read_csv(identity_root / "summary.csv")
    subject_identity = (
        identity.loc[identity.axis == "subject_across_stimuli"]
        .groupby(["representation", "n_features"])
        .agg(
            subject_probe_delta_vs_all=("mean_identity_bacc_delta_vs_all", "mean"),
            subject_probe_delta_max=("mean_identity_bacc_delta_vs_all", "max"),
        )
        .reset_index()
    )
    stimulus_identity = (
        identity.loc[identity.axis == "stimulus_across_subjects"]
        .groupby(["representation", "n_features"])
        .agg(stimulus_probe_delta_vs_all=("mean_identity_bacc_delta_vs_all", "mean"))
        .reset_index()
    )
    output = emotion.merge(subject_identity, on=["representation", "n_features"], validate="one_to_one")
    output = output.merge(stimulus_identity, on=["representation", "n_features"], validate="one_to_one")
    output["retains_emotion"] = output.emotion_bacc_delta_vs_all >= -0.01
    output["reduces_subject_identity"] = output.subject_probe_delta_vs_all <= -0.10
    output["passes_prespecified_gate"] = output.retains_emotion & output.reduces_subject_identity
    return output.sort_values(
        ["passes_prespecified_gate", "emotion_bacc_delta_vs_all", "subject_probe_delta_vs_all"],
        ascending=[False, False, True],
    )


def main() -> None:
    args = parse_args()
    admitted_datasets = require_crossed_datasets(
        args.datasets, context="Representation emotion gate"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.reuse_predictions is None:
        predictions = run(args)
    else:
        predictions = pd.read_csv(args.reuse_predictions, low_memory=False)
        predictions = predictions.loc[
            predictions.dataset.isin(admitted_datasets)
            & predictions.task.isin(args.tasks)
            & predictions.representation.isin(args.representations)
            & predictions.model.isin(args.models)
            & predictions.split_seed.isin(args.seeds)
        ].copy()
    validate(predictions, args)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    per_seed = per_seed_summary(predictions)
    per_seed.to_csv(args.output_root / "per_seed_summary.csv", index=False)
    summary = (
        per_seed.groupby(["dataset", "task", "representation", "n_features", "model"])
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_delta_vs_all=("balanced_accuracy_delta_vs_all", "mean"),
            split_seed_sd=("balanced_accuracy", "std"),
            split_seed_min=("balanced_accuracy", "min"),
            split_seed_max=("balanced_accuracy", "max"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_root / "summary.csv", index=False)
    gate = gate_summary(per_seed, args.identity_root)
    gate.to_csv(args.output_root / "gate_summary.csv", index=False)
    manifest = {
        "datasets": args.datasets,
        "tasks": args.tasks,
        "representations": {name: identity_audit.REPRESENTATIONS[name] for name in args.representations},
        "models": args.models,
        "split_seeds": args.seeds,
        "folds": args.folds,
        "protocol": "joint unseen subject and stimulus",
        "reused_predictions": str(args.reuse_predictions) if args.reuse_predictions else None,
        "identity_root": str(args.identity_root),
        "stimulus_identity_contract": {
            "admitted_datasets": admitted_datasets,
            "eppvr_excluded": True,
        },
        "gate": {
            "mean_emotion_bacc_delta_vs_all_min": -0.01,
            "mean_subject_probe_bacc_delta_vs_all_max": -0.10,
        },
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
