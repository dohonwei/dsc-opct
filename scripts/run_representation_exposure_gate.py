from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm

import run_paired_block_exposure_benchmark as paired
import run_protocol_model_benchmark as benchmark
import run_representation_identity_audit as identity_audit
from stimulus_identity_contract import CROSSED_UNIFIED_DATASETS, require_crossed_datasets


DEFAULT_REPRESENTATIONS = ("absolute_power", "relative_power", "normalized_asymmetry")
DEFAULT_MODELS = ("linear_logistic", "rbf_svm")
DEFAULT_CROSSED_DATASETS = CROSSED_UNIFIED_DATASETS
SUBJECT_SENSITIVE_SETTINGS = {
    ("DEAP", "arousal"),
    ("DEAP", "valence"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired exposure mitigation gate for reduced representations.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_CROSSED_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument(
        "--representations",
        nargs="+",
        default=list(DEFAULT_REPRESENTATIONS),
        choices=list(identity_audit.REPRESENTATIONS),
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(paired.DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--unseen-predictions",
        type=Path,
        default=Path("outputs/representation_emotion_gate_crossed_valid/predictions.csv"),
    )
    parser.add_argument(
        "--all-summary",
        type=Path,
        default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid/per_seed_summary.csv"),
    )
    parser.add_argument(
        "--identity-summary",
        type=Path,
        default=Path("outputs/representation_identity_audit_crossed_valid/summary.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/representation_exposure_gate_crossed_valid"),
    )
    parser.add_argument(
        "--reuse-predictions",
        type=Path,
        default=None,
        help="Reuse a completed prediction CSV and recompute only the admitted-dataset gate.",
    )
    return parser.parse_args()


def load_unseen(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.unseen_predictions, low_memory=False)
    dataset_names = [item.split("=", 1)[0] for item in args.datasets]
    frame = frame.loc[
        frame.dataset.isin(dataset_names)
        & frame.task.isin(args.tasks)
        & frame.representation.isin(args.representations)
        & frame.model.isin(args.models)
        & frame.split_seed.isin(args.seeds)
    ].copy()
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["trial_id"] = frame.trial_id.astype(int)
    frame["exposure"] = "unseen_both"
    return frame


def load_reused_predictions(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.reuse_predictions, low_memory=False)
    dataset_names = [item.split("=", 1)[0] for item in args.datasets]
    return frame.loc[
        frame.dataset.isin(dataset_names)
        & frame.task.isin(args.tasks)
        & frame.representation.isin(args.representations)
        & frame.model.isin(args.models)
        & frame.split_seed.isin(args.seeds)
    ].copy()


def run(args: argparse.Namespace, unseen: pd.DataFrame) -> pd.DataFrame:
    specifications = [item.split("=", 1) for item in args.datasets]
    total = (
        len(specifications)
        * len(args.tasks)
        * len(args.representations)
        * len(args.models)
        * len(args.seeds)
        * args.folds**2
        * len(paired.FITTED_EXPOSURES)
    )
    progress = tqdm(total=total, desc="Representation exposure gate", unit="fit", dynamic_ncols=True)
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
            subject_fold, stimulus_fold = paired.matched.fold_assignments(frame, args.folds, split_seed)
            for task in args.tasks:
                labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
                fold_sets = {}
                for subject_test in range(args.folds):
                    for stimulus_test in range(args.folds):
                        test, train_sets, _ = paired.paired_train_sets(
                            frame,
                            labels,
                            subject_fold,
                            stimulus_fold,
                            subject_test,
                            stimulus_test,
                            dataset,
                            task,
                            split_seed,
                        )
                        fold_sets[(subject_test, stimulus_test)] = (test, train_sets)
                for representation in args.representations:
                    feature_columns = identity_audit.columns_for(frame, representation)
                    features = frame[feature_columns].to_numpy(float)
                    for subject_test in range(args.folds):
                        for stimulus_test in range(args.folds):
                            test, train_sets = fold_sets[(subject_test, stimulus_test)]
                            fold = f"s{subject_test}_t{stimulus_test}"
                            for exposure in paired.FITTED_EXPOSURES:
                                train = train_sets[exposure]
                                for model_name, template in templates.items():
                                    model = clone(template).fit(features[train], labels[train])
                                    probability = benchmark.model_probability(model, features[test])
                                    for index, estimate in zip(test, probability, strict=True):
                                        rows.append(
                                            {
                                                "dataset": dataset,
                                                "task": task,
                                                "representation": representation,
                                                "n_features": len(feature_columns),
                                                "model": model_name,
                                                "split_seed": split_seed,
                                                "exposure": exposure,
                                                "fold": fold,
                                                "row_id": int(frame.iloc[index].row_id),
                                                "subject_id": str(frame.iloc[index].subject_id),
                                                "trial_id": int(frame.iloc[index].trial_id),
                                                "target": int(labels[index]),
                                                "probability": float(estimate),
                                            }
                                        )
                                    progress.update(1)
    progress.close()
    fitted = pd.DataFrame(rows)
    columns = list(fitted.columns)
    return pd.concat([unseen[columns], fitted], ignore_index=True)


def validate(predictions: pd.DataFrame, args: argparse.Namespace) -> None:
    key = ["dataset", "task", "representation", "model", "split_seed", "exposure", "row_id"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate representation-exposure OOF rows")
    if not np.isfinite(predictions.probability.to_numpy(float)).all():
        raise ValueError("Non-finite representation-exposure probability")
    dataset_counts = {
        dataset: len(pd.read_csv(path, usecols=["subject_id"]))
        for dataset, path in (item.split("=", 1) for item in args.datasets)
    }
    groups = predictions.groupby(
        ["dataset", "task", "representation", "model", "split_seed", "exposure"]
    ).size()
    expected = (
        len(args.datasets)
        * len(args.tasks)
        * len(args.representations)
        * len(args.models)
        * len(args.seeds)
        * len(paired.EXPOSURES)
    )
    if len(groups) != expected:
        raise ValueError(f"Observed {len(groups)} groups, expected {expected}")
    for (dataset, _, _, _, _, _), count in groups.items():
        if count != dataset_counts[dataset]:
            raise ValueError(f"Incomplete representation-exposure group for {dataset}: {count}")


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(
        ["dataset", "task", "representation", "n_features", "model", "split_seed", "exposure"]
    ):
        rows.append(
            {
                "dataset": keys[0],
                "task": keys[1],
                "representation": keys[2],
                "n_features": keys[3],
                "model": keys[4],
                "split_seed": keys[5],
                "exposure": keys[6],
                "balanced_accuracy": benchmark.balanced_accuracy(
                    group.target.to_numpy(int), group.probability.to_numpy(float)
                ),
            }
        )
    return pd.DataFrame(rows)


def factorial_effects(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in summary.groupby(
        ["dataset", "task", "representation", "n_features", "model", "split_seed"]
    ):
        scores = group.set_index("exposure").balanced_accuracy
        rows.append(
            {
                "dataset": keys[0],
                "task": keys[1],
                "representation": keys[2],
                "n_features": keys[3],
                "model": keys[4],
                "split_seed": keys[5],
                "unseen_both_balanced_accuracy": scores.unseen_both,
                "subject_exposure_effect": 0.5
                * (scores.seen_subject + scores.seen_both - scores.unseen_both - scores.seen_stimulus),
                "stimulus_exposure_effect": 0.5
                * (scores.seen_stimulus + scores.seen_both - scores.unseen_both - scores.seen_subject),
                "interaction_effect": scores.unseen_both
                - scores.seen_subject
                - scores.seen_stimulus
                + scores.seen_both,
            }
        )
    return pd.DataFrame(rows)


def all_factorial_effects(path: Path, models: list[str]) -> pd.DataFrame:
    summary = pd.read_csv(path)
    summary = summary.loc[summary.model.isin(models)].rename(
        columns={"pooled_balanced_accuracy": "balanced_accuracy"}
    )
    summary["representation"] = "all"
    summary["n_features"] = 72
    return factorial_effects(
        summary[
            [
                "dataset",
                "task",
                "representation",
                "n_features",
                "model",
                "split_seed",
                "exposure",
                "balanced_accuracy",
            ]
        ]
    )


def gate_summary(
    reduced_effects: pd.DataFrame,
    all_effects: pd.DataFrame,
    identity_path: Path,
    admitted_datasets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_reference = all_effects.rename(
        columns={
            "unseen_both_balanced_accuracy": "all_unseen_both_balanced_accuracy",
            "subject_exposure_effect": "all_subject_exposure_effect",
            "stimulus_exposure_effect": "all_stimulus_exposure_effect",
            "interaction_effect": "all_interaction_effect",
        }
    )[
        [
            "dataset",
            "task",
            "model",
            "split_seed",
            "all_unseen_both_balanced_accuracy",
            "all_subject_exposure_effect",
            "all_stimulus_exposure_effect",
            "all_interaction_effect",
        ]
    ]
    comparison = reduced_effects.merge(
        all_reference,
        on=["dataset", "task", "model", "split_seed"],
        validate="many_to_one",
    )
    comparison["joint_bacc_delta_vs_all"] = (
        comparison.unseen_both_balanced_accuracy - comparison.all_unseen_both_balanced_accuracy
    )
    comparison["subject_effect_reduction"] = (
        comparison.all_subject_exposure_effect - comparison.subject_exposure_effect
    )
    setting_summary = (
        comparison.groupby(["dataset", "task", "representation", "n_features"])
        .agg(
            joint_bacc_delta_vs_all=("joint_bacc_delta_vs_all", "mean"),
            subject_effect_all=("all_subject_exposure_effect", "mean"),
            subject_effect_reduced=("subject_exposure_effect", "mean"),
            subject_effect_reduction=("subject_effect_reduction", "mean"),
            subject_effect_reduction_positive_fraction=("subject_effect_reduction", lambda x: np.mean(x > 0)),
        )
        .reset_index()
    )
    identity = pd.read_csv(identity_path)
    identity = (
        identity.loc[
            (identity.axis == "subject_across_stimuli")
            & identity.dataset.isin(admitted_datasets)
        ]
        .groupby(["representation", "n_features"])
        .mean_identity_bacc_delta_vs_all.mean()
        .rename("subject_probe_delta_vs_all")
        .reset_index()
    )
    rows = []
    for representation, group in setting_summary.groupby("representation"):
        sensitive = group.loc[
            group.apply(lambda row: (row.dataset, row.task) in SUBJECT_SENSITIVE_SETTINGS, axis=1)
        ]
        subject_reductions = sensitive.set_index(["dataset", "task"]).subject_effect_reduction
        row = {
            "representation": representation,
            "n_features": int(group.n_features.iloc[0]),
            "mean_joint_bacc_delta_vs_all": group.joint_bacc_delta_vs_all.mean(),
            "worst_setting_joint_bacc_delta_vs_all": group.joint_bacc_delta_vs_all.min(),
            "mean_subject_effect_reduction_sensitive_settings": sensitive.subject_effect_reduction.mean(),
            "min_subject_effect_reduction_sensitive_settings": sensitive.subject_effect_reduction.min(),
            "deap_arousal_subject_effect_reduction": subject_reductions.loc[("DEAP", "arousal")],
            "deap_valence_subject_effect_reduction": subject_reductions.loc[("DEAP", "valence")],
        }
        rows.append(row)
    gate = pd.DataFrame(rows).merge(identity, on=["representation", "n_features"], validate="one_to_one")
    gate["retains_mean_joint_performance"] = gate.mean_joint_bacc_delta_vs_all >= -0.01
    gate["retains_each_setting"] = gate.worst_setting_joint_bacc_delta_vs_all >= -0.03
    gate["reduces_mean_subject_effect"] = gate.mean_subject_effect_reduction_sensitive_settings >= 0.02
    gate["reduces_all_sensitive_settings"] = gate.min_subject_effect_reduction_sensitive_settings > 0
    gate["reduces_subject_probe"] = gate.subject_probe_delta_vs_all <= -0.10
    gate["passes_prespecified_mitigation_gate"] = gate[
        [
            "retains_mean_joint_performance",
            "retains_each_setting",
            "reduces_mean_subject_effect",
            "reduces_all_sensitive_settings",
            "reduces_subject_probe",
        ]
    ].all(axis=1)
    return setting_summary, gate.sort_values(
        ["passes_prespecified_mitigation_gate", "mean_subject_effect_reduction_sensitive_settings"],
        ascending=[False, False],
    )


def main() -> None:
    args = parse_args()
    admitted_datasets = require_crossed_datasets(
        args.datasets, context="Representation exposure gate"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.reuse_predictions is None:
        unseen = load_unseen(args)
        predictions = run(args, unseen)
    else:
        predictions = load_reused_predictions(args)
    validate(predictions, args)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    summary = summarize(predictions)
    summary.to_csv(args.output_root / "per_seed_summary.csv", index=False)
    effects = factorial_effects(summary)
    effects.to_csv(args.output_root / "factorial_effects_per_seed_model.csv", index=False)
    all_effects = all_factorial_effects(args.all_summary, args.models)
    setting_summary, gate = gate_summary(
        effects,
        all_effects,
        args.identity_summary,
        admitted_datasets,
    )
    setting_summary.to_csv(args.output_root / "dataset_task_summary.csv", index=False)
    gate.to_csv(args.output_root / "gate_summary.csv", index=False)
    manifest = {
        "datasets": args.datasets,
        "tasks": args.tasks,
        "representations": args.representations,
        "models": args.models,
        "split_seeds": args.seeds,
        "folds": args.folds,
        "design": "paired block identity exposure",
        "reused_predictions": str(args.reuse_predictions) if args.reuse_predictions else None,
        "subject_sensitive_settings": sorted([list(item) for item in SUBJECT_SENSITIVE_SETTINGS]),
        "exclusion_note": (
            "EPPVR is excluded because the available record positions cannot be verified as "
            "shared physical stimuli across participants."
        ),
        "gate": {
            "mean_joint_bacc_delta_vs_all_min": -0.01,
            "worst_setting_joint_bacc_delta_vs_all_min": -0.03,
            "mean_subject_effect_reduction_min": 0.02,
            "subject_effect_reduction_required_in_each_sensitive_setting": True,
            "subject_probe_bacc_delta_vs_all_max": -0.10,
        },
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
