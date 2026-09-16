from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm

import analyze_matched_factorial_effects as factorial
import run_matched_exposure_benchmark as matched
import run_paired_block_exposure_benchmark as paired
import run_protocol_model_benchmark as benchmark
from stimulus_identity_contract import CROSSED_UNIFIED_DATASETS, require_crossed_datasets


DEFAULT_DATASETS = CROSSED_UNIFIED_DATASETS
DEFAULT_MODELS = (
    "linear_logistic",
    "rbf_svm",
    "extra_trees",
    "hist_gradient_boosting",
)
DEFAULT_SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)
STRATEGIES = ("gt5", "exclude5")
EXPOSURES = tuple(matched.EXPOSURES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label-threshold sensitivity for the paired 2x2 identity-exposure design."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--feature-prefix", default="stimulus__")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/paired_block_threshold_sensitivity"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def strategy_labels(frame: pd.DataFrame, task: str, strategy: str) -> tuple[pd.DataFrame, np.ndarray]:
    score = frame[f"{task}_score"].to_numpy(float)
    if strategy == "gt5":
        keep = np.ones(len(frame), dtype=bool)
    elif strategy == "exclude5":
        keep = score != 5
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")
    subset = frame.loc[keep].copy()
    labels = (score[keep] > 5).astype(int)
    if np.unique(labels).size != 2:
        raise ValueError(f"{task}/{strategy} has fewer than two classes")
    return subset, labels


def fit_seed(
    args: argparse.Namespace,
    strategy: str,
    split_seed: int,
    progress: tqdm,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    templates = {
        name: model
        for name, model in benchmark.make_models(split_seed, args.n_jobs).items()
        if name in args.models
    }
    prediction_rows: list[dict] = []
    audit_rows: list[dict] = []
    label_rows: list[dict] = []
    for specification in args.datasets:
        dataset, path = specification.split("=", 1)
        full_frame = pd.read_csv(path).reset_index(drop=True)
        full_frame["row_id"] = np.arange(len(full_frame))
        feature_columns = [column for column in full_frame if column.startswith(args.feature_prefix)]
        if not feature_columns:
            raise ValueError(f"No {args.feature_prefix!r} columns for {dataset}")
        for task in args.tasks:
            frame, labels = strategy_labels(full_frame, task, strategy)
            frame = frame.reset_index(drop=True)
            features = frame[feature_columns].to_numpy(float)
            subject_fold, stimulus_fold = matched.fold_assignments(frame, args.folds, split_seed)
            label_rows.append(
                {
                    "strategy": strategy,
                    "split_seed": split_seed,
                    "dataset": dataset,
                    "task": task,
                    "n_rows": len(frame),
                    "n_removed_exact_5": len(full_frame) - len(frame),
                    "class_0": int(np.sum(labels == 0)),
                    "class_1": int(np.sum(labels == 1)),
                }
            )
            for subject_test in range(args.folds):
                for stimulus_test in range(args.folds):
                    test, train_sets, fold_audit = paired.paired_train_sets(
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
                    for row in fold_audit:
                        row["strategy"] = strategy
                    audit_rows.extend(fold_audit)
                    fold = f"s{subject_test}_t{stimulus_test}"
                    for exposure in EXPOSURES:
                        train = train_sets[exposure]
                        if np.unique(labels[train]).size < 2:
                            raise ValueError(
                                f"Single-class training set: {strategy}/{dataset}/{task}/{fold}/{exposure}"
                            )
                        for model_name, template in templates.items():
                            model = clone(template).fit(features[train], labels[train])
                            probability = benchmark.model_probability(model, features[test])
                            for index, estimate in zip(test, probability, strict=True):
                                prediction_rows.append(
                                    {
                                        "strategy": strategy,
                                        "dataset": dataset,
                                        "task": task,
                                        "exposure": exposure,
                                        "model": model_name,
                                        "fold": fold,
                                        "row_id": int(frame.iloc[index].row_id),
                                        "subject_id": str(frame.iloc[index].subject_id),
                                        "trial_id": int(frame.iloc[index].trial_id),
                                        "target": int(labels[index]),
                                        "probability": float(estimate),
                                    }
                                )
                            progress.update(1)
    predictions = pd.DataFrame(prediction_rows)
    predictions["split_seed"] = split_seed
    return predictions, pd.DataFrame(audit_rows), pd.DataFrame(label_rows)


def validate_seed(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
    args: argparse.Namespace,
    strategy: str,
    split_seed: int,
) -> None:
    if not np.isfinite(predictions.probability.to_numpy(float)).all():
        raise ValueError(f"{strategy}/{split_seed} has non-finite probabilities")
    key = ["dataset", "task", "exposure", "model", "row_id"]
    if predictions.duplicated(key).any():
        raise ValueError(f"{strategy}/{split_seed} has duplicate OOF predictions")
    expected_groups = len(args.datasets) * len(args.tasks) * len(EXPOSURES) * len(args.models)
    groups = predictions.groupby(["dataset", "task", "exposure", "model"]).size()
    if len(groups) != expected_groups:
        raise ValueError(
            f"{strategy}/{split_seed} has {len(groups)} prediction groups, expected {expected_groups}"
        )
    task_counts = predictions.groupby(["dataset", "task"])["row_id"].nunique()
    for (dataset, task, _, _), count in groups.items():
        if count != task_counts.loc[(dataset, task)]:
            raise ValueError(f"Incomplete OOF predictions for {strategy}/{dataset}/{task}")
    if not audit.matches_reference_size.all() or not audit.matches_reference_class_counts.all():
        raise ValueError(f"{strategy}/{split_seed} violates matched-training contracts")
    if (audit.row_overlap_with_test != 0).any():
        raise ValueError(f"{strategy}/{split_seed} contains train-test row overlap")
    if (audit.shared_core_fraction <= 0).any():
        raise ValueError(f"{strategy}/{split_seed} has an empty shared training core")
    expected_coverage = {
        "unseen_both": (0.0, 0.0),
        "seen_subject": (1.0, 0.0),
        "seen_stimulus": (0.0, 1.0),
        "seen_both": (1.0, 1.0),
    }
    for exposure, (subject_coverage, stimulus_coverage) in expected_coverage.items():
        subset = audit.loc[audit.exposure == exposure]
        if not np.allclose(subset.test_subject_coverage, subject_coverage):
            raise ValueError(f"Invalid subject coverage for {strategy}/{split_seed}/{exposure}")
        if not np.allclose(subset.test_stimulus_coverage, stimulus_coverage):
            raise ValueError(f"Invalid stimulus coverage for {strategy}/{split_seed}/{exposure}")


def strategy_manifest(args: argparse.Namespace, strategy: str) -> dict:
    return {
        "strategy": strategy,
        "label_rule": "score > 5" if strategy == "gt5" else "score > 5 after excluding score == 5",
        "datasets": args.datasets,
        "tasks": args.tasks,
        "models": args.models,
        "split_seeds": args.seeds,
        "folds": args.folds,
        "exposures": list(EXPOSURES),
        "design": "paired label-matched block replacement with a shared training core",
        "stimulus_contract": "Only datasets with verified cross-subject physical stimulus identities",
    }


def load_primary_effects(args: argparse.Namespace) -> pd.DataFrame:
    path = args.primary_root / "factorial_effects_crossed_valid.csv"
    primary = pd.read_csv(path)
    dataset_names = [item.split("=", 1)[0] for item in args.datasets]
    primary = primary.loc[
        primary.dataset.isin(dataset_names) & primary.task.isin(args.tasks)
    ].copy()
    primary["strategy"] = "ge5_primary"
    primary["label_rule"] = "score >= 5"
    return add_current_family_correction(primary)


def add_current_family_correction(effects: pd.DataFrame) -> pd.DataFrame:
    output = effects.copy()
    output["p_holm_current_dataset_task_family"] = np.nan
    for _, indices in output.groupby("contrast").groups.items():
        positions = np.asarray(list(indices), dtype=int)
        output.loc[positions, "p_holm_current_dataset_task_family"] = factorial.holm_adjust(
            output.loc[positions, "p_bootstrap"].to_numpy(float)
        )
    return output


def build_comparison(effects: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "task", "contrast"]
    compact = effects[
        keys
        + [
            "strategy",
            "balanced_accuracy_effect",
            "ci_low",
            "ci_high",
            "p_holm_current_dataset_task_family",
        ]
    ].copy()
    primary = compact.loc[compact.strategy == "ge5_primary", keys + ["balanced_accuracy_effect"]].rename(
        columns={"balanced_accuracy_effect": "primary_effect"}
    )
    compact = compact.merge(primary, on=keys, validate="many_to_one")
    compact["effect_change_from_primary"] = (
        compact.balanced_accuracy_effect - compact.primary_effect
    )
    compact["same_direction_as_primary"] = (
        np.sign(compact.balanced_accuracy_effect) == np.sign(compact.primary_effect)
    )
    grouped = compact.groupby(keys)["balanced_accuracy_effect"]
    stability = grouped.agg(effect_min="min", effect_max="max").reset_index()
    stability["effect_range"] = stability.effect_max - stability.effect_min
    sign_counts = compact.assign(sign=np.sign(compact.balanced_accuracy_effect)).groupby(keys).sign.nunique()
    stability = stability.merge(
        sign_counts.rename("n_effect_signs").reset_index(), on=keys, validate="one_to_one"
    )
    stability["direction_stable_all_strategies"] = stability.n_effect_signs == 1
    return compact.merge(stability, on=keys, validate="many_to_one")


def main() -> None:
    args = parse_args()
    require_crossed_datasets(args.datasets, context="Threshold sensitivity")
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed: dict[str, set[int]] = {}
    for strategy in args.strategies:
        completed[strategy] = set()
        for split_seed in args.seeds:
            seed_root = args.output_root / strategy / f"seed_{split_seed}"
            if (
                (seed_root / "predictions.csv").exists()
                and (seed_root / "split_audit.csv").exists()
                and not args.force
            ):
                completed[strategy].add(split_seed)
    fits_per_seed = (
        len(args.datasets)
        * len(args.tasks)
        * args.folds**2
        * len(EXPOSURES)
        * len(args.models)
    )
    total_fits = sum(
        (len(args.seeds) - len(completed[strategy])) * fits_per_seed
        for strategy in args.strategies
    )
    progress = tqdm(total=total_fits, desc="Threshold sensitivity", unit="fit", dynamic_ncols=True)
    all_effects = [load_primary_effects(args)]
    all_label_audits = []
    for strategy in args.strategies:
        strategy_root = args.output_root / strategy
        strategy_root.mkdir(parents=True, exist_ok=True)
        prediction_frames = []
        audit_frames = []
        summary_frames = []
        label_frames = []
        for split_seed in args.seeds:
            seed_root = strategy_root / f"seed_{split_seed}"
            prediction_path = seed_root / "predictions.csv"
            audit_path = seed_root / "split_audit.csv"
            label_path = seed_root / "label_audit.csv"
            if split_seed in completed[strategy]:
                predictions = pd.read_csv(prediction_path, low_memory=False)
                audit = pd.read_csv(audit_path)
                label_audit = pd.read_csv(label_path)
                predictions["subject_id"] = predictions.subject_id.astype(str)
                predictions["trial_id"] = predictions.trial_id.astype(int)
            else:
                predictions, audit, label_audit = fit_seed(
                    args, strategy, split_seed, progress
                )
                seed_root.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(prediction_path, index=False)
                audit.to_csv(audit_path, index=False)
                label_audit.to_csv(label_path, index=False)
            validate_seed(predictions, audit, args, strategy, split_seed)
            summary = matched.summarize(predictions)
            summary["strategy"] = strategy
            summary["split_seed"] = split_seed
            summary.to_csv(seed_root / "summary.csv", index=False)
            prediction_frames.append(predictions)
            audit_frames.append(audit)
            label_frames.append(label_audit)
            summary_frames.append(summary)
        predictions = pd.concat(prediction_frames, ignore_index=True)
        audits = pd.concat(audit_frames, ignore_index=True)
        labels = pd.concat(label_frames, ignore_index=True)
        summaries = pd.concat(summary_frames, ignore_index=True)
        audits.to_csv(strategy_root / "split_audit.csv", index=False)
        labels.to_csv(strategy_root / "label_audit.csv", index=False)
        summaries.to_csv(strategy_root / "per_seed_summary.csv", index=False)
        matched.per_seed_effects(summaries, args.models).to_csv(
            strategy_root / "per_seed_effects.csv", index=False
        )
        (strategy_root / "run_manifest.json").write_text(
            json.dumps(strategy_manifest(args, strategy), indent=2), encoding="utf-8"
        )
        per_seed, effects = factorial.analyze(
            predictions,
            args.models,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        )
        effects["strategy"] = strategy
        effects["label_rule"] = strategy_manifest(args, strategy)["label_rule"]
        effects = add_current_family_correction(effects)
        per_seed["strategy"] = strategy
        per_seed.to_csv(strategy_root / "factorial_effects_per_seed.csv", index=False)
        effects.to_csv(strategy_root / "factorial_effects.csv", index=False)
        all_effects.append(effects)
        all_label_audits.append(labels)
    progress.close()
    combined = pd.concat(all_effects, ignore_index=True)
    combined.to_csv(args.output_root / "factorial_effects_all_thresholds.csv", index=False)
    build_comparison(combined).to_csv(
        args.output_root / "factorial_effect_threshold_stability.csv", index=False
    )
    if all_label_audits:
        pd.concat(all_label_audits, ignore_index=True).to_csv(
            args.output_root / "label_audit.csv", index=False
        )
    manifest = {
        "primary_results_read_only": str(args.primary_root),
        "alternative_strategies": args.strategies,
        "datasets": args.datasets,
        "tasks": args.tasks,
        "models": args.models,
        "split_seeds": args.seeds,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed_scheme": "sha256(base_seed|dataset|task|factorial)",
        "multiplicity_family": (
            f"{len(args.datasets) * len(args.tasks)} dataset-task comparisons, separately within "
            "each factorial contrast and threshold strategy"
        ),
        "note": "Post-hoc threshold sensitivity; EPPVR excluded because physical stimulus identity is unverified.",
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(build_comparison(combined).to_string(index=False))
    print(f"\nResults written to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
