from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from identity_shortcut.core import (  # noqa: E402
    REPRESENTATIONS,
    build_dose_blocks,
    class_matched_control_split,
    metadata_prior_opportunity,
    representation_columns,
    stable_seed,
)
from identity_shortcut.models import make_shortcut_models  # noqa: E402
import run_matched_exposure_benchmark as matched  # noqa: E402
import run_paired_block_exposure_benchmark as paired  # noqa: E402
import run_protocol_model_benchmark as benchmark  # noqa: E402
from stimulus_identity_contract import CROSSED_UNIFIED_DATASETS, require_crossed_datasets  # noqa: E402


MODEL_NAMES = (
    "linear_logistic",
    "rbf_svm",
    "extra_trees",
    "hist_gradient_boosting",
    "gpu_mlp",
)
AXES = ("subject", "stimulus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Counterfactual identity-label opportunity dose-response experiment."
    )
    parser.add_argument("--datasets", nargs="+", default=list(CROSSED_UNIFIED_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=5.0,
        help="Scores greater than or equal to this value are assigned to the high class.",
    )
    parser.add_argument("--axes", nargs="+", default=list(AXES), choices=list(AXES))
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["all", "relative_power", "normalized_asymmetry"],
        choices=[name for name in REPRESENTATIONS if name != "baseline_delta"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear_logistic", "gpu_mlp"],
        choices=list(MODEL_NAMES),
        help="The dose experiment defaults to low- and high-capacity sentinel models.",
    )
    parser.add_argument("--doses", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(matched.DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--candidate-draws", type=int, default=256)
    parser.add_argument(
        "--intervention-block-fraction",
        type=float,
        default=0.5,
        help="Fraction of the exposure-axis candidate quadrant used for each dose block.",
    )
    parser.add_argument("--gpu-epochs", type=int, default=40)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--max-fold-cells",
        type=int,
        default=None,
        help="Limit subject-by-stimulus fold cells per dataset/task/seed for a smoke test.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/counterfactual_identity_dose_crossed_valid"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def axis_plan(
    frame: pd.DataFrame,
    labels: np.ndarray,
    subject_fold: np.ndarray,
    stimulus_fold: np.ndarray,
    subject_test: int,
    stimulus_test: int,
    train_sets: dict[str, np.ndarray],
    test: np.ndarray,
    axis: str,
    doses: list[float],
    draws: int,
    intervention_block_fraction: float,
    seed: int,
):
    unseen = train_sets["unseen_both"]
    if axis == "subject":
        candidate_mask = (subject_fold == subject_test) & (stimulus_fold != stimulus_test)
        identities = frame.subject_id.astype(str).to_numpy()
    else:
        candidate_mask = (subject_fold != subject_test) & (stimulus_fold == stimulus_test)
        identities = frame.trial_id.astype(str).to_numpy()
    candidates = np.flatnonzero(candidate_mask)
    required = identities[test]
    candidate_counts = np.bincount(labels[candidates], minlength=2)
    class_counts = np.floor(candidate_counts * intervention_block_fraction).astype(int)
    control_counts = np.bincount(labels[unseen], minlength=2)
    if class_counts.sum() < len(np.unique(required)):
        raise ValueError(
            f"Intervention block is too small to cover every {axis} identity; "
            "increase --intervention-block-fraction"
        )
    if np.any(class_counts > control_counts):
        raise ValueError(
            f"Unseen control cannot supply a class-matched {axis} replacement block"
        )
    fixed, removed_control = class_matched_control_split(
        unseen,
        labels,
        class_counts,
        stable_seed(seed, "control_removal"),
    )
    blocks = build_dose_blocks(
        candidates,
        labels,
        identities,
        required,
        class_counts,
        doses,
        draws,
        seed,
    )
    return fixed, identities, blocks, removed_control, candidates, class_counts


def prediction_rows(
    frame: pd.DataFrame,
    test: np.ndarray,
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    dataset: str,
    task: str,
    axis: str,
    representation: str,
    model: str,
    split_seed: int,
    fold: str,
    condition: str,
    nominal_dose: float | None,
    achieved_dose: float | None,
) -> list[dict]:
    rows = []
    for index, estimate in zip(test, probability, strict=True):
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "axis": axis,
                "representation": representation,
                "model": model,
                "split_seed": split_seed,
                "fold": fold,
                "condition": condition,
                "nominal_dose": nominal_dose,
                "achieved_dose": achieved_dose,
                "row_id": int(index),
                "subject_id": str(frame.iloc[index].subject_id),
                "trial_id": str(frame.iloc[index].trial_id),
                "target": int(labels[index]),
                "probability": float(estimate),
            }
        )
    return rows


def summarize(predictions: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dataset", "task", "axis", "representation", "model", "split_seed"]
    for key_values, group in predictions.groupby(keys, sort=True):
        control = group.loc[group.condition == "unseen"]
        control_score = benchmark.balanced_accuracy(
            control.target.to_numpy(int), control.probability.to_numpy(float)
        )
        for nominal_dose, dose_group in group.loc[group.condition == "dose"].groupby("nominal_dose"):
            dose_score = benchmark.balanced_accuracy(
                dose_group.target.to_numpy(int), dose_group.probability.to_numpy(float)
            )
            rows.append(
                dict(
                    zip(keys, key_values, strict=True),
                    nominal_dose=float(nominal_dose),
                    achieved_dose=float(dose_group.achieved_dose.mean()),
                    unseen_balanced_accuracy=control_score,
                    dose_balanced_accuracy=dose_score,
                    exposure_effect=dose_score - control_score,
                )
            )
    summary = pd.DataFrame(rows)
    opportunity = (
        audit.groupby(["dataset", "task", "axis", "split_seed", "nominal_dose"])
        .agg(
            achieved_opportunity=("normalized_mutual_information", "mean"),
            metadata_prior_opportunity=("metadata_prior_opportunity", "mean"),
            opportunity_min=("candidate_min", "mean"),
            opportunity_max=("candidate_max", "mean"),
        )
        .reset_index()
    )
    return summary.merge(
        opportunity,
        on=["dataset", "task", "axis", "split_seed", "nominal_dose"],
        validate="many_to_one",
    )


def preflight_plan_cache(args: argparse.Namespace) -> dict[tuple, tuple]:
    cells_per_setting = args.folds**2
    if args.max_fold_cells is not None:
        cells_per_setting = min(cells_per_setting, args.max_fold_cells)
    total = (
        len(args.datasets)
        * len(args.tasks)
        * len(args.seeds)
        * cells_per_setting
        * len(args.axes)
    )
    progress = tqdm(total=total, desc="Preflight dose plans", unit="plan", dynamic_ncols=True)
    cache: dict[tuple, tuple] = {}
    for specification in args.datasets:
        dataset, path = specification.split("=", 1)
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["subject_id"] = frame.subject_id.astype(str)
        frame["trial_id"] = frame.trial_id.astype(str)
        for split_seed in args.seeds:
            subject_fold, stimulus_fold = matched.fold_assignments(frame, args.folds, split_seed)
            fold_cells = [(s, t) for s in range(args.folds) for t in range(args.folds)]
            if args.max_fold_cells is not None:
                fold_cells = fold_cells[: args.max_fold_cells]
            for task in args.tasks:
                labels = (
                    frame[f"{task}_score"].to_numpy(float) >= args.label_threshold
                ).astype(int)
                for subject_test, stimulus_test in fold_cells:
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
                    fold = f"s{subject_test}_t{stimulus_test}"
                    for axis in args.axes:
                        key = (dataset, task, split_seed, subject_test, stimulus_test, axis)
                        cache[key] = axis_plan(
                            frame,
                            labels,
                            subject_fold,
                            stimulus_fold,
                            subject_test,
                            stimulus_test,
                            train_sets,
                            test,
                            axis,
                            args.doses,
                            args.candidate_draws,
                            args.intervention_block_fraction,
                            stable_seed(dataset, task, axis, split_seed, fold),
                        )
                        progress.update(1)
    progress.close()
    return cache


def main() -> None:
    args = parse_args()
    admitted = require_crossed_datasets(args.datasets, context="Counterfactual dose experiment")
    if len(set(args.doses)) != len(args.doses):
        raise ValueError("Duplicate doses are not allowed")
    if not 0.0 < args.intervention_block_fraction < 1.0:
        raise ValueError("--intervention-block-fraction must lie strictly between 0 and 1")
    output_files = [args.output_root / name for name in ("predictions.csv", "split_audit.csv", "summary.csv")]
    if any(path.exists() for path in output_files) and not args.force:
        raise FileExistsError(f"Output exists under {args.output_root}; pass --force to replace it")
    plan_cache = preflight_plan_cache(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    cells_per_setting = args.folds**2
    if args.max_fold_cells is not None:
        cells_per_setting = min(cells_per_setting, args.max_fold_cells)
    total = (
        len(args.datasets)
        * len(args.tasks)
        * len(args.seeds)
        * cells_per_setting
        * len(args.representations)
        * len(args.models)
        * (1 + len(args.axes) * len(args.doses))
    )
    progress = tqdm(total=total, desc="Counterfactual identity dose", unit="fit", dynamic_ncols=True)
    rows: list[dict] = []
    audit_rows: list[dict] = []
    for specification in args.datasets:
        dataset, path = specification.split("=", 1)
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["subject_id"] = frame.subject_id.astype(str)
        frame["trial_id"] = frame.trial_id.astype(str)
        feature_sets = {
            name: frame[representation_columns(frame, name)].to_numpy(float)
            for name in args.representations
        }
        for split_seed in args.seeds:
            subject_fold, stimulus_fold = matched.fold_assignments(frame, args.folds, split_seed)
            templates = make_shortcut_models(
                split_seed,
                args.n_jobs,
                args.models,
                device=args.device,
                gpu_epochs=args.gpu_epochs,
            )
            fold_cells = [(s, t) for s in range(args.folds) for t in range(args.folds)]
            if args.max_fold_cells is not None:
                fold_cells = fold_cells[: args.max_fold_cells]
            for task in args.tasks:
                labels = (
                    frame[f"{task}_score"].to_numpy(float) >= args.label_threshold
                ).astype(int)
                for subject_test, stimulus_test in fold_cells:
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
                    fold = f"s{subject_test}_t{stimulus_test}"
                    plans = {}
                    for axis in args.axes:
                        key = (dataset, task, split_seed, subject_test, stimulus_test, axis)
                        (
                            fixed,
                            identities,
                            blocks,
                            removed_control,
                            candidates,
                            intervention_class_counts,
                        ) = plan_cache[key]
                        plans[axis] = (fixed, blocks)
                        reference_counts = np.bincount(labels[train_sets["unseen_both"]], minlength=2)
                        for block in blocks:
                            train = np.concatenate([fixed, block.indices])
                            counts = np.bincount(labels[train], minlength=2)
                            coverage = len(set(identities[test]) & set(identities[train])) / len(set(identities[test]))
                            if len(train) != len(train_sets["unseen_both"]) or not np.array_equal(counts, reference_counts):
                                raise ValueError(f"Dose contract failed for {dataset}/{task}/{axis}/{fold}")
                            if set(train) & set(test) or not np.isclose(coverage, 1.0):
                                raise ValueError(f"Identity exposure contract failed for {dataset}/{task}/{axis}/{fold}")
                            audit_rows.append(
                                {
                                    "dataset": dataset,
                                    "task": task,
                                    "axis": axis,
                                    "split_seed": split_seed,
                                    "fold": fold,
                                    "nominal_dose": block.nominal_dose,
                                    "achieved_dose": block.achieved_dose,
                                    "normalized_mutual_information": block.opportunity["normalized_mutual_information"],
                                    "metadata_prior_opportunity": metadata_prior_opportunity(
                                        labels[block.indices],
                                        identities[block.indices],
                                        labels[test],
                                        identities[test],
                                    ),
                                    "conditional_label_entropy": block.opportunity["conditional_label_entropy"],
                                    "identity_rate_std": block.opportunity["identity_rate_std"],
                                    "identity_rate_range": block.opportunity["identity_rate_range"],
                                    "candidate_min": block.candidate_min,
                                    "candidate_max": block.candidate_max,
                                    "candidate_size": len(candidates),
                                    "intervention_block_size": len(block.indices),
                                    "intervention_class_0": intervention_class_counts[0],
                                    "intervention_class_1": intervention_class_counts[1],
                                    "removed_control_size": len(removed_control),
                                    "intervention_block_fraction": args.intervention_block_fraction,
                                    "n_train": len(train),
                                    "n_test": len(test),
                                    "class_0_train": counts[0],
                                    "class_1_train": counts[1],
                                    "identity_coverage": coverage,
                                    "row_overlap_with_test": 0,
                                }
                            )
                    for representation, features in feature_sets.items():
                        for model_name, template in templates.items():
                            baseline = clone(template).fit(features[train_sets["unseen_both"]], labels[train_sets["unseen_both"]])
                            baseline_probability = benchmark.model_probability(baseline, features[test])
                            progress.update(1)
                            for axis in args.axes:
                                rows.extend(
                                    prediction_rows(
                                        frame,
                                        test,
                                        labels,
                                        baseline_probability,
                                        dataset=dataset,
                                        task=task,
                                        axis=axis,
                                        representation=representation,
                                        model=model_name,
                                        split_seed=split_seed,
                                        fold=fold,
                                        condition="unseen",
                                        nominal_dose=None,
                                        achieved_dose=None,
                                    )
                                )
                                fixed, blocks = plans[axis]
                                for block in blocks:
                                    train = np.concatenate([fixed, block.indices])
                                    fitted = clone(template).fit(features[train], labels[train])
                                    probability = benchmark.model_probability(fitted, features[test])
                                    rows.extend(
                                        prediction_rows(
                                            frame,
                                            test,
                                            labels,
                                            probability,
                                            dataset=dataset,
                                            task=task,
                                            axis=axis,
                                            representation=representation,
                                            model=model_name,
                                            split_seed=split_seed,
                                            fold=fold,
                                            condition="dose",
                                            nominal_dose=block.nominal_dose,
                                            achieved_dose=block.achieved_dose,
                                        )
                                    )
                                    progress.update(1)
    progress.close()
    predictions = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    result = summarize(predictions, audit)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    audit.to_csv(args.output_root / "split_audit.csv", index=False)
    result.to_csv(args.output_root / "summary.csv", index=False)
    manifest = {
        "datasets": args.datasets,
        "admitted_datasets": admitted,
        "tasks": args.tasks,
        "label_threshold": args.label_threshold,
        "axes": args.axes,
        "representations": args.representations,
        "models": args.models,
        "doses": args.doses,
        "candidate_draws": args.candidate_draws,
        "intervention_block_fraction": args.intervention_block_fraction,
        "split_seeds": args.seeds,
        "folds": args.folds,
        "max_fold_cells": args.max_fold_cells,
        "device": args.device,
        "gpu_epochs": args.gpu_epochs,
        "intervention": (
            "replace a fixed class-matched unseen-control sub-block with an "
            "identity-complete real-row exposure block"
        ),
        "dose_metric": "normalized mutual information between identity and label within the exposure block",
        "fixed_quantities": ["test rows", "training size", "binary class counts", "identity coverage"],
        "eppvr_excluded_reason": "No verified cross-participant physical stimulus map; validated separately on the subject axis.",
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Results written to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
