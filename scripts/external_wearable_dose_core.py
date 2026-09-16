from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from identity_shortcut.core import metadata_prior_opportunity, stable_seed  # noqa: E402
from identity_shortcut.models import make_shortcut_models  # noqa: E402
import run_counterfactual_identity_dose as dose  # noqa: E402
import run_crossed_identity_probes as probes  # noqa: E402
import run_paired_block_exposure_benchmark as paired  # noqa: E402
import run_protocol_model_benchmark as benchmark  # noqa: E402


@dataclass(frozen=True)
class DoseRunConfig:
    dataset: str
    task: str
    subject_folds: int
    stimulus_folds: int
    seeds: tuple[int, ...]
    doses: tuple[float, ...]
    representations: tuple[str, ...]
    models: tuple[str, ...]
    gpu_epochs: int
    candidate_draws: int = 256
    intervention_block_fraction: float = 0.5
    axis: str = "subject"


def validate_inputs(
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    config: DoseRunConfig,
) -> None:
    if not config.dataset.strip() or not config.task.strip():
        raise ValueError("Dataset and task identifiers must be non-empty")
    if config.subject_folds < 2 or config.stimulus_folds < 2:
        raise ValueError("Subject and stimulus fold counts must both be at least two")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Split seeds must be non-empty and unique")
    if not config.representations or len(set(config.representations)) != len(
        config.representations
    ):
        raise ValueError("Representations must be non-empty and unique")
    if not config.models or len(set(config.models)) != len(config.models):
        raise ValueError("Models must be non-empty and unique")
    if config.gpu_epochs < 1:
        raise ValueError("GPU epochs must be positive")
    if config.candidate_draws < max(32, 4 * len(config.doses)):
        raise ValueError("Candidate draws do not satisfy the dose-construction minimum")
    required = {"subject_id", "trial_id"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Standard trial table is missing columns: {sorted(missing)}")
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(frame) != len(labels):
        raise ValueError("Trial table and labels have different row counts")
    if not np.isfinite(labels).all() or not np.allclose(labels, np.round(labels)):
        raise ValueError("Labels must be finite binary integers")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("External wearable task must contain both binary classes")
    if config.axis != "subject":
        raise ValueError("The external wearable confirmation is restricted to subject identity")
    if (
        not config.doses
        or len(set(config.doses)) != len(config.doses)
        or config.doses[0] != 0.0
        or any(left >= right for left, right in zip(config.doses, config.doses[1:]))
        or any(value < 0.0 or value > 1.0 for value in config.doses)
    ):
        raise ValueError(
            "Dose grid must be strictly increasing, unique, within [0, 1], and start at zero"
        )
    if not 0 < config.intervention_block_fraction < 1:
        raise ValueError("Intervention block fraction must lie strictly between zero and one")
    if set(feature_sets) != set(config.representations):
        raise ValueError("Feature representations do not match the registered configuration")
    for name, values in feature_sets.items():
        if values.ndim != 2 or values.shape[0] != len(frame) or values.shape[1] == 0:
            raise ValueError(f"Representation {name} has an invalid shape: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Representation {name} contains non-finite values")
    numeric_trial = pd.to_numeric(frame.trial_id, errors="coerce")
    if numeric_trial.isna().any() or not np.allclose(numeric_trial, np.round(numeric_trial)):
        raise ValueError("trial_id must be a stable integer physical-stimulus identifier")
    if frame.subject_id.isna().any() or frame.subject_id.astype(str).str.strip().eq("").any():
        raise ValueError("subject_id contains a missing or empty identifier")
    if frame.duplicated(["subject_id", "trial_id"]).any():
        raise ValueError("The standard trial table must contain one row per subject-trial pair")
    if frame.subject_id.nunique() < config.subject_folds:
        raise ValueError("There are fewer subjects than registered subject folds")
    if numeric_trial.nunique() < config.stimulus_folds:
        raise ValueError("There are fewer physical stimuli than registered stimulus folds")


def fold_assignments(
    frame: pd.DataFrame, config: DoseRunConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    subject_map = benchmark.shuffled_fold_map(
        frame.subject_id.astype(str).to_numpy(), config.subject_folds, seed
    )
    stimulus_map = benchmark.shuffled_fold_map(
        frame.trial_id.astype(int).to_numpy(), config.stimulus_folds, seed + 1
    )
    return (
        frame.subject_id.astype(str).map(subject_map).to_numpy(int),
        frame.trial_id.astype(int).map(stimulus_map).to_numpy(int),
    )


def preflight(
    frame: pd.DataFrame, labels: np.ndarray, config: DoseRunConfig
) -> dict[tuple[int, int, int], tuple]:
    cache = {}
    total = len(config.seeds) * config.subject_folds * config.stimulus_folds
    progress = tqdm(
        total=total,
        desc=f"{config.dataset} preflight dose plans",
        unit="plan",
        dynamic_ncols=True,
    )
    for split_seed in config.seeds:
        subject_fold, stimulus_fold = fold_assignments(frame, config, split_seed)
        for subject_test in range(config.subject_folds):
            for stimulus_test in range(config.stimulus_folds):
                test, train_sets, _ = paired.paired_train_sets(
                    frame,
                    labels,
                    subject_fold,
                    stimulus_fold,
                    subject_test,
                    stimulus_test,
                    config.dataset,
                    config.task,
                    split_seed,
                )
                fold = f"s{subject_test}_t{stimulus_test}"
                cache[(split_seed, subject_test, stimulus_test)] = dose.axis_plan(
                    frame,
                    labels,
                    subject_fold,
                    stimulus_fold,
                    subject_test,
                    stimulus_test,
                    train_sets,
                    test,
                    config.axis,
                    list(config.doses),
                    config.candidate_draws,
                    config.intervention_block_fraction,
                    stable_seed(
                        config.dataset,
                        config.task,
                        config.axis,
                        split_seed,
                        fold,
                    ),
                )
                progress.update(1)
    progress.close()
    return cache


def identity_encoding_table(
    frame: pd.DataFrame,
    feature_sets: dict[str, np.ndarray],
    config: DoseRunConfig,
) -> pd.DataFrame:
    rows = []
    total = len(feature_sets) * len(config.seeds) * config.stimulus_folds * 2
    progress = tqdm(
        total=total,
        desc=f"{config.dataset} subject-identity probe",
        unit="fit",
        dynamic_ncols=True,
    )
    targets = frame.subject_id.astype(str).to_numpy()
    for representation, features in feature_sets.items():
        for split_seed in config.seeds:
            group_map = benchmark.shuffled_fold_map(
                frame.trial_id.astype(int).to_numpy(),
                config.stimulus_folds,
                split_seed,
            )
            folds = frame.trial_id.astype(int).map(group_map).to_numpy(int)
            for model_name, template in probes.make_models(split_seed).items():
                scores = []
                for fold in range(config.stimulus_folds):
                    test = folds == fold
                    missing = set(targets[test]) - set(targets[~test])
                    if missing:
                        raise ValueError(
                            f"Identity probe has unseen training classes: {sorted(missing)}"
                        )
                    fitted = clone(template).fit(features[~test], targets[~test])
                    scores.append(
                        probes.balanced_multiclass_accuracy(
                            targets[test], fitted.predict(features[test])
                        )
                    )
                    progress.update(1)
                rows.append(
                    {
                        "dataset": config.dataset,
                        "representation": representation,
                        "n_features": features.shape[1],
                        "model": model_name,
                        "split_seed": split_seed,
                        "chance": 1.0 / frame.subject_id.nunique(),
                        "balanced_accuracy": float(np.mean(scores)),
                    }
                )
    progress.close()
    table = pd.DataFrame(rows)
    table["encoding_margin"] = table.balanced_accuracy - table.chance
    return table


def audit_rows_for_plan(
    frame: pd.DataFrame,
    labels: np.ndarray,
    test: np.ndarray,
    train_sets: dict[str, np.ndarray],
    plan: tuple,
    config: DoseRunConfig,
    split_seed: int,
    fold: str,
) -> tuple[tuple[np.ndarray, list], list[dict[str, object]]]:
    fixed, identities, blocks, removed, candidates, class_counts = plan
    unseen = np.asarray(train_sets["unseen_both"], dtype=int)
    fixed = np.asarray(fixed, dtype=int)
    removed = np.asarray(removed, dtype=int)
    candidates = np.asarray(candidates, dtype=int)
    if len(np.unique(fixed)) != len(fixed) or len(np.unique(removed)) != len(removed):
        raise ValueError(f"Dose control partition contains duplicate rows: seed={split_seed}/{fold}")
    if set(fixed) & set(removed) or set(np.concatenate([fixed, removed])) != set(unseen):
        raise ValueError(f"Dose control partition is not exhaustive: seed={split_seed}/{fold}")
    if set(candidates) & set(test):
        raise ValueError(f"Dose candidate rows overlap the test set: seed={split_seed}/{fold}")
    if len(blocks) != len(config.doses) or not np.allclose(
        [block.nominal_dose for block in blocks], config.doses
    ):
        raise ValueError(f"Dose plan does not match the registered grid: seed={split_seed}/{fold}")
    reference_counts = np.bincount(labels[train_sets["unseen_both"]], minlength=2)
    output = []
    for block in blocks:
        if len(np.unique(block.indices)) != len(block.indices) or not set(
            block.indices
        ).issubset(set(candidates)):
            raise ValueError(f"Dose block is not a unique candidate subset: seed={split_seed}/{fold}")
        block_counts = np.bincount(labels[block.indices], minlength=2)
        if not np.array_equal(block_counts, class_counts):
            raise ValueError(f"Dose block class counts changed: seed={split_seed}/{fold}")
        if not (
            0.0 <= block.achieved_dose <= 1.0
            and block.candidate_min <= block.opportunity["normalized_mutual_information"]
            <= block.candidate_max
        ):
            raise ValueError(f"Dose block falls outside its candidate range: seed={split_seed}/{fold}")
        train = np.concatenate([fixed, block.indices])
        counts = np.bincount(labels[train], minlength=2)
        required_identities = set(identities[test])
        coverage = len(required_identities & set(identities[train])) / len(
            required_identities
        )
        if len(train) != len(train_sets["unseen_both"]) or not np.array_equal(
            counts, reference_counts
        ):
            raise ValueError(f"Dose matching failed: seed={split_seed}/{fold}")
        if set(train) & set(test) or not np.isclose(coverage, 1.0):
            raise ValueError(f"Identity-exposure contract failed: seed={split_seed}/{fold}")
        output.append(
            {
                "dataset": config.dataset,
                "task": config.task,
                "axis": config.axis,
                "split_seed": split_seed,
                "fold": fold,
                "nominal_dose": block.nominal_dose,
                "achieved_dose": block.achieved_dose,
                "normalized_mutual_information": block.opportunity[
                    "normalized_mutual_information"
                ],
                "metadata_prior_opportunity": metadata_prior_opportunity(
                    labels[block.indices],
                    identities[block.indices],
                    labels[test],
                    identities[test],
                ),
                "conditional_label_entropy": block.opportunity[
                    "conditional_label_entropy"
                ],
                "identity_rate_std": block.opportunity["identity_rate_std"],
                "identity_rate_range": block.opportunity["identity_rate_range"],
                "candidate_min": block.candidate_min,
                "candidate_max": block.candidate_max,
                "candidate_size": len(candidates),
                "intervention_block_size": len(block.indices),
                "intervention_class_0": class_counts[0],
                "intervention_class_1": class_counts[1],
                "removed_control_size": len(removed),
                "intervention_block_fraction": config.intervention_block_fraction,
                "n_train": len(train),
                "n_test": len(test),
                "class_0_train": counts[0],
                "class_1_train": counts[1],
                "identity_coverage": coverage,
                "row_overlap_with_test": 0,
            }
        )
    return (fixed, blocks), output


def mechanism_summary(
    split_audit: pd.DataFrame, config: DoseRunConfig
) -> pd.DataFrame:
    opportunity = (
        split_audit.groupby(
            ["dataset", "task", "axis", "split_seed", "nominal_dose"],
            as_index=False,
        )
        .agg(
            achieved_opportunity=("normalized_mutual_information", "mean"),
            metadata_prior_opportunity=("metadata_prior_opportunity", "mean"),
            opportunity_min=("candidate_min", "mean"),
            opportunity_max=("candidate_max", "mean"),
        )
    )
    frames = []
    for representation in config.representations:
        for model in config.models:
            current = opportunity.copy()
            current["representation"] = representation
            current["model"] = model
            frames.append(current)
    table = pd.concat(frames, ignore_index=True)
    columns = [
        "dataset",
        "task",
        "axis",
        "representation",
        "model",
        "split_seed",
        "nominal_dose",
        "achieved_opportunity",
        "metadata_prior_opportunity",
        "opportunity_min",
        "opportunity_max",
    ]
    table = table[columns].sort_values(columns[:7]).reset_index(drop=True)
    expected = (
        len(config.representations)
        * len(config.models)
        * len(config.seeds)
        * len(config.doses)
    )
    if len(table) != expected or not np.isfinite(
        table.select_dtypes(include=[np.number]).to_numpy(float)
    ).all():
        raise ValueError("Pre-outcome mechanism summary is incomplete or non-finite")
    return table


def prepare_dose_design(
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    config: DoseRunConfig,
) -> dict[str, Any]:
    frame = frame.reset_index(drop=True).copy()
    raw_labels = np.asarray(labels)
    normalized_features = {
        name: np.asarray(values, dtype=float) for name, values in feature_sets.items()
    }
    validate_inputs(frame, raw_labels, normalized_features, config)
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["trial_id"] = frame.trial_id.astype(int)
    labels = raw_labels.astype(int)
    feature_sets = normalized_features
    encoding = identity_encoding_table(frame, feature_sets, config)
    plans = preflight(frame, labels, config)
    audit_rows: list[dict[str, object]] = []
    cells: list[dict[str, Any]] = []
    for split_seed in config.seeds:
        subject_fold, stimulus_fold = fold_assignments(frame, config, split_seed)
        for subject_test in range(config.subject_folds):
            for stimulus_test in range(config.stimulus_folds):
                test, train_sets, _ = paired.paired_train_sets(
                    frame,
                    labels,
                    subject_fold,
                    stimulus_fold,
                    subject_test,
                    stimulus_test,
                    config.dataset,
                    config.task,
                    split_seed,
                )
                fold = f"s{subject_test}_t{stimulus_test}"
                axis_plan, plan_audit = audit_rows_for_plan(
                    frame,
                    labels,
                    test,
                    train_sets,
                    plans[(split_seed, subject_test, stimulus_test)],
                    config,
                    split_seed,
                    fold,
                )
                audit_rows.extend(plan_audit)
                fixed, blocks = axis_plan
                cells.append(
                    {
                        "split_seed": split_seed,
                        "fold": fold,
                        "test": test,
                        "unseen_train": train_sets["unseen_both"],
                        "fixed": fixed,
                        "blocks": blocks,
                    }
                )
    split_audit = pd.DataFrame(audit_rows)
    expected_audit = (
        len(config.seeds)
        * config.subject_folds
        * config.stimulus_folds
        * len(config.doses)
    )
    if len(split_audit) != expected_audit:
        raise ValueError(
            f"Expected {expected_audit} split-audit rows, found {len(split_audit)}"
        )
    return {
        "frame": frame,
        "labels": labels,
        "feature_sets": feature_sets,
        "config": config,
        "cells": cells,
        "split_audit": split_audit,
        "mechanism_summary": mechanism_summary(split_audit, config),
        "identity_encoding_margin": encoding,
    }


def run_prepared_dose_outcomes(
    prepared: dict[str, Any],
    *,
    n_jobs: int = -1,
    device: str = "cuda",
) -> dict[str, pd.DataFrame]:
    frame = prepared["frame"]
    labels = prepared["labels"]
    feature_sets = prepared["feature_sets"]
    config = prepared["config"]
    split_audit = prepared["split_audit"]
    cells = prepared["cells"]
    expected_cells = len(config.seeds) * config.subject_folds * config.stimulus_folds
    if len(cells) != expected_cells:
        raise ValueError("Prepared dose design has an unexpected number of crossed cells")
    rows: list[dict[str, object]] = []
    total = expected_cells * len(config.representations) * len(config.models) * (
        1 + len(config.doses)
    )
    progress = tqdm(
        total=total,
        desc=f"{config.dataset} counterfactual subject-identity dose",
        unit="fit",
        dynamic_ncols=True,
    )
    template_cache = {
        split_seed: make_shortcut_models(
            split_seed,
            n_jobs,
            list(config.models),
            device=device,
            gpu_epochs=config.gpu_epochs,
        )
        for split_seed in config.seeds
    }
    for cell in cells:
        split_seed = int(cell["split_seed"])
        fold = str(cell["fold"])
        test = np.asarray(cell["test"], dtype=int)
        unseen_train = np.asarray(cell["unseen_train"], dtype=int)
        fixed = np.asarray(cell["fixed"], dtype=int)
        blocks = cell["blocks"]
        templates = template_cache[split_seed]
        for representation, features in feature_sets.items():
            for model_name, template in templates.items():
                baseline = clone(template).fit(features[unseen_train], labels[unseen_train])
                baseline_probability = benchmark.model_probability(baseline, features[test])
                rows.extend(
                    dose.prediction_rows(
                        frame,
                        test,
                        labels,
                        baseline_probability,
                        dataset=config.dataset,
                        task=config.task,
                        axis=config.axis,
                        representation=representation,
                        model=model_name,
                        split_seed=split_seed,
                        fold=fold,
                        condition="unseen",
                        nominal_dose=None,
                        achieved_dose=None,
                    )
                )
                progress.update(1)
                for block in blocks:
                    train = np.concatenate([fixed, block.indices])
                    fitted = clone(template).fit(features[train], labels[train])
                    probability = benchmark.model_probability(fitted, features[test])
                    rows.extend(
                        dose.prediction_rows(
                            frame,
                            test,
                            labels,
                            probability,
                            dataset=config.dataset,
                            task=config.task,
                            axis=config.axis,
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
    summary = dose.summarize(predictions, split_audit)
    expected_summary = (
        len(config.representations)
        * len(config.models)
        * len(config.seeds)
        * len(config.doses)
    )
    if len(summary) != expected_summary:
        raise ValueError(f"Expected {expected_summary} summary rows, found {len(summary)}")
    mechanism_columns = prepared["mechanism_summary"].columns.tolist()
    mechanism_keys = [
        "dataset",
        "task",
        "axis",
        "representation",
        "model",
        "split_seed",
        "nominal_dose",
    ]
    locked = prepared["mechanism_summary"].set_index(mechanism_keys).sort_index()
    observed = summary[mechanism_columns].set_index(mechanism_keys).sort_index()
    mechanism_values = [column for column in mechanism_columns if column not in mechanism_keys]
    if not locked.index.equals(observed.index) or not np.allclose(
        locked[mechanism_values].to_numpy(float),
        observed[mechanism_values].to_numpy(float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Post-outcome summary changed the locked mechanism quantities")
    prediction_keys = [
        "dataset",
        "task",
        "axis",
        "representation",
        "model",
        "split_seed",
        "fold",
        "condition",
        "nominal_dose",
        "row_id",
    ]
    duplicate_key = predictions[prediction_keys].copy()
    duplicate_key["nominal_dose"] = duplicate_key.nominal_dose.fillna(-1.0)
    if duplicate_key.duplicated().any():
        raise ValueError("Prediction table contains duplicate condition-row keys")
    if not predictions.probability.between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Prediction table contains invalid probabilities")
    if not np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Dose summary contains non-finite numeric values")
    if not np.allclose(split_audit.identity_coverage, 1.0):
        raise ValueError("Split audit does not preserve complete identity exposure")
    if not split_audit.row_overlap_with_test.eq(0).all():
        raise ValueError("Split audit reports training-test row overlap")
    return {
        "predictions": predictions,
        "split_audit": split_audit,
        "summary": summary,
        "mechanism_summary": prepared["mechanism_summary"],
        "identity_encoding_margin": prepared["identity_encoding_margin"],
    }


def run_dose_experiment(
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    config: DoseRunConfig,
    *,
    n_jobs: int = -1,
    device: str = "cuda",
) -> dict[str, pd.DataFrame]:
    prepared = prepare_dose_design(frame, labels, feature_sets, config)
    return run_prepared_dose_outcomes(prepared, n_jobs=n_jobs, device=device)
