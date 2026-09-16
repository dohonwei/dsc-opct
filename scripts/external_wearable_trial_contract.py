from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrialBundleContract:
    dataset: str
    task: str
    representations: tuple[str, ...]
    minimum_subjects: int
    minimum_trials_per_subject: int
    subject_folds: int
    stimulus_folds: int
    exact_trials_per_subject: int | None = None
    exact_stimulus_count: int | None = None
    require_both_classes_per_subject: bool = False
    require_label_constant_by_stimulus: bool = False
    required_stimuli_per_class: int | None = None


def _normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "trial_id"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Trial table lacks required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Trial table is empty")
    if frame.subject_id.isna().any() or frame.trial_id.isna().any():
        raise ValueError("Subject and trial identifiers must be non-missing")

    normalized = frame.loc[:, ["subject_id", "trial_id"]].copy()
    normalized["subject_id"] = normalized.subject_id.astype(str).str.strip()
    if normalized.subject_id.eq("").any():
        raise ValueError("Subject identifiers must be non-empty")
    trial_numeric = pd.to_numeric(normalized.trial_id, errors="coerce")
    if trial_numeric.isna().any() or not np.equal(trial_numeric, np.floor(trial_numeric)).all():
        raise ValueError("Physical trial identifiers must be finite integers")
    normalized["trial_id"] = trial_numeric.astype(np.int64)
    if normalized.duplicated(["subject_id", "trial_id"]).any():
        raise ValueError("Each subject-by-physical-stimulus pair must be unique")
    return normalized


def _validated_labels(labels: np.ndarray, n_rows: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != n_rows:
        raise ValueError("Task labels must be a one-dimensional row-aligned array")
    try:
        numeric = values.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Task labels must already be binary numeric values") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("Task labels contain missing, midpoint, or non-integer values")
    binary = numeric.astype(np.int64)
    if set(np.unique(binary)) != {0, 1}:
        raise ValueError("Task labels must contain both registered binary classes")
    return binary


def _validated_feature_sets(
    feature_sets: Mapping[str, np.ndarray],
    representations: tuple[str, ...],
    n_rows: int,
) -> dict[str, np.ndarray]:
    if set(feature_sets) != set(representations):
        raise ValueError(
            "Feature bundle must contain exactly the registered representations"
        )
    validated = {}
    for representation in representations:
        values = np.asarray(feature_sets[representation])
        if values.ndim != 2 or values.shape[0] != n_rows or values.shape[1] == 0:
            raise ValueError(
                f"Representation {representation} must be a non-empty row-aligned matrix"
            )
        try:
            numeric = values.astype(float, copy=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Representation {representation} contains nonnumeric values"
            ) from error
        if not np.isfinite(numeric).all():
            raise ValueError(f"Representation {representation} contains nonfinite values")
        validated[representation] = numeric
    return validated


def validate_trial_bundle(
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_sets: Mapping[str, np.ndarray],
    contract: TrialBundleContract,
) -> dict[str, object]:
    if not contract.dataset.strip() or not contract.task.strip():
        raise ValueError("Dataset and task names must be non-empty")
    if contract.minimum_subjects < contract.subject_folds:
        raise ValueError("Minimum subjects cannot be smaller than the subject-fold count")
    if contract.minimum_trials_per_subject < 2:
        raise ValueError("At least two trials per subject are required")

    normalized = _normalized_frame(frame)
    binary = _validated_labels(labels, len(normalized))
    validated_features = _validated_feature_sets(
        feature_sets, contract.representations, len(normalized)
    )
    audit = normalized.assign(label=binary)
    subject_counts = audit.groupby("subject_id", sort=False).size()
    stimulus_count = int(audit.trial_id.nunique())
    if len(subject_counts) < contract.minimum_subjects:
        raise ValueError("Eligible participant count is below the registered minimum")
    if stimulus_count < contract.stimulus_folds:
        raise ValueError("Physical-stimulus count is below the registered fold count")
    if (subject_counts < contract.minimum_trials_per_subject).any():
        raise ValueError("At least one participant has too few retained trials")

    if contract.exact_stimulus_count is not None and stimulus_count != contract.exact_stimulus_count:
        raise ValueError("Physical-stimulus count differs from the registered exact count")
    if contract.exact_trials_per_subject is not None:
        expected = contract.exact_trials_per_subject
        if not subject_counts.eq(expected).all():
            raise ValueError("At least one participant lacks the exact registered trial count")
        common_trials = frozenset(audit.trial_id.unique())
        subject_trial_sets = audit.groupby("subject_id").trial_id.apply(
            lambda values: frozenset(values.tolist())
        )
        if not subject_trial_sets.map(lambda values: values == common_trials).all():
            raise ValueError("Participants do not share the same physical-stimulus set")

    if contract.require_both_classes_per_subject:
        subject_classes = audit.groupby("subject_id").label.nunique()
        if not subject_classes.eq(2).all():
            raise ValueError("Each retained participant must contribute both label classes")

    if contract.require_label_constant_by_stimulus:
        stimulus_classes = audit.groupby("trial_id").label.nunique()
        if not stimulus_classes.eq(1).all():
            raise ValueError("A fixed stimulus label changed across participants")
        if contract.required_stimuli_per_class is not None:
            stimulus_labels = audit.groupby("trial_id").label.first()
            counts = stimulus_labels.value_counts().to_dict()
            expected = contract.required_stimuli_per_class
            if counts != {0: expected, 1: expected}:
                raise ValueError("Fixed stimulus labels do not have the registered class balance")

    return {
        "dataset": contract.dataset,
        "task": contract.task,
        "n_rows": int(len(audit)),
        "n_subjects": int(audit.subject_id.nunique()),
        "n_stimuli": stimulus_count,
        "class_counts": {
            str(key): int(value) for key, value in audit.label.value_counts().sort_index().items()
        },
        "trials_per_subject_min": int(subject_counts.min()),
        "trials_per_subject_max": int(subject_counts.max()),
        "feature_dimensions": {
            name: int(values.shape[1]) for name, values in validated_features.items()
        },
        "status": "adapter_output_contract_passed",
        "claim_boundary": (
            "This audit establishes only structural compatibility of adapter output. "
            "It is not endpoint, effectiveness, non-harm, safety, or replication evidence."
        ),
    }
