from __future__ import annotations

import numpy as np
import pandas as pd

from eegemotions27_v11_contract import (
    MINIMUM_PARTICIPANTS,
    MINIMUM_TRIALS_PER_PARTICIPANT,
    NEGATIVE_EMOTION_IDS,
    POSITIVE_EMOTION_IDS,
    REGISTERED_EMOTION_IDS,
    REPRESENTATIONS,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
)
from external_wearable_trial_contract import TrialBundleContract, validate_trial_bundle


def validate_eegemotions27_task_bundle(
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    task: str,
) -> dict[str, object]:
    if task != "category_polarity":
        raise ValueError(f"Unregistered EEGEmotions-27 task: {task}")
    if "emotion_id" not in frame:
        raise ValueError("EEGEmotions-27 trial table lacks emotion_id")

    emotion_numeric = pd.to_numeric(frame.emotion_id, errors="coerce")
    if emotion_numeric.isna().any() or not np.equal(
        emotion_numeric, np.floor(emotion_numeric)
    ).all():
        raise ValueError("Emotion IDs must be finite integers")
    emotion_ids = emotion_numeric.astype(np.int64).to_numpy()
    trial_ids = pd.to_numeric(frame.trial_id, errors="coerce").to_numpy()
    if not np.array_equal(emotion_ids, trial_ids):
        raise ValueError("Physical trial_id must equal the official emotion_id")
    if not set(np.unique(emotion_ids)).issubset(set(REGISTERED_EMOTION_IDS)):
        raise ValueError("Trial table contains an unregistered emotion category")

    expected_labels = np.array(
        [1 if value in POSITIVE_EMOTION_IDS else 0 for value in emotion_ids],
        dtype=np.int64,
    )
    if not np.array_equal(np.asarray(labels), expected_labels):
        raise ValueError("Labels do not match the frozen category-polarity mapping")

    present_positive = set(emotion_ids) & set(POSITIVE_EMOTION_IDS)
    present_negative = set(emotion_ids) & set(NEGATIVE_EMOTION_IDS)
    if present_positive != set(POSITIVE_EMOTION_IDS):
        raise ValueError("All four registered positive stimuli must be present globally")
    if len(present_negative) < STIMULUS_FOLDS:
        raise ValueError("At least four registered negative stimuli must be present globally")

    audit = validate_trial_bundle(
        frame,
        expected_labels,
        feature_sets,
        TrialBundleContract(
            dataset="EEGEmotions-27",
            task=task,
            representations=REPRESENTATIONS,
            minimum_subjects=MINIMUM_PARTICIPANTS,
            minimum_trials_per_subject=MINIMUM_TRIALS_PER_PARTICIPANT,
            subject_folds=SUBJECT_FOLDS,
            stimulus_folds=STIMULUS_FOLDS,
            require_both_classes_per_subject=True,
            require_label_constant_by_stimulus=True,
        ),
    )
    audit["registered_positive_stimuli_present"] = sorted(present_positive)
    audit["registered_negative_stimuli_present"] = sorted(present_negative)
    audit["construct_boundary"] = (
        "Labels encode fixed category-derived affective polarity, not individual "
        "experienced valence or arousal."
    )
    return audit
