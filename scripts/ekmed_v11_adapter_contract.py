from __future__ import annotations

import numpy as np
import pandas as pd

from ekmed_v11_contract import (
    MINIMUM_PARTICIPANTS,
    MINIMUM_TRIALS_PER_PARTICIPANT,
    REPRESENTATIONS,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
)
from external_wearable_trial_contract import TrialBundleContract, validate_trial_bundle


def validate_ekmed_task_bundle(
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    task: str,
) -> dict[str, object]:
    if task not in {"valence", "arousal"}:
        raise ValueError(f"Unregistered EKM-ED task: {task}")
    return validate_trial_bundle(
        frame,
        labels,
        feature_sets,
        TrialBundleContract(
            dataset="EmoKey Moments Muse EEG Dataset (EKM-ED)",
            task=task,
            representations=REPRESENTATIONS,
            minimum_subjects=MINIMUM_PARTICIPANTS,
            minimum_trials_per_subject=MINIMUM_TRIALS_PER_PARTICIPANT,
            subject_folds=SUBJECT_FOLDS,
            stimulus_folds=STIMULUS_FOLDS,
            exact_stimulus_count=4,
            require_both_classes_per_subject=True,
        ),
    )
