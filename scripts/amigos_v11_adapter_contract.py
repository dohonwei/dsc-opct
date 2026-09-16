from __future__ import annotations

import numpy as np
import pandas as pd

from amigos_v11_contract import (
    MINIMUM_PARTICIPANTS,
    REPRESENTATIONS,
    SHORT_VIDEO_COUNT,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
)
from external_wearable_trial_contract import TrialBundleContract, validate_trial_bundle


def validate_amigos_task_bundle(
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_sets: dict[str, np.ndarray],
    task: str,
) -> dict[str, object]:
    if task not in {"valence", "arousal"}:
        raise ValueError(f"Unregistered AMIGOS task: {task}")
    return validate_trial_bundle(
        frame,
        labels,
        feature_sets,
        TrialBundleContract(
            dataset="AMIGOS",
            task=task,
            representations=REPRESENTATIONS,
            minimum_subjects=MINIMUM_PARTICIPANTS,
            minimum_trials_per_subject=SHORT_VIDEO_COUNT,
            subject_folds=SUBJECT_FOLDS,
            stimulus_folds=STIMULUS_FOLDS,
            exact_trials_per_subject=SHORT_VIDEO_COUNT,
            exact_stimulus_count=SHORT_VIDEO_COUNT,
            require_both_classes_per_subject=True,
            require_label_constant_by_stimulus=True,
            required_stimuli_per_class=SHORT_VIDEO_COUNT // 2,
        ),
    )
