from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ekmed_v11_contract import BANDS_HZ, BILATERAL_PAIRS, REQUIRED_CHANNELS
from wearable_eeg_v11_feature_core import (
    STEP_SECONDS,
    representation_columns,
    trial_features as generic_trial_features,
)


def trial_features_from_key_windows(
    key_windows_uv: Sequence[np.ndarray],
    sampling_rate: float,
    subject_id: str,
    stimulus_index: int,
    stimulus_name: str,
) -> dict[str, object]:
    windows = [np.asarray(window, dtype=float) for window in key_windows_uv]
    expected_samples = int(round(4.0 * sampling_rate))
    if len(windows) < 3:
        raise ValueError("At least three registered key-moment windows are required")
    for window in windows:
        if window.shape != (len(REQUIRED_CHANNELS), expected_samples):
            raise ValueError(
                "Each key-moment window must contain the four registered channels "
                f"and exactly {expected_samples} samples; found {window.shape}"
            )
        if not np.isfinite(window).all():
            raise ValueError("Key-moment windows must be finite before feature construction")

    gap_samples = int(round(STEP_SECONDS * sampling_rate))
    gap = np.full((len(REQUIRED_CHANNELS), gap_samples), np.nan, dtype=float)
    pieces = []
    for index, window in enumerate(windows):
        if index:
            pieces.append(gap)
        pieces.append(window)
    separated = np.concatenate(pieces, axis=1)
    row = generic_trial_features(
        separated,
        sampling_rate,
        subject_id,
        int(stimulus_index),
        REQUIRED_CHANNELS,
        BILATERAL_PAIRS,
        BANDS_HZ,
    )
    if row["n_windows"] != len(windows):
        raise RuntimeError(
            "Separated key-moment construction produced an unexpected valid-window count"
        )
    row["stimulus_index"] = int(stimulus_index)
    row["stimulus_name"] = str(stimulus_name)
    row["key_moment_count"] = len(windows)
    return row


__all__ = ["representation_columns", "trial_features_from_key_windows"]
