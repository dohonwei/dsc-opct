from __future__ import annotations

import numpy as np

from emognition_v11_contract import BANDS_HZ, BILATERAL_PAIRS, REQUIRED_CHANNELS
from wearable_eeg_v11_feature_core import (
    representation_columns,
    trial_features as generic_trial_features,
    window_band_features as generic_window_band_features,
)


def window_band_features(
    data_uv: np.ndarray, sampling_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    return generic_window_band_features(
        data_uv, sampling_rate, REQUIRED_CHANNELS, BANDS_HZ
    )


def trial_features(
    data_uv: np.ndarray,
    sampling_rate: float,
    subject_id: str,
    stimulus_index: int,
    stimulus_name: str,
) -> dict[str, object]:
    row = generic_trial_features(
        data_uv,
        sampling_rate,
        subject_id,
        int(stimulus_index),
        REQUIRED_CHANNELS,
        BILATERAL_PAIRS,
        BANDS_HZ,
    )
    row["stimulus_index"] = int(stimulus_index)
    row["stimulus_name"] = str(stimulus_name)
    return row
