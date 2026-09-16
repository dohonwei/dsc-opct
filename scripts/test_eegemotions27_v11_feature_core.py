from __future__ import annotations

import numpy as np

from eegemotions27_v11_contract import REQUIRED_CHANNELS
from eegemotions27_v11_feature_core import (
    representation_columns,
    trial_features,
    window_band_features,
)
from amigos_v11_feature_core import window_band_features as amigos_window_features


def synthetic_trial(sampling_rate: float = 256.0, seconds: float = 12.0) -> np.ndarray:
    time = np.arange(int(round(sampling_rate * seconds))) / sampling_rate
    rows = []
    for channel_index in range(len(REQUIRED_CHANNELS)):
        rows.append(
            (1.0 + 0.03 * channel_index) * np.sin(2 * np.pi * 6.0 * time)
            + (0.8 + 0.02 * channel_index) * np.sin(2 * np.pi * 10.0 * time)
            + (0.5 + 0.01 * channel_index) * np.sin(2 * np.pi * 20.0 * time)
            + 0.25 * np.sin(2 * np.pi * 35.0 * time)
            + 0.001 * channel_index * time
        )
    return np.asarray(rows, dtype=float)


def main() -> None:
    sampling_rate = 256.0
    data = synthetic_trial(sampling_rate)
    row = trial_features(data, sampling_rate, "synthetic-01", 4)
    car = data - data.mean(axis=0, keepdims=True)
    absolute, relative = window_band_features(car, sampling_rate)
    amigos_absolute, amigos_relative = amigos_window_features(car, sampling_rate)
    np.testing.assert_allclose(absolute, amigos_absolute, rtol=0, atol=0)
    np.testing.assert_allclose(relative, amigos_relative, rtol=0, atol=0)

    columns = representation_columns(row)
    assert len(columns["all"]) == 504
    assert len(columns["relative_power"]) == 168
    assert len(columns["normalized_asymmetry"]) == 84
    assert row["emotion_id"] == 4
    assert row["n_windows"] == 5
    print("EEGEmotions-27 feature-core synthetic equivalence test passed")


if __name__ == "__main__":
    main()
