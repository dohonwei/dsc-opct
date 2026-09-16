from __future__ import annotations

import numpy as np

from amigos_v11_contract import REQUIRED_CHANNELS
from amigos_v11_feature_core import (
    representation_columns,
    trial_features,
    window_band_features,
)
from build_faced_trial_features import window_band_features as faced_window_features


def synthetic_trial(sampling_rate: float = 128.0, seconds: float = 12.0) -> np.ndarray:
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
    sampling_rate = 128.0
    data = synthetic_trial(sampling_rate)
    common_artifact = 3.0 * np.sin(
        2 * np.pi * 1.0 * np.arange(data.shape[1]) / sampling_rate
    )
    base_row = trial_features(data, sampling_rate, "synthetic-01", 1)
    artifact_row = trial_features(
        data + common_artifact[None, :], sampling_rate, "synthetic-01", 1
    )
    base_values = np.asarray(
        [base_row[key] for key in sorted(base_row) if key.startswith("stimulus__")]
    )
    artifact_values = np.asarray(
        [artifact_row[key] for key in sorted(artifact_row) if key.startswith("stimulus__")]
    )
    np.testing.assert_allclose(base_values, artifact_values, rtol=1e-8, atol=1e-8)

    car = data - data.mean(axis=0, keepdims=True)
    absolute, relative = window_band_features(car, sampling_rate)
    faced_absolute, faced_relative = faced_window_features(car, sampling_rate)
    np.testing.assert_allclose(absolute, faced_absolute, rtol=0, atol=0)
    np.testing.assert_allclose(relative, faced_relative, rtol=0, atol=0)

    columns = representation_columns(base_row)
    assert len(columns["all"]) == 504
    assert len(columns["relative_power"]) == 168
    assert len(columns["normalized_asymmetry"]) == 84
    assert base_row["n_windows"] == 5
    print("AMIGOS feature core synthetic equivalence test passed")


if __name__ == "__main__":
    main()
