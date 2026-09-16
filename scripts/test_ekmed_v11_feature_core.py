from __future__ import annotations

import numpy as np

from ekmed_v11_contract import REQUIRED_CHANNELS, sam_binary_label
from ekmed_v11_feature_core import (
    representation_columns,
    trial_features_from_key_windows,
)


def synthetic_window(
    sampling_rate: float,
    phase: float,
) -> np.ndarray:
    time = np.arange(int(round(4.0 * sampling_rate))) / sampling_rate
    rows = []
    for channel_index in range(len(REQUIRED_CHANNELS)):
        rows.append(
            (1.0 + 0.04 * channel_index) * np.sin(2 * np.pi * 6.0 * time + phase)
            + (0.7 + 0.03 * channel_index)
            * np.sin(2 * np.pi * 10.0 * time + phase)
            + (0.45 + 0.02 * channel_index)
            * np.sin(2 * np.pi * 20.0 * time + phase)
            + 0.20 * np.sin(2 * np.pi * 35.0 * time + phase)
            + 0.001 * channel_index * time
        )
    return np.asarray(rows, dtype=float)


def main() -> None:
    sampling_rate = 128.0
    windows = [synthetic_window(sampling_rate, phase) for phase in (0.0, 0.2, 0.4)]
    common = [
        window
        + 2.5
        * np.sin(
            2 * np.pi * 1.0 * np.arange(window.shape[1]) / sampling_rate
        )[None, :]
        for window in windows
    ]
    base = trial_features_from_key_windows(
        windows, sampling_rate, "synthetic-09", 0, "anger"
    )
    artifact = trial_features_from_key_windows(
        common, sampling_rate, "synthetic-09", 0, "anger"
    )
    keys = sorted(key for key in base if key.startswith("stimulus__"))
    np.testing.assert_allclose(
        [base[key] for key in keys],
        [artifact[key] for key in keys],
        rtol=1e-8,
        atol=1e-8,
    )
    columns = representation_columns(base)
    assert len(columns["all"]) == 144
    assert len(columns["relative_power"]) == 48
    assert len(columns["normalized_asymmetry"]) == 24
    assert base["n_windows"] == 3
    assert base["key_moment_count"] == 3
    assert sam_binary_label(1) == 0
    assert sam_binary_label(4) == 0
    assert sam_binary_label(5) is None
    assert sam_binary_label(6) == 1
    assert sam_binary_label(9) == 1
    print("EKM-ED feature-core test passed with 144/48/24 dimensions")


if __name__ == "__main__":
    main()
