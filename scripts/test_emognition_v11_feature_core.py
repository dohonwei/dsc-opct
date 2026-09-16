from __future__ import annotations

import numpy as np

from emognition_v11_contract import REQUIRED_CHANNELS, sam_binary_label
from emognition_v11_feature_core import (
    representation_columns,
    trial_features,
)


def synthetic_trial(sampling_rate: float = 256.0, seconds: float = 12.0) -> np.ndarray:
    time = np.arange(int(round(sampling_rate * seconds))) / sampling_rate
    rows = []
    for channel_index in range(len(REQUIRED_CHANNELS)):
        rows.append(
            (1.0 + 0.04 * channel_index) * np.sin(2 * np.pi * 6.0 * time)
            + (0.7 + 0.03 * channel_index) * np.sin(2 * np.pi * 10.0 * time)
            + (0.45 + 0.02 * channel_index) * np.sin(2 * np.pi * 20.0 * time)
            + 0.20 * np.sin(2 * np.pi * 35.0 * time)
            + 0.001 * channel_index * time
        )
    return np.asarray(rows, dtype=float)


def main() -> None:
    sampling_rate = 256.0
    data = synthetic_trial(sampling_rate)
    common_artifact = 2.5 * np.sin(
        2 * np.pi * 1.0 * np.arange(data.shape[1]) / sampling_rate
    )
    base = trial_features(data, sampling_rate, "synthetic-22", 0, "amusement")
    artifact = trial_features(
        data + common_artifact[None, :],
        sampling_rate,
        "synthetic-22",
        0,
        "amusement",
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
    assert base["n_windows"] == 5
    assert sam_binary_label(1) == 0
    assert sam_binary_label(4) == 0
    assert sam_binary_label(5) is None
    assert sam_binary_label(6) == 1
    assert sam_binary_label(9) == 1
    for invalid in (0, 4.5, 10):
        try:
            sam_binary_label(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid SAM rating accepted: {invalid}")
    print("Emognition feature-core test passed with 144/48/24 dimensions")


if __name__ == "__main__":
    main()
