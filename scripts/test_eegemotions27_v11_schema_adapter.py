from __future__ import annotations

import io

import numpy as np

from eegemotions27_v11_contract import REQUIRED_CHANNELS
from eegemotions27_v11_schema_adapter import (
    EEGEmotions27AdapterProfile,
    build_trial_record,
    parse_raw_bytes,
)


def expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("Unknown EEGEmotions-27 raw schema was accepted")


def synthetic_signal(samples: int = 3072) -> np.ndarray:
    time = np.arange(samples) / 256.0
    return np.stack(
        [
            15.0 * np.sin(2 * np.pi * (6.0 + 0.15 * index) * time)
            + 8.0 * np.sin(2 * np.pi * (10.0 + 0.1 * index) * time)
            + 3.0 * np.sin(2 * np.pi * 20.0 * time)
            for index in range(len(REQUIRED_CHANNELS))
        ],
        axis=1,
    )


def encoded(matrix: np.ndarray, delimiter: str) -> bytes:
    handle = io.StringIO()
    np.savetxt(handle, matrix, delimiter=delimiter, fmt="%.8f")
    return handle.getvalue().encode("ascii")


def main() -> None:
    samples_by_channels = synthetic_signal()
    whitespace = encoded(samples_by_channels, " ")
    parsed, schema = parse_raw_bytes(whitespace)
    assert parsed.shape == (14, 3072)
    assert schema["delimiter"] == "whitespace"
    assert schema["orientation"] == "samples_by_channels"

    comma = encoded(samples_by_channels.T, ",")
    parsed_comma, schema_comma = parse_raw_bytes(comma)
    assert parsed_comma.shape == (14, 3072)
    assert schema_comma["delimiter"] == "comma"
    assert schema_comma["orientation"] == "channels_by_samples"

    row, _ = build_trial_record(
        whitespace,
        subject_id="synthetic-01",
        emotion_id=4,
    )
    assert row["emotion_id"] == 4
    assert len([key for key in row if key.startswith("stimulus__")]) == 504

    expect_value_error(parse_raw_bytes, encoded(samples_by_channels[:, :13], " "))
    expect_value_error(parse_raw_bytes, encoded(samples_by_channels[:1024], " "))
    nonfinite = samples_by_channels.copy()
    nonfinite[100, 2] = np.nan
    expect_value_error(parse_raw_bytes, encoded(nonfinite, " "))
    unknown_profile = EEGEmotions27AdapterProfile(sampling_rate_hz=128)
    expect_value_error(parse_raw_bytes, whitespace, unknown_profile)
    print(
        "EEGEmotions-27 schema adapter synthetic test passed: two registered "
        "delimiters, two orientations, 504 features, and fail-closed checks"
    )


if __name__ == "__main__":
    main()
