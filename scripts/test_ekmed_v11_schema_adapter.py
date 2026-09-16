from __future__ import annotations

import numpy as np
import pandas as pd

from ekmed_v11_schema_adapter import (
    EKMAdapterProfile,
    QUESTIONNAIRE_COLUMNS,
    SIGNAL_COLUMNS,
    parse_questionnaire_frame,
    parse_signal_frame,
)


def expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("Unknown EKM-ED schema or invalid trial was accepted")


def questionnaire_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [index, 1, 4 if index % 2 == 0 else 6, 6 if index % 2 == 0 else 4, stimulus]
            for index, stimulus in enumerate(("ANGER", "SADNESS", "HAPPINESS", "FEAR"))
        ],
        columns=["Unnamed: 0", "ID", "VALENCE", "AROUSAL", "EMOTION"],
    )


def signal_frame(samples: int = 31_000) -> pd.DataFrame:
    time = np.arange(samples, dtype=float) / 128.0
    data = {column: np.zeros(samples, dtype=float) for column in SIGNAL_COLUMNS}
    data["TimeStamp"] = [f"sample-{index}" for index in range(samples)]
    for index, channel in enumerate(("TP9", "AF7", "AF8", "TP10")):
        data[f"RAW_{channel}"] = (
            20.0 * np.sin(2.0 * np.pi * (6.0 + index) * time)
            + 5.0 * np.sin(2.0 * np.pi * 18.0 * time)
            + 0.01 * time
        )
    return pd.DataFrame(data, columns=SIGNAL_COLUMNS)


def main() -> None:
    ratings = parse_questionnaire_frame(questionnaire_frame())
    assert set(ratings) == {"1"}
    assert ratings["1"]["ANGER"] == {"valence": 4.0, "arousal": 6.0}

    row, quality = parse_signal_frame(
        signal_frame(), subject_id="1", stimulus_name="FEAR", stimulus_index=3
    )
    assert row["subject_id"] == "1"
    assert row["trial_id"] == 3
    assert row["key_moment_count"] == 3
    assert quality["key_moment_windows"] == 3
    assert len([key for key in row if key.startswith("stimulus__")]) == 144

    duplicate = questionnaire_frame()
    duplicate = pd.concat([duplicate, duplicate.iloc[[0]]], ignore_index=True)
    expect_value_error(parse_questionnaire_frame, duplicate)

    unknown_columns = signal_frame(100)
    unknown_columns = unknown_columns.rename(columns={"RAW_AF7": "EEG_AF7"})
    expect_value_error(
        parse_signal_frame,
        unknown_columns,
        subject_id="1",
        stimulus_name="ANGER",
        stimulus_index=0,
    )

    duplicate_time = signal_frame()
    duplicate_time.loc[1, "TimeStamp"] = duplicate_time.loc[0, "TimeStamp"]
    expect_value_error(
        parse_signal_frame,
        duplicate_time,
        subject_id="1",
        stimulus_name="ANGER",
        stimulus_index=0,
    )

    nonfinite = signal_frame()
    nonfinite.loc[int(83 * 128), "RAW_TP9"] = np.nan
    expect_value_error(
        parse_signal_frame,
        nonfinite,
        subject_id="1",
        stimulus_name="ANGER",
        stimulus_index=0,
    )

    expect_value_error(
        parse_signal_frame,
        signal_frame(),
        subject_id="1",
        stimulus_name="ANGER",
        stimulus_index=0,
        profile=EKMAdapterProfile(sampling_rate_hz=256),
    )
    assert tuple(QUESTIONNAIRE_COLUMNS) == (
        "__index__", "ID", "VALENCE", "AROUSAL", "EMOTION"
    )
    print(
        "EKM-ED schema-adapter synthetic test passed: exact headers, frozen row-order "
        "alignment, public key moments, 144 features, and fail-closed checks"
    )


if __name__ == "__main__":
    main()
