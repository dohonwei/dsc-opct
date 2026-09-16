from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from emognition_v11_schema_adapter import (
    EmognitionAdapterProfile,
    build_trial_record,
    parse_muse_payload,
    parse_questionnaire_payload,
)
from emognition_v11_contract import QUALITY_KEYS, RAW_CHANNEL_KEYS, STIMULUS_NAMES


def expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("Unknown Emognition schema was accepted")


def pair_series(values: np.ndarray) -> list[list[object]]:
    start = datetime(2026, 1, 1, 12, 0, 0)
    return [
        [(start + timedelta(microseconds=3906 * index)).strftime("%Y-%m-%dT%H:%M:%S:%f"), float(value)]
        for index, value in enumerate(values)
    ]


def muse_payload() -> dict[str, object]:
    samples = 2048
    time = np.arange(samples) / 256.0
    payload: dict[str, object] = {}
    for index, key in enumerate(RAW_CHANNEL_KEYS):
        payload[key] = pair_series(
            25.0 * np.sin(2 * np.pi * (6.0 + index) * time) + 0.1 * index
        )
    payload["HeadBandOn"] = pair_series(np.ones(samples))
    for key in QUALITY_KEYS[1:]:
        payload[key] = pair_series(np.ones(samples))
    return payload


def questionnaire_payload() -> dict[str, object]:
    return {
        "metadata": {"id": 22, "movie_order": ["BASELINE", *STIMULUS_NAMES]},
        "questionnaires": [
            {
                "movie": stimulus,
                "emotions": {},
                "sam": {
                    "VALENCE": 6 if index % 2 else 4,
                    "AROUSAL": 4 if index % 2 else 6,
                    "MOTIVATION": 5,
                },
            }
            for index, stimulus in enumerate(STIMULUS_NAMES)
        ],
    }


def main() -> None:
    questionnaire = parse_questionnaire_payload(questionnaire_payload(), "22")
    assert set(questionnaire["ratings"]) == set(STIMULUS_NAMES)

    payload = muse_payload()
    data, quality = parse_muse_payload(payload)
    assert data.shape == (4, 2048)
    assert quality["usable_samples"] == 2048
    row, _ = build_trial_record(
        payload,
        subject_id="22",
        stimulus_name="AMUSEMENT",
        stimulus_index=0,
    )
    assert row["n_windows"] == 3
    assert len([key for key in row if key.startswith("stimulus__")]) == 144

    unknown_profile = EmognitionAdapterProfile(raw_value_policy="assumed_microvolts")
    expect_value_error(parse_muse_payload, payload, unknown_profile)

    incomplete_questionnaire = questionnaire_payload()
    incomplete_questionnaire["questionnaires"][0]["sam"] = {"VALENCE": 6}
    parsed_incomplete = parse_questionnaire_payload(incomplete_questionnaire, "22")
    assert parsed_incomplete["ratings"]["AMUSEMENT"]["arousal"] is None

    bad_questionnaire = questionnaire_payload()
    bad_questionnaire["questionnaires"][0]["sam"] = []
    expect_value_error(parse_questionnaire_payload, bad_questionnaire, "22")

    unknown_hsi = muse_payload()
    unknown_hsi["HSI_AF7"][0][1] = 3
    expect_value_error(parse_muse_payload, unknown_hsi)

    misaligned = muse_payload()
    misaligned["RAW_AF8"][0][0] = "2026-01-01T13:00:00:000000"
    expect_value_error(parse_muse_payload, misaligned)
    print(
        "Emognition schema adapter synthetic test passed: documented nesting, "
        "144-feature construction, and fail-closed schema checks"
    )


if __name__ == "__main__":
    main()
