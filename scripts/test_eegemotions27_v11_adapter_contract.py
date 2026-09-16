from __future__ import annotations

import numpy as np
import pandas as pd

from eegemotions27_v11_adapter_contract import validate_eegemotions27_task_bundle
from eegemotions27_v11_contract import NEGATIVE_EMOTION_IDS, POSITIVE_EMOTION_IDS


REPRESENTATION_DIMS = {
    "all": 12,
    "relative_power": 6,
    "normalized_asymmetry": 3,
}


def features(n_rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260909)
    return {
        name: rng.normal(size=(n_rows, dimension))
        for name, dimension in REPRESENTATION_DIMS.items()
    }


def expect_value_error(function, *args) -> None:
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("Invalid EEGEmotions-27 adapter output was accepted")


def main() -> None:
    emotion_ids = POSITIVE_EMOTION_IDS + NEGATIVE_EMOTION_IDS
    frame = pd.DataFrame(
        [
            {
                "subject_id": f"E{subject:02d}",
                "trial_id": emotion_id,
                "emotion_id": emotion_id,
            }
            for subject in range(30)
            for emotion_id in emotion_ids
        ]
    )
    labels = frame.emotion_id.isin(POSITIVE_EMOTION_IDS).to_numpy(dtype=int)
    feature_sets = features(len(frame))
    audit = validate_eegemotions27_task_bundle(
        frame, labels, feature_sets, "category_polarity"
    )
    assert audit["n_rows"] == 300
    assert audit["n_subjects"] == 30
    assert audit["n_stimuli"] == 10

    ambiguous = frame.copy()
    ambiguous.loc[0, ["trial_id", "emotion_id"]] = 1
    expect_value_error(
        validate_eegemotions27_task_bundle,
        ambiguous,
        labels,
        feature_sets,
        "category_polarity",
    )
    wrong_label = labels.copy()
    wrong_label[0] = 1 - wrong_label[0]
    expect_value_error(
        validate_eegemotions27_task_bundle,
        frame,
        wrong_label,
        feature_sets,
        "category_polarity",
    )
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    expect_value_error(
        validate_eegemotions27_task_bundle,
        duplicate,
        np.append(labels, labels[0]),
        features(len(duplicate)),
        "category_polarity",
    )
    missing_representation = dict(feature_sets)
    missing_representation.pop("normalized_asymmetry")
    expect_value_error(
        validate_eegemotions27_task_bundle,
        frame,
        labels,
        missing_representation,
        "category_polarity",
    )
    print("EEGEmotions-27 adapter-output contract test passed")


if __name__ == "__main__":
    main()
