from __future__ import annotations

import numpy as np
import pandas as pd

from emognition_v11_adapter_contract import validate_emognition_task_bundle


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
    raise AssertionError("Invalid Emognition adapter output was accepted")


def main() -> None:
    frame = pd.DataFrame(
        [
            {"subject_id": f"E{subject:02d}", "trial_id": trial}
            for subject in range(30)
            for trial in range(10)
            if trial != subject % 10
        ]
    )
    subject_number = frame.subject_id.str[1:].astype(int)
    labels = ((frame.trial_id + subject_number) % 2).to_numpy()
    feature_sets = features(len(frame))
    audit = validate_emognition_task_bundle(frame, labels, feature_sets, "arousal")
    assert audit["n_rows"] == 270
    assert audit["trials_per_subject_min"] == 9

    midpoint = labels.astype(object)
    midpoint[0] = None
    expect_value_error(
        validate_emognition_task_bundle,
        frame,
        midpoint,
        feature_sets,
        "arousal",
    )
    one_class = labels.copy()
    one_class[:9] = 0
    expect_value_error(
        validate_emognition_task_bundle,
        frame,
        one_class,
        feature_sets,
        "arousal",
    )
    nonfinite = dict(feature_sets)
    nonfinite["all"] = nonfinite["all"].copy()
    nonfinite["all"][0, 0] = np.nan
    expect_value_error(
        validate_emognition_task_bundle,
        frame,
        labels,
        nonfinite,
        "arousal",
    )
    print("Emognition adapter-output contract test passed")


if __name__ == "__main__":
    main()
