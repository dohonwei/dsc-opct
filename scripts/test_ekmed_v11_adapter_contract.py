from __future__ import annotations

import numpy as np
import pandas as pd

from ekmed_v11_adapter_contract import validate_ekmed_task_bundle


REPRESENTATION_DIMS = {
    "all": 144,
    "relative_power": 48,
    "normalized_asymmetry": 24,
}


def features(n_rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260916)
    return {
        name: rng.normal(size=(n_rows, dimension))
        for name, dimension in REPRESENTATION_DIMS.items()
    }


def expect_value_error(function, *args) -> None:
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("Invalid EKM-ED adapter output was accepted")


def main() -> None:
    frame = pd.DataFrame(
        [
            {"subject_id": f"K{subject:02d}", "trial_id": trial}
            for subject in range(30)
            for trial in range(4)
        ]
    )
    subject_number = frame.subject_id.str[1:].astype(int)
    labels = ((frame.trial_id + subject_number) % 2).to_numpy()
    feature_sets = features(len(frame))
    audit = validate_ekmed_task_bundle(frame, labels, feature_sets, "valence")
    assert audit["n_rows"] == 120
    assert audit["n_stimuli"] == 4
    assert audit["trials_per_subject_min"] == 4

    missing_trials = frame.drop(index=[0, 1]).reset_index(drop=True)
    missing_labels = labels[2:]
    missing_features = {name: values[2:] for name, values in feature_sets.items()}
    expect_value_error(
        validate_ekmed_task_bundle,
        missing_trials,
        missing_labels,
        missing_features,
        "valence",
    )
    midpoint = labels.astype(object)
    midpoint[0] = None
    expect_value_error(
        validate_ekmed_task_bundle,
        frame,
        midpoint,
        feature_sets,
        "valence",
    )
    nonfinite = dict(feature_sets)
    nonfinite["all"] = nonfinite["all"].copy()
    nonfinite["all"][0, 0] = np.nan
    expect_value_error(
        validate_ekmed_task_bundle,
        frame,
        labels,
        nonfinite,
        "valence",
    )
    print("EKM-ED adapter-output contract test passed")


if __name__ == "__main__":
    main()
