from __future__ import annotations

import numpy as np
import pandas as pd

from amigos_v11_adapter_contract import validate_amigos_task_bundle


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
    raise AssertionError("Invalid AMIGOS adapter output was accepted")


def main() -> None:
    frame = pd.DataFrame(
        [
            {"subject_id": f"A{subject:02d}", "trial_id": trial}
            for subject in range(30)
            for trial in range(16)
        ]
    )
    labels = (frame.trial_id >= 8).to_numpy(dtype=int)
    feature_sets = features(len(frame))
    audit = validate_amigos_task_bundle(frame, labels, feature_sets, "valence")
    assert audit["n_rows"] == 480
    assert audit["n_subjects"] == 30
    assert audit["n_stimuli"] == 16

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    expect_value_error(
        validate_amigos_task_bundle,
        duplicate,
        np.append(labels, labels[0]),
        features(len(duplicate)),
        "valence",
    )
    drifting_labels = labels.copy()
    drifting_labels[16] = 1 - drifting_labels[16]
    expect_value_error(
        validate_amigos_task_bundle,
        frame,
        drifting_labels,
        feature_sets,
        "valence",
    )
    missing_representation = dict(feature_sets)
    missing_representation.pop("normalized_asymmetry")
    expect_value_error(
        validate_amigos_task_bundle,
        frame,
        labels,
        missing_representation,
        "valence",
    )
    print("AMIGOS adapter-output contract test passed")


if __name__ == "__main__":
    main()
