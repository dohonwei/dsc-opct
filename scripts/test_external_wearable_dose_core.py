from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from external_wearable_dose_core import (
    DoseRunConfig,
    prepare_dose_design,
    run_prepared_dose_outcomes,
    validate_inputs,
)


def synthetic_table() -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(20260909)
    rows = []
    labels = []
    features = []
    for subject in range(24):
        subject_signature = rng.normal(size=8)
        for stimulus in range(10):
            label = (subject + stimulus) % 2
            rows.append({"subject_id": f"S{subject:02d}", "trial_id": stimulus})
            labels.append(label)
            features.append(
                subject_signature
                + 0.35 * label
                + 0.10 * stimulus
                + rng.normal(scale=0.12, size=8)
            )
    return (
        pd.DataFrame(rows),
        np.asarray(labels, dtype=int),
        {"all": np.asarray(features, dtype=float)},
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the external wearable dry run")
    frame, labels, feature_sets = synthetic_table()
    config = DoseRunConfig(
        dataset="SYNTHETIC-WEARABLE",
        task="valence",
        subject_folds=2,
        stimulus_folds=2,
        seeds=(20260909,),
        doses=(0.0, 0.25, 0.5, 0.75, 1.0),
        representations=("all",),
        models=("linear_logistic", "gpu_mlp"),
        gpu_epochs=40,
        candidate_draws=256,
        intervention_block_fraction=0.5,
    )
    prepared = prepare_dose_design(frame, labels, feature_sets, config)
    assert len(prepared["mechanism_summary"]) == 10
    assert not {
        "probability",
        "exposure_effect",
        "dose_induced_amplification",
        "material_optimism_event",
    } & set(prepared["mechanism_summary"])
    outputs = run_prepared_dose_outcomes(prepared, n_jobs=1, device="cuda")
    summary = outputs["summary"]
    audit = outputs["split_audit"]
    predictions = outputs["predictions"]
    encoding = outputs["identity_encoding_margin"]
    assert len(summary) == 10
    assert len(audit) == 20
    assert len(predictions) == 2880
    assert set(summary.model) == {"linear_logistic", "gpu_mlp"}
    assert set(summary.nominal_dose) == set(config.doses)
    assert np.isfinite(summary.exposure_effect).all()
    assert summary.groupby(["representation", "model", "split_seed"]).size().eq(5).all()
    assert summary.groupby(["representation", "model", "split_seed"]).nominal_dose.apply(
        lambda values: np.allclose(values.sort_values().to_numpy(), config.doses)
    ).all()
    assert (audit.identity_coverage == 1.0).all()
    assert (audit.row_overlap_with_test == 0).all()
    assert (audit.n_train == 60).all()
    assert (audit.n_test == 60).all()
    assert (audit.intervention_block_size == 30).all()
    assert (audit.removed_control_size == 30).all()
    assert (audit.class_0_train + audit.class_1_train == audit.n_train).all()
    assert (
        audit.intervention_class_0 + audit.intervention_class_1
        == audit.intervention_block_size
    ).all()
    assert np.allclose(audit.loc[audit.nominal_dose.eq(0), "achieved_dose"], 0.0)
    assert np.allclose(audit.loc[audit.nominal_dose.eq(1), "achieved_dose"], 1.0)
    assert audit.groupby(["split_seed", "fold"]).nominal_dose.apply(
        lambda values: np.allclose(values.sort_values().to_numpy(), config.doses)
    ).all()
    assert predictions.probability.between(0, 1).all()
    assert predictions.groupby("row_id").size().eq(12).all()
    assert predictions.loc[predictions.condition.eq("unseen")].nominal_dose.isna().all()
    assert predictions.loc[predictions.condition.eq("dose")].nominal_dose.notna().all()
    assert len(encoding) == 2
    assert encoding.encoding_margin.notna().all()
    assert set(encoding.model) == {"linear", "rbf"}

    invalid_cases = (
        (labels.astype(float) + 0.25, config),
        (labels, replace(config, doses=(0.0, 0.5, 0.25, 1.0))),
        (labels, replace(config, subject_folds=25)),
    )
    for invalid_labels, invalid_config in invalid_cases:
        try:
            validate_inputs(frame, invalid_labels, feature_sets, invalid_config)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid standard-table contract was accepted")

    duplicate_frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    duplicate_labels = np.append(labels, labels[0])
    duplicate_features = {"all": np.vstack([feature_sets["all"], feature_sets["all"][[0]]])}
    try:
        validate_inputs(duplicate_frame, duplicate_labels, duplicate_features, config)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate subject-trial rows were accepted")
    print(
        "External wearable dose-core GPU dry run passed: "
        f"{len(predictions)} predictions, {len(summary)} summary rows"
    )


if __name__ == "__main__":
    main()
