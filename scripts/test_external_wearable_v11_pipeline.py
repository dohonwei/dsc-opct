from __future__ import annotations

import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd

from develop_distribution_covered_stratified_opct_v11 import (
    cluster_geometry,
    select_distribution_covered_audit,
)
from external_wearable_dcs_opct_core import (
    ExternalApplicationConfig,
    load_unlabeled_configuration_table,
    write_blinded_action_bundle,
)
from external_wearable_dose_core import DoseRunConfig, prepare_dose_design
from external_wearable_v11_pipeline import (
    finalize_locked_outcomes,
    load_prepared_payload,
    lock_unlabeled_action_from_bundle,
    run_locked_prepared_outcomes,
    sha256,
    verify_preoutcome_design_bundle,
    write_preoutcome_design_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
RISK_ROOT = ROOT / "outputs/identity_shortcut_risk_classifier"


def synthetic_trial_table() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(20260909)
    rows = []
    features = []
    valence = []
    arousal = []
    for subject in range(24):
        signature = rng.normal(size=8)
        for stimulus in range(10):
            rows.append({"subject_id": f"S{subject:02d}", "trial_id": stimulus})
            valence.append((subject + stimulus) % 2)
            arousal.append((subject // 2 + stimulus) % 2)
            features.append(
                signature
                + 0.2 * valence[-1]
                + 0.1 * arousal[-1]
                + rng.normal(scale=0.15, size=8)
            )
    return (
        pd.DataFrame(rows),
        {"valence": np.asarray(valence), "arousal": np.asarray(arousal)},
        {"all": np.asarray(features, dtype=float)},
    )


def main() -> None:
    frame, task_labels, feature_sets = synthetic_trial_table()
    seeds = (20260813, 20260829, 20260911, 20260923, 20261007)
    prepared = {}
    for task, labels in task_labels.items():
        prepared[task] = prepare_dose_design(
            frame,
            labels,
            feature_sets,
            DoseRunConfig(
                dataset="SYNTHETIC-WEARABLE",
                task=task,
                subject_folds=2,
                stimulus_folds=2,
                seeds=seeds,
                doses=(0.0, 0.25, 0.5, 0.75, 1.0),
                representations=("all",),
                models=("linear_logistic",),
                gpu_epochs=40,
                candidate_draws=32,
                intervention_block_fraction=0.5,
            ),
        )
    config = ExternalApplicationConfig(
        dataset="SYNTHETIC-WEARABLE",
        tasks=("valence", "arousal"),
        representations=("all",),
        models=("linear_logistic",),
        seeds=seeds,
    )
    mechanism_summary = pd.concat(
        [prepared[task]["mechanism_summary"] for task in config.tasks],
        ignore_index=True,
    )
    identity_encoding = prepared[config.tasks[0]]["identity_encoding_margin"]
    split_audit = pd.concat(
        [prepared[task]["split_audit"] for task in config.tasks],
        ignore_index=True,
    )
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    risk_manifest = json.loads(
        (RISK_ROOT / "freeze_manifest.json").read_text(encoding="utf-8")
    )
    risk_model = joblib.load(RISK_ROOT / "frozen_risk_model.joblib")
    public_risk_table = pd.read_csv(RISK_ROOT / "risk_learning_table.csv")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "pipeline"
        lock_path, lock_hash = write_preoutcome_design_bundle(
            root,
            mechanism_summary=mechanism_summary,
            identity_encoding=identity_encoding,
            split_audit=split_audit,
            prepared_payload=prepared,
            config=config,
            provenance={"fixture": "synthetic_preoutcome_state_machine"},
        )
        lock, recovered_config = verify_preoutcome_design_bundle(root, lock_hash)
        assert recovered_config == config
        assert lock["status"] == "external_preoutcome_design_locked"
        recovered = load_prepared_payload(root, lock_hash)
        assert set(recovered) == set(config.tasks)

        api_root = Path(directory) / "formal_action_api"
        _, api_preoutcome_hash = write_preoutcome_design_bundle(
            api_root,
            mechanism_summary=mechanism_summary,
            identity_encoding=identity_encoding,
            split_audit=split_audit,
            prepared_payload=prepared,
            config=config,
            provenance={"fixture": "formal_action_api_regression"},
        )
        api_action_path, api_action_hash = lock_unlabeled_action_from_bundle(
            api_root,
            preoutcome_lock_sha256=api_preoutcome_hash,
            public_risk_table=public_risk_table,
            risk_model=risk_model,
            risk_manifest=risk_manifest,
            freeze=freeze,
            freeze_path=FREEZE,
            device="cuda",
        )
        assert api_action_path.is_file()
        assert sha256(api_action_path) == api_action_hash
        api_action = json.loads(api_action_path.read_text(encoding="utf-8"))
        assert api_action["provenance"]["preoutcome_lock_sha256"] == api_preoutcome_hash
        assert api_action["provenance"]["freeze_sha256"] == sha256(FREEZE)

        original_lock = lock_path.read_bytes()
        lock_path.write_bytes(original_lock + b"\n")
        try:
            verify_preoutcome_design_bundle(root, lock_hash)
        except RuntimeError:
            pass
        else:
            raise AssertionError("A modified pre-outcome lock was accepted")
        lock_path.write_bytes(original_lock)
        assert sha256(lock_path) == lock_hash

        target = load_unlabeled_configuration_table(
            root / "preoutcome/mechanism_summary.csv",
            root / "preoutcome/identity_encoding_margin.csv",
            config,
        )
        dose_index = target.nominal_dose.map({0.25: 0, 0.5: 1, 0.75: 2, 1.0: 3})
        target["probability_identity"] = dose_index.map(
            {0: 0.35, 1: 0.45, 2: 0.55, 3: 0.65}
        )
        target["probability_wg_opct"] = dose_index.map(
            {0: 0.15, 1: 0.30, 2: 0.70, 3: 0.85}
        )
        components = pd.DataFrame(
            [
                {
                    "method": "coral",
                    "applicable": True,
                    "probability_rank": 1.0,
                    "order_inversions": 0,
                },
                {
                    "method": "quantile_mapping",
                    "applicable": True,
                    "probability_rank": 1.0,
                    "order_inversions": 0,
                },
            ]
        )
        geometry = cluster_geometry(target)
        assignment = select_distribution_covered_audit(geometry)
        action_lock_path, action_lock_hash = write_blinded_action_bundle(
            root / "action",
            {
                "mechanism": target,
                "components": components,
                "witness": {
                    "directional_agreement": 1.0,
                    "displacement_cosine": 1.0,
                    "normalized_disagreement": 0.0,
                    "witness_pass": True,
                },
                "candidate_applicable": True,
                "candidate_method": "wg_opct",
                "geometry": geometry,
                "assignment": assignment,
            },
            {
                "preoutcome_lock_sha256": lock_hash,
                "freeze_sha256": sha256(FREEZE),
            },
        )
        assert action_lock_path.is_file()

        original_action_lock = action_lock_path.read_bytes()
        action_lock_path.write_bytes(original_action_lock + b"\n")
        try:
            run_locked_prepared_outcomes(
                root,
                preoutcome_lock_sha256=lock_hash,
                action_lock_sha256=action_lock_hash,
                n_jobs=1,
                device="cuda",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Model training accepted a modified action lock")
        action_lock_path.write_bytes(original_action_lock)

        mechanism_path = root / "action/mechanism_probabilities_blinded.csv"
        original_mechanism = mechanism_path.read_bytes()
        mechanism_path.write_bytes(original_mechanism + b"\n")
        try:
            run_locked_prepared_outcomes(
                root,
                preoutcome_lock_sha256=lock_hash,
                action_lock_sha256=action_lock_hash,
                n_jobs=1,
                device="cuda",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Model training accepted a modified action artifact")
        mechanism_path.write_bytes(original_mechanism)

        generated_summary = run_locked_prepared_outcomes(
            root,
            preoutcome_lock_sha256=lock_hash,
            action_lock_sha256=action_lock_hash,
            n_jobs=1,
            device="cuda",
        )
        assert generated_summary.is_file()
        model_manifest = json.loads(
            (root / "model_outcomes/model_outcome_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert model_manifest["preoutcome_lock_sha256"] == lock_hash
        assert model_manifest["action_lock_sha256"] == action_lock_hash
        assert model_manifest["artifact_sha256"]["summary"] == sha256(
            generated_summary
        )

        full_summary = mechanism_summary.copy()
        full_summary["exposure_effect"] = np.where(
            full_summary.nominal_dose.eq(0.0),
            0.0,
            np.where(full_summary.nominal_dose.ge(0.75), 0.03, 0.01),
        )
        bad_summary = full_summary.copy()
        bad_summary.loc[0, "achieved_opportunity"] += 0.01
        bad_path = Path(directory) / "bad_summary.csv"
        bad_summary.to_csv(bad_path, index=False)
        try:
            finalize_locked_outcomes(
                root,
                preoutcome_lock_sha256=lock_hash,
                action_lock_sha256=action_lock_hash,
                full_summary_path=bad_path,
                freeze=freeze,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("A changed post-outcome mechanism value was accepted")

        summary_path = Path(directory) / "full_summary.csv"
        full_summary.to_csv(summary_path, index=False)
        gate = finalize_locked_outcomes(
            root,
            preoutcome_lock_sha256=lock_hash,
            action_lock_sha256=action_lock_hash,
            full_summary_path=summary_path,
            freeze=freeze,
        )
        assert gate["claim_supported"] is True
        assert gate["selected_method"] == "wg_opct"
        assert gate["released_action_nonharm_pass"] is True
        manifest = json.loads(
            (root / "outcome/external_confirmation_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["preoutcome_lock_sha256"] == lock_hash
        assert manifest["action_lock_sha256"] == action_lock_hash

    with tempfile.TemporaryDirectory() as directory:
        changed = mechanism_summary.copy()
        changed.loc[0, "achieved_opportunity"] += 0.01
        try:
            write_preoutcome_design_bundle(
                Path(directory) / "invalid",
                mechanism_summary=changed,
                identity_encoding=identity_encoding,
                split_audit=split_audit,
                prepared_payload=prepared,
                config=config,
                provenance={"fixture": "must_fail"},
            )
        except ValueError:
            pass
        else:
            raise AssertionError("A mechanism table inconsistent with its prepared payload was accepted")

    print(
        "External wearable v11 state-machine test passed: prepared-design lock, "
        "action lock, tamper rejection, mechanism immutability, and final gate"
    )


if __name__ == "__main__":
    main()
