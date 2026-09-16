from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from external_wearable_dcs_opct_core import ExternalApplicationConfig
from external_wearable_dose_core import DoseRunConfig
from external_wearable_implementation_lock import (
    ImplementationLockSpec,
    build_implementation_lock,
)
from external_wearable_one_shot_core import (
    RegisteredTaskBundle,
    finalize_registered_outcomes,
    lock_registered_action,
    prepare_and_lock_preoutcome,
    run_registered_outcomes,
)
from external_wearable_trial_contract import TrialBundleContract, validate_trial_bundle


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
RESERVATION = ROOT / "docs/amigos_v11_external_confirmation_reservation.json"
RISK_ROOT = ROOT / "outputs/identity_shortcut_risk_classifier"
SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)


def validator(frame, labels, feature_sets, task):
    return validate_trial_bundle(
        frame,
        labels,
        feature_sets,
        TrialBundleContract(
            dataset="SYNTHETIC-WEARABLE",
            task=task,
            representations=("all",),
            minimum_subjects=20,
            minimum_trials_per_subject=10,
            subject_folds=2,
            stimulus_folds=2,
            exact_trials_per_subject=10,
            exact_stimulus_count=10,
            require_both_classes_per_subject=True,
            require_label_constant_by_stimulus=True,
            required_stimuli_per_class=5,
        ),
    )


def main() -> None:
    frame = pd.DataFrame(
        [
            {"subject_id": f"S{subject:02d}", "trial_id": trial}
            for subject in range(24)
            for trial in range(10)
        ]
    )
    rng = np.random.default_rng(20260909)
    subject_signal = np.repeat(rng.normal(size=(24, 8)), 10, axis=0)
    stimulus_signal = np.tile(rng.normal(scale=0.3, size=(10, 8)), (24, 1))
    feature_sets = {
        "all": subject_signal + stimulus_signal + rng.normal(scale=0.1, size=(len(frame), 8))
    }
    task_bundles = {
        "valence": RegisteredTaskBundle(
            frame, (frame.trial_id >= 5).to_numpy(dtype=int), feature_sets
        ),
        "arousal": RegisteredTaskBundle(
            frame, (frame.trial_id % 2).to_numpy(dtype=int), feature_sets
        ),
    }
    application_config = ExternalApplicationConfig(
        dataset="SYNTHETIC-WEARABLE",
        tasks=("valence", "arousal"),
        representations=("all",),
        models=("linear_logistic",),
        seeds=SEEDS,
    )
    dose_configs = {
        task: DoseRunConfig(
            dataset=application_config.dataset,
            task=task,
            subject_folds=2,
            stimulus_folds=2,
            seeds=application_config.seeds,
            doses=application_config.doses,
            representations=application_config.representations,
            models=application_config.models,
            gpu_epochs=40,
            candidate_draws=32,
        )
        for task in application_config.tasks
    }

    (ROOT / "outputs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="_synthetic_external_one_shot_", dir=ROOT / "outputs"
    ) as directory:
        workspace = Path(directory)
        reservation_path = workspace / "reservation.json"
        schema_path = workspace / "schema_manifest.json"
        adapter_path = workspace / "synthetic_adapter.py"
        adapter_test_path = workspace / "test_synthetic_adapter.py"
        implementation_path = workspace / "implementation_lock.json"

        reservation = json.loads(RESERVATION.read_text(encoding="utf-8"))
        reservation["dataset"] = application_config.dataset
        reservation["registration_id"] = "synthetic-one-shot-registration"
        reservation_path.write_text(json.dumps(reservation), encoding="utf-8")
        schema_path.write_text(
            json.dumps(
                {
                    "status": "synthetic_schema_complete",
                    "reservation_id": reservation["registration_id"],
                    "files": [{"path": "synthetic.mat", "shape": [24, 10, 8]}],
                }
            ),
            encoding="utf-8",
        )
        adapter_path.write_text("ADAPTER = 'synthetic'\n", encoding="utf-8")
        adapter_test_path.write_text(
            "print('synthetic schema adapter passed')\n", encoding="utf-8"
        )
        implementation_spec = ImplementationLockSpec(
            dataset=application_config.dataset,
            lock_id="synthetic-full-one-shot-lock",
            reservation_path=reservation_path,
            schema_manifest_path=schema_path,
            freeze_path=FREEZE,
            output_path=implementation_path,
            expected_schema_status="synthetic_schema_complete",
            analysis_code=(
                adapter_path,
                ROOT / "scripts/external_wearable_one_shot_core.py",
            ),
            dependency_code=(
                ROOT / "scripts/external_wearable_v11_pipeline.py",
                ROOT / "scripts/external_wearable_dose_core.py",
                ROOT / "scripts/external_wearable_dcs_opct_core.py",
            ),
            tests=(adapter_test_path,),
            registered_execution={
                "tasks": list(application_config.tasks),
                "representations": list(application_config.representations),
                "models": list(application_config.models),
                "split_seeds": list(application_config.seeds),
                "doses": list(application_config.doses),
                "device": "cuda",
                "material_threshold": application_config.material_threshold,
            },
        )
        _, implementation_sha = build_implementation_lock(
            implementation_spec,
            root=ROOT,
            operator_attests_no_participant_values_accessed=True,
        )
        output_root = workspace / "pipeline"
        _, preoutcome_sha = prepare_and_lock_preoutcome(
            output_root,
            task_bundles=task_bundles,
            task_validators={"valence": validator, "arousal": validator},
            dose_configs=dose_configs,
            application_config=application_config,
            implementation_lock_path=implementation_path,
            implementation_lock_sha256=implementation_sha,
            project_root=ROOT,
            provenance={"fixture": "full_synthetic_one_shot"},
        )
        action_path, action_sha = lock_registered_action(
            output_root,
            preoutcome_lock_sha256=preoutcome_sha,
            implementation_lock_path=implementation_path,
            implementation_lock_sha256=implementation_sha,
            project_root=ROOT,
            risk_root=RISK_ROOT,
            device="cuda",
        )
        action = json.loads(action_path.read_text(encoding="utf-8"))
        assert action["provenance"]["implementation_lock_sha256"] == implementation_sha
        summary_path = run_registered_outcomes(
            output_root,
            preoutcome_lock_sha256=preoutcome_sha,
            action_lock_sha256=action_sha,
            implementation_lock_path=implementation_path,
            implementation_lock_sha256=implementation_sha,
            project_root=ROOT,
            n_jobs=1,
            device="cuda",
        )
        gate = finalize_registered_outcomes(
            output_root,
            preoutcome_lock_sha256=preoutcome_sha,
            action_lock_sha256=action_sha,
            full_summary_path=summary_path,
            implementation_lock_path=implementation_path,
            implementation_lock_sha256=implementation_sha,
            project_root=ROOT,
        )
        assert (output_root / "outcome/external_confirmation_manifest.json").is_file()
        assert isinstance(gate["claim_supported"], bool)
        if gate["selected_method"] == "identity":
            assert gate["effectiveness_pass"] is False
            assert gate["released_action_nonharm_pass"] is None
            assert gate["safety_success"] is False

    print(
        "External wearable full one-shot CUDA test passed: adapter audit, frozen action, "
        "model outcomes, endpoint gate, and identity boundary"
    )


if __name__ == "__main__":
    main()
