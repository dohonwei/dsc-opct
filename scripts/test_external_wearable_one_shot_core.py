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
    sha256,
)
from external_wearable_one_shot_core import (
    RegisteredTaskBundle,
    prepare_and_lock_preoutcome,
)
from external_wearable_trial_contract import TrialBundleContract, validate_trial_bundle
from external_wearable_v11_pipeline import verify_preoutcome_design_bundle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESERVATION = ROOT / "docs/amigos_v11_external_confirmation_reservation.json"


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


def expect_error(error_type, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__} was not raised")


def main() -> None:
    frame = pd.DataFrame(
        [
            {"subject_id": f"S{subject:02d}", "trial_id": trial}
            for subject in range(24)
            for trial in range(10)
        ]
    )
    rng = np.random.default_rng(20260909)
    feature_sets = {"all": rng.normal(size=(len(frame), 8))}
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
        seeds=(20260909,),
    )
    dose_configs = {
        task: DoseRunConfig(
            dataset="SYNTHETIC-WEARABLE",
            task=task,
            subject_folds=2,
            stimulus_folds=2,
            seeds=(20260909,),
            doses=(0.0, 0.25, 0.5, 0.75, 1.0),
            representations=("all",),
            models=("linear_logistic",),
            gpu_epochs=40,
            candidate_draws=32,
        )
        for task in application_config.tasks
    }

    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        (project / "docs").mkdir()
        (project / "scripts").mkdir()
        (project / "outputs").mkdir()
        freeze = project / "docs/freeze.json"
        reservation = project / "docs/reservation.json"
        schema = project / "outputs/schema.json"
        adapter = project / "scripts/adapter.py"
        test = project / "scripts/test_adapter.py"
        frozen_artifact = project / "frozen_model.bin"
        frozen_artifact.write_bytes(b"synthetic frozen model")
        freeze.write_text(
            json.dumps(
                {
                    "locked_artifacts": {
                        "frozen_model.bin": sha256(frozen_artifact),
                    }
                }
            ),
            encoding="utf-8",
        )
        reservation_data = json.loads(SOURCE_RESERVATION.read_text(encoding="utf-8"))
        reservation_data["dataset"] = "SYNTHETIC-WEARABLE"
        reservation_data["dcs_opct_v11_contract"]["freeze_sha256"] = sha256(freeze)
        reservation.write_text(json.dumps(reservation_data), encoding="utf-8")
        schema.write_text(
            json.dumps(
                {
                    "status": "synthetic_schema_complete",
                    "reservation_id": reservation_data["registration_id"],
                    "files": [{"path": "synthetic.mat", "shape": [24, 10]}],
                }
            ),
            encoding="utf-8",
        )
        adapter.write_text("ADAPTER = 'synthetic'\n", encoding="utf-8")
        test.write_text("print('adapter passed')\n", encoding="utf-8")
        implementation_path = project / "docs/implementation_lock.json"
        implementation_spec = ImplementationLockSpec(
            dataset="SYNTHETIC-WEARABLE",
            lock_id="synthetic-one-shot-lock",
            reservation_path=reservation,
            schema_manifest_path=schema,
            freeze_path=freeze,
            output_path=implementation_path,
            expected_schema_status="synthetic_schema_complete",
            analysis_code=(adapter,),
            dependency_code=(test,),
            tests=(test,),
            registered_execution={"material_threshold": 0.02},
        )
        _, implementation_sha = build_implementation_lock(
            implementation_spec,
            root=project,
            operator_attests_no_participant_values_accessed=True,
        )
        output_root = project / "outputs/one_shot"
        lock_path, lock_sha = prepare_and_lock_preoutcome(
            output_root,
            task_bundles=task_bundles,
            task_validators={"valence": validator, "arousal": validator},
            dose_configs=dose_configs,
            application_config=application_config,
            implementation_lock_path=implementation_path,
            implementation_lock_sha256=implementation_sha,
            project_root=project,
            provenance={"fixture": "synthetic_adapter_to_preoutcome"},
        )
        lock, recovered = verify_preoutcome_design_bundle(output_root, lock_sha)
        assert recovered == application_config
        assert lock["artifact_sha256"]["adapter_audits"] == sha256(
            output_root / "preoutcome/adapter_output_audits.json"
        )
        audits = json.loads(
            (output_root / "preoutcome/adapter_output_audits.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(audits) == {"valence", "arousal"}
        assert lock_path.is_file()

        original_adapter = adapter.read_bytes()
        adapter.write_bytes(original_adapter + b"\n")
        expect_error(
            RuntimeError,
            prepare_and_lock_preoutcome,
            project / "outputs/tampered",
            task_bundles=task_bundles,
            task_validators={"valence": validator, "arousal": validator},
            dose_configs=dose_configs,
            application_config=application_config,
            implementation_lock_path=implementation_path,
            implementation_lock_sha256=implementation_sha,
            project_root=project,
            provenance={"fixture": "must_fail_before_preparation"},
        )

    print(
        "External wearable one-shot preparation test passed: implementation lock, "
        "adapter audits, dose preparation, and pre-outcome lock"
    )


if __name__ == "__main__":
    main()
