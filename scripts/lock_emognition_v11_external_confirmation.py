from __future__ import annotations

import argparse
from pathlib import Path

from emognition_v11_contract import (
    DOSES,
    FREEZE,
    GPU_EPOCHS,
    MATERIAL_THRESHOLD,
    MODELS,
    REPRESENTATIONS,
    RESERVATION,
    ROOT,
    SEEDS,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
)
from external_wearable_implementation_lock import (
    ImplementationLockSpec,
    build_implementation_lock,
)


SCHEMA_MANIFEST = ROOT / "outputs/emognition_v11_preaccess/schema_only_acquisition_manifest.json"
OUTPUT = ROOT / "docs/emognition_v11_external_confirmation_implementation_lock.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock the schema-bound Emognition v11 implementation"
    )
    parser.add_argument("--schema-manifest", type=Path, default=SCHEMA_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--confirm-no-participant-values-accessed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ImplementationLockSpec(
        dataset="Emognition Wearable Dataset 2020",
        lock_id="emognition-dcs-opct-v11-schema-bound-implementation",
        reservation_path=RESERVATION,
        schema_manifest_path=args.schema_manifest,
        freeze_path=FREEZE,
        output_path=args.output,
        expected_schema_status="emognition_schema_only_acquisition_manifest_complete",
        analysis_code=(
            ROOT / "scripts/emognition_v11_schema_adapter.py",
            ROOT / "scripts/run_emognition_v11_external_confirmation.py",
            ROOT / "scripts/test_emognition_v11_schema_adapter.py",
        ),
        dependency_code=(
            ROOT / "scripts/external_wearable_implementation_lock.py",
            ROOT / "scripts/lock_emognition_v11_external_confirmation.py",
            ROOT / "scripts/emognition_v11_contract.py",
            ROOT / "scripts/emognition_v11_feature_core.py",
            ROOT / "scripts/emognition_v11_adapter_contract.py",
            ROOT / "docs/emognition_v11_schema_adapter_preaccess_specification.md",
            ROOT / "scripts/external_wearable_trial_contract.py",
            ROOT / "scripts/external_wearable_dose_core.py",
            ROOT / "scripts/external_wearable_dcs_opct_core.py",
            ROOT / "scripts/external_wearable_v11_pipeline.py",
            ROOT / "scripts/external_wearable_one_shot_core.py",
        ),
        tests=(
            ROOT / "scripts/validate_emognition_v11_preaccess_stack.py",
            ROOT / "scripts/test_emognition_v11_schema_adapter.py",
        ),
        registered_execution={
            "tasks": ["valence", "arousal"],
            "representations": list(REPRESENTATIONS),
            "models": list(MODELS),
            "split_seeds": list(SEEDS),
            "subject_folds": SUBJECT_FOLDS,
            "stimulus_folds": STIMULUS_FOLDS,
            "doses": list(DOSES),
            "gpu_epochs": GPU_EPOCHS,
            "candidate_draws": 256,
            "intervention_block_fraction": 0.5,
            "device": "cuda",
            "material_threshold": MATERIAL_THRESHOLD,
            "entry_point": "scripts/run_emognition_v11_external_confirmation.py",
        },
    )
    path, digest = build_implementation_lock(
        spec,
        root=ROOT,
        operator_attests_no_participant_values_accessed=(
            args.confirm_no_participant_values_accessed
        ),
    )
    print(f"Locked Emognition implementation: {path} ({digest})")


if __name__ == "__main__":
    main()
