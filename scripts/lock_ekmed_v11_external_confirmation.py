from __future__ import annotations

import argparse
from pathlib import Path

from ekmed_v11_contract import (
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


SCHEMA_RECEIPT = ROOT / "outputs/ekmed_v11_archive_schema/header_schema_receipt.json"
OUTPUT = ROOT / "docs/ekmed_v11_external_confirmation_implementation_lock.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock the schema-bound EKM-ED v11 implementation"
    )
    parser.add_argument("--schema-receipt", type=Path, default=SCHEMA_RECEIPT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--confirm-no-participant-values-accessed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ImplementationLockSpec(
        dataset="EmoKey Moments Muse EEG Dataset (EKM-ED)",
        lock_id="ekmed-dcs-opct-v11-schema-bound-implementation",
        reservation_path=RESERVATION,
        schema_manifest_path=args.schema_receipt,
        freeze_path=FREEZE,
        output_path=args.output,
        expected_schema_status="header_schema_recorded_without_data_rows",
        analysis_code=(
            ROOT / "scripts/ekmed_v11_schema_adapter.py",
            ROOT / "scripts/run_ekmed_v11_external_confirmation.py",
            ROOT / "scripts/test_ekmed_v11_schema_adapter.py",
            ROOT / "scripts/test_ekmed_v11_archive_adapter.py",
        ),
        dependency_code=(
            ROOT / "scripts/external_wearable_implementation_lock.py",
            ROOT / "scripts/lock_ekmed_v11_external_confirmation.py",
            ROOT / "scripts/ekmed_v11_contract.py",
            ROOT / "scripts/ekmed_v11_feature_core.py",
            ROOT / "scripts/ekmed_v11_adapter_contract.py",
            ROOT / "scripts/inspect_ekmed_v11_archive_schema.py",
            ROOT / "scripts/inspect_ekmed_v11_header_schema.py",
            ROOT / "docs/ekmed_v11_public_key_moment_receipt_20260916.json",
            ROOT / "scripts/external_wearable_trial_contract.py",
            ROOT / "scripts/wearable_eeg_v11_feature_core.py",
            ROOT / "scripts/external_wearable_dose_core.py",
            ROOT / "scripts/external_wearable_dcs_opct_core.py",
            ROOT / "scripts/external_wearable_v11_pipeline.py",
            ROOT / "scripts/external_wearable_one_shot_core.py",
        ),
        tests=(
            ROOT / "scripts/test_ekmed_v11_schema_adapter.py",
            ROOT / "scripts/test_ekmed_v11_archive_adapter.py",
            ROOT / "scripts/test_ekmed_v11_adapter_contract.py",
            ROOT / "scripts/test_ekmed_v11_feature_core.py",
            ROOT / "scripts/test_external_wearable_dose_core.py",
            ROOT / "scripts/test_external_wearable_dcs_opct_core.py",
            ROOT / "scripts/test_external_wearable_v11_pipeline.py",
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
            "entry_point": "scripts/run_ekmed_v11_external_confirmation.py",
        },
    )
    path, digest = build_implementation_lock(
        spec,
        root=ROOT,
        operator_attests_no_participant_values_accessed=(
            args.confirm_no_participant_values_accessed
        ),
    )
    print(f"Locked EKM-ED implementation: {path} ({digest})")


if __name__ == "__main__":
    main()
