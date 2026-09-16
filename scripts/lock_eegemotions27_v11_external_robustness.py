from __future__ import annotations

import argparse
from pathlib import Path

from eegemotions27_presignal_implementation_lock import (
    build_presignal_implementation_lock,
)
from eegemotions27_v11_contract import (
    DOSES,
    FREEZE,
    GPU_EPOCHS,
    LOCKED_COMMIT,
    MATERIAL_THRESHOLD,
    MODELS,
    REPRESENTATIONS,
    RESERVATION,
    ROOT,
    SEEDS,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
)
from external_wearable_implementation_lock import ImplementationLockSpec


SCHEMA_MANIFEST = ROOT / "outputs/eegemotions27_v11_presignal/repository_metadata_manifest.json"
OUTPUT = ROOT / "docs/eegemotions27_v11_external_robustness_implementation_lock.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock the EEGEmotions-27 v11 implementation before signal access"
    )
    parser.add_argument("--schema-manifest", type=Path, default=SCHEMA_MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--confirm-no-signal-or-outcome-values-accessed", action="store_true"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = ImplementationLockSpec(
        dataset="EEGEmotions-27",
        lock_id="eegemotions27-dcs-opct-v11-presignal-schema-bound-implementation",
        reservation_path=RESERVATION,
        schema_manifest_path=args.schema_manifest,
        freeze_path=FREEZE,
        output_path=args.output,
        expected_schema_status="eegemotions27_repository_metadata_manifest_complete",
        analysis_code=(
            ROOT / "scripts/eegemotions27_v11_schema_adapter.py",
            ROOT / "scripts/eegemotions27_v11_feature_core.py",
            ROOT / "scripts/run_eegemotions27_v11_external_robustness.py",
            ROOT / "scripts/test_eegemotions27_v11_schema_adapter.py",
        ),
        dependency_code=(
            ROOT / "scripts/eegemotions27_presignal_implementation_lock.py",
            ROOT / "scripts/lock_eegemotions27_v11_external_robustness.py",
            ROOT / "scripts/eegemotions27_v11_contract.py",
            ROOT / "scripts/eegemotions27_v11_adapter_contract.py",
            ROOT / "scripts/inspect_eegemotions27_v11_repository.py",
            ROOT / "scripts/external_wearable_trial_contract.py",
            ROOT / "scripts/wearable_eeg_v11_feature_core.py",
            ROOT / "scripts/external_wearable_implementation_lock.py",
            ROOT / "scripts/external_wearable_dose_core.py",
            ROOT / "scripts/external_wearable_dcs_opct_core.py",
            ROOT / "scripts/external_wearable_v11_pipeline.py",
            ROOT / "scripts/external_wearable_one_shot_core.py",
            ROOT / "outputs/eegemotions27_v11_presignal/official_eeg_raw_tree.json",
            ROOT / "docs/evidence/eegemotions27/emotivX_channels_location.ced",
            ROOT / "docs/eegemotions27_v11_category_polarity_mapping.md",
            ROOT / "docs/eegemotions27_v11_preaccess_history_audit.json",
        ),
        tests=(
            ROOT / "scripts/test_eegemotions27_v11_adapter_contract.py",
            ROOT / "scripts/test_eegemotions27_v11_feature_core.py",
            ROOT / "scripts/test_eegemotions27_v11_schema_adapter.py",
            ROOT / "scripts/test_eegemotions27_v11_repository_inspector.py",
            ROOT / "scripts/test_external_wearable_dose_core.py",
            ROOT / "scripts/test_external_wearable_dcs_opct_core.py",
        ),
        registered_execution={
            "tasks": ["category_polarity"],
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
            "locked_commit": LOCKED_COMMIT,
            "entry_point": "scripts/run_eegemotions27_v11_external_robustness.py",
            "claim_role": "separate pre-signal external robustness test",
        },
    )
    path, digest = build_presignal_implementation_lock(
        spec,
        root=ROOT,
        operator_attests_no_signal_or_outcome_values_accessed=(
            args.confirm_no_signal_or_outcome_values_accessed
        ),
    )
    print(f"Locked EEGEmotions-27 implementation: {path} ({digest})")


if __name__ == "__main__":
    main()
