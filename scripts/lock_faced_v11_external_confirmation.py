from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import torch

from faced_v11_contract import (
    ARCHIVED_MANIFEST,
    DEFAULT_DATA_ROOT,
    DOSES,
    FREEZE,
    GPU_EPOCHS,
    IMPLEMENTATION_LOCK,
    MATERIAL_THRESHOLD,
    MINIMUM_PARTICIPANTS,
    MODELS,
    NON_NEUTRAL_VIDEOS,
    REPRESENTATIONS,
    RESERVATION,
    SEEDS,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
    ROOT,
    sha256,
    verify_frozen_artifacts,
)


ANALYSIS_CODE = (
    "docs/faced_v11_external_confirmation_implementation_protocol.md",
    "scripts/acquire_faced_v11_dataset.py",
    "scripts/build_faced_trial_features.py",
    "scripts/run_faced_v11_counterfactual_identity_dose.py",
    "scripts/apply_frozen_dcs_opct_v11_to_faced.py",
    "scripts/validate_faced_v11_preaccess.py",
    "scripts/lock_faced_v11_external_confirmation.py",
)
DEPENDENCY_CODE = (
    "scripts/faced_v11_contract.py",
    "scripts/run_counterfactual_identity_dose.py",
    "scripts/run_crossed_identity_probes.py",
    "scripts/run_paired_block_exposure_benchmark.py",
    "scripts/run_protocol_model_benchmark.py",
    "src/identity_shortcut/core.py",
    "src/identity_shortcut/models.py",
)
SOURCE_METADATA = (
    "outputs/faced_v11_preaccess_metadata/nemar_manifest.json",
    "outputs/faced_v11_preaccess_metadata/zarr_index.json",
    "outputs/faced_v11_preaccess_metadata/official_README.md",
    "outputs/faced_v11_preaccess_metadata/official_faced.py",
    "outputs/faced_v11_preaccess_metadata/official_dataset_description.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the FACED v11 implementation before participant/event/signal access."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def hashes(paths: tuple[str, ...]) -> dict[str, str]:
    output = {}
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative] = sha256(path)
    return output


def assert_no_protected_values(data_root: Path) -> None:
    if not data_root.exists():
        return
    protected = []
    protected.extend(data_root.glob("participants.tsv"))
    protected.extend(data_root.glob("sub-*/eeg/*_events.tsv"))
    protected.extend(data_root.glob("sub-*/eeg/*.bdf"))
    if protected:
        raise RuntimeError(
            "FACED participant/event/signal files already exist at the target root; "
            "the pre-access implementation lock cannot be asserted"
        )


def package_versions() -> dict[str, str]:
    names = (
        "mne",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "tqdm",
        "requests",
    )
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    args = parse_args()
    if IMPLEMENTATION_LOCK.exists():
        raise FileExistsError(
            f"Refusing to overwrite FACED implementation lock: {IMPLEMENTATION_LOCK}"
        )
    assert_no_protected_values(args.data_root)
    verify_frozen_artifacts()
    reservation = json.loads(RESERVATION.read_text(encoding="utf-8"))
    if reservation.get("status") != "reserved_before_participant_value_access":
        raise RuntimeError("FACED reservation status is invalid")
    archived_manifest = json.loads(ARCHIVED_MANIFEST.read_text(encoding="utf-8"))
    if len(archived_manifest) != 745:
        raise ValueError("FACED archived manifest file count changed")

    lock = {
        "status": "locked_before_faced_participant_event_or_signal_value_access",
        "lock_id": "faced-dcs-opct-v11-implementation-20260909",
        "locked_at_local": "2026-09-09",
        "reservation_sha256": sha256(RESERVATION),
        "freeze_sha256": sha256(FREEZE),
        "analysis_code_sha256": hashes(ANALYSIS_CODE),
        "dependency_code_sha256": hashes(DEPENDENCY_CODE),
        "source_metadata_sha256": hashes(SOURCE_METADATA),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        },
        "registered_execution": {
            "minimum_usable_participants": MINIMUM_PARTICIPANTS,
            "physical_non_neutral_videos": list(NON_NEUTRAL_VIDEOS),
            "representations": list(REPRESENTATIONS),
            "models": list(MODELS),
            "split_seeds": list(SEEDS),
            "subject_folds": SUBJECT_FOLDS,
            "stimulus_folds": STIMULUS_FOLDS,
            "doses": list(DOSES),
            "gpu_epochs": GPU_EPOCHS,
            "material_threshold": MATERIAL_THRESHOLD,
            "primary_axis": "subject",
            "task": "stimulus-category affective polarity",
        },
        "commands": [
            "python scripts/acquire_faced_v11_dataset.py",
            "python scripts/build_faced_trial_features.py",
            "python scripts/run_faced_v11_counterfactual_identity_dose.py",
            "python scripts/apply_frozen_dcs_opct_v11_to_faced.py",
        ],
        "claim_boundary": (
            "This lock fixes construction and analysis code before FACED participant, event, "
            "or EEG values are accessed. It does not establish compatibility, effectiveness, or safety."
        ),
    }
    if not lock["runtime"]["cuda_available"]:
        raise RuntimeError(
            "CUDA must be available when the FACED implementation is locked"
        )
    IMPLEMENTATION_LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
