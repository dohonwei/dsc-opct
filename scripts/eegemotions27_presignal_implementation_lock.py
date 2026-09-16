from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from external_wearable_implementation_lock import (
    ImplementationLockSpec,
    hash_inventory,
    relative_name,
    run_lock_tests,
    runtime_versions,
    sha256,
    validate_schema_manifest,
    verify_frozen_inventory,
    verify_implementation_lock,
)


LOCK_DATE = "2026-09-09"
RESERVATION_STATUS = "reserved_after_repository_metadata_before_signal_and_outcome_access"


def build_presignal_implementation_lock(
    spec: ImplementationLockSpec,
    *,
    root: Path,
    operator_attests_no_signal_or_outcome_values_accessed: bool,
) -> tuple[Path, str]:
    if spec.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite implementation lock: {spec.output_path}")
    if not operator_attests_no_signal_or_outcome_values_accessed:
        raise RuntimeError(
            "Implementation locking requires an explicit attestation that signal, "
            "derived-feature, prediction, endpoint, and transfer-outcome values have "
            "not been opened or inspected"
        )
    for path in (spec.reservation_path, spec.schema_manifest_path, spec.freeze_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    reservation = json.loads(spec.reservation_path.read_text(encoding="utf-8"))
    manifest = json.loads(spec.schema_manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(spec.freeze_path.read_text(encoding="utf-8"))
    if reservation.get("status") != RESERVATION_STATUS:
        raise RuntimeError("EEGEmotions-27 pre-signal reservation status is invalid")
    if reservation.get("dataset") != spec.dataset:
        raise RuntimeError("Implementation-lock dataset differs from the reservation")
    validate_schema_manifest(manifest, reservation, spec)
    expected_freeze = reservation.get("dcs_opct_v11_contract", {}).get("freeze_sha256")
    if not expected_freeze or sha256(spec.freeze_path) != expected_freeze:
        raise RuntimeError("Implementation lock does not match the reserved v11 freeze")

    frozen_artifacts = verify_frozen_inventory(freeze, root)
    analysis_hashes = hash_inventory(spec.analysis_code, root)
    dependency_hashes = hash_inventory(spec.dependency_code, root)
    test_results = run_lock_tests(spec.tests, root)
    runtime = runtime_versions()
    if not runtime["cuda_available"]:
        raise RuntimeError("CUDA must be available when the implementation is locked")

    payload: dict[str, Any] = {
        "status": "implementation_locked_before_participant_value_access",
        "status_semantics": (
            "Compatibility status for the shared one-shot verifier; here participant-value "
            "access means signal samples, demographic rows, derived features, predictions, "
            "events, and outcomes. Prior filename-level coverage metadata is disclosed."
        ),
        "prospective_status": "locked_before_signal_and_outcome_access_after_repository_metadata_disclosure",
        "lock_id": spec.lock_id,
        "locked_at_local": LOCK_DATE,
        "dataset": spec.dataset,
        "reservation_path": relative_name(spec.reservation_path, root),
        "reservation_sha256": sha256(spec.reservation_path),
        "schema_manifest_path": relative_name(spec.schema_manifest_path, root),
        "schema_manifest_sha256": sha256(spec.schema_manifest_path),
        "schema_manifest_status": manifest["status"],
        "freeze_path": relative_name(spec.freeze_path, root),
        "freeze_sha256": sha256(spec.freeze_path),
        "frozen_artifact_sha256": frozen_artifacts,
        "analysis_code_sha256": analysis_hashes,
        "dependency_code_sha256": dependency_hashes,
        "pre_lock_tests": test_results,
        "runtime": runtime,
        "registered_execution": spec.registered_execution,
        "prior_metadata_access": {
            "disclosed": True,
            "scope": (
                "README, channel metadata, repository filenames, blob identifiers, byte "
                "sizes, and participant-by-emotion availability encoded in filenames"
            ),
        },
        "participant_value_access_attestation": {
            "operator_attested": True,
            "scope": (
                "No raw EEG sample, participant demographic row, derived feature, model "
                "prediction, material event, risk probability, calibration outcome, or "
                "held-out result was opened or inspected before this lock."
            ),
        },
        "change_policy": (
            "After this lock, no algorithm, feature, label, quality rule, threshold, split, "
            "seed, dose, candidate, audit assignment, model setting, or analysis-code change "
            "is permitted before the one-shot gate. Defects require preserved failures and "
            "numbered non-overwriting amendments."
        ),
        "claim_boundary": (
            "This lock establishes pre-signal implementation provenance with disclosed "
            "metadata access only. It supplies no structural, endpoint, effectiveness, "
            "non-harm, safety, or replication evidence."
        ),
    }
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    spec.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    verify_implementation_lock(spec.output_path, root=root)
    return spec.output_path, sha256(spec.output_path)
