from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_external_access_evidence_v5"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"

SOURCES = {
    "v11_final_freeze.json": "docs/distribution_covered_stratified_opct_v11_final_freeze.json",
    "dataset_availability_matrix.md": "docs/dcs_opct_v11_dataset_availability_matrix.md",
    "claim_evidence_matrix.md": "docs/dcs_opct_v11_claim_evidence_matrix.md",
    "package_v1_failure_001.json": "docs/dcs_opct_v11_external_access_evidence_failure_001.json",
    "package_v2_superseded.json": "docs/dcs_opct_v11_external_access_evidence_v2_superseded.json",
    "package_v3_superseded.json": "docs/dcs_opct_v11_external_access_evidence_v3_superseded.json",
    "package_v4_superseded.json": "docs/dcs_opct_v11_external_access_evidence_v4_superseded.json",
    "external_family_registry.json": "docs/v11_prospective_external_family_registry.json",
    "external_standardized_contract.md": "docs/external_wearable_v11_standardized_analysis_contract.md",
    "amigos_reservation.json": "docs/amigos_v11_external_confirmation_reservation.json",
    "amigos_preaccess_history.json": "docs/amigos_v11_preaccess_history_audit.json",
    "amigos_implementation_protocol.md": "docs/amigos_v11_external_confirmation_implementation_protocol.md",
    "amigos_access_attempt_001.json": "docs/amigos_v11_access_attempt_001.json",
    "amigos_access_attempt_002.json": "docs/amigos_v11_access_attempt_002.json",
    "amigos_access_attempt_003.json": "docs/amigos_v11_access_attempt_003.json",
    "amigos_access_request.md": "docs/amigos_v11_official_access_request_email.md",
    "amigos_preaccess_stack_report.json": "outputs/amigos_v11_preaccess/stack_validation_report.json",
    "emognition_reservation.json": "docs/emognition_v11_external_confirmation_reservation.json",
    "emognition_preaccess_history.json": "docs/emognition_v11_preaccess_history_audit.json",
    "emognition_protocol.md": "docs/emognition_v11_external_confirmation_protocol.md",
    "emognition_adapter_specification.md": "docs/emognition_v11_schema_adapter_preaccess_specification.md",
    "emognition_metadata_receipt.json": "docs/emognition_v11_dataverse_metadata_receipt_20260910.json",
    "emognition_access_readiness_failure_001.json": "docs/emognition_v11_access_readiness_failure_001.json",
    "emognition_access_request.md": "docs/emognition_v11_official_access_request_email.md",
    "emognition_official_eula.pdf": "docs/evidence/emognition/Emognition_EULA_Dataverse_v6.0.pdf",
    "emognition_access_readiness_report.json": "outputs/emognition_v11_access_readiness/report.json",
    "emognition_preaccess_stack_report.json": "outputs/emognition_v11_preaccess/stack_validation_report.json",
    "emognition_contract.py": "scripts/emognition_v11_contract.py",
    "emognition_schema_inspector.py": "scripts/inspect_emognition_v11_archive_schema.py",
    "emognition_schema_adapter.py": "scripts/emognition_v11_schema_adapter.py",
    "emognition_schema_adapter_test.py": "scripts/test_emognition_v11_schema_adapter.py",
    "emognition_archive_adapter_test.py": "scripts/test_emognition_v11_archive_adapter.py",
    "emognition_gpu_runner.py": "scripts/run_emognition_v11_external_confirmation.py",
    "emognition_implementation_locker.py": "scripts/lock_emognition_v11_external_confirmation.py",
    "emognition_preaccess_validator.py": "scripts/validate_emognition_v11_preaccess_stack.py",
    "amigos_preaccess_validator.py": "scripts/validate_amigos_v11_preaccess_stack.py",
    "package_builder.py": "scripts/build_dcs_opct_v11_external_access_evidence.py",
    "package_validator.py": "scripts/validate_dcs_opct_v11_external_access_evidence.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite external-access package: {OUT}")
    if sha256(FREEZE) != EXPECTED_FREEZE:
        raise RuntimeError("Frozen v11 hash changed")
    missing = [relative for relative in SOURCES.values() if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"External-access evidence is incomplete: {missing}")

    OUT.mkdir(parents=True)
    source_hashes = {}
    output_hashes = {}
    source_to_output = {}
    for output_name, relative in tqdm(
        sorted(SOURCES.items()),
        desc="Package external-access evidence",
        unit="file",
        dynamic_ncols=True,
    ):
        source = ROOT / relative
        target = OUT / output_name
        shutil.copy2(source, target)
        source_hashes[relative] = sha256(source)
        output_hashes[output_name] = sha256(target)
        source_to_output[relative] = output_name
    manifest = {
        "status": "external_access_evidence_packaged_pending_validation",
        "date": "2026-09-09",
        "method": "DCS-OPCT v11 prospective external-access readiness sidecar",
        "frozen_v11_sha256": EXPECTED_FREEZE,
        "file_count_excluding_manifest_and_validation_report": len(output_hashes),
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
        "source_to_output": source_to_output,
        "restricted_participant_data_included": False,
        "claim_boundary": (
            "This package establishes access-route history, protocol provenance, "
            "and implementation readiness only. It is not external validation, "
            "effectiveness, non-harm, safety, or replication evidence."
        ),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"External-access evidence packaged: {len(output_hashes)} files")
    print(f"Package: {OUT}")


if __name__ == "__main__":
    main()
