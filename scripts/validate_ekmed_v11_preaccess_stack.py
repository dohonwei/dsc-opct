from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ekmed_v11_contract import verify_preaccess_contract


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/ekmed_v11_preaccess/stack_validation_report.json"
PREACCESS_REPORT = ROOT / "outputs/ekmed_v11_preaccess/validation_report.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
TESTS = (
    ("schema_inspector", "scripts/test_ekmed_v11_schema_inspector.py"),
    ("header_schema_inspector", "scripts/test_ekmed_v11_header_schema.py"),
    ("feature_core", "scripts/test_ekmed_v11_feature_core.py"),
    ("adapter_output_contract", "scripts/test_ekmed_v11_adapter_contract.py"),
    ("gpu_model", "scripts/test_ekmed_v11_gpu_model.py"),
    ("shared_dose_core_gpu", "scripts/test_external_wearable_dose_core.py"),
    ("shared_dcs_opct_core_gpu", "scripts/test_external_wearable_dcs_opct_core.py"),
    ("shared_state_machine", "scripts/test_external_wearable_v11_pipeline.py"),
)
EVIDENCE_FILES = (
    "scripts/validate_ekmed_v11_preaccess_stack.py",
    "docs/ekmed_v11_preaccess_history_audit.json",
    "docs/ekmed_v11_zenodo_metadata_receipt_20260916.json",
    "docs/ekmed_v11_public_key_moment_receipt_20260916.json",
    "docs/ekmed_v11_external_confirmation_reservation.json",
    "docs/ekmed_v11_external_confirmation_reservation_amendment_001.json",
    "docs/ekmed_v11_external_confirmation_protocol.md",
    "scripts/ekmed_v11_contract.py",
    "scripts/ekmed_v11_feature_core.py",
    "scripts/ekmed_v11_adapter_contract.py",
    "scripts/inspect_ekmed_v11_archive_schema.py",
    "scripts/inspect_ekmed_v11_header_schema.py",
    "scripts/test_ekmed_v11_schema_inspector.py",
    "scripts/test_ekmed_v11_header_schema.py",
    "scripts/test_ekmed_v11_feature_core.py",
    "scripts/test_ekmed_v11_adapter_contract.py",
    "scripts/test_ekmed_v11_gpu_model.py",
    "scripts/validate_ekmed_v11_preaccess.py",
    "outputs/ekmed_v11_preaccess/validation_report.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    verify_preaccess_contract(require_archive_absent=False)
    preaccess = json.loads(PREACCESS_REPORT.read_text(encoding="utf-8"))
    results = []
    progress = tqdm(TESTS, desc="EKM-ED pre-member stack", unit="check", dynamic_ncols=True)
    for name, relative in progress:
        completed = subprocess.run(
            [sys.executable, str(ROOT / relative)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "name": name,
                "script": relative,
                "passed": completed.returncode == 0,
                "stdout": completed.stdout.strip(),
                "stderr_tail": completed.stderr.strip()[-2000:],
            }
        )
    progress.close()
    freeze_hash = sha256(FREEZE)
    files = {relative: sha256(ROOT / relative) for relative in EVIDENCE_FILES}
    checks = {
        "reservation_snapshot_27_of_27": preaccess.get("status") == "passed"
        and preaccess.get("checks_passed") == preaccess.get("checks_total") == 27,
        "reservation_amendment_valid": True,
        "all_tests_passed": all(item["passed"] for item in results),
        "cuda_available": torch.cuda.is_available(),
        "freeze_hash_unchanged": freeze_hash == EXPECTED_FREEZE_HASH,
        "all_evidence_files_present": all((ROOT / path).is_file() for path in EVIDENCE_FILES),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "date": "2026-09-16",
        "dataset": "EmoKey Moments Muse EEG Dataset (EKM-ED)",
        "phase": "post-reservation, pre-member-access implementation readiness",
        "checks": checks,
        "tests": results,
        "gpu": {
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "registered_epochs": 40,
        },
        "python_executable": sys.executable,
        "freeze_sha256": freeze_hash,
        "evidence_sha256": files,
        "remaining_before_participant_value_access": [
            "complete the official archive download and verify byte size plus MD5",
            "record the ZIP central directory without opening members",
            "implement and synthetic-test the schema adapter from names and sizes only",
            "write the implementation lock before opening any member",
        ],
        "claim_boundary": (
            "Passing this report establishes only protocol and implementation readiness. "
            "It supplies no EKM-ED participant, structural, endpoint, effectiveness, "
            "non-harm, replication, or safety evidence."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError(f"EKM-ED pre-member stack failed: {checks}")
    print(f"EKM-ED pre-member stack passed: {len(TESTS)}/{len(TESTS)} test groups")


if __name__ == "__main__":
    main()
