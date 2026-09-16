from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/amigos_v11_preaccess/stack_validation_report.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"

TESTS = (
    ("reservation_contract", "scripts/validate_amigos_v11_preaccess.py"),
    ("schema_inspector", "scripts/test_amigos_v11_schema_inspector.py"),
    ("feature_core", "scripts/test_amigos_v11_feature_core.py"),
    ("adapter_output_contract", "scripts/test_amigos_v11_adapter_contract.py"),
    ("implementation_lock", "scripts/test_external_wearable_implementation_lock.py"),
    ("one_shot_preparation", "scripts/test_external_wearable_one_shot_core.py"),
    ("one_shot_cuda_end_to_end", "scripts/test_external_wearable_one_shot_end_to_end.py"),
    ("gpu_model", "scripts/test_amigos_v11_gpu_model.py"),
    ("shared_dose_core_gpu", "scripts/test_external_wearable_dose_core.py"),
    ("shared_dcs_opct_core_gpu", "scripts/test_external_wearable_dcs_opct_core.py"),
    ("shared_state_machine", "scripts/test_external_wearable_v11_pipeline.py"),
    ("endpoint_eligibility", "scripts/test_amigos_v11_endpoint_eligibility.py"),
    ("external_family_registry", "scripts/validate_v11_prospective_external_family.py"),
)
EVIDENCE_FILES = (
    "scripts/validate_amigos_v11_preaccess_stack.py",
    "docs/amigos_v11_preaccess_history_audit.json",
    "docs/amigos_v11_external_confirmation_reservation.json",
    "docs/amigos_v11_external_confirmation_implementation_protocol.md",
    "docs/amigos_v11_access_attempt_001.json",
    "docs/amigos_v11_access_attempt_002.json",
    "docs/amigos_v11_access_attempt_003.json",
    "docs/amigos_v11_official_access_request_email.md",
    "docs/amigos_v11_endpoint_eligibility_planning.md",
    "docs/external_wearable_v11_standardized_analysis_contract.md",
    "docs/v11_prospective_external_family_registry.json",
    "scripts/validate_v11_prospective_external_family.py",
    "scripts/wearable_eeg_v11_feature_core.py",
    "scripts/external_wearable_dose_core.py",
    "scripts/external_wearable_dcs_opct_core.py",
    "scripts/external_wearable_v11_pipeline.py",
    "scripts/external_wearable_trial_contract.py",
    "scripts/external_wearable_implementation_lock.py",
    "scripts/external_wearable_one_shot_core.py",
    "scripts/lock_amigos_v11_external_confirmation.py",
    "scripts/amigos_v11_contract.py",
    "scripts/amigos_v11_adapter_contract.py",
    "scripts/inspect_amigos_v11_schema.py",
    "scripts/amigos_v11_feature_core.py",
    "scripts/analyze_amigos_v11_endpoint_eligibility.py",
    "scripts/test_amigos_v11_schema_inspector.py",
    "scripts/test_amigos_v11_feature_core.py",
    "scripts/test_amigos_v11_gpu_model.py",
    "scripts/test_external_wearable_dose_core.py",
    "scripts/test_external_wearable_dcs_opct_core.py",
    "scripts/test_external_wearable_v11_pipeline.py",
    "scripts/test_amigos_v11_adapter_contract.py",
    "scripts/test_external_wearable_implementation_lock.py",
    "scripts/test_external_wearable_one_shot_core.py",
    "scripts/test_external_wearable_one_shot_end_to_end.py",
    "scripts/test_amigos_v11_endpoint_eligibility.py",
    "scripts/validate_amigos_v11_preaccess.py",
    "outputs/amigos_v11_preaccess/endpoint_eligibility_operating_characteristics.csv",
    "outputs/amigos_v11_preaccess/endpoint_eligibility_report.json",
    "outputs/amigos_v11_preaccess/amigos_v11_endpoint_eligibility.png",
    "outputs/amigos_v11_preaccess/amigos_v11_endpoint_eligibility.pdf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    results = []
    progress = tqdm(TESTS, desc="AMIGOS pre-access stack", unit="check", dynamic_ncols=True)
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
        "all_tests_passed": all(item["passed"] for item in results),
        "cuda_available": torch.cuda.is_available(),
        "freeze_hash_unchanged": freeze_hash == EXPECTED_FREEZE_HASH,
        "all_evidence_files_present": all((ROOT / path).is_file() for path in EVIDENCE_FILES),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "date": "2026-09-09",
        "dataset": "AMIGOS",
        "phase": "pre-access implementation readiness",
        "checks": checks,
        "tests": results,
        "gpu": {
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "registered_epochs": 40,
        },
        "freeze_sha256": freeze_hash,
        "evidence_sha256": files,
        "remaining_before_participant_value_access": [
            "obtain the official EULA-authorized archive",
            "run the schema-only inspector",
            "implement and test the schema-specific MAT adapter and one-shot runner",
            "run the prepared implementation-lock command before participant-value access"
        ],
        "claim_boundary": (
            "Passing this report establishes only pre-access protocol and implementation "
            "readiness. It supplies no AMIGOS structural, endpoint, effectiveness, or safety evidence."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError(f"AMIGOS pre-access stack failed: {checks}")
    print(f"AMIGOS pre-access stack passed: {len(TESTS)}/{len(TESTS)} test groups")


if __name__ == "__main__":
    main()
