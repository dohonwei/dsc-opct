from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_release_readiness/report.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
DATA_MATRIX = ROOT / "docs/dcs_opct_v11_dataset_availability_matrix.md"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
STACK_REPORTS = {
    "AMIGOS": ROOT / "outputs/amigos_v11_preaccess/stack_validation_report.json",
    "Emognition": ROOT / "outputs/emognition_v11_preaccess/stack_validation_report.json",
}
STACK_EXPECTED_TESTS = {"AMIGOS": 13, "Emognition": 15}
RUFF_TARGETS = (
    "scripts/external_wearable_v11_pipeline.py",
    "scripts/external_wearable_implementation_lock.py",
    "scripts/external_wearable_one_shot_core.py",
    "scripts/test_external_wearable_one_shot_end_to_end.py",
    "scripts/emognition_v11_schema_adapter.py",
    "scripts/run_emognition_v11_external_confirmation.py",
    "scripts/test_emognition_v11_schema_adapter.py",
    "scripts/test_emognition_v11_archive_adapter.py",
    "scripts/validate_emognition_v11_preaccess_stack.py",
    "scripts/validate_amigos_v11_preaccess_stack.py",
    "scripts/validate_dcs_opct_v11_release_readiness.py",
    "scripts/detect_dcs_opct_v11_external_arrival.py",
    "scripts/validate_dcs_opct_v11_external_arrival.py",
    "scripts/audit_dcs_opct_v11_q1_submission_readiness.py",
)
ARRIVAL_REPORT = ROOT / "outputs/dcs_opct_v11_external_arrival_readiness/report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_step(name: str, command: list[str]) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {
        "name": name,
        "command": command,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "stdout_tail": completed.stdout.strip()[-4000:],
        "stderr_tail": completed.stderr.strip()[-4000:],
    }


def verify_stack_report(dataset: str, path: Path) -> dict:
    if not path.is_file():
        return {"dataset": dataset, "passed": False, "reason": "report_missing"}
    report = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {}
    evidence = report.get("evidence_sha256", {})
    for relative, recorded in evidence.items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if recorded != actual:
            mismatches[relative] = {"recorded": recorded, "actual": actual}
    tests = report.get("tests", [])
    checks = {
        "status_passed": report.get("status") == "passed",
        "expected_test_groups": len(tests) == STACK_EXPECTED_TESTS[dataset],
        "all_test_groups_passed": all(item.get("passed") for item in tests),
        "freeze_hash_unchanged": report.get("freeze_sha256") == EXPECTED_FREEZE_HASH,
        "current_code_hashes_match": not mismatches,
    }
    return {
        "dataset": dataset,
        "passed": all(checks.values()),
        "checks": checks,
        "mismatches": mismatches,
        "report_sha256": sha256(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DCS-OPCT v11 pre-submission integrity audit."
    )
    parser.add_argument(
        "--skip-preaccess-refresh",
        action="store_true",
        help="Verify existing AMIGOS and Emognition reports without rerunning their CUDA stacks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    commands: list[tuple[str, list[str]]] = [
        ("compile_scripts", [sys.executable, "-m", "compileall", "-q", "scripts"]),
        ("ruff_safety_stack", [sys.executable, "-m", "ruff", "check", *RUFF_TARGETS]),
        (
            "validate_external_arrival_detector",
            [sys.executable, "scripts/validate_dcs_opct_v11_external_arrival.py"],
        ),
        (
            "refresh_external_arrival_status",
            [sys.executable, "scripts/detect_dcs_opct_v11_external_arrival.py"],
        ),
    ]
    if not args.skip_preaccess_refresh:
        commands.extend(
            [
                ("refresh_amigos_preaccess", [sys.executable, "scripts/validate_amigos_v11_preaccess_stack.py"]),
                ("refresh_emognition_preaccess", [sys.executable, "scripts/validate_emognition_v11_preaccess_stack.py"]),
            ]
        )
    commands.append(
        ("validate_submission_artifacts", [sys.executable, "scripts/validate_dcs_opct_v11_submission_artifacts.py"])
    )

    command_results = []
    progress = tqdm(commands, desc="DCS-OPCT v11 release audit", unit="step", dynamic_ncols=True)
    for name, command in progress:
        progress.set_postfix_str(name)
        command_results.append(run_step(name, command))
    progress.close()

    freeze_hash = sha256(FREEZE)
    stack_results = [verify_stack_report(name, path) for name, path in STACK_REPORTS.items()]
    arrival_report = (
        json.loads(ARRIVAL_REPORT.read_text(encoding="utf-8"))
        if ARRIVAL_REPORT.is_file()
        else {}
    )
    checks = {
        "all_commands_passed": all(item["passed"] for item in command_results),
        "freeze_hash_unchanged": freeze_hash == EXPECTED_FREEZE_HASH,
        "cuda_available": torch.cuda.is_available(),
        "both_preaccess_stacks_current": all(item["passed"] for item in stack_results),
        "dataset_availability_matrix_present": DATA_MATRIX.is_file(),
        "external_arrival_detector_fail_closed": (
            arrival_report.get("automatic_execution") is False
            and arrival_report.get("participant_values_accessed") is False
            and arrival_report.get("freeze_sha256") == EXPECTED_FREEZE_HASH
            and len(arrival_report.get("datasets", [])) == 2
        ),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "pre-submission software and evidence readiness",
        "claim_boundary": (
            "This audit establishes internal reproducibility and pre-access software readiness only. "
            "It is not external effectiveness, non-harm, or safety evidence."
        ),
        "checks": checks,
        "commands": command_results,
        "preaccess_stack_reports": stack_results,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "freeze_sha256": freeze_hash,
        "driver_sha256": sha256(Path(__file__)),
        "dataset_availability_matrix_sha256": sha256(DATA_MATRIX) if DATA_MATRIX.is_file() else None,
        "external_arrival_report_sha256": (
            sha256(ARRIVAL_REPORT) if ARRIVAL_REPORT.is_file() else None
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"DCS-OPCT v11 release audit failed: {failed}")
    print(f"DCS-OPCT v11 release audit passed: {len(checks)}/{len(checks)} gates")
    print(f"Report: {OUT}")


if __name__ == "__main__":
    main()
