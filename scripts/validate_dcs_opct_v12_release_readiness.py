from __future__ import annotations

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
OUT = ROOT / "outputs/dcs_opct_v12_release_readiness/report.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
PACKAGE_REPORT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/independent_validation_report.json"
V12_GATE = ROOT / "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/retrospective_robustness_gate.json"


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


def main() -> None:
    started = time.perf_counter()
    commands = [
        ("compile_scripts", [sys.executable, "-m", "compileall", "-q", "scripts"]),
        (
            "ruff_v12_release_stack",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "scripts/make_dcs_opct_v15_submission_artifacts.py",
                "scripts/validate_dcs_opct_v15_submission_artifacts.py",
                "scripts/validate_dcs_opct_v12_release_readiness.py",
                "scripts/analyze_dcs_opct_v12_multidomain_risk_robustness.py",
                "scripts/validate_dcs_opct_v12_multidomain_risk_robustness.py",
                "scripts/make_dcs_opct_v12_multidomain_artifacts_v3.py",
                "scripts/validate_dcs_opct_v12_multidomain_artifacts_v3.py",
            ],
        ),
        ("v11_full_gpu_release_audit", [sys.executable, "scripts/validate_dcs_opct_v11_release_readiness.py"]),
        ("validate_v12_multidomain_analysis", [sys.executable, "scripts/validate_dcs_opct_v12_multidomain_risk_robustness.py"]),
        ("validate_v12_multidomain_artifacts", [sys.executable, "scripts/validate_dcs_opct_v12_multidomain_artifacts_v3.py"]),
        ("validate_v15_submission_package", [sys.executable, "scripts/validate_dcs_opct_v15_submission_artifacts.py"]),
    ]

    results = []
    progress = tqdm(commands, desc="DCS-OPCT v12 release audit", unit="step", dynamic_ncols=True)
    for name, command in progress:
        progress.set_postfix_str(name)
        results.append(run_step(name, command))
    progress.close()

    package_report = json.loads(PACKAGE_REPORT.read_text(encoding="utf-8"))
    gate = json.loads(V12_GATE.read_text(encoding="utf-8"))
    freeze_hash = sha256(FREEZE)
    checks = {
        "all_commands_passed": all(result["passed"] for result in results),
        "freeze_hash_unchanged": freeze_hash == EXPECTED_FREEZE_HASH,
        "cuda_available": torch.cuda.is_available(),
        "v12_package_validated": package_report.get("status") == "passed"
        and package_report.get("checks_passed") == package_report.get("checks_total") == 21,
        "failed_robustness_gate_preserved": gate.get("status") == "failed",
        "claim_boundary_preserved": "does not establish prospective external effectiveness"
        in package_report.get("claim_boundary", ""),
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "v12 submission software, GPU, artifact, and claim-boundary readiness",
        "claim_boundary": (
            "Passing establishes internal reproducibility and release readiness only. "
            "The seven-domain retrospective gate remains failed, and prospective external "
            "effectiveness or universal safety is not established."
        ),
        "checks": checks,
        "commands": results,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "freeze_sha256": freeze_hash,
        "package_report_sha256": sha256(PACKAGE_REPORT),
        "driver_sha256": sha256(Path(__file__)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"DCS-OPCT v12 release audit failed: {failed}")
    print(f"DCS-OPCT v12 release audit passed: {len(checks)}/{len(checks)} gates")
    print(f"Report: {OUT}")


if __name__ == "__main__":
    main()
