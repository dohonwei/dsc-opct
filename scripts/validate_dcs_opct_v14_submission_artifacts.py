from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v14"
MANIFEST = OUT / "submission_artifact_manifest.json"
REPORT = OUT / "independent_validation_report.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_idempotent(report: dict) -> None:
    if REPORT.is_file():
        previous = read_json(REPORT)
        old = {key: value for key, value in previous.items() if key != "generated_at_utc"}
        new = {key: value for key, value in report.items() if key != "generated_at_utc"}
        if old == new:
            report["generated_at_utc"] = previous.get("generated_at_utc")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    manifest = read_json(MANIFEST)
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record(
        "freeze_hash_declared",
        manifest.get("frozen_v11_sha256") == EXPECTED_FREEZE_HASH,
        manifest.get("frozen_v11_sha256"),
    )
    record(
        "v12_failed_gate_preserved",
        manifest.get("v12_gate_status") == "failed",
        manifest.get("v12_gate_status"),
    )

    output_hashes = manifest.get("output_sha256", {})
    mismatches = {}
    progress = tqdm(
        sorted(output_hashes.items()),
        desc="Validate v14 package hashes",
        unit="file",
        dynamic_ncols=True,
    )
    for relative, expected in progress:
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    progress.close()
    record("all_packaged_hashes_match", not mismatches, mismatches)
    record(
        "declared_file_count",
        manifest.get("file_count_excluding_manifest_and_validation_report")
        == len(output_hashes),
        len(output_hashes),
    )

    source_mismatches = {}
    for relative, expected in manifest.get("source_sha256", {}).items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            source_mismatches[relative] = {"expected": expected, "actual": actual}
    record("all_source_hashes_current", not source_mismatches, source_mismatches)

    v11_report = read_json(OUT / "v11_base_independent_validation_report.json")
    v12_report = read_json(OUT / "v12_analysis_independent_validation_report.json")
    artifact_report = read_json(OUT / "v12_artifact_independent_validation_report.json")
    v13_report = read_json(OUT / "v13_base_independent_validation_report.json")
    external_report = read_json(OUT / "external_access_v5_validation_report.json")
    v11_readiness = read_json(OUT / "v11_release_readiness_report.json")
    v11_inventory = read_json(OUT / "v11_reproducibility_validation_report.json")
    amigos = read_json(OUT / "amigos_preaccess_stack_report.json")
    emognition = read_json(OUT / "emognition_preaccess_stack_report.json")
    gate = read_json(OUT / "v12_analysis_retrospective_robustness_gate.json")

    record(
        "v11_base_validation_78_of_78",
        v11_report.get("status") == "passed"
        and v11_report.get("checks_passed") == v11_report.get("checks_total") == 78,
        [v11_report.get("checks_passed"), v11_report.get("checks_total")],
    )
    record(
        "v12_analysis_validation_17_of_17",
        v12_report.get("status") == "passed"
        and v12_report.get("checks_passed") == v12_report.get("checks_total") == 17,
        [v12_report.get("checks_passed"), v12_report.get("checks_total")],
    )
    record(
        "v12_artifact_validation_11_of_11",
        artifact_report.get("status") == "passed"
        and artifact_report.get("checks_passed")
        == artifact_report.get("checks_total")
        == 11,
        [artifact_report.get("checks_passed"), artifact_report.get("checks_total")],
    )
    record(
        "v13_historical_package_14_of_14",
        v13_report.get("status") == "passed"
        and v13_report.get("checks_passed") == v13_report.get("checks_total") == 14,
        [v13_report.get("checks_passed"), v13_report.get("checks_total")],
    )
    record(
        "external_access_v5_12_of_12",
        external_report.get("status") == "passed"
        and external_report.get("checks_passed")
        == external_report.get("checks_total")
        == 12,
        [external_report.get("checks_passed"), external_report.get("checks_total")],
    )
    record(
        "v11_release_readiness_passed",
        v11_readiness.get("status") == "passed"
        and all(v11_readiness.get("checks", {}).values()),
        v11_readiness.get("checks"),
    )
    record(
        "v11_reproducibility_inventory_16_of_16",
        v11_inventory.get("status") == "passed"
        and v11_inventory.get("checks_passed")
        == v11_inventory.get("checks_total")
        == 16,
        [v11_inventory.get("checks_passed"), v11_inventory.get("checks_total")],
    )
    record(
        "amigos_preaccess_cuda_13_of_13",
        amigos.get("status") == "passed"
        and len(amigos.get("tests", [])) == 13
        and amigos.get("checks", {}).get("cuda_available") is True,
        {"tests": len(amigos.get("tests", [])), "cuda": amigos.get("checks", {}).get("cuda_available")},
    )
    record(
        "emognition_preaccess_cuda_15_of_15",
        emognition.get("status") == "passed"
        and len(emognition.get("tests", [])) == 15
        and emognition.get("checks", {}).get("cuda_available") is True,
        {"tests": len(emognition.get("tests", [])), "cuda": emognition.get("checks", {}).get("cuda_available")},
    )
    record("prespecified_gate_failed", gate.get("status") == "failed", gate.get("checks"))

    pdfs = [
        OUT / "manuscript.pdf",
        OUT / "supplementary.pdf",
        OUT / "v12_fig_multidomain_robustness.pdf",
    ]
    record(
        "pdf_signatures_valid",
        all(
            path.stat().st_size > 10_000 and path.read_bytes()[:4] == b"%PDF"
            for path in pdfs
        ),
        [path.name for path in pdfs],
    )
    record(
        "no_restricted_participant_data",
        manifest.get("restricted_participant_data_included") is False,
        manifest.get("restricted_participant_data_included"),
    )
    packaging_failure = read_json(OUT / "v14_packaging_failure_001.json")
    record(
        "packaging_failure_preserved_without_scientific_impact",
        packaging_failure.get("status") == "preserved_packaging_execution_failure"
        and packaging_failure.get("scientific_impact", "").startswith("none;"),
        packaging_failure.get("error"),
    )
    record(
        "claim_boundary_explicit",
        "does not establish prospective external effectiveness"
        in manifest.get("claim_boundary", ""),
        manifest.get("claim_boundary"),
    )
    record(
        "anti_cycle_rule_explicit",
        "not package members" in manifest.get("anti_cycle_rule", ""),
        manifest.get("anti_cycle_rule"),
    )

    passed = sum(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(MANIFEST),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "claim_boundary": manifest.get("claim_boundary"),
    }
    write_idempotent(report)
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"DCS-OPCT v14 package validation failed: {failed}")
    print(f"DCS-OPCT v14 package validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
