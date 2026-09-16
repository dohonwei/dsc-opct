from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v12"
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

    record("freeze_hash_declared", manifest.get("frozen_v11_sha256") == EXPECTED_FREEZE_HASH, manifest.get("frozen_v11_sha256"))
    record("v12_failed_gate_preserved", manifest.get("v12_gate_status") == "failed", manifest.get("v12_gate_status"))

    output_hashes = manifest.get("output_sha256", {})
    mismatches = {}
    progress = tqdm(sorted(output_hashes.items()), desc="Validate v12 package hashes", unit="file", dynamic_ncols=True)
    for relative, expected in progress:
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    progress.close()
    record("all_packaged_hashes_match", not mismatches, mismatches)
    record("declared_file_count", manifest.get("file_count_excluding_manifest_and_validation_report") == len(output_hashes), len(output_hashes))

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
    gate = read_json(OUT / "v12_analysis_retrospective_robustness_gate.json")
    coefficients = (OUT / "v12_analysis_fold_coefficients.csv").read_text(encoding="utf-8")

    record("v11_base_validation_78_of_78", v11_report.get("status") == "passed" and v11_report.get("checks_passed") == v11_report.get("checks_total") == 78, [v11_report.get("checks_passed"), v11_report.get("checks_total")])
    record("v12_analysis_validation_17_of_17", v12_report.get("status") == "passed" and v12_report.get("checks_passed") == v12_report.get("checks_total") == 17, [v12_report.get("checks_passed"), v12_report.get("checks_total")])
    record("v12_artifact_validation_11_of_11", artifact_report.get("status") == "passed" and artifact_report.get("checks_passed") == artifact_report.get("checks_total") == 11, [artifact_report.get("checks_passed"), artifact_report.get("checks_total")])
    record("prespecified_gate_failed", gate.get("status") == "failed", gate.get("checks"))
    selected = gate.get("selected_method", {})
    record("failed_metrics_preserved", abs(selected.get("median_auroc", 0) - 0.6158936918955827) < 1e-12 and abs(selected.get("median_balanced_accuracy", 0) - 0.5922882427307206) < 1e-12 and selected.get("domains_auroc_ge_060") == 4.0, selected)
    record("all_35_mechanism_coefficients_present", coefficients.count("domain_balanced_gpu") >= 35, coefficients.count("domain_balanced_gpu"))
    record("failure_records_complete", all((OUT / f"v12_failure_{name}.json").is_file() for name in ("execution_001", "execution_002", "execution_003", "figure_001")), "four preserved records")
    pdfs = [OUT / "manuscript.pdf", OUT / "supplementary.pdf", OUT / "v12_fig_multidomain_robustness.pdf"]
    record("pdf_signatures_valid", all(path.stat().st_size > 10_000 and path.read_bytes()[:4] == b"%PDF" for path in pdfs), [path.name for path in pdfs])
    record("claim_boundary_explicit", "does not establish prospective external effectiveness" in manifest.get("claim_boundary", ""), manifest.get("claim_boundary"))

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
        raise RuntimeError(f"DCS-OPCT v12 package validation failed: {failed}")
    print(f"DCS-OPCT v12 package validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
