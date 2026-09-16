from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/dcs_opct_v12_reproducibility_inventory"
INVENTORY = OUT_DIR / "inventory.json"
REPORT = OUT_DIR / "validation_report.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
READINESS = ROOT / "outputs/dcs_opct_v12_release_readiness/report.json"
PACKAGE = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/independent_validation_report.json"
V12_ANALYSIS = ROOT / "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/independent_validation_report.json"
V12_ARTIFACTS = ROOT / "outputs/dcs_opct_v12_multidomain_artifacts_v3/independent_validation_report.json"
V12_GATE = ROOT / "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/retrospective_robustness_gate.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EXPECTED_GROUPS = {
    "release_and_inventory_drivers",
    "submission_package",
    "v12_multidomain_analysis",
    "v12_artifacts",
    "preserved_v12_failures",
    "prospective_external_readiness",
    "manuscript",
    "frozen_v11_core",
}


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
    inventory = read_json(INVENTORY)
    freeze = read_json(FREEZE)
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    entries = inventory.get("entries", [])
    paths = [entry.get("path") for entry in entries]
    record("schema_version", inventory.get("schema_version") == "dcs-opct-v12-reproducibility-inventory-1.0", inventory.get("schema_version"))
    record("unique_paths", len(paths) == len(set(paths)), len(paths))
    record("declared_file_count", inventory.get("unique_file_count") == len(entries), len(entries))
    record("required_groups_present", EXPECTED_GROUPS.issubset(inventory.get("group_counts", {})), sorted(inventory.get("group_counts", {})))
    forbidden = {
        "outputs/dcs_opct_v12_reproducibility_inventory/inventory.json",
        "outputs/dcs_opct_v12_reproducibility_inventory/validation_report.json",
    }
    record("no_self_or_report_hash", forbidden.isdisjoint(paths), sorted(forbidden.intersection(paths)))

    mismatches = {}
    progress = tqdm(entries, desc="Validate v12 inventory hashes", unit="file", dynamic_ncols=True)
    for entry in progress:
        path = ROOT / entry["path"]
        actual = sha256(path) if path.is_file() else None
        actual_bytes = path.stat().st_size if path.is_file() else None
        if actual != entry.get("sha256") or actual_bytes != entry.get("bytes"):
            mismatches[entry["path"]] = {"sha256": actual, "bytes": actual_bytes}
    progress.close()
    record("all_inventory_hashes_match", not mismatches, mismatches)

    freeze_hash = sha256(FREEZE)
    record("freeze_hash_unchanged", freeze_hash == EXPECTED_FREEZE_HASH == inventory.get("freeze_sha256"), freeze_hash)
    entry_map = {entry["path"]: entry for entry in entries}
    missing_frozen = []
    changed_frozen = {}
    for relative, expected in freeze["locked_artifacts"].items():
        relative = Path(relative.replace("\\", "/")).as_posix()
        entry = entry_map.get(relative)
        if entry is None:
            missing_frozen.append(relative)
        elif entry.get("sha256") != expected:
            changed_frozen[relative] = {"expected": expected, "actual": entry.get("sha256")}
    record("all_frozen_artifacts_present", not missing_frozen, missing_frozen)
    record("all_frozen_artifact_hashes_match", not changed_frozen, changed_frozen)

    readiness = read_json(READINESS)
    package = read_json(PACKAGE)
    analysis = read_json(V12_ANALYSIS)
    artifacts = read_json(V12_ARTIFACTS)
    gate = read_json(V12_GATE)
    record("release_readiness_6_of_6", readiness.get("status") == "passed" and len(readiness.get("checks", {})) == 6 and all(readiness.get("checks", {}).values()), readiness.get("checks"))
    record("submission_package_21_of_21", package.get("status") == "passed" and package.get("checks_passed") == package.get("checks_total") == 21, [package.get("checks_passed"), package.get("checks_total")])
    record("v12_analysis_17_of_17", analysis.get("status") == "passed" and analysis.get("checks_passed") == analysis.get("checks_total") == 17, [analysis.get("checks_passed"), analysis.get("checks_total")])
    record("v12_artifacts_11_of_11", artifacts.get("status") == "passed" and artifacts.get("checks_passed") == artifacts.get("checks_total") == 11, [artifacts.get("checks_passed"), artifacts.get("checks_total")])
    record("failed_gate_preserved", gate.get("status") == "failed", gate.get("checks"))
    failure_paths = [path for path in paths if "failure" in path and path.startswith("docs/dcs_opct_v12")]
    record("eight_v12_failure_records_present", len(failure_paths) == 8, failure_paths)
    pdf_paths = [ROOT / path for path in paths if path.endswith(".pdf")]
    record("compiled_pdfs_valid", len(pdf_paths) == 3 and all(path.stat().st_size > 10_000 and path.read_bytes()[:4] == b"%PDF" for path in pdf_paths), [path.name for path in pdf_paths])
    record("claim_boundary_explicit", "not prospective external effectiveness" in inventory.get("claim_boundary", ""), inventory.get("claim_boundary"))

    passed = sum(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_sha256": sha256(INVENTORY),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "claim_boundary": inventory.get("claim_boundary"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_idempotent(report)
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"DCS-OPCT v12 reproducibility validation failed: {failed}")
    print(f"DCS-OPCT v12 reproducibility validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
