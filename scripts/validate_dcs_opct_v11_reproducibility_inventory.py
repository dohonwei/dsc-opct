from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/dcs_opct_v11_reproducibility_inventory"
INVENTORY = OUT_DIR / "inventory.json"
REPORT = OUT_DIR / "validation_report.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
READINESS = ROOT / "outputs/dcs_opct_v11_release_readiness/report.json"
SUBMISSION_MANIFEST = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v11/submission_artifact_manifest.json"
SUBMISSION_REPORT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v11/independent_validation_report.json"
LATEST_SUBMISSION_MANIFEST = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/submission_artifact_manifest.json"
LATEST_SUBMISSION_REPORT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/independent_validation_report.json"
Q1_READINESS = ROOT / "outputs/dcs_opct_v11_q1_submission_readiness/readiness_report.json"
ARRIVAL_REPORT = ROOT / "outputs/dcs_opct_v11_external_arrival_readiness/report.json"
CLUSTER_REPORT = ROOT / "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910/independent_validation_report.json"
AGGREGATION_REPORT = ROOT / "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910/independent_validation_report.json"
PREVIEW_REPORT = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/independent_validation_report.json"
PREVIEW_MANIFEST = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/preview_manifest.json"
HIGHER_LEVEL_REPORT = ROOT / "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/independent_validation_report.json"
STACK_REPORTS = {
    "AMIGOS": ROOT / "outputs/amigos_v11_preaccess/stack_validation_report.json",
    "Emognition": ROOT / "outputs/emognition_v11_preaccess/stack_validation_report.json",
}
STACK_EXPECTED_TESTS = {"AMIGOS": 13, "Emognition": 15}
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EXPECTED_SCHEMA = "dcs-opct-v11-reproducibility-inventory-1.0"
REQUIRED_GROUPS = {
    "public_repository_finalization",
    "reproducibility_driver",
    "access_and_readiness",
    "prospective_external_software",
    "submission_package",
    "latest_submission_package",
    "component_diagnostic",
    "cluster_dependence_audit_yield",
    "cluster_score_aggregation_sensitivity",
    "evidence_integration_preview_v2",
    "nature_review_and_preview_v3",
    "higher_level_dependence_and_preview_v4",
    "reviewer_closure_20260914",
    "manuscript",
    "frozen_core_evidence",
}
EXPECTED_FIXED_PATH_GROUPS = {
    "docs/dcs_opct_v11_public_repository_readme.md": "public_repository_finalization",
    "docs/dcs_opct_v11_public_repository_receipt_20260916.json": "public_repository_finalization",
    "docs/dcs_opct_v11_reviewer_task_traceability_20260916.md": "public_repository_finalization",
    "docs/elsarticle/dcs_opct_v11_author_confirmation_checklist.md": "public_repository_finalization",
    "scripts/finalize_dcs_opct_repository_link.py": "public_repository_finalization",
    "scripts/build_dcs_opct_v11_reviewer_task_traceability.py": "public_repository_finalization",
    "scripts/validate_dcs_opct_v11_reviewer_task_traceability.py": "public_repository_finalization",
    "scripts/make_dcs_opct_v23_submission_artifacts.py": "public_repository_finalization",
    "scripts/validate_dcs_opct_v23_submission_artifacts.py": "public_repository_finalization",
    "outputs/dcs_opct_v11_reviewer_task_traceability_20260916/reviewer_task_traceability.json": "public_repository_finalization",
    "outputs/dcs_opct_v11_reviewer_task_traceability_20260916/independent_validation_report.json": "public_repository_finalization",
    "outputs/dcs_opct_v11_submission_artifacts_crossfit_v22_zip_verification.json": "public_repository_finalization",
    "scripts/validate_dcs_opct_v11_release_readiness.py": "reproducibility_driver",
    "scripts/build_dcs_opct_v11_reproducibility_inventory.py": "reproducibility_driver",
    "scripts/validate_dcs_opct_v11_reproducibility_inventory.py": "reproducibility_driver",
    "scripts/audit_dcs_opct_v11_q1_submission_readiness.py": "reproducibility_driver",
    "scripts/detect_dcs_opct_v11_external_arrival.py": "reproducibility_driver",
    "scripts/validate_dcs_opct_v11_external_arrival.py": "reproducibility_driver",
    "scripts/validate_dcs_opct_v11_reviewer_closure.py": "reproducibility_driver",
    "docs/dcs_opct_v11_dataset_availability_matrix.md": "access_and_readiness",
    "outputs/dcs_opct_v11_release_readiness/report.json": "access_and_readiness",
    "outputs/amigos_v11_preaccess/stack_validation_report.json": "access_and_readiness",
    "outputs/emognition_v11_preaccess/stack_validation_report.json": "access_and_readiness",
    "outputs/dcs_opct_v11_q1_submission_readiness/readiness_report.json": "access_and_readiness",
    "outputs/dcs_opct_v11_external_arrival_readiness/report.json": "access_and_readiness",
    "scripts/external_wearable_implementation_lock.py": "prospective_external_software",
    "scripts/external_wearable_one_shot_core.py": "prospective_external_software",
    "scripts/test_external_wearable_implementation_lock.py": "prospective_external_software",
    "scripts/test_external_wearable_one_shot_core.py": "prospective_external_software",
    "scripts/test_external_wearable_one_shot_end_to_end.py": "prospective_external_software",
    "outputs/dcs_opct_v11_submission_artifacts_crossfit_v11/submission_artifact_manifest.json": "submission_package",
    "outputs/dcs_opct_v11_submission_artifacts_crossfit_v11/independent_validation_report.json": "submission_package",
    "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/submission_artifact_manifest.json": "latest_submission_package",
    "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/independent_validation_report.json": "latest_submission_package",
    "scripts/analyze_dcs_opct_v11_cluster_dependence_and_audit_yield.py": "cluster_dependence_audit_yield",
    "scripts/validate_dcs_opct_v11_cluster_dependence_and_audit_yield.py": "cluster_dependence_audit_yield",
    "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910/manifest.json": "cluster_dependence_audit_yield",
    "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910/independent_validation_report.json": "cluster_dependence_audit_yield",
    "scripts/analyze_dcs_opct_v11_cluster_score_aggregation_sensitivity.py": "cluster_score_aggregation_sensitivity",
    "scripts/validate_dcs_opct_v11_cluster_score_aggregation_sensitivity.py": "cluster_score_aggregation_sensitivity",
    "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910/manifest.json": "cluster_score_aggregation_sensitivity",
    "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910/independent_validation_report.json": "cluster_score_aggregation_sensitivity",
    "scripts/build_dcs_opct_v11_evidence_integration_preview.py": "evidence_integration_preview_v2",
    "scripts/validate_dcs_opct_v11_evidence_integration_preview.py": "evidence_integration_preview_v2",
    "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/preview_manifest.json": "evidence_integration_preview_v2",
    "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/independent_validation_report.json": "evidence_integration_preview_v2",
    "scripts/build_dcs_opct_v11_nature_review.py": "nature_review_and_preview_v3",
    "scripts/build_dcs_opct_v11_evidence_integration_preview_v3.py": "nature_review_and_preview_v3",
    "scripts/validate_dcs_opct_v11_evidence_integration_preview_v3.py": "nature_review_and_preview_v3",
    "outputs/dcs_opct_v11_nature_review_20260910/review_dcs_opct_v11_20260910.docx": "nature_review_and_preview_v3",
    "outputs/dcs_opct_v11_nature_review_20260910/review_dcs_opct_v11_20260910.md": "nature_review_and_preview_v3",
    "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/preview_manifest.json": "nature_review_and_preview_v3",
    "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/independent_validation_report.json": "nature_review_and_preview_v3",
    "scripts/analyze_dcs_opct_v11_higher_level_cluster_sensitivity.py": "higher_level_dependence_and_preview_v4",
    "scripts/validate_dcs_opct_v11_higher_level_cluster_sensitivity.py": "higher_level_dependence_and_preview_v4",
    "scripts/build_dcs_opct_v11_evidence_integration_preview_v4.py": "higher_level_dependence_and_preview_v4",
    "scripts/validate_dcs_opct_v11_evidence_integration_preview_v4.py": "higher_level_dependence_and_preview_v4",
    "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/manifest.json": "higher_level_dependence_and_preview_v4",
    "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/independent_validation_report.json": "higher_level_dependence_and_preview_v4",
    "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/preview_manifest.json": "higher_level_dependence_and_preview_v4",
    "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/independent_validation_report.json": "higher_level_dependence_and_preview_v4",
    "scripts/analyze_dcs_opct_v11_leave_one_component.py": "component_diagnostic",
    "scripts/validate_dcs_opct_v11_leave_one_component.py": "component_diagnostic",
    "outputs/dcs_opct_v11_leave_one_component/same_assignment_leave_one_component.csv": "component_diagnostic",
    "outputs/dcs_opct_v11_leave_one_component/random_half_allocation_repetitions.csv": "component_diagnostic",
    "outputs/dcs_opct_v11_leave_one_component/distribution_coverage_summary.csv": "component_diagnostic",
    "outputs/dcs_opct_v11_leave_one_component/table_leave_one_component.tex": "component_diagnostic",
    "outputs/dcs_opct_v11_leave_one_component/table_distribution_coverage_diagnostic.tex": "component_diagnostic",
    "outputs/dcs_opct_v11_leave_one_component/manifest.json": "component_diagnostic",
    "outputs/dcs_opct_v11_leave_one_component/independent_validation_report.json": "component_diagnostic",
    "scripts/analyze_dcs_opct_v11_eppvr_raw_partition_first.py": "reviewer_closure_20260914",
    "scripts/validate_dcs_opct_v11_eppvr_raw_partition_first.py": "reviewer_closure_20260914",
    "scripts/build_dcs_opct_v11_independence_ladder.py": "reviewer_closure_20260914",
    "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/independent_validation_report.json": "reviewer_closure_20260914",
    "docs/elsarticle/figures/fig_v11_independence_ladder.pdf": "reviewer_closure_20260914",
    "docs/elsarticle/tables/table_v11_independence_ladder.tex": "reviewer_closure_20260914",
    "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex": "manuscript",
    "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.pdf": "manuscript",
    "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex": "manuscript",
    "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.pdf": "manuscript",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    readiness = read_json(READINESS)
    submission_manifest = read_json(SUBMISSION_MANIFEST)
    submission_report = read_json(SUBMISSION_REPORT)
    latest_submission_manifest = read_json(LATEST_SUBMISSION_MANIFEST)
    latest_submission_report = read_json(LATEST_SUBMISSION_REPORT)
    q1_readiness = read_json(Q1_READINESS)
    arrival_report = read_json(ARRIVAL_REPORT)
    cluster_report = read_json(CLUSTER_REPORT)
    aggregation_report = read_json(AGGREGATION_REPORT)
    higher_level_report = read_json(HIGHER_LEVEL_REPORT)
    preview_report = read_json(PREVIEW_REPORT)
    preview_manifest = read_json(PREVIEW_MANIFEST)
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    entries = inventory.get("entries", [])
    paths = [entry.get("path") for entry in entries]
    record("schema_version", inventory.get("schema_version") == EXPECTED_SCHEMA, inventory.get("schema_version"))
    record("unique_paths", len(paths) == len(set(paths)), len(paths))
    record("declared_file_count", inventory.get("unique_file_count") == len(entries), len(entries))
    record(
        "required_groups_present",
        REQUIRED_GROUPS.issubset(inventory.get("group_counts", {})),
        sorted(inventory.get("group_counts", {})),
    )
    forbidden = {
        "outputs/dcs_opct_v11_reproducibility_inventory/inventory.json",
        "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json",
    }
    record("no_self_or_report_hash", forbidden.isdisjoint(paths), sorted(forbidden.intersection(paths)))

    mismatches = {}
    progress = tqdm(entries, desc="Validate reproducibility hashes", unit="file", dynamic_ncols=True)
    for entry in progress:
        relative = entry["path"]
        path = ROOT / relative
        if not path.is_file():
            mismatches[relative] = {"reason": "missing"}
            continue
        actual = sha256(path)
        if actual != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
            mismatches[relative] = {
                "recorded_sha256": entry.get("sha256"),
                "actual_sha256": actual,
                "recorded_bytes": entry.get("bytes"),
                "actual_bytes": path.stat().st_size,
            }
    progress.close()
    record("all_inventory_hashes_match", not mismatches, mismatches)

    freeze_hash = sha256(FREEZE)
    record(
        "freeze_hash_unchanged",
        freeze_hash == EXPECTED_FREEZE_HASH == inventory.get("freeze_sha256"),
        freeze_hash,
    )
    entry_map = {entry["path"]: entry for entry in entries}
    missing_fixed = []
    wrong_fixed_groups = {}
    for relative, expected_group in EXPECTED_FIXED_PATH_GROUPS.items():
        entry = entry_map.get(relative)
        if entry is None:
            missing_fixed.append(relative)
        elif expected_group not in entry.get("groups", []):
            wrong_fixed_groups[relative] = {
                "expected": expected_group,
                "recorded": entry.get("groups", []),
            }
    record("all_fixed_evidence_included", not missing_fixed, missing_fixed)
    record("fixed_evidence_groups_correct", not wrong_fixed_groups, wrong_fixed_groups)

    missing_frozen = []
    changed_frozen = {}
    for relative, expected in freeze["locked_artifacts"].items():
        normalized = Path(relative.replace("\\", "/")).as_posix()
        entry = entry_map.get(normalized)
        if entry is None:
            missing_frozen.append(normalized)
        elif entry["sha256"] != expected:
            changed_frozen[normalized] = {"freeze": expected, "inventory": entry["sha256"]}
    record("all_frozen_artifacts_included", not missing_frozen, missing_frozen)
    record("frozen_artifact_hashes_match_freeze", not changed_frozen, changed_frozen)

    readiness_checks = readiness.get("checks", {})
    record(
        "release_readiness_passed",
        readiness.get("status") == "passed"
        and readiness.get("freeze_sha256") == EXPECTED_FREEZE_HASH
        and all(readiness_checks.values()),
        {"status": readiness.get("status"), "checks": readiness_checks},
    )
    for dataset, path in STACK_REPORTS.items():
        report = read_json(path)
        record(
            f"{dataset.lower()}_preaccess_stack_passed",
            report.get("status") == "passed"
            and len(report.get("tests", [])) == STACK_EXPECTED_TESTS[dataset]
            and all(test.get("passed") for test in report.get("tests", []))
            and report.get("freeze_sha256") == EXPECTED_FREEZE_HASH,
            {"status": report.get("status"), "test_groups": len(report.get("tests", []))},
        )

    record(
        "submission_package_validated",
        submission_manifest.get("status") == "submission_artifacts_generated_and_validated"
        and submission_manifest.get("input_sha256", {}).get(
            "docs\\distribution_covered_stratified_opct_v11_final_freeze.json"
        )
        == EXPECTED_FREEZE_HASH
        and submission_report.get("status") == "passed"
        and submission_report.get("checks_passed") == submission_report.get("checks_total") == 78,
        {
            "manifest_status": submission_manifest.get("status"),
            "validation": [submission_report.get("checks_passed"), submission_report.get("checks_total")],
        },
    )
    record(
        "latest_submission_package_validated",
        latest_submission_report.get("status") == "passed"
        and latest_submission_report.get("checks_passed")
        == latest_submission_report.get("checks_total")
        == 21
        and latest_submission_manifest.get("frozen_v11_sha256") == EXPECTED_FREEZE_HASH,
        {
            "validation": [
                latest_submission_report.get("checks_passed"),
                latest_submission_report.get("checks_total"),
            ],
            "freeze": latest_submission_manifest.get("frozen_v11_sha256"),
        },
    )
    record(
        "q1_readiness_preserves_claim_boundary",
        q1_readiness.get("status") == "not_ready"
        and q1_readiness.get("claim_readiness", {}).get(
            "q1_strong_effective_safe_transfer"
        )
        is False,
        q1_readiness.get("claim_readiness"),
    )
    record(
        "external_arrival_detector_fail_closed",
        arrival_report.get("automatic_execution") is False
        and arrival_report.get("participant_values_accessed") is False
        and arrival_report.get("freeze_sha256") == EXPECTED_FREEZE_HASH
        and all(
            item.get("participant_values_accessed") is False
            for item in arrival_report.get("datasets", [])
        ),
        [item.get("status") for item in arrival_report.get("datasets", [])],
    )
    record(
        "cluster_dependence_analysis_validated",
        cluster_report.get("status") == "passed"
        and cluster_report.get("checks_passed") == cluster_report.get("checks_total") == 16,
        [cluster_report.get("checks_passed"), cluster_report.get("checks_total")],
    )
    record(
        "cluster_aggregation_sensitivity_validated",
        aggregation_report.get("status") == "passed"
        and aggregation_report.get("checks_passed")
        == aggregation_report.get("checks_total")
        == 8,
        [aggregation_report.get("checks_passed"), aggregation_report.get("checks_total")],
    )
    record(
        "higher_level_cluster_sensitivity_validated",
        higher_level_report.get("status") == "passed"
        and higher_level_report.get("checks_passed")
        == higher_level_report.get("checks_total")
        == 13,
        [higher_level_report.get("checks_passed"), higher_level_report.get("checks_total")],
    )
    record(
        "staged_preview_v4_validated",
        preview_report.get("status") == "passed"
        and preview_report.get("checks_passed") == preview_report.get("checks_total") == 11
        and preview_manifest.get("status")
        == "staged_preview_not_merged_into_authoritative_manuscript"
        and preview_manifest.get("authoritative_sources_unchanged") is True,
        {
            "checks": [preview_report.get("checks_passed"), preview_report.get("checks_total")],
            "status": preview_manifest.get("status"),
        },
    )
    pdf_paths = [path for path in paths if path.endswith(".pdf")]
    pdf_checks = {}
    for relative in pdf_paths:
        path = ROOT / relative
        pdf_checks[relative] = path.is_file() and path.stat().st_size > 10_000 and path.read_bytes()[:4] == b"%PDF"
    required_compiled_pdfs = {
        "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.pdf",
        "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/manuscript_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/supplementary_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/manuscript_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/supplementary_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/manuscript_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/supplementary_preview.pdf",
    }
    record(
        "compiled_pdfs_valid",
        required_compiled_pdfs.issubset(pdf_paths) and all(pdf_checks.values()),
        {"required_present": required_compiled_pdfs.issubset(pdf_paths), "count": len(pdf_paths)},
    )

    passed = sum(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": inventory.get("claim_boundary"),
        "inventory_sha256": sha256(INVENTORY),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_idempotent(report)
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"Reproducibility inventory validation failed: {failed}")
    print(f"Reproducibility inventory validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
