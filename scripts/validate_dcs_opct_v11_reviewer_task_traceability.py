from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp/nature_review_studio/review_dcs_opct_v11_bspc_20260914.json"
OUT = ROOT / "outputs/dcs_opct_v11_reviewer_task_traceability_20260916"
TRACE = OUT / "reviewer_task_traceability.json"
MARKDOWN = ROOT / "docs/dcs_opct_v11_reviewer_task_traceability_20260916.md"
REPORT = OUT / "independent_validation_report.json"
MANUSCRIPT = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex"


EXPECTED_STATUS = {
    "T1": "RESOLVED_WITH_BOUNDED_CLAIM",
    "T2": "RESOLVED_WITH_FAIL_CLOSED_RESULT",
    "T3": "RESOLVED_WITH_FORMAL_NONCOMPARABILITY_BOUNDARY",
    "T4": "RESOLVED_AS_MEASUREMENT_ERROR_SENSITIVITY",
    "T5": "PARTIALLY_RESOLVED_PROSPECTIVE_ATTEMPT_STRUCTURALLY_INELIGIBLE",
    "T6": "RESOLVED",
    "T7": "RESOLVED",
    "T8": "READY_PENDING_AUTHOR_DEPOSITION",
    "T9": "RESOLVED_FROM_OFFICIAL_PROJECT_RECORDS",
    "T10": "RESOLVED",
    "T11": "RESOLVED_UNAVAILABLE_TRANSPARENTLY_DISCLOSED",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    payload = read_json(TRACE)
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    markdown = MARKDOWN.read_text(encoding="utf-8")
    tasks = payload["tasks"]
    task_map = {task["id"]: task for task in tasks}
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    expected_ids = [f"T{i}" for i in range(1, 12)]
    record(
        "schema_version",
        payload.get("schema_version") == "dcs-opct-v11-reviewer-task-traceability-v1",
        payload.get("schema_version"),
    )
    record(
        "authoritative_source_hash",
        payload.get("authoritative_review_sha256") == sha256(SOURCE),
        payload.get("authoritative_review_sha256"),
    )
    record("exact_t1_t11_sequence", [task["id"] for task in tasks] == expected_ids, [task["id"] for task in tasks])
    record("task_count_11", payload.get("task_count") == len(tasks) == 11, payload.get("task_count"))
    observed_status = {key: task_map[key]["closure_status"] for key in expected_ids}
    record("closure_statuses_exact", observed_status == EXPECTED_STATUS, observed_status)

    missing_evidence = {}
    for task in tasks:
        absent = [path for path in task["evidence"] if not (ROOT / path).is_file()]
        if absent:
            missing_evidence[task["id"]] = absent
    record("all_evidence_files_exist", not missing_evidence, missing_evidence)

    missing_anchors = {}
    for task in tasks:
        absent = [anchor for anchor in task["manuscript_anchors"] if anchor not in manuscript]
        if absent:
            missing_anchors[task["id"]] = absent
    record("all_manuscript_anchors_present", not missing_anchors, missing_anchors)

    component_reports = {
        "inductive": ROOT / "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/independent_validation_report.json",
        "raw_group": ROOT / "outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914/independent_validation_report.json",
        "raw_partition": ROOT / "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/independent_validation_report.json",
        "probabilistic_event": ROOT / "outputs/dcs_opct_v11_probabilistic_event_risk_20260914/independent_validation_report.json",
        "target_endpoint": ROOT / "outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914/independent_validation_report.json",
        "nearest_neighbor": ROOT / "outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/independent_validation_report.json",
        "reviewer_closure": ROOT / "outputs/dcs_opct_v11_reviewer_closure_20260914/validation_report.json",
        "ekmed_confirmatory_failure": ROOT / "outputs/ekmed_v11_external_confirmation/independent_validation_report.json",
    }
    failed_components = {
        name: read_json(path).get("status")
        for name, path in component_reports.items()
        if read_json(path).get("status") != "passed"
    }
    record("component_validators_pass", not failed_components, failed_components)

    raw_partition = read_json(component_reports["raw_partition"])
    reviewer_closure = read_json(component_reports["reviewer_closure"])
    record(
        "raw_partition_21_of_21",
        raw_partition.get("checks_passed") == raw_partition.get("checks_total") == 21,
        [raw_partition.get("checks_passed"), raw_partition.get("checks_total")],
    )
    record(
        "reviewer_closure_14_of_14",
        reviewer_closure.get("checks_passed") == reviewer_closure.get("checks_total") == 14,
        [reviewer_closure.get("checks_passed"), reviewer_closure.get("checks_total")],
    )

    record(
        "submission_blockers_exact",
        payload.get("submission_blockers") == ["T8"],
        payload.get("submission_blockers"),
    )
    record(
        "author_record_item_exact",
        payload.get("author_record_item") == [],
        payload.get("author_record_item"),
    )
    record(
        "strong_claim_blocker_exact",
        payload.get("strong_claim_scientific_blocker") == ["T5"],
        payload.get("strong_claim_scientific_blocker"),
    )

    boundary = payload.get("bounded_submission_position", "").lower()
    required_boundaries = (
        "retrospective selective configuration-level audit-risk calibration",
        "do not establish positive fully training-data-independent calibration",
        "prospective external effectiveness",
        "universal safety",
        "direct improvement of trial-level eeg emotion predictions",
    )
    record("bounded_claim_complete", all(term in boundary for term in required_boundaries), required_boundaries)

    prohibited = (
        "universally safe transfer",
        "externally validated safety",
        "positive fully training-data-independent release achieved",
    )
    combined = (json.dumps(payload, ensure_ascii=False) + "\n" + markdown).lower()
    record("no_overclaim_in_traceability", not any(term in combined for term in prohibited), prohibited)
    record("markdown_has_all_tasks", all(f"### {task_id}:" in markdown for task_id in expected_ids), expected_ids)

    passed = sum(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "traceability_sha256": sha256(TRACE),
        "markdown_sha256": sha256(MARKDOWN),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "submission_blockers": payload.get("submission_blockers"),
        "author_record_item": payload.get("author_record_item"),
        "strong_claim_scientific_blocker": payload.get("strong_claim_scientific_blocker"),
        "bounded_submission_position": payload.get("bounded_submission_position"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"Reviewer-task traceability validation failed: {failed}")
    print(f"Reviewer-task traceability validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
