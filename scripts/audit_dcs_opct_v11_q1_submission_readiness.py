from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from detect_dcs_opct_v11_external_arrival import (
    REPORT_JSON as ARRIVAL_REPORT_JSON,
    REPORT_MD as ARRIVAL_REPORT_MD,
    build_report as build_arrival_report,
    write_idempotent as write_arrival_report,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_q1_submission_readiness"
REPORT_JSON = OUT / "readiness_report.json"
REPORT_MD = OUT / "readiness_report.md"

EXPECTED_FREEZE_SHA256 = (
    "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def status_row(
    requirement: str,
    status: str,
    evidence: list[str],
    interpretation: str,
) -> dict:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "interpretation": interpretation,
    }


def write_idempotent(path: Path, payload: dict) -> None:
    if path.is_file():
        previous = read_json(path)
        old = {key: value for key, value in previous.items() if key != "generated_at_utc"}
        new = {key: value for key, value in payload.items() if key != "generated_at_utc"}
        if old == new:
            payload["generated_at_utc"] = previous.get("generated_at_utc")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    freeze = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
    package_report_path = (
        ROOT
        / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15"
        / "independent_validation_report.json"
    )
    retrospective_gate_path = (
        ROOT
        / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15"
        / "v12_analysis_retrospective_robustness_gate.json"
    )
    eegemotions_report_path = (
        ROOT
        / "outputs/eegemotions27_v11_external_robustness"
        / "independent_validation_report.json"
    )
    amigos_stack_path = ROOT / "outputs/amigos_v11_preaccess/stack_validation_report.json"
    emognition_stack_path = (
        ROOT / "outputs/emognition_v11_preaccess/stack_validation_report.json"
    )
    manuscript_path = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex"
    cluster_report_path = (
        ROOT
        / "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910"
        / "independent_validation_report.json"
    )
    aggregation_report_path = (
        ROOT
        / "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910"
        / "independent_validation_report.json"
    )
    higher_level_report_path = (
        ROOT
        / "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910"
        / "independent_validation_report.json"
    )
    preview_report_path = (
        ROOT
        / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4"
        / "independent_validation_report.json"
    )
    preview_manifest_path = preview_report_path.parent / "preview_manifest.json"

    required_files = [
        freeze,
        package_report_path,
        retrospective_gate_path,
        eegemotions_report_path,
        amigos_stack_path,
        emognition_stack_path,
        manuscript_path,
        cluster_report_path,
        aggregation_report_path,
        higher_level_report_path,
        preview_report_path,
        preview_manifest_path,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing readiness evidence: {missing}")

    package_report = read_json(package_report_path)
    retrospective_gate = read_json(retrospective_gate_path)
    eegemotions_report = read_json(eegemotions_report_path)
    amigos_stack = read_json(amigos_stack_path)
    emognition_stack = read_json(emognition_stack_path)
    cluster_report = read_json(cluster_report_path)
    aggregation_report = read_json(aggregation_report_path)
    higher_level_report = read_json(higher_level_report_path)
    preview_report = read_json(preview_report_path)
    preview_manifest = read_json(preview_manifest_path)
    manuscript = manuscript_path.read_text(encoding="utf-8")

    freeze_current = sha256(freeze) == EXPECTED_FREEZE_SHA256
    package_current = (
        package_report.get("status") == "passed"
        and package_report.get("checks_passed") == package_report.get("checks_total") == 21
    )
    eegemotions_valid = (
        eegemotions_report.get("status") == "passed"
        and eegemotions_report.get("n_passed")
        == eegemotions_report.get("n_checks")
        == 12
        and all(check.get("passed") is True for check in eegemotions_report["checks"])
        and eegemotions_report.get("claim_supported") is False
    )
    amigos_ready = (
        amigos_stack.get("status") == "passed"
        and amigos_stack.get("checks", {}).get("cuda_available") is True
        and len(amigos_stack.get("tests", [])) == 13
    )
    emognition_ready = (
        emognition_stack.get("status") == "passed"
        and emognition_stack.get("checks", {}).get("cuda_available") is True
        and len(emognition_stack.get("tests", [])) == 15
    )

    cluster_evidence_valid = (
        cluster_report.get("status") == "passed"
        and cluster_report.get("checks_passed")
        == cluster_report.get("checks_total")
        == 16
    )
    aggregation_evidence_valid = (
        aggregation_report.get("status") == "passed"
        and aggregation_report.get("checks_passed")
        == aggregation_report.get("checks_total")
        == 8
    )
    higher_level_evidence_valid = (
        higher_level_report.get("status") == "passed"
        and higher_level_report.get("checks_passed")
        == higher_level_report.get("checks_total")
        == 13
    )
    preview_valid = (
        preview_report.get("status") == "passed"
        and preview_report.get("checks_passed")
        == preview_report.get("checks_total")
        == 11
        and preview_manifest.get("status")
        == "staged_preview_not_merged_into_authoritative_manuscript"
        and preview_manifest.get("authoritative_sources_unchanged") is True
    )

    amigos_root = Path(r"E:\AA发表论文的数据\dataset\AMIGOS")
    emognition_archive = Path(
        r"E:\AA发表论文的数据\dataset\Emognition\study_data.zip"
    )
    amigos_acquired = amigos_root.is_dir() and any(amigos_root.iterdir())
    emognition_acquired = emognition_archive.is_file()
    arrival_report = build_arrival_report(amigos_root, emognition_archive)
    write_arrival_report(arrival_report, ARRIVAL_REPORT_JSON, ARRIVAL_REPORT_MD)
    arrival_detector_valid = (
        arrival_report.get("automatic_execution") is False
        and arrival_report.get("participant_values_accessed") is False
        and arrival_report.get("freeze_sha256") == EXPECTED_FREEZE_SHA256
        and len(arrival_report.get("datasets", [])) == 2
        and all(
            item.get("participant_values_accessed") is False
            for item in arrival_report.get("datasets", [])
        )
    )

    eegemotions_integrated = all(
        token in manuscript
        for token in ["EEGEmotions", "category-derived polarity", "pre-signal"]
    )
    credit_complete = (
        "Author contributions will be finalized" not in manuscript
        and "CRediT authorship contribution statement" in manuscript
    )

    requirements = [
        status_row(
            "Frozen method integrity",
            "PASS" if freeze_current else "FAIL",
            [str(freeze.relative_to(ROOT)), EXPECTED_FREEZE_SHA256],
            "The v11 method and all registered decision thresholds remain unchanged.",
        ),
        status_row(
            "Latest packaged evidence integrity",
            "PASS" if package_current else "FAIL",
            [str(package_report_path.relative_to(ROOT))],
            "The v15 package passes 21/21 independent checks; this is artifact integrity, not external effectiveness.",
        ),
        status_row(
            "Retrospective seven-domain robustness",
            "FAIL" if retrospective_gate.get("status") == "failed" else "PASS",
            [str(retrospective_gate_path.relative_to(ROOT))],
            "The prespecified retrospective gate failed and must remain visible.",
        ),
        status_row(
            "Separate EEGEmotions-27 robustness boundary",
            "PASS_BOUNDARY_ONLY" if eegemotions_valid else "FAIL",
            [str(eegemotions_report_path.relative_to(ROOT))],
            "The external pipeline is internally valid but returned identity; it supports abstention-boundary evidence only.",
        ),
        status_row(
            "Complete-cluster dependence-aware audit-yield evidence",
            "PASS_POSTHOC" if cluster_evidence_valid else "FAIL",
            [str(cluster_report_path.relative_to(ROOT))],
            "The 16/16 validated post-hoc analysis uses complete clusters, dependence-aware intervals, and exact stratified inference; it is not prospective utility evidence.",
        ),
        status_row(
            "Cluster-score aggregation sensitivity",
            "PASS_POSTHOC" if aggregation_evidence_valid else "FAIL",
            [str(aggregation_report_path.relative_to(ROOT))],
            "Maximum, mean, and median scores selected the same clusters; the 8/8 result supports retrospective aggregation robustness only.",
        ),
        status_row(
            "Higher-level dataset-task-seed sensitivity",
            "PASS_POSTHOC_BOUNDARY" if higher_level_evidence_valid else "FAIL",
            [str(higher_level_report_path.relative_to(ROOT))],
            "The 13/13 validated whole-block analysis retains the non-significant exact result; configuration-level audit yield cannot be promoted to workflow-level utility.",
        ),
        status_row(
            "Evidence-integration preview v4",
            "PASS_STAGED_NOT_AUTHORITATIVE" if preview_valid else "FAIL",
            [str(preview_report_path.relative_to(ROOT))],
            "The 37-page manuscript and 31-page supplement previews pass 11/11 checks and preserve the non-significant higher-level result while authoritative TEX remains unchanged.",
        ),
        status_row(
            "EEGEmotions-27 manuscript integration",
            "PASS" if eegemotions_integrated else "PENDING_AUTHOR_APPROVAL",
            [str(manuscript_path.relative_to(ROOT))],
            "The reviewed figures and tables are staged, but the result has not yet been inserted into the manuscript.",
        ),
        status_row(
            "AMIGOS pre-access GPU readiness",
            "PASS" if amigos_ready else "FAIL",
            [str(amigos_stack_path.relative_to(ROOT))],
            "The locked implementation is ready; no participant-level result exists.",
        ),
        status_row(
            "AMIGOS authorized archive acquisition",
            "PASS" if amigos_acquired else "BLOCKED_EXTERNAL_ACCESS",
            [str(amigos_root)],
            "The official archive is not present locally, so the registered one-shot test cannot start.",
        ),
        status_row(
            "Emognition pre-access GPU readiness",
            "PASS" if emognition_ready else "FAIL",
            [str(emognition_stack_path.relative_to(ROOT))],
            "The locked implementation is ready; no participant-level result exists.",
        ),
        status_row(
            "Emognition authorized archive acquisition",
            "PASS" if emognition_acquired else "BLOCKED_EXTERNAL_ACCESS",
            [str(emognition_archive)],
            "The EULA-controlled archive is not present locally, so the registered one-shot replication cannot start.",
        ),
        status_row(
            "Fail-closed external-arrival detector",
            "PASS" if arrival_detector_valid else "FAIL",
            [str(ARRIVAL_REPORT_JSON.relative_to(ROOT))],
            "The detector checks availability metadata only, never auto-runs analysis, and emits a schema-only or locked CUDA command only when its prerequisites exist.",
        ),
        status_row(
            "Prospective external effective release",
            "NOT_SUPPORTED",
            [
                "docs/v11_prospective_external_family_registry.json",
                str(eegemotions_report_path.relative_to(ROOT)),
            ],
            "No prospectively reserved compatible dataset has released a non-identity action and passed every frozen effectiveness and non-harm criterion.",
        ),
        status_row(
            "CRediT authorship statement",
            "PASS" if credit_complete else "PENDING_AUTHOR_APPROVAL",
            [str(manuscript_path.relative_to(ROOT))],
            "The placeholder requires agreement from all authors before submission.",
        ),
    ]

    strong_claim_ready = all(
        row["status"] == "PASS"
        for row in requirements
        if row["requirement"]
        in {
            "Frozen method integrity",
            "Latest packaged evidence integrity",
            "AMIGOS authorized archive acquisition",
            "Emognition authorized archive acquisition",
            "Prospective external effective release",
        }
    )
    bounded_submission_ready = (
        freeze_current
        and package_current
        and eegemotions_valid
        and cluster_evidence_valid
        and aggregation_evidence_valid
        and higher_level_evidence_valid
        and preview_valid
        and arrival_detector_valid
        and eegemotions_integrated
        and credit_complete
    )

    report = {
        "status": "ready" if strong_claim_ready else "not_ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "submission-evidence readiness without changing frozen v11",
        "freeze_sha256": sha256(freeze),
        "requirements": requirements,
        "claim_readiness": {
            "q1_strong_effective_safe_transfer": strong_claim_ready,
            "bounded_high_q2_selective_calibration_submission": bounded_submission_ready,
        },
        "next_actions": [
            "Obtain explicit author approval before integrating the reviewed EEGEmotions-27 figures and tables.",
            "Finalize and approve the CRediT contribution statement.",
            "Obtain AMIGOS only through the verified official or author-authorized route.",
            "Complete the Emognition EULA and obtain the official study_data.zip archive.",
            "Run every acquired family member once under its locked implementation and report all outcomes, including ineligibility or abstention.",
            "Keep the complete-cluster and aggregation analyses explicitly post-hoc in every manuscript claim.",
            "Keep the non-significant dataset-task-seed whole-block result visible and do not claim workflow-level audit utility.",
        ],
        "claim_boundary": (
            "The current evidence supports retrospective selective audit-risk calibration and externally tested abstention boundaries. "
            "It does not support prospective external effectiveness, replicated effectiveness, universal safety, or transferable operating points."
        ),
    }

    lines = [
        "# DCS-OPCT v11 Submission Readiness Audit",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Verdict",
        "",
        f"- Q1 strong effective-and-safe transfer claim: **{'READY' if strong_claim_ready else 'NOT READY'}**",
        f"- Bounded high-Q2 selective-calibration submission: **{'READY' if bounded_submission_ready else 'NOT READY'}**",
        "",
        "## Requirement Audit",
        "",
        "| Requirement | Status | Interpretation |",
        "|---|---|---|",
    ]
    for row in requirements:
        lines.append(
            f"| {row['requirement']} | {row['status']} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            *[f"{index}. {action}" for index, action in enumerate(report["next_actions"], 1)],
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    write_idempotent(REPORT_JSON, report)
    lines[2] = f"Generated: {report['generated_at_utc']}"
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"Q1 strong-claim readiness: {strong_claim_ready}")
    print(f"Bounded high-Q2 readiness: {bounded_submission_ready}")
    print(f"Report: {REPORT_JSON}")


if __name__ == "__main__":
    main()
