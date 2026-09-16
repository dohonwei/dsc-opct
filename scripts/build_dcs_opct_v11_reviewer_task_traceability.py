from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVIEW = ROOT / "tmp/nature_review_studio/review_dcs_opct_v11_bspc_20260914.json"
OUT = ROOT / "outputs/dcs_opct_v11_reviewer_task_traceability_20260916"
JSON_OUT = OUT / "reviewer_task_traceability.json"
MD_OUT = ROOT / "docs/dcs_opct_v11_reviewer_task_traceability_20260916.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


TASK_DETAILS = {
    "T1": {
        "closure_status": "RESOLVED_WITH_BOUNDED_CLAIM",
        "resolution": (
            "Added an inductive target-transform sensitivity with seed-only assignment and audit-side "
            "transform fitting. EPPVR and CASE retained positive held-out gains; CEAP returned original."
        ),
        "evidence": [
            "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/analysis_report.json",
            "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/independent_validation_report.json",
            "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/inductive_vs_transductive.csv",
        ],
        "manuscript_anchors": [
            "We added a stricter configuration-level inductive sensitivity.",
            "The covariate-unseen sensitivity separated results that required transductive target-batch access",
        ],
        "remaining_boundary": (
            "This removes held-out-covariate access but not all underlying training-data dependence; the "
            "stronger raw-partition-first EPPVR analysis is tracked under T2."
        ),
    },
    "T2": {
        "closure_status": "RESOLVED_WITH_FAIL_CLOSED_RESULT",
        "resolution": (
            "Added raw-group endpoint reconstruction and a 15/15 EPPVR raw-partition-first rebuild. The "
            "latter used disjoint participants, raw rows, training rows, identity probes, and fitted emotion "
            "models, requiring 9,600 emotion-model fits and 160 identity-probe fits."
        ),
        "evidence": [
            "outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914/analysis_report.json",
            "outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914/independent_validation_report.json",
            "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/analysis_manifest.json",
            "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/independent_validation_report.json",
            "docs/elsarticle/figures/fig_v11_independence_ladder.pdf",
        ],
        "manuscript_anchors": [
            "We therefore added a second, nested raw-group sensitivity.",
            "Finally, we performed a raw-partition-first EPPVR sensitivity.",
            "No participant, global raw row, training row, or fitted target-domain emotion model crossed the population boundary.",
        ],
        "remaining_boundary": (
            "The fully disjoint analysis returned original at the applicability gate. It verifies fail-closed "
            "execution, not positive training-data-independent calibration."
        ),
    },
    "T3": {
        "closure_status": "RESOLVED_WITH_FORMAL_NONCOMPARABILITY_BOUNDARY",
        "resolution": (
            "Implemented matched-resource CPCS-style calibration and native ATC-style target-accuracy "
            "estimation. TransCal was not forced into an invalid comparison because no target-adapted task "
            "classifier logits exist in the frozen configuration-risk pipeline."
        ),
        "evidence": [
            "outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/analysis_report.json",
            "outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/independent_validation_report.json",
            "outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/cpcs_style_certified_outcomes.csv",
            "outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/atc_style_accuracy_estimation.csv",
        ],
        "manuscript_anchors": [
            "The configuration-level CPCS-style candidate used source labels and audit-side unlabeled target geometry",
            "The ATC-style estimator produced absolute target-accuracy errors",
            "TransCal was not instantiated because its primary formulation calibrates logits from a target-adapted task classifier",
        ],
        "remaining_boundary": (
            "The result is a concrete matched-contract comparison, not superiority over every CPCS or "
            "TransCal implementation."
        ),
    },
    "T4": {
        "closure_status": "RESOLVED_AS_MEASUREMENT_ERROR_SENSITIVITY",
        "resolution": (
            "Propagated endpoint measurement error with paired crossed bootstrap resampling and soft event "
            "probabilities, then repeated source-risk and target-action sensitivity analyses."
        ),
        "evidence": [
            "outputs/dcs_opct_v11_probabilistic_event_risk_20260914/analysis_report.json",
            "outputs/dcs_opct_v11_probabilistic_event_risk_20260914/independent_validation_report.json",
            "outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914/analysis_report.json",
            "outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914/independent_validation_report.json",
        ],
        "manuscript_anchors": [
            "We separately propagated endpoint measurement error by resampling test observations with crossed participant and physical-stimulus multinomial weights",
            "The frozen event is therefore retained for protocol continuity, not treated as a precisely observed biological state.",
        ],
        "remaining_boundary": (
            "The sensitivity covers test-sample uncertainty, not repeated model fitting or all shared "
            "training-data uncertainty; the frozen binary endpoint was not re-optimized."
        ),
    },
    "T5": {
        "closure_status": "PARTIALLY_RESOLVED_PROSPECTIVE_ATTEMPT_STRUCTURALLY_INELIGIBLE",
        "resolution": (
            "Narrowed every central claim to retrospective selective calibration within evaluated development "
            "support and reported the failed seven-domain gate. Completed a checksum-verified, schema-bound, "
            "implementation-locked EKM-ED prospective attempt without post-access retuning. The frozen "
            "key-moment endpoint yielded zero eligible participants in both tasks and stopped before model "
            "fitting. AMIGOS and Emognition remain reserved, and no compatible dataset has produced a positive "
            "non-identity external release."
        ),
        "evidence": [
            "outputs/dcs_opct_v11_reviewer_closure_20260914/validation_report.json",
            "outputs/dcs_opct_v11_release_readiness/report.json",
            "outputs/amigos_v11_preaccess/validation_report.json",
            "outputs/emognition_v11_preaccess/validation_report.json",
            "docs/ekmed_v11_external_confirmation_implementation_lock.json",
            "outputs/ekmed_v11_external_confirmation/failure_001_structural_ineligibility.json",
            "outputs/ekmed_v11_external_confirmation/independent_validation_report.json",
            "docs/dcs_opct_v11_claim_evidence_matrix.md",
        ],
        "manuscript_anchors": [
            "A post-hoc seven-domain refit retained positive mechanism directions and improved pooled probability error, but it failed its prespecified leave-dataset-out discrimination and operating-point gate.",
            "The EKM-ED attempt failed one stage earlier under a stronger prospective information boundary.",
            "EKM-ED met the prospective reservation, checksum, header-only schema, implementation-lock, and no-retuning requirements, but failed the frozen trial-completeness and participant-eligibility gate before model fitting.",
            "They do not establish a positive fully training-data-independent release, universal transport safety, successful prospective external effectiveness",
        ],
        "remaining_boundary": (
            "Scientific blocker for any prospective external-effectiveness or universal-safety claim. It does "
            "not prevent submission of the explicitly bounded retrospective paper."
        ),
    },
    "T6": {
        "closure_status": "RESOLVED",
        "resolution": (
            "Revised the framework figure and caption to distinguish source, target, audit, held-out, "
            "transductive, inductive, raw-group, and raw-partition-first information boundaries."
        ),
        "evidence": [
            "docs/elsarticle/figures/fig_v11_three_stage_framework.pdf",
            "docs/elsarticle/figures/fig_v11_independence_ladder.pdf",
            "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
        ],
        "manuscript_anchors": [
            "Three-stage DCS-OPCT evidence chain and information-access boundaries.",
            "The primary analysis is transductive because the complete unlabeled target batch may enter transformation fitting",
        ],
        "remaining_boundary": "None beyond the claim limits already shown in the figure caption.",
    },
    "T7": {
        "closure_status": "RESOLVED",
        "resolution": (
            "Reframed Table 1 as a neutral resource and decision-level matrix reporting each method family's "
            "prediction object, target information, assumptions, and present-study implementation status."
        ),
        "evidence": [
            "docs/dcs_opct_v11_closest_method_capability_matrix.md",
            "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
            "outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/analysis_report.json",
        ],
        "manuscript_anchors": [
            "Neutral resource and decision-level positioning of DCS-OPCT relative to neighboring method families.",
            "Rows describe primary formulations rather than every possible extension",
        ],
        "remaining_boundary": "The table is positioning evidence, not an empirical superiority table.",
    },
    "T8": {
        "closure_status": "RESOLVED_PUBLIC_REPOSITORY_VERIFIED",
        "resolution": (
            "Published the checksum-verified v22 reproducibility package, source code, aggregate outputs, and "
            "validation reports in a public author-identifiable GitHub repository. Independently verified "
            "public visibility, the remote commit, the SHA-256 sidecar, and the uploaded archive hash."
        ),
        "evidence": [
            "docs/dcs_opct_v11_public_repository_receipt_20260916.json",
            "docs/elsarticle/dcs_opct_v11_author_confirmation_checklist.md",
            "outputs/dcs_opct_v11_submission_artifacts_crossfit_v22_zip_verification.json",
            "scripts/finalize_dcs_opct_repository_link.py",
        ],
        "manuscript_anchors": [
            "A checksum-verified reproducibility package, source code, aggregate outputs, and validation reports are publicly available",
        ],
        "remaining_boundary": (
            "The GitHub repository is author-identifiable and is therefore suitable only when BSPC permits "
            "non-anonymous repository disclosure or after identity masking is no longer required."
        ),
    },
    "T9": {
        "closure_status": "RESOLVED_FROM_OFFICIAL_PROJECT_RECORDS",
        "resolution": (
            "Official project plan and approval records identify National Natural Science Foundation of China "
            "project No. 62172081; the same EPPVR study also acknowledges that project. A bounded Funding "
            "section was added without importing unrelated grants from the prior paper."
        ),
        "evidence": [
            "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
            "docs/dcs_opct_v11_eppvr_metadata_provenance_20260916.md",
            "docs/elsarticle/dcs_opct_v11_author_confirmation_checklist.md",
        ],
        "manuscript_anchors": [
            "This work was supported by the National Natural Science Foundation of China (No. 62172081).",
        ],
        "remaining_boundary": (
            "Only grant No. 62172081 is included; two Guangxi grants from the prior paper were excluded because "
            "their contribution to the present study was not established."
        ),
    },
    "T10": {
        "closure_status": "RESOLVED",
        "resolution": (
            "Standardized outcome-held-out and transductive/inductive terminology, compressed comparator "
            "detail in the abstract, moved repeated diagnostics to the supplement, and aligned manuscript, "
            "highlights, and cover letter claim boundaries."
        ),
        "evidence": [
            "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
            "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex",
            "docs/elsarticle/dcs_opct_v11_bspc_highlights.txt",
            "docs/elsarticle/dcs_opct_v11_bspc_cover_letter.md",
            "outputs/dcs_opct_v11_reviewer_closure_20260914/validation_report.json",
        ],
        "manuscript_anchors": [
            "Positive transport evidence remains retrospective and conditional on the evaluated target batch",
            "The study does not establish prospective external effectiveness",
        ],
        "remaining_boundary": "Further compression is editorial rather than a missing analysis.",
    },
    "T11": {
        "closure_status": "RESOLVED_UNAVAILABLE_TRANSPARENTLY_DISCLOSED",
        "resolution": (
            "Reported the verified collective composition of channels 5--9 and explicitly declined to guess "
            "their within-block index order. These channels are not used in the present analysis."
        ),
        "evidence": [
            "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
            "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex",
            "docs/elsarticle/dcs_opct_v11_author_confirmation_checklist.md",
        ],
        "manuscript_anchors": [
            "Channels 5--9 collectively comprise two EOG channels, one PPG channel, one GSR channel, and one temperature channel;",
            "their exact within-block order could not be recovered from the inspected acquisition and analysis records",
            "Neither channels 5--9 nor dominance enters the present analysis.",
        ],
        "remaining_boundary": (
            "The exact within-block order is unavailable in the inspected acquisition and analysis records. "
            "This is explicitly disclosed and does not affect the executed FP1/FP2 analysis."
        ),
    },
}


def build_markdown(payload: dict) -> str:
    lines = [
        "# DCS-OPCT v11 Reviewer Task Traceability",
        "",
        "Date: 2026-09-16",
        "",
        "Authoritative source: `tmp/nature_review_studio/review_dcs_opct_v11_bspc_20260914.json`.",
        "This file distinguishes implemented closure, bounded partial resolution, author-dependent submission",
        "items, and the remaining external scientific blocker. A fail-closed return-original action is never",
        "counted as positive calibration effectiveness.",
        "",
        "| ID | Source concern | Closure status | Resolution and remaining boundary |",
        "|---|---|---|---|",
    ]
    for task in payload["tasks"]:
        summary = f'{task["resolution"]} Boundary: {task["remaining_boundary"]}'
        summary = summary.replace("|", "\\|")
        concern = task["source_concern"].replace("|", "\\|")
        lines.append(f'| {task["id"]} | {concern} | `{task["closure_status"]}` | {summary} |')

    lines.extend(["", "## Evidence map", ""])
    for task in payload["tasks"]:
        lines.extend(
            [
                f'### {task["id"]}: {task["closure_status"]}',
                "",
                f'**Reviewer strategy:** {task["source_strategy"]}',
                "",
                "**Evidence files:**",
                "",
                *[f"- `{path}`" for path in task["evidence"]],
                "",
                "**Manuscript anchors:**",
                "",
                *[f"- `{anchor}`" for anchor in task["manuscript_anchors"]],
                "",
            ]
        )

    lines.extend(
        [
            "## Submission decision boundary",
            "",
            "- T8 is resolved by the verified public author-identifiable GitHub repository; T9 is resolved from official project records.",
            "- T11 is resolved by explicitly recording that the exact within-block order is unavailable; the",
            "  collective composition is disclosed and the channels are outside the executed analysis.",
            "- T5 now includes a completed negative prospective EKM-ED endpoint-transport test, but remains the",
            "  decisive scientific blocker for a positive prospective external-effectiveness claim. The bounded",
            "  BSPC manuscript can be submitted without claiming that result.",
            "- No task closure upgrades identity fallback into effectiveness, universal safety, or future-domain",
            "  non-harm.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    source = json.loads(SOURCE_REVIEW.read_text(encoding="utf-8"))
    table = source["revision_task_table"]
    headers = table["headers"]
    rows = [dict(zip(headers, row)) for row in table["rows"]]
    ids = [row["ID"] for row in rows]
    if ids != [f"T{i}" for i in range(1, 12)]:
        raise RuntimeError(f"Unexpected reviewer task IDs: {ids}")

    tasks = []
    for row in rows:
        task_id = row["ID"]
        tasks.append(
            {
                "id": task_id,
                "reviewer": row["Reviewer"],
                "source_concern": row["Concern"],
                "source_strategy": row["Strategy"],
                "source_status": row["Status"],
                "source_blocks_response": row["Blocks response?"],
                **TASK_DETAILS[task_id],
            }
        )

    payload = {
        "schema_version": "dcs-opct-v11-reviewer-task-traceability-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": "2026-09-16",
        "authoritative_review_source": SOURCE_REVIEW.relative_to(ROOT).as_posix(),
        "authoritative_review_sha256": sha256(SOURCE_REVIEW),
        "task_count": len(tasks),
        "tasks": tasks,
        "submission_blockers": [],
        "author_record_item": [],
        "strong_claim_scientific_blocker": ["T5"],
        "bounded_submission_position": (
            "The implemented analyses support submission as a retrospective selective configuration-level "
            "audit-risk calibration study within evaluated development support. They do not establish positive "
            "fully training-data-independent calibration, prospective external effectiveness, universal safety, "
            "future-domain non-harm, or direct improvement of trial-level EEG emotion predictions."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    MD_OUT.write_text(build_markdown(payload), encoding="utf-8")
    print(f"Traceability JSON: {JSON_OUT}")
    print(f"Traceability Markdown: {MD_OUT}")


if __name__ == "__main__":
    main()
