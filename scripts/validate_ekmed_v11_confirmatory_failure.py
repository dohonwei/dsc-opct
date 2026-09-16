from __future__ import annotations

import json
from pathlib import Path

from external_wearable_implementation_lock import sha256, verify_implementation_lock


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "docs/ekmed_v11_external_confirmation_implementation_lock.json"
FAILURE = (
    ROOT
    / "outputs/ekmed_v11_external_confirmation/failure_001_structural_ineligibility.json"
)
MANUSCRIPT = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex"
SUPPLEMENT = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex"
EVIDENCE_TABLE = ROOT / "docs/elsarticle/tables/table_evidence_role_matrix.tex"
CLAIM_MATRIX = ROOT / "docs/dcs_opct_v11_claim_evidence_matrix.md"
OUTPUT = (
    ROOT
    / "outputs/ekmed_v11_external_confirmation/independent_validation_report.json"
)

LOCK_SHA256 = "ec729f31627033819af91368995089780c2825dac3cee3f0ae233d73bf380235"
FAILURE_SHA256 = "8f621780673a87e455594ec46e2bb38ef1538470f355471b46de0f64633cb070"


def main() -> None:
    verify_implementation_lock(LOCK, root=ROOT, expected_sha256=LOCK_SHA256)
    failure = json.loads(FAILURE.read_text(encoding="utf-8"))
    structural = failure["structural_audit"]
    exclusions = structural["exclusions"]
    gate = failure["registered_gate_interpretation"]
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    supplement_semantic = supplement.replace(r"\allowbreak{}", "")
    evidence_table = EVIDENCE_TABLE.read_text(encoding="utf-8")
    claim_matrix = CLAIM_MATRIX.read_text(encoding="utf-8")

    checks = {
        "failure_record_hash_frozen": sha256(FAILURE) == FAILURE_SHA256,
        "failure_status_immutable": failure.get("status") == "immutable_confirmatory_failure",
        "implementation_lock_matches": failure["implementation_lock"]["sha256"] == LOCK_SHA256,
        "archive_sha256_matches": failure["archive"]["sha256"]
        == "0b6b4cc003cf4b94b8af970590aecfd8f05dc98522c2b2ecd92bc523ae5471fa",
        "all_47_participants_attempted": structural["questionnaire_participants_attempted"] == 47,
        "participant_film_accounting_complete": (
            structural["theoretical_participant_film_units"]
            == structural["constructed_trials_before_task_filter"]
            + sum(exclusions.values())
            == 188
        ),
        "both_tasks_zero_eligible": structural["valence"]["eligible_subjects"] == 0
        and structural["arousal"]["eligible_subjects"] == 0,
        "no_preoutcome_lock": gate["preoutcome_lock_created"] is False,
        "no_action_lock": gate["unlabeled_action_created"] is False,
        "no_model_fit": gate["model_fitting_started"] is False,
        "no_outcome_access": gate["material_event_outcomes_accessed"] is False,
        "no_generated_preoutcome_artifact": not (
            ROOT / "outputs/ekmed_v11_external_confirmation/preoutcome_design_lock.json"
        ).exists(),
        "manuscript_reports_86_of_188": "only 86 of 188 participant--film units" in manuscript,
        "manuscript_reports_zero_eligible_boundary": "no participant in either task satisfied" in manuscript,
        "manuscript_blocks_effectiveness_claim": "They do not establish a positive fully training-data-independent release, universal transport safety, successful prospective external effectiveness" in manuscript,
        "supplement_contains_failure_section": "EKM-ED prospective structural endpoint failure" in supplement,
        "supplement_contains_failure_hash": FAILURE_SHA256 in supplement_semantic,
        "evidence_table_labels_structural_test": "Prospective structural test & EKM-ED" in evidence_table,
        "claim_matrix_forbids_safety_upgrade": "Prospective structural failure; no effectiveness result" in claim_matrix,
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not failed else "failed",
        "date": "2026-09-16",
        "dataset": "EmoKey Moments Muse EEG Dataset (EKM-ED)",
        "phase": "post-lock confirmatory structural-failure validation",
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "implementation_lock_sha256": LOCK_SHA256,
        "failure_record_sha256": FAILURE_SHA256,
        "confirmatory_interpretation": "structurally_ineligible_and_non_estimable",
        "claim_boundary": (
            "Validation preserves a prospective endpoint-transport failure. It does not "
            "establish DCS-OPCT effectiveness, ineffectiveness, non-harm, safety, or replication."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed:
        raise RuntimeError(f"EKM-ED failure validation failed: {failed}")
    print(f"EKM-ED confirmatory failure validation passed: {len(checks)}/{len(checks)}")


if __name__ == "__main__":
    main()
