from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_scientific_claim_readiness_20260910"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
MECH = ROOT / "outputs/dcs_opct_v11_mechanism_falsification_20260910"
PACKAGE = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15"
SIMPLIFIED = ROOT / "outputs/dcs_opct_v11_end_to_end_simplified_comparators_20260910_v2"
HIGHER = ROOT / "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910"
EEG = ROOT / "outputs/eegemotions27_v11_external_robustness"
NOVELTY = ROOT / "docs/dcs_opct_v11_closest_method_capability_matrix.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_passed(path: Path, total: int) -> bool:
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("status") == "passed" and report.get("checks_passed") == report.get("checks_total") == total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mechanism_report = json.loads((MECH / "mechanism_falsification_report.json").read_text(encoding="utf-8"))
    mechanism_validation = validation_passed(MECH / "independent_validation_report.json", 15)
    package_validation = validation_passed(PACKAGE / "independent_validation_report.json", 21)
    simplified_validation = validation_passed(SIMPLIFIED / "independent_validation_report.json", 13)
    higher_validation = validation_passed(HIGHER / "independent_validation_report.json", 13)
    eeg_validation = json.loads((EEG / "independent_validation_report.json").read_text(encoding="utf-8"))
    eeg_boundary = (
        eeg_validation.get("status") == "passed"
        and eeg_validation.get("n_passed") == eeg_validation.get("n_checks") == 12
    )

    paired = pd.read_csv(MECH / "dataset_level_paired_inference.csv").set_index("comparison_model")
    contrasts = pd.read_csv(MECH / "cluster_bootstrap_quartile_contrasts.csv").set_index("feature")
    actions = pd.read_csv(PACKAGE / "six_dataset_action_outcome.csv")
    budget = pd.read_csv(PACKAGE / "strong_calibration_decision_curve_summary.csv")
    dcs_budget = budget.loc[budget.method.eq("dcs_selective")]
    higher = pd.read_csv(HIGHER / "higher_level_audit_yield.csv")

    opportunity_mechanism = bool(
        paired.loc["additive_mechanism", "exact_p_auc_gain"] <= 0.05
        and paired.loc["additive_mechanism", "domains_auc_improved"] >= 6
        and contrasts.loc["encoding_margin", "bootstrap_ci_low"] < 0 < contrasts.loc["encoding_margin", "bootstrap_ci_high"]
        and contrasts.loc["opportunity_delta", "bootstrap_ci_low"] > 0
        and contrasts.loc["metadata_prior_opportunity", "bootstrap_ci_low"] > 0
    )
    bounded_release = bool(
        actions.loc[actions.outcome.eq("certified adaptation")].dataset.tolist() == ["EPPVR", "CASE", "CEAP"]
        and (actions.loc[actions.outcome.eq("certified adaptation"), "heldout_brier_gain"] > 0).all()
        and actions.loc[actions.selected_action.eq("identity"), "outcome"].str.contains("non-intervention").all()
    )
    retrospective_nonharm = bool(dcs_budget.material_negative_transfer_frequency.eq(0).all())
    higher_level_failure_preserved = bool(
        higher_validation
        and np.isclose(higher.exact_random_p_event_row_sensitivity.iloc[0], 0.207, atol=0.001)
        and np.isclose(higher.exact_random_p_material_excess.iloc[0], 0.260, atol=0.001)
    )
    novelty_matrix_complete = NOVELTY.is_file() and all(
        token in NOVELTY.read_text(encoding="utf-8")
        for token in ["Counterfactual identity-utilization endpoint", "Limited-label release certificate", "Domain-level return-original fallback"]
    )
    prospective_effective_release = False

    high_q2_ready = all([
        sha256(FREEZE) == EXPECTED_FREEZE,
        package_validation,
        mechanism_validation,
        opportunity_mechanism,
        bounded_release,
        retrospective_nonharm,
        simplified_validation,
        higher_level_failure_preserved,
        eeg_boundary,
        novelty_matrix_complete,
    ])
    q1_strong_ready = high_q2_ready and prospective_effective_release

    requirements = [
        ["Frozen v11 integrity", sha256(FREEZE) == EXPECTED_FREEZE, "All thresholds, actions, and failures remain frozen."],
        ["Validated evidence package", package_validation, "The v15 package passes 21/21 independent checks."],
        ["Mechanism separation", opportunity_mechanism, "Opportunity gradients are positive while the encoding gradient crosses zero; additive versus encoding-only LODO AUROC exact p=0.03125."],
        ["Bounded retrospective release", bounded_release, "EPPVR, CASE, and CEAP have positive held-out Brier gains; unsupported domains retain identity."],
        ["Retrospective released-action non-harm", retrospective_nonharm, "No material negative transfer occurred for DCS-OPCT in the matched-budget repetitions."],
        ["Component capability evidence", simplified_validation, "Unlabeled gates reduced labels reaching certification by one third while preserving the retrospective release set."],
        ["Higher-level failure retained", higher_level_failure_preserved, "Whole-block exact p-values remain approximately 0.207 and 0.260; workflow-level utility is not claimed."],
        ["External abstention boundary", eeg_boundary, "EEGEmotions-27 remains a separate pre-signal external robustness test with identity non-intervention."],
        ["Closest-method capability distinction", novelty_matrix_complete, "Novelty is stated as an end-to-end capability difference, not component invention."],
        ["Prospective external effective release", prospective_effective_release, "No reserved compatible external dataset has produced a certified non-identity release."],
    ]
    report = {
        "status": "high_q2_scientific_basis_ready" if high_q2_ready else "not_ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "innovation and result sufficiency only; manuscript layout and author metadata excluded",
        "claim_readiness": {
            "q1_effective_safe_cross_domain_transfer": q1_strong_ready,
            "high_q2_mechanism_and_risk_controlled_release": high_q2_ready,
        },
        "requirements": [
            {"requirement": name, "status": "PASS" if passed else "FAIL", "interpretation": interpretation}
            for name, passed, interpretation in requirements
        ],
        "recommended_primary_claim": (
            "Identity decodability alone is insufficient evidence of material shortcut utilization. "
            "Across seven retrospective domains, opportunity-conditioned mechanism variables improved cross-domain "
            "risk discrimination, and DCS-OPCT provided bounded, risk-controlled calibration release in three "
            "development domains while returning non-intervention outside its applicability support."
        ),
        "forbidden_claims": [
            "universally safe cross-domain transfer",
            "prospectively validated external effectiveness",
            "interaction-term superiority",
            "workflow-level audit utility across tasks and seeds",
            "identity fallback as effectiveness or safety success",
        ],
        "journal_level_interpretation": (
            "The innovation and results are sufficient for a high-level Q2 engineering or biomedical-signal methodology "
            "submission under the bounded mechanism-plus-release claim. They are not sufficient for a Q1 paper whose "
            "headline claim is effective and safe cross-domain transfer."
        ),
        "decisive_upgrade_for_q1": (
            "A prospectively reserved compatible external dataset must release a non-identity action and pass every "
            "frozen effectiveness and non-harm criterion without changing v11."
        ),
        "mechanism_composite_flag_preserved": mechanism_report["mechanism_falsification_supported"],
    }
    (OUT / "scientific_claim_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# DCS-OPCT v11 scientific claim readiness", "", f"Generated: {report['generated_at_utc']}", "",
        "## Verdict", "",
        f"- Q1 effective-safe transfer: **{'READY' if q1_strong_ready else 'NOT READY'}**",
        f"- High-level Q2 mechanism and risk-controlled release: **{'READY' if high_q2_ready else 'NOT READY'}**", "",
        "## Evidence audit", "", "| Requirement | Status | Interpretation |", "|---|---|---|",
    ]
    lines.extend(f"| {item['requirement']} | {item['status']} | {item['interpretation']} |" for item in report["requirements"])
    lines.extend(["", "## Recommended primary claim", "", report["recommended_primary_claim"], "", "## Q1 upgrade", "", report["decisive_upgrade_for_q1"]])
    (OUT / "scientific_claim_readiness.md").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    print(json.dumps(report["claim_readiness"], indent=2))


if __name__ == "__main__":
    main()
