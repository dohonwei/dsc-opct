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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report = json.loads((OUT / "scientific_claim_readiness.json").read_text(encoding="utf-8"))
    mechanism = json.loads((ROOT / "outputs/dcs_opct_v11_mechanism_falsification_20260910/independent_validation_report.json").read_text(encoding="utf-8"))
    paired = pd.read_csv(ROOT / "outputs/dcs_opct_v11_mechanism_falsification_20260910/dataset_level_paired_inference.csv").set_index("comparison_model")
    contrasts = pd.read_csv(ROOT / "outputs/dcs_opct_v11_mechanism_falsification_20260910/cluster_bootstrap_quartile_contrasts.csv").set_index("feature")
    actions = pd.read_csv(ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/six_dataset_action_outcome.csv")
    higher = pd.read_csv(ROOT / "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/higher_level_audit_yield.csv")
    requirements = {item["requirement"]: item["status"] for item in report["requirements"]}

    checks = {
        "freeze_unchanged": sha256(FREEZE) == EXPECTED_FREEZE,
        "mechanism_validation_15_of_15": mechanism.get("status") == "passed" and mechanism.get("checks_passed") == mechanism.get("checks_total") == 15,
        "high_q2_true": report["claim_readiness"]["high_q2_mechanism_and_risk_controlled_release"] is True,
        "q1_strong_false": report["claim_readiness"]["q1_effective_safe_cross_domain_transfer"] is False,
        "prospective_release_failure_visible": requirements.get("Prospective external effective release") == "FAIL",
        "additive_vs_encoding_exact_result": np.isclose(paired.loc["additive_mechanism", "exact_p_auc_gain"], 0.03125) and paired.loc["additive_mechanism", "domains_auc_improved"] == 6,
        "opportunity_gradient_only": contrasts.loc["encoding_margin", "bootstrap_ci_low"] < 0 < contrasts.loc["encoding_margin", "bootstrap_ci_high"] and contrasts.loc["opportunity_delta", "bootstrap_ci_low"] > 0 and contrasts.loc["metadata_prior_opportunity", "bootstrap_ci_low"] > 0,
        "three_positive_development_releases": set(actions.loc[actions.outcome.eq("certified adaptation"), "dataset"]) == {"EPPVR", "CASE", "CEAP"} and (actions.loc[actions.outcome.eq("certified adaptation"), "heldout_brier_gain"] > 0).all(),
        "identity_is_non_intervention": actions.loc[actions.selected_action.eq("identity"), "outcome"].str.contains("non-intervention").all(),
        "higher_level_failure_visible": np.isclose(higher.exact_random_p_event_row_sensitivity.iloc[0], 0.2074074074) and np.isclose(higher.exact_random_p_material_excess.iloc[0], 0.2597530864),
        "forbidden_claims_complete": all(token in report["forbidden_claims"] for token in ["universally safe cross-domain transfer", "interaction-term superiority", "identity fallback as effectiveness or safety success"]),
        "scope_excludes_layout": "manuscript layout" in report["scope"],
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    validation = {
        "status": "passed" if all(checks.values()) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "checks": checks,
        "conclusion": (
            "The current evidence supports a bounded high-level Q2 mechanism-and-risk-controlled-release submission, "
            "but not a Q1 claim of prospectively effective and universally safe cross-domain transfer."
        ),
    }
    (OUT / "independent_validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if validation["status"] != "passed":
        raise RuntimeError([name for name, passed in checks.items() if not passed])
    print(f"Scientific readiness validation passed: {validation['checks_passed']}/{validation['checks_total']}")


if __name__ == "__main__":
    main()
