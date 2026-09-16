from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = ROOT / "outputs" / "transport_policy_public_development"
EPPVR_ROOT = ROOT / "outputs" / "eppvr_transport_policy_retrospective"
LOCKED_ROOT = ROOT / "outputs" / "eppvr_frozen_risk_validation"
SEEDIV_ROOT = ROOT / "outputs" / "seediv_transport_policy_external"
SEEDIV_REGISTRATION = ROOT / "docs" / "seediv_external_freeze.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str, checks: list[dict[str, object]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks: list[dict[str, object]] = []
    manifest_path = POLICY_ROOT / "transport_policy_freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = POLICY_ROOT / manifest["policy_artifact"]
    artifact = joblib.load(artifact_path)
    check(manifest["admissible"], "public transport policy passed every frozen gate", checks)
    check(
        sha256(artifact_path) == manifest["policy_artifact_sha256"],
        "frozen policy artifact hash matches manifest",
        checks,
    )
    check(
        sha256(ROOT / "src" / "identity_shortcut" / "transport_policy.py")
        == manifest["policy_code_sha256"],
        "policy implementation hash matches freeze",
        checks,
    )
    check(
        set(artifact["candidate_methods"])
        == {"identity", "mean_shift", "coral", "quantile_mapping", "support_clipping", "scmt"},
        "policy contains all six candidate actions",
        checks,
    )
    results = pd.read_csv(POLICY_ROOT / "synthetic_candidate_results.csv")
    check(
        results[["seed", "shift_family", "shift_dose"]].drop_duplicates().shape[0] == 120,
        "public development contains 120 unique synthetic scenarios",
        checks,
    )
    check(
        results.groupby("scenario_id").method.nunique().eq(6).all(),
        "every scenario contains every candidate action",
        checks,
    )
    decisions = pd.read_csv(POLICY_ROOT / "losfo_policy_decisions.csv")
    check(len(decisions) == decisions.scenario_id.nunique() == 120, "LOSFO has one decision per scenario", checks)
    check(
        set(decisions.outer_held_family) == set(manifest["shift_families"]),
        "every shift family is held out exactly once",
        checks,
    )
    summary = pd.read_csv(POLICY_ROOT / "losfo_policy_summary.csv")
    overall = summary.loc[summary.outer_held_family == "OVERALL"].iloc[0]
    family = summary.loc[summary.outer_held_family != "OVERALL"]
    check(overall.mean_brier_gain > 0, "LOSFO mean Brier gain is positive", checks)
    check(overall.negative_transfer_rate <= 0.20, "LOSFO negative-transfer rate is bounded", checks)
    check(overall.auc_noninferiority_rate >= 0.90, "LOSFO AUROC safety criterion passes", checks)
    check(
        family.mean_brier_gain.min() >= -0.0001 - 1e-12,
        "worst held-out shift family meets Brier tolerance",
        checks,
    )
    check(
        family.auc_noninferiority_rate.min() >= 0.90,
        "every held-out shift family meets AUROC safety criterion",
        checks,
    )
    intervals = pd.read_csv(POLICY_ROOT / "policy_cluster_bootstrap_intervals.csv")
    check(
        intervals.bootstrap_valid_repetitions.eq(5000).all(),
        "all public policy intervals use 5000 valid cluster draws",
        checks,
    )
    check(
        set(intervals.metric)
        >= {"mean_policy_brier_gain", "policy_minus_fixed_gain", "negative_transfer_rate"},
        "bootstrap table contains benefit and safety endpoints",
        checks,
    )
    utility = pd.read_csv(POLICY_ROOT / "decision_utility_curve.csv")
    check(
        set(utility.strategy) == {"policy", "fixed", "identity", "oracle"},
        "decision utility compares policy, fixed, identity, and oracle strategies",
        checks,
    )
    retrospective = json.loads(
        (EPPVR_ROOT / "retrospective_policy_manifest.json").read_text(encoding="utf-8")
    )
    check(
        retrospective["status"] == "post_external_retrospective_policy_evaluation",
        "EPPVR application is explicitly retrospective",
        checks,
    )
    check(
        retrospective["policy_manifest_sha256"] == sha256(manifest_path),
        "EPPVR application references the frozen public policy",
        checks,
    )
    locked_gate = LOCKED_ROOT / "external_validation_gate.json"
    check(
        retrospective["locked_external_gate_unchanged"]
        and retrospective["locked_external_gate_sha256_before"]
        == retrospective["locked_external_gate_sha256_after"]
        == sha256(locked_gate),
        "original locked EPPVR gate remains byte-identical",
        checks,
    )
    decision = pd.read_csv(EPPVR_ROOT / "frozen_policy_decision.csv")
    check(len(decision) == 1, "EPPVR has exactly one domain-level policy action", checks)
    check(
        decision.selected_method.iloc[0] == retrospective["selected_method"],
        "EPPVR decision and manifest agree",
        checks,
    )
    registration = json.loads(SEEDIV_REGISTRATION.read_text(encoding="utf-8"))
    check(
        registration["status"] == "registered_before_seediv_outcome_generation",
        "SEED-IV protocol was registered before outcome generation",
        checks,
    )
    check(
        registration["policy_manifest_sha256"] == sha256(manifest_path),
        "SEED-IV registration references the unchanged policy manifest",
        checks,
    )
    check(
        registration["policy_artifact_sha256"] == sha256(artifact_path),
        "SEED-IV registration references the unchanged policy artifact",
        checks,
    )
    risk_model_path = ROOT / "outputs" / "identity_shortcut_risk_classifier" / "frozen_risk_model.joblib"
    check(
        registration["risk_model_sha256"] == sha256(risk_model_path),
        "SEED-IV registration references the unchanged public risk model",
        checks,
    )
    for task, item in registration["feature_files"].items():
        check(
            sha256(ROOT / item["path"]) == item["sha256"],
            f"registered SEED-IV {task} feature file is byte-identical",
            checks,
        )
    seediv_manifest = json.loads(
        (SEEDIV_ROOT / "external_validation_manifest.json").read_text(encoding="utf-8")
    )
    check(
        seediv_manifest["status"] == "untouched_external_validation_complete",
        "SEED-IV is recorded as the untouched external validation",
        checks,
    )
    check(
        seediv_manifest["registration_sha256"] == sha256(SEEDIV_REGISTRATION),
        "SEED-IV result references the registered protocol",
        checks,
    )
    decision_lock_path = SEEDIV_ROOT / "decision_lock.json"
    check(
        seediv_manifest["decision_lock_sha256"] == sha256(decision_lock_path),
        "SEED-IV pre-outcome action lock is byte-identical",
        checks,
    )
    seediv_decision = pd.read_csv(SEEDIV_ROOT / "frozen_policy_decision.csv")
    seediv_signatures = pd.read_csv(SEEDIV_ROOT / "unlabeled_candidate_signatures.csv")
    check(len(seediv_decision) == 1, "SEED-IV has exactly one frozen domain action", checks)
    check(
        set(seediv_signatures.method) == set(manifest["candidate_methods"]),
        "SEED-IV preserves every registered candidate action signature",
        checks,
    )
    predictions = pd.read_csv(SEEDIV_ROOT / "seediv_policy_predictions.csv")
    check(len(predictions) == 240, "SEED-IV contains all 240 registered configurations", checks)
    check(
        set(predictions.task) == {"arousal", "valence"}
        and set(predictions.representation) == set(registration["representations"])
        and set(predictions.model) == set(registration["models"])
        and set(predictions.split_seed) == set(registration["seeds"])
        and set(predictions.nominal_dose) == {0.25, 0.5, 0.75, 1.0},
        "SEED-IV result contains the complete frozen factorial grid",
        checks,
    )
    seediv_gate = json.loads(
        (SEEDIV_ROOT / "external_validation_gate.json").read_text(encoding="utf-8")
    )
    check(
        isinstance(seediv_gate["safety_pass"], bool)
        and isinstance(seediv_gate["benefit_pass"], bool),
        "SEED-IV reports both prespecified safety and benefit gates",
        checks,
    )
    utility = pd.read_csv(SEEDIV_ROOT / "audit_budget_utility.csv")
    check(
        set(utility.audit_budget_fraction) == {0.05, 0.1, 0.2, 0.3, 0.5},
        "SEED-IV reports frozen-policy utility across five audit budgets",
        checks,
    )
    report = {
        "status": "passed",
        "n_checks": len(checks),
        "checks": checks,
        "scientific_claim_status": (
            "untouched_seediv_safety_and_benefit_passed"
            if seediv_gate["safety_pass"] and seediv_gate["benefit_pass"]
            else "untouched_seediv_external_gate_failed"
        ),
        "required_next_validation": (
            None
            if seediv_gate["safety_pass"] and seediv_gate["benefit_pass"]
            else (
                "Any successor policy must be developed without SEED-IV outcomes and "
                "confirmed on another untouched dataset."
            )
        ),
    }
    (POLICY_ROOT / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
