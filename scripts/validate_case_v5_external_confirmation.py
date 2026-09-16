from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_minimal_intervention_transport_v5 import RULE, rank_minimal_interventions  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    certify_checkpoint,
    probability_methods,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CASE frozen v5 confirmation.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/case_v5_external_confirmation"))
    parser.add_argument("--registration", type=Path, default=Path("docs/case_v5_external_confirmation_registration.json"))
    parser.add_argument("--registration-amendment", type=Path, default=Path("docs/case_v5_external_confirmation_registration_amendment_003.json"))
    parser.add_argument("--freeze", type=Path, default=Path("docs/risk_controlled_transport_v5_freeze.json"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(checks, name, passed, observed, required) -> None:
    checks.append({"check": name, "passed": bool(passed), "observed": observed, "required": required})


def main() -> None:
    args = parse_args()
    output = args.output_root
    registration = json.loads(args.registration.read_text("utf-8"))
    amendment = json.loads(args.registration_amendment.read_text("utf-8"))
    freeze = json.loads(args.freeze.read_text("utf-8"))
    manifest = json.loads((output / "external_confirmation_manifest.json").read_text("utf-8"))
    lock = json.loads((output / "primary_action_lock.json").read_text("utf-8"))
    certificate_payload = json.loads((output / "primary_candidate_certificate.json").read_text("utf-8"))
    gate = json.loads((output / "external_confirmation_gate.json").read_text("utf-8"))
    signatures = pd.read_csv(output / "candidate_signatures.csv")
    rankings = pd.read_csv(output / "candidate_distortion_rankings.csv")
    mechanism = pd.read_csv(output / "mechanism_probabilities_blinded.csv")
    assignments = pd.read_csv(output / "primary_audit_cluster_order.csv")
    audit = pd.read_csv(output / "primary_audit_labeled.csv")
    heldout = pd.read_csv(output / "primary_heldout_results.csv")
    checks = []

    record(
        checks,
        "registration, amendment chain, and frozen v5 hashes match",
        manifest["registration_sha256"] == sha256(args.registration)
        and manifest["registration_amendment_sha256"] == amendment["parent_amendment_sha256"]
        and manifest["freeze_sha256"] == sha256(args.freeze)
        and registration["v5_freeze"]["sha256"] == sha256(args.freeze),
        {
            "registration": manifest["registration_sha256"],
            "outcome_amendment": manifest["registration_amendment_sha256"],
            "validation_amendment": sha256(args.registration_amendment),
            "freeze": manifest["freeze_sha256"],
        },
        "exact registration/freeze hashes and an amendment parent matching the outcome manifest",
    )
    code_ok = all(
        sha256(ROOT / relative) == expected
        for relative, expected in amendment["analysis_code_sha256"].items()
    )
    record(checks, "registered CASE analysis code is unchanged", code_ok, code_ok, "true")
    lock_files_ok = (
        lock["candidate_signatures_sha256"] == sha256(output / "candidate_signatures.csv")
        and lock["mechanism_probabilities_sha256"] == sha256(output / "mechanism_probabilities_blinded.csv")
        and manifest["action_lock_sha256"] == sha256(output / "primary_action_lock.json")
    )
    record(checks, "pre-outcome action-lock artifacts retain exact hashes", lock_files_ok, lock_files_ok, "true")

    config = RiskControlledConfig(**freeze["config"])
    rebuilt, candidate, reason = rank_minimal_interventions(
        signatures, probability_methods(mechanism), config, RULE
    )
    ranking_ok = (
        rebuilt.sort_values("method").distortion_score.round(12).tolist()
        == rankings.sort_values("method").distortion_score.round(12).tolist()
    )
    record(
        checks,
        "unlabeled CASE action reconstructs exactly",
        ranking_ok and candidate == lock["candidate_method"] and reason == lock["action_lock_reason"],
        {"candidate": candidate, "reason": reason},
        {"candidate": lock["candidate_method"], "reason": lock["action_lock_reason"]},
    )
    cluster_columns = list(config.cluster_columns)
    audit_assignments = assignments.loc[assignments.partition.eq("audit")]
    heldout_assignments = assignments.loc[assignments.partition.eq("heldout")]
    audit_keys = set(audit_assignments[cluster_columns].itertuples(index=False, name=None))
    heldout_keys = set(heldout_assignments[cluster_columns].itertuples(index=False, name=None))
    partition_ok = (
        len(mechanism) == 320
        and len(audit) == 160
        and len(heldout) == 160
        and len(audit_assignments) == 40
        and len(heldout_assignments) == 40
        and audit_keys.isdisjoint(heldout_keys)
    )
    record(
        checks,
        "primary CASE audit and held-out partitions satisfy the frozen contract",
        partition_ok,
        {
            "mechanism_rows": len(mechanism), "audit_rows": len(audit),
            "heldout_rows": len(heldout), "audit_clusters": len(audit_assignments),
            "heldout_clusters": len(heldout_assignments), "disjoint": audit_keys.isdisjoint(heldout_keys),
        },
        "320 total, 160/160 rows, 40/40 disjoint clusters",
    )

    if candidate == "identity":
        rebuilt_certificate = None
        selected = "identity"
    else:
        rebuilt_certificate = certify_checkpoint(
            audit,
            [candidate],
            config,
            int(registration["one_shot_v5_contract"]["primary_audit_seed"])
            + RULE.configuration_budget * 1009,
            total_registered_candidates=1,
        ).iloc[0]
        selected = candidate if bool(rebuilt_certificate.certified) else "identity"
    stored_certificate = certificate_payload["certificate"]
    certificate_ok = (
        (rebuilt_certificate is None and stored_certificate is None)
        or (
            rebuilt_certificate is not None
            and stored_certificate is not None
            and bool(rebuilt_certificate.certified) == bool(stored_certificate["certified"])
            and abs(float(rebuilt_certificate.brier_gain_lcb) - float(stored_certificate["brier_gain_lcb"])) < 1e-12
            and abs(float(rebuilt_certificate.auc_delta_lcb) - float(stored_certificate["auc_delta_lcb"])) < 1e-12
        )
    )
    record(
        checks,
        "primary target-label certificate reconstructs exactly",
        certificate_ok and selected == certificate_payload["selected_method"],
        {"selected": selected, "certificate_present": rebuilt_certificate is not None},
        {"selected": certificate_payload["selected_method"], "certificate_present": stored_certificate is not None},
    )
    metrics = evaluate_action(heldout, selected)
    metric_ok = all(
        metrics[key] == gate["heldout_metrics"][key]
        if isinstance(metrics[key], (bool, str))
        else abs(float(metrics[key]) - float(gate["heldout_metrics"][key])) < 1e-12
        for key in metrics
    )
    record(checks, "held-out metrics reconstruct exactly", metric_ok, metrics, gate["heldout_metrics"])
    certificate_pass = bool(rebuilt_certificate is not None and rebuilt_certificate.certified)
    effectiveness_pass = bool(selected != "identity" and metrics["brier_gain"] > 0.0)
    safety_pass = bool(
        (selected == "identity" or certificate_pass)
        and not metrics["negative_transfer"]
        and metrics["auc_noninferior"]
    )
    claim_supported = bool(certificate_pass and effectiveness_pass and safety_pass)
    gate_ok = (
        certificate_pass == gate["certificate_pass"]
        and effectiveness_pass == gate["effectiveness_pass"]
        and safety_pass == gate["safety_pass"]
        and claim_supported == gate["claim_supported"]
        and claim_supported == manifest["claim_supported"]
    )
    record(
        checks,
        "claim gate follows the registered conjunction",
        gate_ok,
        {
            "certificate_pass": certificate_pass,
            "effectiveness_pass": effectiveness_pass,
            "safety_pass": safety_pass,
            "claim_supported": claim_supported,
        },
        "exact match to stored gate and manifest",
    )

    passed = all(item["passed"] for item in checks)
    report = {
        "status": "passed" if passed else "failed",
        "date": "2026-09-07",
        "n_checks": len(checks),
        "n_passed": sum(item["passed"] for item in checks),
        "checks": checks,
        "claim_supported": claim_supported if passed else False,
        "interpretation": gate["interpretation"] if passed else "artifact validation failed",
    }
    report_path = output / "independent_validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite CASE validation report: {report_path}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
