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

from develop_minimal_intervention_transport_v5 import (  # noqa: E402
    POLICY_PATH,
    PROTOCOL_PATH,
    PUBLIC_PATH,
    RULE,
    rank_minimal_interventions,
)
from develop_risk_controlled_transport_v5 import load_datasets  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    probability_methods,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate minimal-intervention v5 artifacts.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/minimal_intervention_transport_v5_development"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(checks, name, passed, observed, required) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "required": required,
        }
    )


def main() -> None:
    output = parse_args().output_root
    manifest = json.loads((output / "development_manifest.json").read_text("utf-8"))
    actions = pd.read_csv(output / "unlabeled_action_locks.csv")
    rankings = pd.read_csv(output / "candidate_distortion_rankings.csv")
    certificates = pd.read_csv(output / "fixed_horizon_certificates.csv")
    assignments = pd.read_csv(output / "audit_cluster_assignments.csv")
    results = pd.read_csv(output / "repeated_audit_results.csv")
    summary = pd.read_csv(output / "rule_summary.csv")
    checks = []

    script_path = ROOT / "scripts/develop_minimal_intervention_transport_v5.py"
    input_paths = {
        "public_results": ROOT / PUBLIC_PATH,
        "policy": ROOT / POLICY_PATH,
        "protocol": ROOT / PROTOCOL_PATH,
        "eppvr_predictions": ROOT / "outputs/eppvr_transport_policy_retrospective/eppvr_policy_predictions.csv",
        "eppvr_signatures": ROOT / "outputs/eppvr_transport_policy_retrospective/unlabeled_candidate_signatures.csv",
        "seediv_predictions": ROOT / "outputs/seediv_transport_policy_external/seediv_policy_predictions.csv",
        "seediv_signatures": ROOT / "outputs/seediv_transport_policy_external/unlabeled_candidate_signatures.csv",
        "dreamer_mechanisms": ROOT / "outputs/dreamer_v4_external_confirmation/mechanism_probabilities_blinded.csv",
        "dreamer_signatures": ROOT / "outputs/dreamer_v4_external_confirmation/candidate_signatures.csv",
        "dreamer_outcomes": ROOT / "outputs/dreamer_counterfactual_external/summary.csv",
    }
    observed_hashes = {name: sha256(path) for name, path in input_paths.items()}
    record(
        checks,
        "script hash matches manifest",
        manifest["script_sha256"] == sha256(script_path),
        manifest["script_sha256"],
        sha256(script_path),
    )
    record(
        checks,
        "protocol and input hashes match manifest",
        manifest["input_sha256"] == observed_hashes,
        manifest["input_sha256"],
        "all current SHA-256 hashes exactly match",
    )

    repetitions = int(manifest["audit_repetitions"])
    expected_seeds = [
        int(manifest["seed"]) + index * int(manifest["seed_step"])
        for index in range(repetitions)
    ]
    seed_ok = all(
        group.sort_values("audit_repetition").audit_seed.tolist() == expected_seeds
        for _, group in results.groupby("dataset")
    )
    record(
        checks,
        "audit seed schedule is exact",
        seed_ok,
        sorted(results.audit_seed.unique().tolist())[:3]
        + sorted(results.audit_seed.unique().tolist())[-3:],
        "20260907 + repetition * 7919 for every dataset",
    )
    expected_rows = 3 * repetitions
    grid_ok = (
        len(results) == expected_rows
        and len(certificates) == expected_rows
        and len(actions) == 3
    )
    record(
        checks,
        "result grids are complete",
        grid_ok,
        {"results": len(results), "certificates": len(certificates), "actions": len(actions)},
        f"{expected_rows}, {expected_rows}, and 3 rows",
    )

    public = pd.read_csv(ROOT / PUBLIC_PATH)
    policy = joblib.load(ROOT / POLICY_PATH)
    datasets = load_datasets(public, policy)
    config = RiskControlledConfig(**manifest["config"])
    reconstructed = []
    ranking_ok = True
    for dataset, (frame, signatures) in datasets.items():
        rebuilt, candidate, reason = rank_minimal_interventions(
            signatures, probability_methods(frame), config, RULE
        )
        stored = rankings.loc[rankings.dataset.eq(dataset)].sort_values("method")
        rebuilt = rebuilt.sort_values("method")
        ranking_ok &= stored.method.tolist() == rebuilt.method.tolist()
        ranking_ok &= (
            stored.distortion_score.round(12).tolist()
            == rebuilt.distortion_score.round(12).tolist()
        )
        reconstructed.append(
            {"dataset": dataset, "candidate_method": candidate, "action_lock_reason": reason}
        )
    rebuilt_actions = pd.DataFrame(reconstructed).sort_values("dataset").reset_index(drop=True)
    stored_actions = actions[
        ["dataset", "candidate_method", "action_lock_reason"]
    ].sort_values("dataset").reset_index(drop=True)
    record(
        checks,
        "unlabeled locks and rankings reconstruct exactly",
        ranking_ok and stored_actions.equals(rebuilt_actions),
        stored_actions.to_dict(orient="records"),
        rebuilt_actions.to_dict(orient="records"),
    )

    cluster_columns = list(config.cluster_columns)
    cluster_count_ok = True
    separation_ok = True
    heldout_count_ok = True
    for (dataset, repetition), group in assignments.groupby(
        ["dataset", "audit_repetition"], sort=False
    ):
        audit = group.loc[group.partition.eq("audit")]
        heldout = group.loc[group.partition.eq("heldout")]
        cluster_count_ok &= len(audit) == 40
        cluster_count_ok &= audit.cluster_order_position.tolist() == list(range(40))
        audit_keys = set(audit[cluster_columns].itertuples(index=False, name=None))
        heldout_keys = set(heldout[cluster_columns].itertuples(index=False, name=None))
        separation_ok &= audit_keys.isdisjoint(heldout_keys)
        expected_heldout = len(datasets[str(dataset)][0]) - 160
        observed = results.loc[
            results.dataset.eq(dataset) & results.audit_repetition.eq(repetition),
            "n_heldout_configurations",
        ].iloc[0]
        heldout_count_ok &= int(observed) == expected_heldout
    record(
        checks,
        "every audit has 40 complete clusters and 160 configurations",
        cluster_count_ok and certificates.n_audit_configurations.eq(160).all(),
        certificates.n_audit_configurations.unique().tolist(),
        "40 clusters and 160 configurations in every audit",
    )
    record(
        checks,
        "audit and held-out data are cluster-disjoint",
        separation_ok and heldout_count_ok,
        {"cluster_separation": separation_ok, "heldout_counts": heldout_count_ok},
        "no overlap and exact remaining row counts",
    )

    deployed = certificates.loc[certificates.selected_method.ne("identity")]
    certificate_ok = (
        deployed.certified.eq(True).all()
        and deployed.brier_certified.eq(True).all()
        and deployed.auc_noninferior.eq(True).all()
        and deployed.brier_gain_lcb.gt(RULE.minimum_certified_gain).all()
        and deployed.auc_delta_lcb.ge(RULE.auc_noninferiority_margin).all()
    )
    record(
        checks,
        "every deployment satisfies both certificates",
        certificate_ok,
        {
            "n_deployed": len(deployed),
            "minimum_brier_lcb": float(deployed.brier_gain_lcb.min()) if len(deployed) else None,
            "minimum_auc_lcb": float(deployed.auc_delta_lcb.min()) if len(deployed) else None,
        },
        "Brier LCB > 0.001 and AUROC LCB >= -0.02",
    )
    merged = results.merge(
        certificates[["dataset", "audit_repetition", "audit_seed", "selected_method"]],
        on=["dataset", "audit_repetition", "audit_seed"],
        suffixes=("_result", "_certificate"),
        validate="one_to_one",
    )
    action_match = merged.selected_method_result.eq(merged.selected_method_certificate).all()
    record(
        checks,
        "held-out evaluation uses the certified action",
        action_match,
        int(merged.selected_method_result.eq(merged.selected_method_certificate).sum()),
        str(len(merged)),
    )

    by_dataset = summary.set_index("dataset")
    eppvr = by_dataset.loc["EPPVR"]
    eppvr_pass = (
        eppvr.adaptation_coverage >= 0.50
        and eppvr.negative_transfer_frequency == 0.0
        and eppvr.minimum_brier_gain_when_adapted > 0.005
    )
    record(
        checks,
        "EPPVR effectiveness and safety gates pass",
        eppvr_pass,
        {
            "coverage": float(eppvr.adaptation_coverage),
            "negative_transfer_frequency": float(eppvr.negative_transfer_frequency),
            "minimum_adapted_gain": float(eppvr.minimum_brier_gain_when_adapted),
        },
        "coverage >= 0.50, zero negative transfer, minimum adapted gain > 0.005",
    )
    seediv = by_dataset.loc["SEED-IV"]
    seediv_harmful = results.loc[
        results.dataset.eq("SEED-IV") & results.adapted & results.brier_gain.lt(0.0)
    ]
    record(
        checks,
        "SEED-IV releases no harmful action",
        seediv.material_negative_transfer_frequency == 0.0 and seediv_harmful.empty,
        {
            "material_negative_transfer_frequency": float(seediv.material_negative_transfer_frequency),
            "harmful_deployments": len(seediv_harmful),
        },
        "zero material negative transfer and zero harmful deployments",
    )
    dreamer = by_dataset.loc["DREAMER"]
    record(
        checks,
        "DREAMER has zero material negative transfer",
        dreamer.material_negative_transfer_frequency == 0.0,
        float(dreamer.material_negative_transfer_frequency),
        "0",
    )

    passed = all(item["passed"] for item in checks)
    report = {
        "status": "passed" if passed else "failed",
        "date": "2026-09-07",
        "n_checks": len(checks),
        "n_passed": sum(item["passed"] for item in checks),
        "checks": checks,
        "development_acceptance_passed": bool(passed),
        "claim_status": (
            "candidate_can_be_frozen_for_untouched_external_confirmation"
            if passed
            else "candidate_not_eligible_for_freeze"
        ),
        "claim_boundary": (
            "Retrospective development cannot establish effective and safe cross-domain "
            "transport. A frozen one-shot test on a genuinely untouched dataset is required."
        ),
    }
    report_path = output / "validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
