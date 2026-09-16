from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    applicability_table,
    attach_conformal_ood,
    paired_point_metrics,
    probability_methods,
    rows_for_clusters,
    run_registered_checkpoints,
)
from identity_shortcut.transport_policy import committee_distribution  # noqa: E402


DATASETS = {
    "SEED-IV": {
        "predictions": Path(
            "outputs/seediv_transport_policy_external/seediv_policy_predictions.csv"
        ),
        "signatures": Path(
            "outputs/seediv_transport_policy_external/unlabeled_candidate_signatures.csv"
        ),
        "v3_decision": Path(
            "outputs/seediv_transport_policy_external/frozen_policy_decision.csv"
        ),
    },
    "EPPVR": {
        "predictions": Path(
            "outputs/eppvr_transport_policy_retrospective/eppvr_policy_predictions.csv"
        ),
        "signatures": Path(
            "outputs/eppvr_transport_policy_retrospective/unlabeled_candidate_signatures.csv"
        ),
        "v3_decision": Path(
            "outputs/eppvr_transport_policy_retrospective/frozen_policy_decision.csv"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop and stress-test the risk-controlled selective transport v4 policy."
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path("outputs/transport_policy_public_development"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/risk_controlled_transport_v4_development"),
    )
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, probability))


def evaluate_action(frame: pd.DataFrame, method: str) -> dict[str, float | bool]:
    labels = frame.material_optimism_event.to_numpy(int)
    identity = frame.probability_identity.to_numpy(float)
    selected = frame[f"probability_{method}"].to_numpy(float)
    identity_brier = float(np.mean((labels - identity) ** 2))
    selected_brier = float(np.mean((labels - selected) ** 2))
    identity_auc = safe_auc(labels, identity)
    selected_auc = safe_auc(labels, selected)
    gain = identity_brier - selected_brier
    auc_delta = (
        selected_auc - identity_auc
        if np.isfinite(identity_auc) and np.isfinite(selected_auc)
        else np.nan
    )
    return {
        "selected_method": method,
        "adapted": method != "identity",
        "brier_gain": gain,
        "auc_delta": auc_delta,
        "negative_transfer": bool(method != "identity" and gain < 0.0),
        "auc_noninferior": bool(np.isfinite(auc_delta) and auc_delta >= -0.02),
    }


def select_naive(audit: pd.DataFrame, methods: list[str]) -> str:
    if not methods:
        return "identity"
    metrics = paired_point_metrics(audit, methods)
    eligible = metrics.loc[
        metrics.brier_gain.gt(0.0) & metrics.auc_delta.ge(-0.02)
    ]
    if eligible.empty:
        return "identity"
    return str(
        eligible.sort_values(["brier_gain", "method"], ascending=[False, True]).iloc[0].method
    )


def select_oracle(test: pd.DataFrame, methods: list[str]) -> str:
    metrics = paired_point_metrics(test, methods)
    best = metrics.sort_values(["brier_gain", "method"], ascending=[False, True]).iloc[0]
    return str(best.method) if float(best.brier_gain) > 0.0 else "identity"


def enrich_signatures(
    raw_signatures: pd.DataFrame,
    development: pd.DataFrame,
    policy_artifact: dict[str, object],
) -> pd.DataFrame:
    signatures = attach_conformal_ood(development, raw_signatures)
    signatures["predicted_brier_gain"] = 0.0
    signatures["predicted_auc_delta"] = 0.0
    candidate_mask = signatures.method.ne("identity")
    candidates = signatures.loc[candidate_mask]
    if len(candidates):
        gain = committee_distribution(policy_artifact["gain_committee"], candidates)
        auc = committee_distribution(policy_artifact["auc_committee"], candidates)
        signatures.loc[candidate_mask, "predicted_brier_gain"] = gain.mean(axis=1)
        signatures.loc[candidate_mask, "predicted_auc_delta"] = auc.mean(axis=1)
    return signatures


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in results.groupby(
        ["dataset", "configuration_budget", "strategy"], sort=False
    ):
        adapted = group.loc[group.adapted]
        rows.append(
            {
                "dataset": keys[0],
                "configuration_budget": keys[1],
                "strategy": keys[2],
                "audit_repetitions": len(group),
                "adaptation_coverage": float(group.adapted.mean()),
                "negative_transfer_rate": float(group.negative_transfer.mean()),
                "negative_transfer_rate_when_adapted": (
                    float(adapted.negative_transfer.mean()) if len(adapted) else 0.0
                ),
                "mean_brier_gain_when_adapted": (
                    float(adapted.brier_gain.mean()) if len(adapted) else 0.0
                ),
                "minimum_brier_gain_when_adapted": (
                    float(adapted.brier_gain.min()) if len(adapted) else 0.0
                ),
                "mean_brier_gain": float(group.brier_gain.mean()),
                "median_brier_gain": float(group.brier_gain.median()),
                "split_distribution_q025": float(group.brier_gain.quantile(0.025)),
                "split_distribution_q975": float(group.brier_gain.quantile(0.975)),
                "mean_auc_delta": float(group.auc_delta.mean()),
                "auc_noninferiority_rate": float(group.auc_noninferior.mean()),
                "mean_oracle_regret": float(group.oracle_regret.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 5)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)
    args.output_root.mkdir(parents=True, exist_ok=True)
    public_results_path = args.public_root / "synthetic_candidate_results.csv"
    policy_path = args.public_root / "frozen_transport_policy.joblib"
    public = pd.read_csv(public_results_path)
    policy_artifact = joblib.load(policy_path)
    fixed_method = str(
        public.loc[public.method.ne("identity")]
        .groupby("method")
        .brier_gain.mean()
        .sort_values(ascending=False)
        .index[0]
    )
    config = RiskControlledConfig(
        bootstrap_repetitions=args.bootstrap_repetitions,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )
    result_rows = []
    decision_rows = []
    certificate_rows = []
    applicability_rows = []
    progress = tqdm(
        total=len(DATASETS) * args.audit_repetitions,
        desc="Risk-controlled transport development",
        unit="audit split",
        dynamic_ncols=True,
    )
    for dataset, paths in DATASETS.items():
        frame = pd.read_csv(paths["predictions"])
        raw_signatures = pd.read_csv(paths["signatures"])
        signatures = enrich_signatures(raw_signatures, public, policy_artifact)
        methods = probability_methods(frame)
        applicability = applicability_table(signatures, methods, config)
        applicability.insert(0, "dataset", dataset)
        applicability_rows.append(applicability)
        applicable = applicability.loc[
            applicability.applicable & applicability.method.ne("identity"), "method"
        ].tolist()
        v3_method = str(pd.read_csv(paths["v3_decision"]).selected_method.iloc[0])
        for repetition in range(args.audit_repetitions):
            split_seed = args.seed + repetition * 7919 + sum(map(ord, dataset))
            decisions, certificates, order = run_registered_checkpoints(
                frame, signatures, config, split_seed
            )
            decisions.insert(0, "dataset", dataset)
            decisions.insert(1, "audit_repetition", repetition)
            decisions.insert(2, "audit_seed", split_seed)
            decision_rows.append(decisions)
            if not certificates.empty:
                certificates.insert(0, "dataset", dataset)
                certificates.insert(1, "audit_repetition", repetition)
                certificate_rows.append(certificates)
            for decision in decisions.itertuples(index=False):
                audit_clusters = order[: decision.actual_cluster_budget]
                audit_mask = rows_for_clusters(frame, config.cluster_columns, audit_clusters)
                audit = frame.loc[audit_mask].reset_index(drop=True)
                test = frame.loc[~audit_mask].reset_index(drop=True)
                naive_method = select_naive(audit, applicable)
                oracle_method = select_oracle(test, methods)
                oracle_gain = float(evaluate_action(test, oracle_method)["brier_gain"])
                strategies = {
                    "identity": "identity",
                    "v3_unlabeled_policy": v3_method,
                    "fixed_public_adapter": fixed_method,
                    "naive_target_validation": naive_method,
                    "v4_risk_controlled": decision.selected_method,
                    "full_label_oracle": oracle_method,
                }
                for strategy, method in strategies.items():
                    metrics = evaluate_action(test, method)
                    result_rows.append(
                        {
                            "dataset": dataset,
                            "audit_repetition": repetition,
                            "audit_seed": split_seed,
                            "configuration_budget": decision.configuration_budget,
                            "cluster_budget": decision.cluster_budget,
                            "actual_configuration_budget": decision.actual_configuration_budget,
                            "actual_cluster_budget": decision.actual_cluster_budget,
                            "n_test_configurations": len(test),
                            "n_test_events": int(test.material_optimism_event.sum()),
                            "strategy": strategy,
                            **metrics,
                            "oracle_regret": oracle_gain - float(metrics["brier_gain"]),
                        }
                    )
            progress.update(1)
            progress.set_postfix(dataset=dataset, refresh=False)
    progress.close()
    results = pd.DataFrame(result_rows)
    decisions = pd.concat(decision_rows, ignore_index=True)
    certificates = (
        pd.concat(certificate_rows, ignore_index=True) if certificate_rows else pd.DataFrame()
    )
    applicability = pd.concat(applicability_rows, ignore_index=True)
    summary = summarize(results)
    results.to_csv(args.output_root / "repeated_audit_results.csv", index=False)
    decisions.to_csv(args.output_root / "checkpoint_decisions.csv", index=False)
    certificates.to_csv(args.output_root / "candidate_certificates.csv", index=False)
    applicability.to_csv(args.output_root / "applicability_gates.csv", index=False)
    summary.to_csv(args.output_root / "strategy_summary.csv", index=False)
    manifest = {
        "status": "v4_retrospective_development_complete",
        "claim_boundary": (
            "SEED-IV and EPPVR outcomes are development-only. DREAMER remains required for "
            "untouched one-shot confirmation after v4 thresholds and code are frozen."
        ),
        "config": config.to_dict(),
        "fixed_public_adapter": fixed_method,
        "audit_repetitions": args.audit_repetitions,
        "seed": args.seed,
        "public_results_sha256": sha256(public_results_path),
        "v3_policy_artifact_sha256": sha256(policy_path),
        "v4_code_sha256": sha256(
            ROOT / "src" / "identity_shortcut" / "risk_controlled_transport.py"
        ),
        "development_script_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nApplicability gates:")
    print(applicability.to_string(index=False))
    print("\nStrategy summary:")
    print(summary.to_string(index=False))
    print("\nManifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
