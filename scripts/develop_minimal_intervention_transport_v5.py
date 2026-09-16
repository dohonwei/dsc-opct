from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import joblib
import pandas as pd
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_risk_controlled_transport_v5 import (  # noqa: E402
    evaluate_action,
    load_datasets,
    summarize,
)
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    applicability_table,
    balanced_cluster_order,
    certify_checkpoint,
    probability_methods,
    rows_for_clusters,
    validate_prediction_contract,
)


@dataclass(frozen=True)
class MinimalInterventionRule:
    name: str = "minimal_intervention_h160_m001"
    configuration_budget: int = 160
    minimum_predicted_gain: float = 0.001
    minimum_certified_gain: float = 0.001
    auc_noninferiority_margin: float = -0.02


RULE = MinimalInterventionRule()
PUBLIC_PATH = Path(
    "outputs/transport_policy_public_development/synthetic_candidate_results.csv"
)
POLICY_PATH = Path(
    "outputs/transport_policy_public_development/frozen_transport_policy.joblib"
)
PROTOCOL_PATH = Path("docs/risk_controlled_transport_v5_development_protocol.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop the fixed-horizon minimal-intervention v5 transport rule."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/minimal_intervention_transport_v5_development"),
    )
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--seed-step", type=int, default=7919)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_minimal_interventions(
    signatures: pd.DataFrame,
    methods: list[str],
    config: RiskControlledConfig,
    rule: MinimalInterventionRule,
) -> tuple[pd.DataFrame, str, str]:
    applicability = applicability_table(signatures, methods, config)
    candidates = signatures.merge(
        applicability[["method", "applicable", "applicability_reason"]],
        on="method",
        how="right",
        validate="one_to_one",
    )
    candidates["distortion_score"] = (
        candidates.probability_mean_shift.abs()
        + (1.0 - candidates.probability_rank)
        + candidates.decision_flip_rate
    )
    candidates["passes_actionability"] = (
        candidates.method.ne("identity")
        & candidates.applicable
        & candidates.predicted_brier_gain.ge(rule.minimum_predicted_gain)
        & candidates.predicted_auc_delta.ge(rule.auc_noninferiority_margin)
    )
    candidates["minimal_intervention_eligible"] = (
        candidates.method.ne("identity")
        & candidates.applicable
        & candidates.predicted_brier_gain.ge(0.0)
        & candidates.predicted_auc_delta.ge(rule.auc_noninferiority_margin)
    )
    candidates["distortion_rank"] = pd.NA
    eligible = candidates.loc[candidates.minimal_intervention_eligible].sort_values(
        ["distortion_score", "predicted_brier_gain", "method"],
        ascending=[True, False, True],
    )
    if len(eligible):
        candidates.loc[eligible.index, "distortion_rank"] = range(1, len(eligible) + 1)
    if not bool(candidates.passes_actionability.any()):
        selected = "identity"
        reason = "abstain_unlabeled_actionability_gate_failed"
    else:
        selected = str(eligible.iloc[0].method)
        reason = "minimal_intervention_locked_before_target_outcomes"
    candidates["locked_candidate"] = candidates.method.eq(selected)
    return candidates, selected, reason


def cluster_assignment_rows(
    dataset: str,
    repetition: int,
    seed: int,
    order: list[tuple[object, ...]],
    config: RiskControlledConfig,
) -> list[dict[str, object]]:
    audit_budget = config.cluster_budgets[0]
    rows = []
    for position, cluster in enumerate(order):
        rows.append(
            {
                "dataset": dataset,
                "rule": RULE.name,
                "audit_repetition": repetition,
                "audit_seed": seed,
                "cluster_order_position": position,
                "partition": "audit" if position < audit_budget else "heldout",
                **dict(zip(config.cluster_columns, cluster, strict=True)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for debugging.")
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 5)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)

    public = pd.read_csv(PUBLIC_PATH)
    policy = joblib.load(POLICY_PATH)
    datasets = load_datasets(public, policy)
    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )

    action_rows: list[dict[str, object]] = []
    ranking_tables: list[pd.DataFrame] = []
    certificate_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    progress = tqdm(
        total=len(datasets) * args.audit_repetitions,
        desc="v5 minimal-intervention certification",
        unit="audit",
        dynamic_ncols=True,
    )

    for dataset, (frame, signatures) in datasets.items():
        validate_prediction_contract(frame, config)
        rankings, candidate, action_reason = rank_minimal_interventions(
            signatures, probability_methods(frame), config, RULE
        )
        rankings.insert(0, "dataset", dataset)
        rankings.insert(1, "rule", RULE.name)
        ranking_tables.append(rankings)
        selected_signature = rankings.loc[rankings.method.eq(candidate)].iloc[0]
        action_rows.append(
            {
                "dataset": dataset,
                "rule": RULE.name,
                "candidate_method": candidate,
                "action_lock_reason": action_reason,
                "predicted_brier_gain": float(selected_signature.predicted_brier_gain),
                "predicted_auc_delta": float(selected_signature.predicted_auc_delta),
                "distortion_score": float(selected_signature.distortion_score),
            }
        )

        for repetition in range(args.audit_repetitions):
            seed = args.seed + repetition * args.seed_step
            order = balanced_cluster_order(frame, config, seed)
            audit_clusters = order[: config.cluster_budgets[0]]
            audit_mask = rows_for_clusters(frame, config.cluster_columns, audit_clusters)
            audit = frame.loc[audit_mask].reset_index(drop=True)
            heldout = frame.loc[~audit_mask].reset_index(drop=True)
            assignment_rows.extend(
                cluster_assignment_rows(dataset, repetition, seed, order, config)
            )

            if candidate == "identity":
                certificate = None
                selected = "identity"
                selection_reason = "abstain_no_unlabeled_candidate"
            else:
                certificate = certify_checkpoint(
                    audit,
                    [candidate],
                    config,
                    seed + RULE.configuration_budget * 1009,
                    total_registered_candidates=1,
                ).iloc[0]
                selected = candidate if bool(certificate.certified) else "identity"
                selection_reason = (
                    "selected_fixed_horizon_minimal_intervention_certificate"
                    if selected != "identity"
                    else f"abstain_confirmation_failed:{certificate.failure_reason}"
                )

            certificate_rows.append(
                {
                    "dataset": dataset,
                    "rule": RULE.name,
                    "audit_repetition": repetition,
                    "audit_seed": seed,
                    "candidate_method": candidate,
                    "selected_method": selected,
                    "selection_reason": selection_reason,
                    "configuration_budget": RULE.configuration_budget,
                    "cluster_budget": config.cluster_budgets[0],
                    "n_audit_configurations": len(audit),
                    "n_heldout_configurations": len(heldout),
                    "brier_gain": None if certificate is None else float(certificate.brier_gain),
                    "brier_gain_lcb": None if certificate is None else float(certificate.brier_gain_lcb),
                    "brier_certified": None if certificate is None else bool(certificate.brier_certified),
                    "auc_delta": None if certificate is None else float(certificate.auc_delta),
                    "auc_delta_lcb": None if certificate is None else float(certificate.auc_delta_lcb),
                    "auc_noninferior": None if certificate is None else bool(certificate.auc_noninferior),
                    "certified": None if certificate is None else bool(certificate.certified),
                    "tail_alpha": None if certificate is None else float(certificate.tail_alpha),
                    "valid_auc_bootstraps": None if certificate is None else int(certificate.valid_auc_bootstraps),
                }
            )
            result_rows.append(
                {
                    "dataset": dataset,
                    "rule": RULE.name,
                    "audit_repetition": repetition,
                    "audit_seed": seed,
                    "candidate_method": candidate,
                    "actual_configuration_budget": len(audit),
                    "n_heldout_configurations": len(heldout),
                    **evaluate_action(heldout, selected),
                }
            )
            progress.update(1)
            progress.set_postfix(dataset=dataset, method=selected, refresh=False)
    progress.close()

    args.output_root.mkdir(parents=True, exist_ok=False)
    actions = pd.DataFrame(action_rows)
    rankings = pd.concat(ranking_tables, ignore_index=True)
    certificates = pd.DataFrame(certificate_rows)
    assignments = pd.DataFrame(assignment_rows)
    results = pd.DataFrame(result_rows)
    summary = summarize(results)
    actions.to_csv(args.output_root / "unlabeled_action_locks.csv", index=False)
    rankings.to_csv(args.output_root / "candidate_distortion_rankings.csv", index=False)
    certificates.to_csv(args.output_root / "fixed_horizon_certificates.csv", index=False)
    assignments.to_csv(args.output_root / "audit_cluster_assignments.csv", index=False)
    results.to_csv(args.output_root / "repeated_audit_results.csv", index=False)
    summary.to_csv(args.output_root / "rule_summary.csv", index=False)

    input_paths = {
        "public_results": PUBLIC_PATH,
        "policy": POLICY_PATH,
        "protocol": PROTOCOL_PATH,
        "eppvr_predictions": Path("outputs/eppvr_transport_policy_retrospective/eppvr_policy_predictions.csv"),
        "eppvr_signatures": Path("outputs/eppvr_transport_policy_retrospective/unlabeled_candidate_signatures.csv"),
        "seediv_predictions": Path("outputs/seediv_transport_policy_external/seediv_policy_predictions.csv"),
        "seediv_signatures": Path("outputs/seediv_transport_policy_external/unlabeled_candidate_signatures.csv"),
        "dreamer_mechanisms": Path("outputs/dreamer_v4_external_confirmation/mechanism_probabilities_blinded.csv"),
        "dreamer_signatures": Path("outputs/dreamer_v4_external_confirmation/candidate_signatures.csv"),
        "dreamer_outcomes": Path("outputs/dreamer_counterfactual_external/summary.csv"),
    }
    manifest = {
        "status": "minimal_intervention_v5_retrospective_development_complete",
        "date": "2026-09-07",
        "claim_status": "development_only_external_confirmation_required",
        "rule": asdict(RULE),
        "config": config.to_dict(),
        "target_label_role": "single_fixed_horizon_confirmation_only",
        "action_selection_role": "unlabeled_actionability_then_minimum_distortion",
        "audit_repetitions": args.audit_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "cuda_verified": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "statistical_backend": "numpy_cpu_cluster_bootstrap",
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
    }
    (args.output_root / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(actions.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
