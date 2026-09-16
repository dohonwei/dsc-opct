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

from develop_risk_controlled_transport_v5 import evaluate_action, load_datasets, summarize  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    applicability_table,
    balanced_cluster_order,
    certify_checkpoint,
    probability_methods,
    rows_for_clusters,
)


@dataclass(frozen=True)
class LabelIndependentRule:
    name: str
    minimum_predicted_gain: float
    minimum_certified_gain: float


RULES = (
    LabelIndependentRule("unlabeled_top1_m0", 0.0, 0.0),
    LabelIndependentRule("unlabeled_top1_m0005", 0.0005, 0.0005),
    LabelIndependentRule("unlabeled_top1_m001", 0.001, 0.001),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop label-independent action selection with target-label certification."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/label_independent_transport_v5_development"),
    )
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
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


def choose_unlabeled_action(
    signatures: pd.DataFrame,
    methods: list[str],
    config: RiskControlledConfig,
    rule: LabelIndependentRule,
) -> tuple[str, str]:
    applicability = applicability_table(signatures, methods, config)
    candidates = signatures.merge(
        applicability[["method", "applicable", "applicability_reason"]],
        on="method",
        validate="one_to_one",
    )
    eligible = candidates.loc[
        candidates.method.ne("identity")
        & candidates.applicable
        & candidates.predicted_brier_gain.gt(rule.minimum_predicted_gain)
        & candidates.predicted_auc_delta.ge(config.auc_noninferiority_margin)
    ]
    if eligible.empty:
        return "identity", "no_applicable_unlabeled_candidate_above_practical_gain"
    selected = eligible.sort_values(
        ["predicted_brier_gain", "predicted_auc_delta", "method"],
        ascending=[False, False, True],
    ).iloc[0]
    return str(selected.method), "unlabeled_top1_locked_before_target_outcomes"


def run_checkpoints(
    frame: pd.DataFrame,
    candidate: str,
    base_config: RiskControlledConfig,
    seed: int,
    rule: LabelIndependentRule,
) -> tuple[pd.DataFrame, list[tuple[object, ...]]]:
    order = balanced_cluster_order(frame, base_config, seed)
    locked_method: str | None = None
    locked_cluster_budget: int | None = None
    rows: list[dict[str, object]] = []
    certification_config = RiskControlledConfig(
        configuration_budgets=base_config.configuration_budgets,
        configurations_per_cluster=base_config.configurations_per_cluster,
        familywise_alpha=base_config.familywise_alpha,
        bootstrap_repetitions=base_config.bootstrap_repetitions,
        auc_noninferiority_margin=base_config.auc_noninferiority_margin,
        minimum_brier_gain=rule.minimum_certified_gain,
        minimum_valid_auc_bootstraps=base_config.minimum_valid_auc_bootstraps,
    )
    for configuration_budget, cluster_budget in zip(
        base_config.configuration_budgets,
        base_config.cluster_budgets,
        strict=True,
    ):
        if locked_method is not None:
            rows.append(
                {
                    "configuration_budget": configuration_budget,
                    "cluster_budget": cluster_budget,
                    "actual_cluster_budget": locked_cluster_budget,
                    "candidate_method": candidate,
                    "selected_method": locked_method,
                    "selection_reason": "carried_forward_label_independent_certificate",
                    "brier_gain": None,
                    "brier_gain_lcb": None,
                    "auc_delta": None,
                    "auc_delta_lcb": None,
                    "valid_auc_bootstraps": None,
                }
            )
            continue
        if candidate == "identity":
            rows.append(
                {
                    "configuration_budget": configuration_budget,
                    "cluster_budget": cluster_budget,
                    "actual_cluster_budget": cluster_budget,
                    "candidate_method": candidate,
                    "selected_method": "identity",
                    "selection_reason": "abstain_no_unlabeled_candidate",
                    "brier_gain": None,
                    "brier_gain_lcb": None,
                    "auc_delta": None,
                    "auc_delta_lcb": None,
                    "valid_auc_bootstraps": None,
                }
            )
            continue
        audit_clusters = order[:cluster_budget]
        audit = frame.loc[
            rows_for_clusters(frame, base_config.cluster_columns, audit_clusters)
        ].reset_index(drop=True)
        certificate = certify_checkpoint(
            audit,
            [candidate],
            certification_config,
            seed + configuration_budget * 1009,
            total_registered_candidates=1,
        ).iloc[0]
        if bool(certificate.certified):
            selected = candidate
            reason = "selected_target_labels_used_for_confirmation_only"
            locked_method = candidate
            locked_cluster_budget = cluster_budget
        else:
            selected = "identity"
            reason = f"abstain_confirmation_failed:{certificate.failure_reason}"
        rows.append(
            {
                "configuration_budget": configuration_budget,
                "cluster_budget": cluster_budget,
                "actual_cluster_budget": locked_cluster_budget or cluster_budget,
                "candidate_method": candidate,
                "selected_method": selected,
                "selection_reason": reason,
                "brier_gain": float(certificate.brier_gain),
                "brier_gain_lcb": float(certificate.brier_gain_lcb),
                "auc_delta": float(certificate.auc_delta),
                "auc_delta_lcb": float(certificate.auc_delta_lcb),
                "valid_auc_bootstraps": int(certificate.valid_auc_bootstraps),
            }
        )
    return pd.DataFrame(rows), order


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for debugging.")
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 5)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)

    public_path = Path(
        "outputs/transport_policy_public_development/synthetic_candidate_results.csv"
    )
    policy_path = Path(
        "outputs/transport_policy_public_development/frozen_transport_policy.joblib"
    )
    public = pd.read_csv(public_path)
    policy = joblib.load(policy_path)
    datasets = load_datasets(public, policy)
    config = RiskControlledConfig(
        bootstrap_repetitions=args.bootstrap_repetitions,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )

    action_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    decision_tables: list[pd.DataFrame] = []
    total = len(datasets) * len(RULES) * args.audit_repetitions
    progress = tqdm(
        total=total,
        desc="v5 label-independent certification",
        unit="audit",
        dynamic_ncols=True,
    )
    for dataset, (frame, signatures) in datasets.items():
        methods = probability_methods(frame)
        for rule in RULES:
            candidate, action_reason = choose_unlabeled_action(
                signatures, methods, config, rule
            )
            action_rows.append(
                {
                    "dataset": dataset,
                    "rule": rule.name,
                    "candidate_method": candidate,
                    "action_lock_reason": action_reason,
                }
            )
            for repetition in range(args.audit_repetitions):
                seed = args.seed + repetition * args.seed_step
                decisions, order = run_checkpoints(
                    frame, candidate, config, seed, rule
                )
                final = decisions.iloc[-1]
                used_clusters = order[: int(final.actual_cluster_budget)]
                test = frame.loc[
                    ~rows_for_clusters(frame, config.cluster_columns, used_clusters)
                ].reset_index(drop=True)
                result_rows.append(
                    {
                        "dataset": dataset,
                        "rule": rule.name,
                        "audit_repetition": repetition,
                        "audit_seed": seed,
                        "candidate_method": candidate,
                        "actual_configuration_budget": (
                            int(final.actual_cluster_budget)
                            * config.configurations_per_cluster
                        ),
                        "n_heldout_configurations": len(test),
                        **evaluate_action(test, str(final.selected_method)),
                    }
                )
                decisions.insert(0, "dataset", dataset)
                decisions.insert(1, "rule", rule.name)
                decisions.insert(2, "audit_repetition", repetition)
                decisions.insert(3, "audit_seed", seed)
                decision_tables.append(decisions)
                progress.update(1)
                progress.set_postfix(dataset=dataset, rule=rule.name, refresh=False)
    progress.close()

    args.output_root.mkdir(parents=True, exist_ok=False)
    results = pd.DataFrame(result_rows)
    summary = summarize(results)
    pd.DataFrame(action_rows).to_csv(
        args.output_root / "unlabeled_action_locks.csv", index=False
    )
    pd.concat(decision_tables, ignore_index=True).to_csv(
        args.output_root / "checkpoint_certificates.csv", index=False
    )
    results.to_csv(args.output_root / "repeated_audit_results.csv", index=False)
    summary.to_csv(args.output_root / "rule_summary.csv", index=False)
    manifest = {
        "status": "label_independent_v5_retrospective_development_complete",
        "date": "2026-09-07",
        "claim_status": "development_only_external_confirmation_required",
        "target_label_role": "confirmation_only_after_unlabeled_action_lock",
        "rules": [asdict(rule) for rule in RULES],
        "audit_repetitions": args.audit_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "cuda_verified": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "statistical_backend": "numpy_cpu_cluster_bootstrap",
        "script_sha256": sha256(Path(__file__)),
        "public_results_sha256": sha256(public_path),
        "policy_sha256": sha256(policy_path),
    }
    (args.output_root / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(action_rows).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
