from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from apply_frozen_risk_controlled_transport_to_dreamer import KEYS, load_counterfactual_outcomes  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    applicability_table,
    attach_conformal_ood,
    balanced_cluster_order,
    certify_checkpoint,
    paired_point_metrics,
    probability_methods,
    rows_for_clusters,
)
from identity_shortcut.transport_policy import committee_distribution  # noqa: E402


DATASETS = {
    "EPPVR": {
        "predictions": Path("outputs/eppvr_transport_policy_retrospective/eppvr_policy_predictions.csv"),
        "signatures": Path("outputs/eppvr_transport_policy_retrospective/unlabeled_candidate_signatures.csv"),
    },
    "SEED-IV": {
        "predictions": Path("outputs/seediv_transport_policy_external/seediv_policy_predictions.csv"),
        "signatures": Path("outputs/seediv_transport_policy_external/unlabeled_candidate_signatures.csv"),
    },
}


@dataclass(frozen=True)
class CandidateRule:
    name: str
    minimum_practical_gain: float
    minimum_confirmation_clusters: int


RULES = (
    CandidateRule("split_confirm_m0_c2", 0.0, 2),
    CandidateRule("split_confirm_m0005_c4", 0.0005, 4),
    CandidateRule("split_confirm_m001_c4", 0.001, 4),
    CandidateRule("split_confirm_m001_c8", 0.001, 8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrospectively develop v5 selection-confirmation transport rules."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/risk_controlled_transport_v5_development"),
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


def enrich_signatures(
    raw: pd.DataFrame,
    development: pd.DataFrame,
    policy: dict[str, object],
) -> pd.DataFrame:
    output = attach_conformal_ood(development, raw)
    output["predicted_brier_gain"] = 0.0
    output["predicted_auc_delta"] = 0.0
    mask = output.method.ne("identity")
    if mask.any():
        output.loc[mask, "predicted_brier_gain"] = committee_distribution(
            policy["gain_committee"], output.loc[mask]
        ).mean(axis=1)
        output.loc[mask, "predicted_auc_delta"] = committee_distribution(
            policy["auc_committee"], output.loc[mask]
        ).mean(axis=1)
    return output


def load_datasets(
    public: pd.DataFrame,
    policy: dict[str, object],
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, paths in DATASETS.items():
        loaded[name] = (
            pd.read_csv(paths["predictions"]),
            enrich_signatures(pd.read_csv(paths["signatures"]), public, policy),
        )

    mechanism = pd.read_csv(
        "outputs/dreamer_v4_external_confirmation/mechanism_probabilities_blinded.csv"
    )
    outcomes = load_counterfactual_outcomes(
        Path("outputs/dreamer_counterfactual_external/summary.csv"), 0.02, None
    )
    dreamer = mechanism.merge(
        outcomes, on=KEYS + ["nominal_dose"], validate="one_to_one"
    )
    dreamer_signatures = pd.read_csv(
        "outputs/dreamer_v4_external_confirmation/candidate_signatures.csv"
    )
    loaded["DREAMER"] = (dreamer, dreamer_signatures)
    return loaded


def evaluate_action(frame: pd.DataFrame, method: str) -> dict[str, object]:
    point = paired_point_metrics(frame, [method]).iloc[0]
    gain = float(point.brier_gain)
    auc_delta = float(point.auc_delta)
    return {
        "selected_method": method,
        "adapted": method != "identity",
        "brier_gain": gain,
        "auc_delta": auc_delta,
        "negative_transfer": bool(method != "identity" and gain < 0.0),
        "material_negative_transfer": bool(method != "identity" and gain < -0.001),
        "auc_noninferior": bool(np.isfinite(auc_delta) and auc_delta >= -0.02),
    }


def select_on_discovery(
    discovery: pd.DataFrame,
    methods: list[str],
    margin: float,
) -> str:
    if not methods or discovery.material_optimism_event.nunique() < 2:
        return "identity"
    point = paired_point_metrics(discovery, methods)
    eligible = point.loc[point.brier_gain.gt(margin) & point.auc_delta.ge(-0.02)]
    if eligible.empty:
        return "identity"
    return str(
        eligible.sort_values(["brier_gain", "method"], ascending=[False, True]).iloc[0].method
    )


def run_split_confirmation(
    frame: pd.DataFrame,
    signatures: pd.DataFrame,
    base_config: RiskControlledConfig,
    seed: int,
    rule: CandidateRule,
) -> tuple[pd.DataFrame, list[tuple[object, ...]]]:
    methods = probability_methods(frame)
    applicability = applicability_table(signatures, methods, base_config)
    applicable = applicability.loc[
        applicability.applicable & applicability.method.ne("identity"), "method"
    ].tolist()
    order = balanced_cluster_order(frame, base_config, seed)
    decisions: list[dict[str, object]] = []
    locked_method: str | None = None
    locked_budget: int | None = None
    for configuration_budget, cluster_budget in zip(
        base_config.configuration_budgets,
        base_config.cluster_budgets,
        strict=True,
    ):
        audit_clusters = order[:cluster_budget]
        if locked_method is not None:
            decisions.append(
                {
                    "configuration_budget": configuration_budget,
                    "cluster_budget": cluster_budget,
                    "actual_cluster_budget": locked_budget,
                    "selected_method": locked_method,
                    "selection_reason": "carried_forward_independent_confirmation",
                }
            )
            continue

        discovery_clusters = audit_clusters[::2]
        confirmation_clusters = audit_clusters[1::2]
        if len(confirmation_clusters) < rule.minimum_confirmation_clusters:
            selected = "identity"
            reason = "abstain_confirmation_sample_below_floor"
        else:
            discovery = frame.loc[
                rows_for_clusters(frame, base_config.cluster_columns, discovery_clusters)
            ].reset_index(drop=True)
            confirmation = frame.loc[
                rows_for_clusters(frame, base_config.cluster_columns, confirmation_clusters)
            ].reset_index(drop=True)
            candidate = select_on_discovery(
                discovery, applicable, rule.minimum_practical_gain
            )
            if candidate == "identity":
                selected = "identity"
                reason = "abstain_no_discovery_candidate"
            else:
                confirmation_config = RiskControlledConfig(
                    configuration_budgets=base_config.configuration_budgets,
                    configurations_per_cluster=base_config.configurations_per_cluster,
                    familywise_alpha=base_config.familywise_alpha,
                    bootstrap_repetitions=base_config.bootstrap_repetitions,
                    auc_noninferiority_margin=base_config.auc_noninferiority_margin,
                    minimum_brier_gain=rule.minimum_practical_gain,
                    minimum_valid_auc_bootstraps=base_config.minimum_valid_auc_bootstraps,
                )
                certificate = certify_checkpoint(
                    confirmation,
                    [candidate],
                    confirmation_config,
                    seed + configuration_budget * 1009,
                    total_registered_candidates=1,
                ).iloc[0]
                if bool(certificate.certified):
                    selected = candidate
                    reason = "selected_after_independent_confirmation"
                    locked_method = candidate
                    locked_budget = cluster_budget
                else:
                    selected = "identity"
                    reason = f"abstain_confirmation_failed:{certificate.failure_reason}"
        decisions.append(
            {
                "configuration_budget": configuration_budget,
                "cluster_budget": cluster_budget,
                "actual_cluster_budget": locked_budget or cluster_budget,
                "selected_method": selected,
                "selection_reason": reason,
            }
        )
    return pd.DataFrame(decisions), order


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in results.groupby(["dataset", "rule"], sort=False):
        adapted = group.loc[group.adapted]
        rows.append(
            {
                "dataset": keys[0],
                "rule": keys[1],
                "audit_repetitions": len(group),
                "adaptation_coverage": float(group.adapted.mean()),
                "negative_transfer_frequency": float(group.negative_transfer.mean()),
                "material_negative_transfer_frequency": float(
                    group.material_negative_transfer.mean()
                ),
                "negative_transfer_when_adapted": (
                    float(adapted.negative_transfer.mean()) if len(adapted) else 0.0
                ),
                "mean_brier_gain": float(group.brier_gain.mean()),
                "mean_brier_gain_when_adapted": (
                    float(adapted.brier_gain.mean()) if len(adapted) else 0.0
                ),
                "minimum_brier_gain_when_adapted": (
                    float(adapted.brier_gain.min()) if len(adapted) else 0.0
                ),
                "auc_noninferiority_rate": float(group.auc_noninferior.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite v5 development output: {args.output_root}"
        )
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

    rows: list[dict[str, object]] = []
    decisions: list[pd.DataFrame] = []
    total = len(datasets) * len(RULES) * args.audit_repetitions
    progress = tqdm(
        total=total,
        desc="v5 retrospective selection-confirmation",
        unit="audit",
        dynamic_ncols=True,
    )
    for dataset, (frame, signatures) in datasets.items():
        for rule in RULES:
            for repetition in range(args.audit_repetitions):
                seed = args.seed + repetition * args.seed_step
                decision, order = run_split_confirmation(
                    frame, signatures, config, seed, rule
                )
                final = decision.iloc[-1]
                used_clusters = order[: int(final.actual_cluster_budget)]
                test = frame.loc[
                    ~rows_for_clusters(frame, config.cluster_columns, used_clusters)
                ].reset_index(drop=True)
                metrics = evaluate_action(test, str(final.selected_method))
                rows.append(
                    {
                        "dataset": dataset,
                        "rule": rule.name,
                        "audit_repetition": repetition,
                        "audit_seed": seed,
                        "actual_configuration_budget": (
                            int(final.actual_cluster_budget)
                            * config.configurations_per_cluster
                        ),
                        "n_heldout_configurations": len(test),
                        **metrics,
                    }
                )
                decision.insert(0, "dataset", dataset)
                decision.insert(1, "rule", rule.name)
                decision.insert(2, "audit_repetition", repetition)
                decision.insert(3, "audit_seed", seed)
                decisions.append(decision)
                progress.update(1)
                progress.set_postfix(dataset=dataset, rule=rule.name, refresh=False)
    progress.close()

    args.output_root.mkdir(parents=True, exist_ok=False)
    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(args.output_root / "repeated_audit_results.csv", index=False)
    summary.to_csv(args.output_root / "rule_summary.csv", index=False)
    pd.concat(decisions, ignore_index=True).to_csv(
        args.output_root / "checkpoint_decisions.csv", index=False
    )
    manifest = {
        "status": "v5_retrospective_development_complete",
        "date": "2026-09-07",
        "claim_status": "development_only_external_confirmation_required",
        "dreamer_role": "development_failure_analysis_after_v4_confirmation_failed",
        "rules": [rule.__dict__ for rule in RULES],
        "audit_repetitions": args.audit_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "script_sha256": sha256(Path(__file__)),
        "public_results_sha256": sha256(public_path),
        "policy_sha256": sha256(policy_path),
    }
    (args.output_root / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
