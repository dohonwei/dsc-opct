from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from scipy.stats import spearmanr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_minimal_intervention_transport_v5 import (  # noqa: E402
    PUBLIC_PATH,
    POLICY_PATH,
    RULE as V5_RULE,
    rank_minimal_interventions,
)
from develop_risk_controlled_transport_v5 import (  # noqa: E402
    evaluate_action,
    load_datasets,
    summarize,
)
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    balanced_cluster_order,
    certify_checkpoint,
    probability_methods,
    rows_for_clusters,
    validate_prediction_contract,
)


PROTOCOL_PATH = Path("docs/order_preserving_transport_v6_development_protocol.md")
CASE_ROOT = Path("outputs/case_v5_external_confirmation")


@dataclass(frozen=True)
class OPCTConfig:
    name: str = "opct_v6_h160_m001"
    epochs: int = 500
    learning_rate: float = 0.03
    regularization: float = 1e-4
    minimum_scale: float = 0.1
    maximum_scale: float = 5.0
    maximum_absolute_shift: float = 5.0
    maximum_probability_mean_shift: float = 0.10
    minimum_rank: float = 0.999999
    configuration_budget: int = 160
    minimum_certified_gain: float = 0.001
    auc_noninferiority_margin: float = -0.02


RULE = OPCTConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Develop order-preserving calibration transport v6.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/order_preserving_transport_v6_development"),
    )
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--seed-step", type=int, default=7919)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_all_datasets() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    public = pd.read_csv(PUBLIC_PATH)
    policy = joblib.load(POLICY_PATH)
    datasets = load_datasets(public, policy)
    case = pd.concat(
        [
            pd.read_csv(CASE_ROOT / "primary_audit_labeled.csv"),
            pd.read_csv(CASE_ROOT / "primary_heldout_results.csv"),
        ],
        ignore_index=True,
    )
    case = case.sort_values(
        ["task", "representation", "model", "split_seed", "nominal_dose"]
    ).reset_index(drop=True)
    datasets["CASE"] = (
        case,
        pd.read_csv(CASE_ROOT / "candidate_signatures.csv"),
    )
    return datasets


def fit_opct(
    identity_probability: np.ndarray,
    base_probability: np.ndarray,
    config: OPCTConfig,
    seed: int,
    device: str,
    progress: tqdm | None = None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    eps = 1e-6
    identity = np.clip(np.asarray(identity_probability, dtype=np.float64), eps, 1.0 - eps)
    base = np.clip(np.asarray(base_probability, dtype=np.float64), eps, 1.0 - eps)
    logits = torch.as_tensor(logit(identity), dtype=torch.float32, device=device)
    target = torch.as_tensor(base, dtype=torch.float32, device=device)
    log_scale = torch.nn.Parameter(torch.zeros((), device=device))
    shift = torch.nn.Parameter(torch.zeros((), device=device))
    optimizer = torch.optim.Adam([log_scale, shift], lr=config.learning_rate)
    final_loss = float("nan")
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(log_scale).clamp(config.minimum_scale, config.maximum_scale)
        bounded_shift = shift.clamp(
            -config.maximum_absolute_shift, config.maximum_absolute_shift
        )
        prediction = torch.sigmoid(scale * logits + bounded_shift)
        loss = torch.mean((prediction - target) ** 2) + config.regularization * (
            (scale - 1.0) ** 2 + bounded_shift**2
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if progress is not None:
            progress.update(1)
    scale_value = float(
        torch.exp(log_scale).clamp(config.minimum_scale, config.maximum_scale).detach().cpu()
    )
    shift_value = float(
        shift.clamp(-config.maximum_absolute_shift, config.maximum_absolute_shift).detach().cpu()
    )
    projected = expit(scale_value * logit(identity) + shift_value)
    rank = float(spearmanr(identity, projected).statistic)
    order_inversions = int(
        np.sum(np.diff(projected[np.argsort(identity, kind="stable")]) < -1e-12)
    )
    diagnostics: dict[str, float | bool] = {
        "scale": scale_value,
        "shift": shift_value,
        "final_loss": final_loss,
        "probability_mean_shift": float(np.mean(projected - identity)),
        "probability_rank": rank,
        "decision_flip_rate": float(np.mean((identity >= 0.5) != (projected >= 0.5))),
        "order_inversions": order_inversions,
        "applicable": bool(
            abs(float(np.mean(projected - identity))) <= config.maximum_probability_mean_shift
            and rank >= config.minimum_rank
            and order_inversions == 0
        ),
    }
    return projected, diagnostics


def cluster_assignment_rows(
    dataset: str,
    repetition: int,
    seed: int,
    order: list[tuple[object, ...]],
    config: RiskControlledConfig,
) -> list[dict[str, object]]:
    rows = []
    for position, cluster in enumerate(order):
        rows.append(
            {
                "dataset": dataset,
                "rule": RULE.name,
                "audit_repetition": repetition,
                "audit_seed": seed,
                "cluster_order_position": position,
                "partition": "audit" if position < config.cluster_budgets[0] else "heldout",
                **dict(zip(config.cluster_columns, cluster, strict=True)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPCT v6 development")
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 5)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)

    datasets = load_all_datasets()
    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )
    fit_progress = tqdm(
        total=len(datasets) * RULE.epochs,
        desc="v6 CUDA order-preserving projection",
        unit="epoch",
        dynamic_ncols=True,
    )
    prepared: dict[str, tuple[pd.DataFrame, str, dict[str, float | bool]]] = {}
    action_rows = []
    for dataset, (raw_frame, signatures) in datasets.items():
        frame = raw_frame.copy()
        validate_prediction_contract(frame, config)
        rankings, base_method, action_reason = rank_minimal_interventions(
            signatures, probability_methods(frame), config, V5_RULE
        )
        if base_method == "identity":
            projected = frame.probability_identity.to_numpy(float).copy()
            diagnostics: dict[str, float | bool] = {
                "scale": 1.0,
                "shift": 0.0,
                "final_loss": 0.0,
                "probability_mean_shift": 0.0,
                "probability_rank": 1.0,
                "decision_flip_rate": 0.0,
                "order_inversions": 0,
                "applicable": False,
            }
            fit_progress.update(RULE.epochs)
            projection_reason = "abstain_v5_unlabeled_actionability_gate_failed"
        else:
            projected, diagnostics = fit_opct(
                frame.probability_identity.to_numpy(float),
                frame[f"probability_{base_method}"].to_numpy(float),
                RULE,
                args.seed,
                "cuda",
                fit_progress,
            )
            projection_reason = (
                "opct_locked_before_audit_outcomes"
                if bool(diagnostics["applicable"])
                else "abstain_opct_unlabeled_invariant_failed"
            )
        frame["probability_opct"] = projected
        prepared[dataset] = (frame, base_method, diagnostics)
        action_rows.append(
            {
                "dataset": dataset,
                "rule": RULE.name,
                "base_method": base_method,
                "base_action_reason": action_reason,
                "projection_reason": projection_reason,
                **diagnostics,
            }
        )
        fit_progress.set_postfix(dataset=dataset, base=base_method, refresh=False)
    fit_progress.close()

    result_rows = []
    certificate_rows = []
    assignment_rows = []
    audit_progress = tqdm(
        total=len(datasets) * args.audit_repetitions,
        desc="v6 fixed-horizon certification",
        unit="audit",
        dynamic_ncols=True,
    )
    for dataset, (frame, base_method, diagnostics) in prepared.items():
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
            if base_method == "identity" or not bool(diagnostics["applicable"]):
                certificate = None
                selected = "identity"
                selection_reason = "abstain_no_applicable_opct_candidate"
            else:
                certificate = certify_checkpoint(
                    audit,
                    ["opct"],
                    config,
                    seed + RULE.configuration_budget * 1009,
                    total_registered_candidates=1,
                ).iloc[0]
                selected = "opct" if bool(certificate.certified) else "identity"
                selection_reason = (
                    "selected_simultaneously_certified_opct"
                    if selected == "opct"
                    else f"abstain_confirmation_failed:{certificate.failure_reason}"
                )
            certificate_rows.append(
                {
                    "dataset": dataset,
                    "rule": RULE.name,
                    "audit_repetition": repetition,
                    "audit_seed": seed,
                    "base_method": base_method,
                    "selected_method": selected,
                    "selection_reason": selection_reason,
                    "n_audit_configurations": len(audit),
                    "n_heldout_configurations": len(heldout),
                    "brier_gain": None if certificate is None else float(certificate.brier_gain),
                    "brier_gain_lcb": None if certificate is None else float(certificate.brier_gain_lcb),
                    "brier_certified": None if certificate is None else bool(certificate.brier_certified),
                    "auc_delta": None if certificate is None else float(certificate.auc_delta),
                    "auc_delta_lcb": None if certificate is None else float(certificate.auc_delta_lcb),
                    "auc_noninferior": None if certificate is None else bool(certificate.auc_noninferior),
                    "certified": None if certificate is None else bool(certificate.certified),
                    "failure_reason": None if certificate is None else str(certificate.failure_reason),
                }
            )
            result_rows.append(
                {
                    "dataset": dataset,
                    "rule": RULE.name,
                    "audit_repetition": repetition,
                    "audit_seed": seed,
                    "base_method": base_method,
                    "actual_configuration_budget": len(audit),
                    "n_heldout_configurations": len(heldout),
                    **evaluate_action(heldout, selected),
                }
            )
            audit_progress.update(1)
            audit_progress.set_postfix(dataset=dataset, method=selected, refresh=False)
    audit_progress.close()

    args.output_root.mkdir(parents=True, exist_ok=False)
    actions = pd.DataFrame(action_rows)
    certificates = pd.DataFrame(certificate_rows)
    assignments = pd.DataFrame(assignment_rows)
    results = pd.DataFrame(result_rows)
    summary = summarize(results)
    actions.to_csv(args.output_root / "unlabeled_opct_locks.csv", index=False)
    certificates.to_csv(args.output_root / "fixed_horizon_certificates.csv", index=False)
    assignments.to_csv(args.output_root / "audit_cluster_assignments.csv", index=False)
    results.to_csv(args.output_root / "repeated_audit_results.csv", index=False)
    summary.to_csv(args.output_root / "rule_summary.csv", index=False)

    v5 = pd.read_csv(
        "outputs/minimal_intervention_transport_v5_development/rule_summary.csv"
    )
    comparison = summary.merge(
        v5.drop(columns="rule"), on="dataset", how="left", suffixes=("_v6", "_v5")
    )
    comparison.to_csv(args.output_root / "v5_v6_comparison.csv", index=False)
    summary_index = summary.set_index("dataset")
    adapted_datasets = summary.loc[summary.adaptation_coverage.ge(0.5)]
    acceptance = {
        "zero_material_negative_transfer": bool(
            summary.material_negative_transfer_frequency.eq(0).all()
        ),
        "all_released_actions_auc_noninferior": bool(
            results.loc[results.adapted, "auc_noninferior"].all()
        ),
        "at_least_two_datasets_coverage_ge_050": bool(len(adapted_datasets) >= 2),
        "adapted_dataset_mean_gain_gt_0005": bool(
            adapted_datasets.mean_brier_gain_when_adapted.gt(0.005).all()
        ),
        "adapted_dataset_min_gain_gt_0001": bool(
            adapted_datasets.minimum_brier_gain_when_adapted.gt(0.001).all()
        ),
        "case_coverage_ge_080": bool(summary_index.loc["CASE", "adaptation_coverage"] >= 0.80),
        "eppvr_coverage_drop_le_010": bool(
            summary_index.loc["EPPVR", "adaptation_coverage"]
            >= float(v5.loc[v5.dataset.eq("EPPVR"), "adaptation_coverage"].iloc[0]) - 0.10
        ),
    }
    acceptance["all_passed"] = bool(all(acceptance.values()))
    (args.output_root / "development_acceptance_gate.json").write_text(
        json.dumps(acceptance, indent=2), encoding="utf-8"
    )
    input_paths = {
        "protocol": PROTOCOL_PATH,
        "public_results": PUBLIC_PATH,
        "policy": POLICY_PATH,
        "case_gate": CASE_ROOT / "external_confirmation_gate.json",
        "case_audit": CASE_ROOT / "primary_audit_labeled.csv",
        "case_heldout": CASE_ROOT / "primary_heldout_results.csv",
    }
    manifest = {
        "status": "opct_v6_retrospective_development_complete",
        "date": "2026-09-07",
        "claim_status": "development_only_new_untouched_confirmation_required",
        "rule": asdict(RULE),
        "risk_control_config": config.to_dict(),
        "audit_repetitions": args.audit_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "cuda_verified": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "case_role": "retrospective_method_development_after_v5_one_shot_failure",
    }
    (args.output_root / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(actions.to_string(index=False))
    print(summary.to_string(index=False))
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()
