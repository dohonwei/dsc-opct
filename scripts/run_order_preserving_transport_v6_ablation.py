from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from scipy.stats import spearmanr
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_order_preserving_transport_v6 import RULE, load_all_datasets  # noqa: E402
from develop_minimal_intervention_transport_v5 import (  # noqa: E402
    RULE as V5_RULE,
    rank_minimal_interventions,
)
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    balanced_cluster_order,
    certify_checkpoint,
    paired_point_metrics,
    probability_methods,
    rows_for_clusters,
    validate_prediction_contract,
)


PROTOCOL = Path("docs/order_preserving_transport_v6_ablation_protocol.md")
DEVELOPMENT = Path("outputs/order_preserving_transport_v6_development")
VARIANTS = ("direct_base", "intercept_only", "slope_only", "opct")
FACTORS = ("task", "representation", "model", "split_seed", "nominal_dose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen OPCT v6 ablation.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/order_preserving_transport_v6_ablation"),
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


def fit_variant(
    identity_probability: np.ndarray,
    base_probability: np.ndarray,
    variant: str,
    seed: int,
    progress: tqdm,
) -> tuple[np.ndarray, dict[str, object]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    eps = 1e-6
    identity = np.clip(np.asarray(identity_probability, dtype=np.float64), eps, 1 - eps)
    base = np.clip(np.asarray(base_probability, dtype=np.float64), eps, 1 - eps)
    identity_logit = torch.as_tensor(logit(identity), dtype=torch.float32, device="cuda")
    target = torch.as_tensor(base, dtype=torch.float32, device="cuda")
    log_scale = torch.nn.Parameter(torch.zeros((), device="cuda"))
    shift = torch.nn.Parameter(torch.zeros((), device="cuda"))
    parameters = []
    if variant in {"slope_only", "opct"}:
        parameters.append(log_scale)
    if variant in {"intercept_only", "opct"}:
        parameters.append(shift)
    optimizer = torch.optim.Adam(parameters, lr=RULE.learning_rate)
    final_loss = float("nan")
    for _ in range(RULE.epochs):
        optimizer.zero_grad(set_to_none=True)
        scale = (
            torch.exp(log_scale).clamp(RULE.minimum_scale, RULE.maximum_scale)
            if variant in {"slope_only", "opct"}
            else torch.ones((), device="cuda")
        )
        offset = (
            shift.clamp(-RULE.maximum_absolute_shift, RULE.maximum_absolute_shift)
            if variant in {"intercept_only", "opct"}
            else torch.zeros((), device="cuda")
        )
        prediction = torch.sigmoid(scale * identity_logit + offset)
        penalty = torch.zeros((), device="cuda")
        if variant in {"slope_only", "opct"}:
            penalty = penalty + (scale - 1) ** 2
        if variant in {"intercept_only", "opct"}:
            penalty = penalty + offset**2
        loss = torch.mean((prediction - target) ** 2) + RULE.regularization * penalty
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        progress.update(1)
    scale_value = (
        float(torch.exp(log_scale).clamp(RULE.minimum_scale, RULE.maximum_scale).detach().cpu())
        if variant in {"slope_only", "opct"} else 1.0
    )
    shift_value = (
        float(shift.clamp(-RULE.maximum_absolute_shift, RULE.maximum_absolute_shift).detach().cpu())
        if variant in {"intercept_only", "opct"} else 0.0
    )
    projected = expit(scale_value * logit(identity) + shift_value)
    rank = float(spearmanr(identity, projected).statistic)
    inversions = int(np.sum(np.diff(projected[np.argsort(identity, kind="stable")]) < -1e-12))
    mean_shift = float(np.mean(projected - identity))
    return projected, {
        "scale": scale_value,
        "shift": shift_value,
        "final_loss": final_loss,
        "approximation_mse": float(np.mean((projected - base) ** 2)),
        "probability_mean_shift": mean_shift,
        "probability_rank": rank,
        "decision_flip_rate": float(np.mean((identity >= 0.5) != (projected >= 0.5))),
        "order_inversions": inversions,
        "applicable": bool(
            abs(mean_shift) <= RULE.maximum_probability_mean_shift
            and rank >= RULE.minimum_rank and inversions == 0
        ),
    }


def direct_diagnostics(identity: np.ndarray, base: np.ndarray) -> dict[str, object]:
    rank = float(spearmanr(identity, base).statistic)
    mean_shift = float(np.mean(base - identity))
    ordered = base[np.argsort(identity, kind="stable")]
    return {
        "scale": np.nan,
        "shift": np.nan,
        "final_loss": 0.0,
        "approximation_mse": 0.0,
        "probability_mean_shift": mean_shift,
        "probability_rank": rank,
        "decision_flip_rate": float(np.mean((identity >= 0.5) != (base >= 0.5))),
        "order_inversions": int(np.sum(np.diff(ordered) < -1e-12)),
        "applicable": bool(abs(mean_shift) <= 0.10 and rank >= 0.95),
    }


def bootstrap_mean_ci(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 500):
        stop = min(start + 500, repetitions)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def cluster_means(frame: pd.DataFrame, values: np.ndarray, columns: tuple[str, ...]) -> np.ndarray:
    table = frame.loc[:, list(columns)].copy()
    table["value"] = values
    return table.groupby(list(columns), dropna=False, sort=False).value.mean().to_numpy(float)


def full_data_analyses(
    prepared: dict[str, pd.DataFrame], repetitions: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrast_rows = []
    subgroup_rows = []
    cluster_columns = RiskControlledConfig().cluster_columns
    total = sum(
        4 + sum(frame[factor].nunique(dropna=False) for factor in FACTORS)
        for frame in prepared.values()
    )
    progress = tqdm(
        total=total,
        desc="v6 ablation cluster bootstrap",
        unit="analysis",
        dynamic_ncols=True,
    )
    for dataset_index, (dataset, frame) in enumerate(prepared.items()):
        labels = frame.material_optimism_event.to_numpy(int)
        identity = frame.probability_identity.to_numpy(float)
        identity_loss = (labels - identity) ** 2
        opct_gain = identity_loss - (labels - frame.probability_opct.to_numpy(float)) ** 2
        for comparator_index, comparator in enumerate(
            ("identity", "direct_base", "intercept_only", "slope_only")
        ):
            comparator_probability = frame[f"probability_{comparator}"].to_numpy(float)
            comparator_gain = identity_loss - (labels - comparator_probability) ** 2
            units = cluster_means(frame, opct_gain - comparator_gain, cluster_columns)
            lower, upper = bootstrap_mean_ci(
                units, repetitions, seed + dataset_index * 1000 + comparator_index
            )
            contrast_rows.append(
                {
                    "dataset": dataset,
                    "contrast": f"opct_minus_{comparator}",
                    "n_clusters": len(units),
                    "mean_brier_gain_difference": float(units.mean()),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "opct_better": bool(lower > 0),
                }
            )
            progress.update(1)
        for factor in FACTORS:
            for level, subgroup in frame.groupby(factor, dropna=False, sort=False):
                indices = subgroup.index.to_numpy(int)
                gains = opct_gain[indices]
                units = cluster_means(subgroup, gains, cluster_columns)
                lower, upper = bootstrap_mean_ci(
                    units, repetitions, seed + dataset_index * 100000 + len(subgroup_rows)
                )
                point = paired_point_metrics(subgroup, ["opct"]).iloc[0]
                subgroup_rows.append(
                    {
                        "dataset": dataset,
                        "factor": factor,
                        "level": str(level),
                        "n_rows": len(subgroup),
                        "n_clusters": len(units),
                        "brier_gain": float(gains.mean()),
                        "ci95_lower": lower,
                        "ci95_upper": upper,
                        "auc_delta": float(point.auc_delta),
                        "negative_gain": bool(gains.mean() < 0),
                    }
                )
                progress.update(1)
    progress.close()
    return pd.DataFrame(contrast_rows), pd.DataFrame(subgroup_rows)


def summarize(results: pd.DataFrame, locks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    applicable = locks.set_index(["dataset", "method"])["applicable"]
    for (dataset, method), group in results.groupby(["dataset", "method"], sort=False):
        released = group.loc[group.released]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "applicable": bool(applicable.loc[(dataset, method)]),
                "audit_repetitions": len(group),
                "certification_rate": float(group.certified.mean()),
                "release_rate": float(group.released.mean()),
                "mean_heldout_brier_gain": float(group.brier_gain.mean()),
                "mean_brier_gain_when_released": (
                    float(released.brier_gain.mean()) if len(released) else 0.0
                ),
                "minimum_brier_gain_when_released": (
                    float(released.brier_gain.min()) if len(released) else 0.0
                ),
                "material_negative_transfer_when_released": (
                    float(released.material_negative_transfer.mean()) if len(released) else 0.0
                ),
                "auc_violation_when_released": (
                    float((~released.auc_noninferior).mean()) if len(released) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def prepare_datasets(
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: RiskControlledConfig,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    prepared = {}
    lock_rows = []
    progress = tqdm(
        total=len(datasets) * 3 * RULE.epochs,
        desc="v6 CUDA ablation fitting",
        unit="epoch",
        dynamic_ncols=True,
    )
    for dataset, (raw_frame, signatures) in datasets.items():
        frame = raw_frame.copy().reset_index(drop=True)
        validate_prediction_contract(frame, config)
        _, base_method, reason = rank_minimal_interventions(
            signatures, probability_methods(frame), config, V5_RULE
        )
        identity = frame.probability_identity.to_numpy(float)
        upstream_applicable = base_method != "identity"
        base = (
            frame[f"probability_{base_method}"].to_numpy(float)
            if upstream_applicable else identity.copy()
        )
        frame["probability_direct_base"] = base
        diagnostics = direct_diagnostics(identity, base)
        diagnostics["applicable"] = bool(upstream_applicable and diagnostics["applicable"])
        lock_rows.append(
            {
                "dataset": dataset,
                "method": "direct_base",
                "base_method": base_method,
                "base_action_reason": reason,
                **diagnostics,
            }
        )
        for variant in ("intercept_only", "slope_only", "opct"):
            if upstream_applicable:
                probability, diagnostics = fit_variant(identity, base, variant, seed, progress)
            else:
                probability = identity.copy()
                diagnostics = {
                    "scale": 1.0,
                    "shift": 0.0,
                    "final_loss": 0.0,
                    "approximation_mse": 0.0,
                    "probability_mean_shift": 0.0,
                    "probability_rank": 1.0,
                    "decision_flip_rate": 0.0,
                    "order_inversions": 0,
                    "applicable": False,
                }
                progress.update(RULE.epochs)
            frame[f"probability_{variant}"] = probability
            lock_rows.append(
                {
                    "dataset": dataset,
                    "method": variant,
                    "base_method": base_method,
                    "base_action_reason": reason,
                    **diagnostics,
                }
            )
        prepared[dataset] = frame
        progress.set_postfix(dataset=dataset, base=base_method, refresh=False)
    progress.close()
    return prepared, pd.DataFrame(lock_rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for OPCT v6 ablation")
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 3)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)

    validation_path = DEVELOPMENT / "independent_validation_report.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed":
        raise RuntimeError("Independent v6 validation must pass before ablation")
    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )
    prepared, locks = prepare_datasets(load_all_datasets(), config, args.seed)
    lock_index = locks.set_index(["dataset", "method"])
    certificate_rows = []
    result_rows = []
    progress = tqdm(
        total=len(prepared) * args.audit_repetitions,
        desc="v6 simultaneous ablation certification",
        unit="audit",
        dynamic_ncols=True,
    )
    for dataset, frame in prepared.items():
        for repetition in range(args.audit_repetitions):
            audit_seed = args.seed + repetition * args.seed_step
            order = balanced_cluster_order(frame, config, audit_seed)
            audit_clusters = order[: config.cluster_budgets[0]]
            audit_mask = rows_for_clusters(frame, config.cluster_columns, audit_clusters)
            audit = frame.loc[audit_mask].reset_index(drop=True)
            heldout = frame.loc[~audit_mask].reset_index(drop=True)
            candidates = [
                method for method in VARIANTS
                if bool(lock_index.loc[(dataset, method), "applicable"])
            ]
            certificates = (
                certify_checkpoint(
                    audit,
                    candidates,
                    config,
                    audit_seed + RULE.configuration_budget * 1009,
                    total_registered_candidates=len(VARIANTS),
                ).set_index("method")
                if candidates else pd.DataFrame()
            )
            for method in VARIANTS:
                applicable = bool(lock_index.loc[(dataset, method), "applicable"])
                has_certificate = bool(not certificates.empty and method in certificates.index)
                certificate = certificates.loc[method] if has_certificate else None
                certified = bool(applicable and has_certificate and certificate.certified)
                certificate_rows.append(
                    {
                        "dataset": dataset,
                        "audit_repetition": repetition,
                        "audit_seed": audit_seed,
                        "method": method,
                        "applicable": applicable,
                        "certified": certified,
                        "brier_gain": np.nan if certificate is None else float(certificate.brier_gain),
                        "brier_gain_lcb": (
                            np.nan if certificate is None else float(certificate.brier_gain_lcb)
                        ),
                        "auc_delta": np.nan if certificate is None else float(certificate.auc_delta),
                        "auc_delta_lcb": (
                            np.nan if certificate is None else float(certificate.auc_delta_lcb)
                        ),
                        "failure_reason": (
                            "not_applicable" if certificate is None
                            else str(certificate.failure_reason)
                        ),
                    }
                )
                metrics = evaluate_action(heldout, method)
                result_rows.append(
                    {
                        "dataset": dataset,
                        "audit_repetition": repetition,
                        "audit_seed": audit_seed,
                        "method": method,
                        "applicable": applicable,
                        "certified": certified,
                        "released": bool(applicable and certified),
                        **{
                            key: value for key, value in metrics.items()
                            if key not in {"selected_method", "adapted"}
                        },
                    }
                )
            progress.update(1)
            progress.set_postfix(dataset=dataset, candidates=len(candidates), refresh=False)
    progress.close()

    certificates = pd.DataFrame(certificate_rows)
    results = pd.DataFrame(result_rows)
    summary = summarize(results, locks)
    contrasts, heterogeneity = full_data_analyses(
        prepared, args.bootstrap_repetitions, args.seed + 424242
    )
    paired_rows = []
    for dataset, group in results.groupby("dataset", sort=False):
        pivot = group.pivot(index="audit_repetition", columns="method", values="brier_gain")
        for comparator in ("direct_base", "intercept_only", "slope_only"):
            difference = pivot.opct - pivot[comparator]
            paired_rows.append(
                {
                    "dataset": dataset,
                    "contrast": f"opct_minus_{comparator}",
                    "n_paired_audits": len(difference),
                    "mean_difference": float(difference.mean()),
                    "median_difference": float(difference.median()),
                    "opct_win_rate": float((difference > 0).mean()),
                    "tie_rate": float(np.isclose(difference, 0, atol=1e-12).mean()),
                }
            )
    paired = pd.DataFrame(paired_rows)

    args.output_root.mkdir(parents=True, exist_ok=False)
    locks.to_csv(args.output_root / "unlabeled_variant_locks.csv", index=False)
    certificates.to_csv(args.output_root / "simultaneous_certificates.csv", index=False)
    results.to_csv(args.output_root / "heldout_candidate_results.csv", index=False)
    summary.to_csv(args.output_root / "ablation_summary.csv", index=False)
    paired.to_csv(args.output_root / "paired_audit_contrasts.csv", index=False)
    contrasts.to_csv(args.output_root / "full_data_cluster_bootstrap_contrasts.csv", index=False)
    heterogeneity.to_csv(args.output_root / "opct_subgroup_heterogeneity.csv", index=False)

    adapted = summary.loc[summary.dataset.isin(["EPPVR", "CASE"])]
    opct = adapted.loc[adapted.method.eq("opct")]
    relevant_subgroups = heterogeneity.loc[heterogeneity.dataset.isin(["EPPVR", "CASE"])]
    gate = {
        "independent_v6_validation_passed": True,
        "all_opct_order_inversions_zero": bool(
            locks.loc[locks.method.eq("opct"), "order_inversions"].eq(0).all()
        ),
        "opct_zero_released_material_negative_transfer": bool(
            results.loc[
                results.method.eq("opct") & results.released,
                "material_negative_transfer",
            ].eq(False).all()
        ),
        "opct_zero_released_auc_violations": bool(
            results.loc[
                results.method.eq("opct") & results.released,
                "auc_noninferior",
            ].eq(True).all()
        ),
        "opct_release_rate_ge_080_on_eppvr_case": bool(opct.release_rate.ge(0.80).all()),
        "no_negative_overall_opct_subgroup_means": bool(
            relevant_subgroups.negative_gain.eq(False).all()
        ),
    }
    gate["all_passed"] = bool(all(gate.values()))
    (args.output_root / "ablation_acceptance_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "opct_v6_ablation_complete",
        "date": "2026-09-07",
        "claim_status": "retrospective_ablation_only",
        "cuda_verified": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "audit_repetitions": args.audit_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "development_manifest_sha256": sha256(DEVELOPMENT / "development_manifest.json"),
        "independent_validation_sha256": sha256(validation_path),
    }
    (args.output_root / "ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(locks.to_string(index=False))
    print(summary.to_string(index=False))
    print(contrasts.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
