from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_consensus_order_preserving_transport_v7 import load_development  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    CLUSTER_COLUMNS,
    STRATUM_COLUMNS,
    partition_frame,
)
from develop_order_preserving_transport_v6 import RULE  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import prepare as prepare_v9  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    balanced_cluster_order,
    rows_for_clusters,
)


ASSIGNMENTS = Path(
    "outputs/distribution_covered_stratified_opct_v11_development/"
    "distribution_covered_assignments.csv"
)
LOCKS = Path(
    "outputs/distribution_covered_stratified_opct_v11_development/"
    "unlabeled_action_locks.csv"
)
BUDGET_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
METHOD_ORDER = (
    "dcs_selective",
    "dcs_unconditional",
    "target_platt",
    "target_isotonic",
    "split_certified_platt",
)
METHOD_LABELS = {
    "dcs_selective": "DCS selective",
    "dcs_unconditional": "DCS unconditional",
    "target_platt": "Target Platt",
    "target_isotonic": "Target isotonic",
    "split_certified_platt": "Split-certified Platt",
}
METHOD_COLORS = {
    "dcs_selective": "#087f8c",
    "dcs_unconditional": "#63b7af",
    "target_platt": "#3266a8",
    "target_isotonic": "#c17c24",
    "split_certified_platt": "#8f4c8a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrospective fixed-heldout audit-budget decision curves for frozen "
            "DCS-OPCT v11 and matched target-calibration comparators."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dcs_opct_v11_audit_budget_decision_curves"),
    )
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--platt-epochs", type=int, default=1000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--seed-step", type=int, default=7919)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_seed(frame: pd.DataFrame, base_seed: int) -> int:
    keys = (
        frame.loc[:, list(CLUSTER_COLUMNS)]
        .drop_duplicates()
        .sort_values(list(CLUSTER_COLUMNS))
        .astype(str)
    )
    payload = keys.to_csv(index=False).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return int((base_seed + offset) % (2**32 - 1))


def safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        return np.nan
    return float(roc_auc_score(labels, probability))


def fit_positive_platt_batch(
    probabilities: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    seed: int,
    progress: tqdm,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    x = torch.as_tensor(logit(clipped), dtype=torch.float32, device="cuda")
    y = torch.as_tensor(labels, dtype=torch.float32, device="cuda")
    raw_scale = torch.nn.Parameter(torch.zeros(x.shape[0], device="cuda"))
    shift = torch.nn.Parameter(torch.zeros(x.shape[0], device="cuda"))
    optimizer = torch.optim.Adam([raw_scale, shift], lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.nn.functional.softplus(raw_scale) + 1e-4
        prediction = torch.sigmoid(scale[:, None] * x + shift[:, None])
        loss = torch.nn.functional.binary_cross_entropy(prediction, y)
        loss.backward()
        optimizer.step()
        progress.update(1)
    return (
        (torch.nn.functional.softplus(raw_scale) + 1e-4).detach().cpu().numpy(),
        shift.detach().cpu().numpy(),
    )


def apply_platt(probability: np.ndarray, scale: float, shift: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return expit(scale * logit(clipped) + shift)


def stratified_brier_certificate(
    audit: pd.DataFrame,
    probability_column: str,
    bootstrap_repetitions: int,
    seed: int,
    applicable: bool,
) -> dict[str, object]:
    if not applicable or audit.empty:
        return {
            "audit_brier_gain": np.nan,
            "audit_brier_gain_lcb": np.nan,
            "certified": False,
            "failure_reason": (
                "unlabeled_action_inapplicable" if not applicable else "empty_certificate_split"
            ),
        }
    work = audit.copy()
    labels = work.material_optimism_event.to_numpy(int)
    identity = work.probability_identity.to_numpy(float)
    candidate = work[probability_column].to_numpy(float)
    work["paired_brier_gain"] = (labels - identity) ** 2 - (labels - candidate) ** 2
    cluster_gain = (
        work.groupby(list(CLUSTER_COLUMNS), sort=False, dropna=False)
        .paired_brier_gain.mean()
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    draws = np.zeros(bootstrap_repetitions, dtype=float)
    total_clusters = 0
    for _, stratum in cluster_gain.groupby(list(STRATUM_COLUMNS), sort=False, dropna=False):
        gains = stratum.paired_brier_gain.to_numpy(float)
        sampled = rng.integers(0, len(gains), size=(bootstrap_repetitions, len(gains)))
        draws += gains[sampled].sum(axis=1)
        total_clusters += len(gains)
    draws /= total_clusters
    point = float(cluster_gain.paired_brier_gain.mean())
    lcb = float(np.quantile(draws, 0.025))
    certified = bool(lcb > RULE.minimum_certified_gain)
    return {
        "audit_brier_gain": point,
        "audit_brier_gain_lcb": lcb,
        "certified": certified,
        "failure_reason": "certified" if certified else "brier_lower_bound_not_positive",
    }


def heldout_metrics(
    heldout: pd.DataFrame,
    probability: np.ndarray,
    released: bool,
) -> tuple[dict[str, object], np.ndarray]:
    labels = heldout.material_optimism_event.to_numpy(int)
    identity = heldout.probability_identity.to_numpy(float)
    row_gain = (labels - identity) ** 2 - (labels - probability) ** 2
    work = heldout.loc[:, list(CLUSTER_COLUMNS)].copy()
    work["gain"] = row_gain
    cluster_gain = (
        work.groupby(list(CLUSTER_COLUMNS), sort=True, dropna=False)
        .gain.mean()
        .to_numpy(float)
    )
    identity_auc = safe_auc(labels, identity)
    candidate_auc = safe_auc(labels, probability)
    auc_delta = candidate_auc - identity_auc
    gain = float(np.mean(row_gain))
    return (
        {
            "released": bool(released),
            "brier_gain": gain,
            "auc_delta": auc_delta,
            "negative_transfer": bool(released and gain < 0.0),
            "material_negative_transfer": bool(released and gain < -0.001),
            "auc_noninferiority_failure": bool(released and auc_delta < -0.02),
        },
        cluster_gain,
    )


def two_stage_interval(
    gain_matrix: np.ndarray,
    cluster_strata: list[np.ndarray],
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for draw in range(repetitions):
        audit_repetition = int(rng.integers(0, gain_matrix.shape[0]))
        total = 0.0
        count = 0
        for indices in cluster_strata:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            total += float(gain_matrix[audit_repetition, sampled].sum())
            count += len(indices)
        draws[draw] = total / count
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(
    replicates: pd.DataFrame,
    gain_matrices: dict[tuple[str, float, str], list[np.ndarray]],
    heldout_strata: dict[str, list[np.ndarray]],
    bootstrap_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    grouped = replicates.groupby(["dataset", "budget_fraction", "method"], sort=False)
    for index, ((dataset, fraction, method), group) in enumerate(grouped):
        matrix = np.stack(gain_matrices[(dataset, fraction, method)])
        ci_low, ci_high = two_stage_interval(
            matrix,
            heldout_strata[dataset],
            bootstrap_repetitions,
            seed + index * 1009,
        )
        released = group.loc[group.released]
        rows.append(
            {
                "dataset": dataset,
                "budget_fraction": fraction,
                "audit_clusters": int(group.audit_clusters.iloc[0]),
                "audit_configurations": int(group.audit_configurations.iloc[0]),
                "method": method,
                "audit_repetitions": len(group),
                "release_rate": float(group.released.mean()),
                "mean_brier_gain": float(group.brier_gain.mean()),
                "brier_gain_ci_low": ci_low,
                "brier_gain_ci_high": ci_high,
                "mean_brier_gain_when_released": (
                    float(released.brier_gain.mean()) if len(released) else 0.0
                ),
                "negative_transfer_frequency": float(group.negative_transfer.mean()),
                "material_negative_transfer_frequency": float(
                    group.material_negative_transfer.mean()
                ),
                "auc_noninferiority_failure_frequency": float(
                    group.auc_noninferiority_failure.mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_gain_and_risk(summary: pd.DataFrame, output: Path) -> None:
    datasets = summary.dataset.drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(datasets), 2, figsize=(13.2, 2.8 * len(datasets)), constrained_layout=True
    )
    for row, dataset in enumerate(datasets):
        part = summary.loc[summary.dataset.eq(dataset)]
        for method in METHOD_ORDER:
            method_part = part.loc[part.method.eq(method)].sort_values("audit_configurations")
            x = method_part.audit_configurations.to_numpy(float)
            y = method_part.mean_brier_gain.to_numpy(float)
            low = method_part.brier_gain_ci_low.to_numpy(float)
            high = method_part.brier_gain_ci_high.to_numpy(float)
            axes[row, 0].plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            axes[row, 0].fill_between(
                x, low, high, color=METHOD_COLORS[method], alpha=0.10
            )
            axes[row, 1].plot(
                x,
                method_part.material_negative_transfer_frequency,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axes[row, 0].axhline(0.0, color="#303840", linewidth=0.9)
        axes[row, 1].axhline(0.0, color="#303840", linewidth=0.9)
        axes[row, 0].set_ylabel(f"{dataset}\nBrier gain")
        axes[row, 1].set_ylabel(f"{dataset}\nMaterial negative rate")
        axes[row, 1].set_ylim(-0.03, 1.03)
        for axis in axes[row]:
            axis.grid(axis="y", color="#dce2e8", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.set_xlabel("Labeled audit configurations")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.012),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Audit-budget efficacy and material-negative-transfer sensitivity", fontsize=15
    )
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_release(summary: pd.DataFrame, output: Path) -> None:
    datasets = summary.dataset.drop_duplicates().tolist()
    selective = ("dcs_selective", "split_certified_platt")
    fig, axes_grid = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)
    axes = axes_grid.ravel()
    for axis, dataset in zip(axes, datasets, strict=False):
        part = summary.loc[summary.dataset.eq(dataset)]
        for method in selective:
            method_part = part.loc[part.method.eq(method)].sort_values(
                "audit_configurations"
            )
            axis.plot(
                method_part.audit_configurations,
                method_part.release_rate,
                marker="o",
                linewidth=2.0,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(dataset)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Labeled configurations")
        axis.grid(axis="y", color="#dce2e8", linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Certified release rate")
    axes[3].set_ylabel("Certified release rate")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Budget-dependent selective release", fontsize=15, y=1.16)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batched positive-Platt fitting")
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 5)
        args.platt_epochs = min(args.platt_epochs, 50)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)

    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )
    prepared, _, _ = prepare_v9(load_development(), config, args.seed)
    assignments = pd.read_csv(ASSIGNMENTS)
    locks = pd.read_csv(LOCKS).set_index("dataset")
    fit_progress = tqdm(
        total=len(prepared) * len(BUDGET_FRACTIONS) * args.platt_epochs * 2,
        desc="CUDA batched Platt fits",
        unit="epoch",
        dynamic_ncols=True,
    )
    replicate_rows: list[dict[str, object]] = []
    certificate_rows: list[dict[str, object]] = []
    budget_rows: list[dict[str, object]] = []
    gain_matrices: dict[tuple[str, float, str], list[np.ndarray]] = {}
    heldout_strata: dict[str, list[np.ndarray]] = {}

    for dataset_index, (dataset, frame) in enumerate(prepared.items()):
        assignment = assignments.loc[assignments.dataset.eq(dataset)].drop(
            columns="dataset"
        )
        audit_pool, heldout = partition_frame(frame, assignment)
        heldout_clusters = (
            heldout.loc[:, list(CLUSTER_COLUMNS)]
            .drop_duplicates()
            .sort_values(list(CLUSTER_COLUMNS))
            .reset_index(drop=True)
        )
        heldout_clusters["cluster_index"] = np.arange(len(heldout_clusters), dtype=int)
        heldout = heldout.merge(
            heldout_clusters,
            on=list(CLUSTER_COLUMNS),
            how="left",
            validate="many_to_one",
        )
        heldout_strata[dataset] = [
            group.cluster_index.to_numpy(int)
            for _, group in heldout_clusters.groupby(
                list(STRATUM_COLUMNS), sort=False, dropna=False
            )
        ]
        repetition_orders = []
        for repetition in range(args.audit_repetitions):
            repetition_seed = (
                args.seed + dataset_index * 1000003 + repetition * args.seed_step
            )
            repetition_orders.append(
                balanced_cluster_order(audit_pool, config, repetition_seed)
            )
        total_clusters = len(repetition_orders[0])
        identity_heldout = heldout.probability_identity.to_numpy(float)
        dcs_heldout = heldout.probability_wg_opct.to_numpy(float)
        lock = locks.loc[dataset]

        for fraction in BUDGET_FRACTIONS:
            cluster_budget = max(2, int(round(total_clusters * fraction)))
            configuration_budget = cluster_budget * config.configurations_per_cluster
            selected_audits = []
            split_fits = []
            split_certificates = []
            for order in repetition_orders:
                clusters = order[:cluster_budget]
                selected_audits.append(
                    audit_pool.loc[
                        rows_for_clusters(audit_pool, CLUSTER_COLUMNS, clusters)
                    ].reset_index(drop=True)
                )
                split_fits.append(
                    audit_pool.loc[
                        rows_for_clusters(audit_pool, CLUSTER_COLUMNS, clusters[::2])
                    ].reset_index(drop=True)
                )
                split_certificates.append(
                    audit_pool.loc[
                        rows_for_clusters(audit_pool, CLUSTER_COLUMNS, clusters[1::2])
                    ].reset_index(drop=True)
                )
            audit_probability = np.stack(
                [part.probability_identity.to_numpy(float) for part in selected_audits]
            )
            audit_labels = np.stack(
                [part.material_optimism_event.to_numpy(int) for part in selected_audits]
            )
            split_probability = np.stack(
                [part.probability_identity.to_numpy(float) for part in split_fits]
            )
            split_labels = np.stack(
                [part.material_optimism_event.to_numpy(int) for part in split_fits]
            )
            platt_scale, platt_shift = fit_positive_platt_batch(
                audit_probability,
                audit_labels,
                args.platt_epochs,
                args.seed + dataset_index * 1009 + cluster_budget,
                fit_progress,
            )
            split_scale, split_shift = fit_positive_platt_batch(
                split_probability,
                split_labels,
                args.platt_epochs,
                args.seed + dataset_index * 2017 + cluster_budget,
                fit_progress,
            )
            budget_rows.append(
                {
                    "dataset": dataset,
                    "budget_fraction": fraction,
                    "audit_pool_clusters": total_clusters,
                    "audit_pool_configurations": len(audit_pool),
                    "audit_clusters": cluster_budget,
                    "audit_configurations": configuration_budget,
                    "split_fit_clusters": len(repetition_orders[0][:cluster_budget:2]),
                    "split_certificate_clusters": len(
                        repetition_orders[0][1:cluster_budget:2]
                    ),
                    "heldout_clusters": len(heldout_clusters),
                    "heldout_configurations": len(heldout),
                }
            )

            for repetition in range(args.audit_repetitions):
                audit = selected_audits[repetition]
                dcs_certificate = stratified_brier_certificate(
                    audit,
                    "probability_wg_opct",
                    args.bootstrap_repetitions,
                    frame_seed(
                        audit,
                        args.seed + dataset_index * 1000003 + cluster_budget * 1009,
                    ),
                    bool(lock.applicable),
                )
                direct_platt = apply_platt(
                    identity_heldout,
                    float(platt_scale[repetition]),
                    float(platt_shift[repetition]),
                )
                isotonic = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                audit_y = audit.material_optimism_event.to_numpy(int)
                if len(np.unique(audit_y)) == 1:
                    direct_isotonic = np.full(len(heldout), float(audit_y[0]))
                else:
                    isotonic.fit(audit.probability_identity.to_numpy(float), audit_y)
                    direct_isotonic = isotonic.predict(identity_heldout)

                split_certificate_frame = split_certificates[repetition].copy()
                split_certificate_frame["probability_split_platt"] = apply_platt(
                    split_certificate_frame.probability_identity.to_numpy(float),
                    float(split_scale[repetition]),
                    float(split_shift[repetition]),
                )
                split_certificate = stratified_brier_certificate(
                    split_certificate_frame,
                    "probability_split_platt",
                    args.bootstrap_repetitions,
                    frame_seed(
                        split_certificate_frame,
                        args.seed + dataset_index * 2000003 + cluster_budget * 2017,
                    ),
                    True,
                )
                split_heldout = apply_platt(
                    identity_heldout,
                    float(split_scale[repetition]),
                    float(split_shift[repetition]),
                )
                predictions = {
                    "dcs_selective": (
                        dcs_heldout
                        if dcs_certificate["certified"]
                        else identity_heldout,
                        bool(dcs_certificate["certified"]),
                    ),
                    "dcs_unconditional": (
                        dcs_heldout,
                        bool(np.max(np.abs(dcs_heldout - identity_heldout)) > 1e-12),
                    ),
                    "target_platt": (direct_platt, True),
                    "target_isotonic": (direct_isotonic, True),
                    "split_certified_platt": (
                        split_heldout
                        if split_certificate["certified"]
                        else identity_heldout,
                        bool(split_certificate["certified"]),
                    ),
                }
                for method, (probability, released) in predictions.items():
                    metrics, cluster_gains = heldout_metrics(
                        heldout, probability, released
                    )
                    replicate_rows.append(
                        {
                            "dataset": dataset,
                            "budget_fraction": fraction,
                            "audit_clusters": cluster_budget,
                            "audit_configurations": configuration_budget,
                            "audit_repetition": repetition,
                            "method": method,
                            **metrics,
                        }
                    )
                    gain_matrices.setdefault(
                        (dataset, fraction, method), []
                    ).append(cluster_gains)
                certificate_rows.extend(
                    [
                        {
                            "dataset": dataset,
                            "budget_fraction": fraction,
                            "audit_repetition": repetition,
                            "method": "dcs_selective",
                            "fit_configurations": 0,
                            "certificate_configurations": len(audit),
                            **dcs_certificate,
                        },
                        {
                            "dataset": dataset,
                            "budget_fraction": fraction,
                            "audit_repetition": repetition,
                            "method": "split_certified_platt",
                            "fit_configurations": len(split_fits[repetition]),
                            "certificate_configurations": len(
                                split_certificate_frame
                            ),
                            "platt_scale": float(split_scale[repetition]),
                            "platt_shift": float(split_shift[repetition]),
                            **split_certificate,
                        },
                    ]
                )
    fit_progress.close()

    replicates = pd.DataFrame(replicate_rows)
    summary = summarize(
        replicates,
        gain_matrices,
        heldout_strata,
        args.bootstrap_repetitions,
        args.seed,
    )
    summary_values = summary[
        ["mean_brier_gain", "brier_gain_ci_low", "brier_gain_ci_high"]
    ].to_numpy(float)
    if not np.isfinite(summary_values).all():
        raise RuntimeError("Non-finite decision-curve summary detected")

    args.output_root.mkdir(parents=True, exist_ok=False)
    replicates.to_csv(
        args.output_root / "decision_curve_replicates.csv", index=False
    )
    summary.to_csv(args.output_root / "decision_curve_summary.csv", index=False)
    pd.DataFrame(certificate_rows).to_csv(
        args.output_root / "certification_diagnostics.csv", index=False
    )
    pd.DataFrame(budget_rows).drop_duplicates().to_csv(
        args.output_root / "budget_design.csv", index=False
    )
    plot_gain_and_risk(
        summary, args.output_root / "fig_audit_budget_gain_and_risk"
    )
    plot_release(
        summary, args.output_root / "fig_audit_budget_selective_release"
    )
    manifest = {
        "status": "retrospective_audit_budget_decision_curve_complete",
        "date": "2026-09-08",
        "claim_boundary": (
            "Sensitivity analysis on repeated nested subsets of the frozen v11 audit "
            "half, with the frozen held-out half reused for evaluation. Intervals "
            "combine random audit-design draws and stratified complete-cluster "
            "resampling of that held-out half; they are not prospective external-"
            "validation intervals."
        ),
        "frozen_v11_unchanged": True,
        "budget_fractions": list(BUDGET_FRACTIONS),
        "methods": list(METHOD_ORDER),
        "audit_repetitions": args.audit_repetitions,
        "platt_epochs": args.platt_epochs,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "material_negative_transfer_threshold": -0.001,
        "auc_noninferiority_margin": -0.02,
        "minimum_certificate_brier_gain_lcb": RULE.minimum_certified_gain,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "gpu_name": torch.cuda.get_device_name(0),
        "input_sha256": {
            "assignments": sha256(ASSIGNMENTS),
            "locks": sha256(LOCKS),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
