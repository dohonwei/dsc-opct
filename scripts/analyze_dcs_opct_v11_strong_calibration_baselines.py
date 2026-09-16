from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_dcs_opct_v11_audit_budget_decision_curves import (  # noqa: E402
    ASSIGNMENTS,
    BUDGET_FRACTIONS,
    LOCKS,
    apply_platt,
    fit_positive_platt_batch,
    frame_seed,
    heldout_metrics,
    sha256,
    stratified_brier_certificate,
    summarize,
)
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


LEGACY_OUTPUT = Path("outputs/dcs_opct_v11_audit_budget_decision_curves")
FREEZE_MANIFEST = Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json")
EXPECTED_FREEZE_SHA256 = (
    "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
)
LEGACY_METHODS = (
    "dcs_selective",
    "dcs_unconditional",
    "target_platt",
    "target_isotonic",
    "split_certified_platt",
)
NEW_METHODS = ("target_beta", "target_temperature", "split_certified_beta")
METHOD_ORDER = LEGACY_METHODS + NEW_METHODS
METHOD_LABELS = {
    "dcs_selective": "DCS selective",
    "dcs_unconditional": "DCS unconditional",
    "target_platt": "Target Platt",
    "target_isotonic": "Target isotonic",
    "split_certified_platt": "Split-certified Platt",
    "target_beta": "Target beta",
    "target_temperature": "Target temperature",
    "split_certified_beta": "Split-certified beta",
}
METHOD_COLORS = {
    "dcs_selective": "#087f8c",
    "dcs_unconditional": "#69aaa3",
    "target_platt": "#3266a8",
    "target_isotonic": "#c17c24",
    "split_certified_platt": "#8f4c8a",
    "target_beta": "#b03a48",
    "target_temperature": "#5c6f2c",
    "split_certified_beta": "#6c4ea1",
}
DIRECT_METHODS = (
    "dcs_selective",
    "target_platt",
    "target_isotonic",
    "target_beta",
    "target_temperature",
)
SELECTIVE_METHODS = (
    "dcs_selective",
    "split_certified_platt",
    "split_certified_beta",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrospective matched-budget comparison of frozen DCS-OPCT v11 "
            "against stronger GPU target-calibration baselines."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dcs_opct_v11_strong_calibration_baselines"),
    )
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--calibration-epochs", type=int, default=1000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--seed-step", type=int, default=7919)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def fit_monotone_beta_batch(
    probabilities: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    seed: int,
    progress: tqdm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit beta calibration with nonnegative shape terms, preserving rank order."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    log_p = torch.as_tensor(np.log(clipped), dtype=torch.float32, device="cuda")
    neg_log_one_minus_p = torch.as_tensor(
        -np.log1p(-clipped), dtype=torch.float32, device="cuda"
    )
    y = torch.as_tensor(labels, dtype=torch.float32, device="cuda")
    initialization = inverse_softplus(1.0 - 1e-4)
    raw_a = torch.nn.Parameter(
        torch.full((log_p.shape[0],), initialization, device="cuda")
    )
    raw_b = torch.nn.Parameter(
        torch.full((log_p.shape[0],), initialization, device="cuda")
    )
    shift = torch.nn.Parameter(torch.zeros(log_p.shape[0], device="cuda"))
    optimizer = torch.optim.Adam([raw_a, raw_b, shift], lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        a = torch.nn.functional.softplus(raw_a) + 1e-4
        b = torch.nn.functional.softplus(raw_b) + 1e-4
        logits = a[:, None] * log_p + b[:, None] * neg_log_one_minus_p + shift[:, None]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        progress.update(1)
    return (
        (torch.nn.functional.softplus(raw_a) + 1e-4).detach().cpu().numpy(),
        (torch.nn.functional.softplus(raw_b) + 1e-4).detach().cpu().numpy(),
        shift.detach().cpu().numpy(),
    )


def apply_beta(
    probability: np.ndarray,
    shape_positive: float,
    shape_negative: float,
    shift: float,
) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    calibrated_logit = (
        shape_positive * np.log(clipped)
        - shape_negative * np.log1p(-clipped)
        + shift
    )
    return expit(calibrated_logit)


def fit_temperature_batch(
    probabilities: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    seed: int,
    progress: tqdm,
) -> np.ndarray:
    """Fit positive inverse temperature without an intercept."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    x = torch.as_tensor(logit(clipped), dtype=torch.float32, device="cuda")
    y = torch.as_tensor(labels, dtype=torch.float32, device="cuda")
    initialization = inverse_softplus(1.0 - 1e-4)
    raw_scale = torch.nn.Parameter(
        torch.full((x.shape[0],), initialization, device="cuda")
    )
    optimizer = torch.optim.Adam([raw_scale], lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        inverse_temperature = torch.nn.functional.softplus(raw_scale) + 1e-4
        prediction = torch.sigmoid(inverse_temperature[:, None] * x)
        loss = torch.nn.functional.binary_cross_entropy(prediction, y)
        loss.backward()
        optimizer.step()
        progress.update(1)
    return (
        torch.nn.functional.softplus(raw_scale) + 1e-4
    ).detach().cpu().numpy()


def apply_temperature(probability: np.ndarray, inverse_temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return expit(inverse_temperature * logit(clipped))


def summarize_methods(
    replicates: pd.DataFrame,
    gain_matrices: dict[tuple[str, float, str], list[np.ndarray]],
    heldout_strata: dict[str, list[np.ndarray]],
    methods: tuple[str, ...],
    bootstrap_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    subset = replicates.loc[replicates.method.isin(methods)].copy()
    selected_matrices = {
        key: value for key, value in gain_matrices.items() if key[2] in methods
    }
    return summarize(
        subset,
        selected_matrices,
        heldout_strata,
        bootstrap_repetitions,
        seed,
    )


def verify_legacy_reproduction(summary: pd.DataFrame) -> dict[str, object]:
    reference_path = LEGACY_OUTPUT / "decision_curve_summary.csv"
    if not reference_path.exists():
        return {
            "status": "reference_missing",
            "reference": str(reference_path),
            "maximum_absolute_discrepancy": None,
        }
    reference = pd.read_csv(reference_path)
    keys = ["dataset", "budget_fraction", "method"]
    columns = [
        "audit_clusters",
        "audit_configurations",
        "audit_repetitions",
        "release_rate",
        "mean_brier_gain",
        "brier_gain_ci_low",
        "brier_gain_ci_high",
        "mean_brier_gain_when_released",
        "negative_transfer_frequency",
        "material_negative_transfer_frequency",
        "auc_noninferiority_failure_frequency",
    ]
    candidate = summary.loc[summary.method.isin(LEGACY_METHODS), keys + columns]
    merged = reference[keys + columns].merge(
        candidate,
        on=keys,
        how="outer",
        suffixes=("_reference", "_candidate"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged._merge.eq("both").all():
        raise RuntimeError("Legacy reproduction failed: row keys do not match")
    discrepancies = []
    for column in columns:
        left = merged[f"{column}_reference"].to_numpy(float)
        right = merged[f"{column}_candidate"].to_numpy(float)
        discrepancies.append(float(np.nanmax(np.abs(left - right))))
    maximum = max(discrepancies)
    if maximum > 1e-12:
        raise RuntimeError(
            f"Legacy reproduction discrepancy {maximum:.3e} exceeds tolerance"
        )
    return {
        "status": "passed",
        "reference": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "rows_compared": len(merged),
        "columns_compared": columns,
        "maximum_absolute_discrepancy": maximum,
        "tolerance": 1e-12,
    }


def paired_comparisons(replicates: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["dataset", "budget_fraction", "audit_repetition"]
    wide = replicates.pivot(index=index_columns, columns="method")
    rows: list[dict[str, object]] = []
    comparators = tuple(method for method in METHOD_ORDER if method != "dcs_selective")
    for (dataset, fraction), group in replicates.groupby(
        ["dataset", "budget_fraction"], sort=False
    ):
        indexed = wide.loc[(dataset, fraction)]
        dcs_gain = indexed["brier_gain"]["dcs_selective"].to_numpy(float)
        dcs_material = indexed["material_negative_transfer"][
            "dcs_selective"
        ].to_numpy(float)
        for comparator in comparators:
            comparator_gain = indexed["brier_gain"][comparator].to_numpy(float)
            comparator_material = indexed["material_negative_transfer"][
                comparator
            ].to_numpy(float)
            difference = dcs_gain - comparator_gain
            rows.append(
                {
                    "dataset": dataset,
                    "budget_fraction": fraction,
                    "audit_configurations": int(group.audit_configurations.iloc[0]),
                    "reference_method": "dcs_selective",
                    "comparator_method": comparator,
                    "mean_paired_brier_gain_difference": float(difference.mean()),
                    "median_paired_brier_gain_difference": float(np.median(difference)),
                    "dcs_higher_gain_frequency": float(np.mean(difference > 0.0)),
                    "dcs_lower_gain_frequency": float(np.mean(difference < 0.0)),
                    "material_negative_transfer_frequency_difference": float(
                        dcs_material.mean() - comparator_material.mean()
                    ),
                    "descriptive_only": True,
                }
            )
    return pd.DataFrame(rows)


def low_label_summary(summary: pd.DataFrame) -> pd.DataFrame:
    low = summary.loc[summary.budget_fraction.isin((0.2, 0.4))].copy()
    return (
        low.groupby(["dataset", "method"], sort=False, observed=True)
        .agg(
            mean_release_rate=("release_rate", "mean"),
            mean_brier_gain=("mean_brier_gain", "mean"),
            mean_material_negative_transfer_frequency=(
                "material_negative_transfer_frequency",
                "mean",
            ),
            mean_auc_noninferiority_failure_frequency=(
                "auc_noninferiority_failure_frequency",
                "mean",
            ),
        )
        .reset_index()
    )


def plot_direct(summary: pd.DataFrame, output: Path) -> None:
    datasets = summary.dataset.drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12.7, 2.55 * len(datasets)))
    for row, dataset in enumerate(datasets):
        part = summary.loc[summary.dataset.eq(dataset)]
        for method in DIRECT_METHODS:
            method_part = part.loc[part.method.eq(method)].sort_values(
                "audit_configurations"
            )
            x = method_part.audit_configurations.to_numpy(float)
            axes[row, 0].plot(
                x,
                method_part.mean_brier_gain,
                marker="o",
                linewidth=1.7,
                markersize=4.2,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            axes[row, 1].plot(
                x,
                method_part.material_negative_transfer_frequency,
                marker="o",
                linewidth=1.7,
                markersize=4.2,
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
        bbox_to_anchor=(0.5, 0.003),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Direct target-calibration strength and low-label risk", fontsize=15, y=0.995
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.975))
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_selective(summary: pd.DataFrame, output: Path) -> None:
    datasets = summary.dataset.drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 3, figsize=(14.2, 2.55 * len(datasets)))
    for row, dataset in enumerate(datasets):
        part = summary.loc[summary.dataset.eq(dataset)]
        for method in SELECTIVE_METHODS:
            method_part = part.loc[part.method.eq(method)].sort_values(
                "audit_configurations"
            )
            x = method_part.audit_configurations.to_numpy(float)
            axes[row, 0].plot(
                x,
                method_part.release_rate,
                marker="o",
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            axes[row, 1].plot(
                x,
                method_part.mean_brier_gain,
                marker="o",
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            axes[row, 2].plot(
                x,
                method_part.material_negative_transfer_frequency,
                marker="o",
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axes[row, 0].set_ylabel(f"{dataset}\nRelease rate")
        axes[row, 1].set_ylabel(f"{dataset}\nBrier gain")
        axes[row, 2].set_ylabel(f"{dataset}\nMaterial negative rate")
        axes[row, 0].set_ylim(-0.03, 1.03)
        axes[row, 2].set_ylim(-0.03, 1.03)
        axes[row, 1].axhline(0.0, color="#303840", linewidth=0.9)
        for axis in axes[row]:
            axis.grid(axis="y", color="#dce2e8", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.set_xlabel("Labeled audit configurations")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Selective release against split-certified calibration", fontsize=15, y=0.995
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.975))
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the strong calibration baselines")
    freeze_hash = sha256(FREEZE_MANIFEST)
    if freeze_hash != EXPECTED_FREEZE_SHA256:
        raise RuntimeError(
            "Frozen v11 manifest hash mismatch; refusing retrospective extension"
        )
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 5)
        args.calibration_epochs = min(args.calibration_epochs, 50)
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
        total=len(prepared) * len(BUDGET_FRACTIONS) * args.calibration_epochs * 5,
        desc="CUDA strong calibration fits",
        unit="epoch",
        dynamic_ncols=True,
    )
    replicate_rows: list[dict[str, object]] = []
    certificate_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
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
                args.calibration_epochs,
                args.seed + dataset_index * 1009 + cluster_budget,
                fit_progress,
            )
            split_platt_scale, split_platt_shift = fit_positive_platt_batch(
                split_probability,
                split_labels,
                args.calibration_epochs,
                args.seed + dataset_index * 2017 + cluster_budget,
                fit_progress,
            )
            beta_a, beta_b, beta_shift = fit_monotone_beta_batch(
                audit_probability,
                audit_labels,
                args.calibration_epochs,
                args.seed + dataset_index * 3011 + cluster_budget,
                fit_progress,
            )
            split_beta_a, split_beta_b, split_beta_shift = fit_monotone_beta_batch(
                split_probability,
                split_labels,
                args.calibration_epochs,
                args.seed + dataset_index * 4001 + cluster_budget,
                fit_progress,
            )
            inverse_temperature = fit_temperature_batch(
                audit_probability,
                audit_labels,
                args.calibration_epochs,
                args.seed + dataset_index * 5003 + cluster_budget,
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
                    float(split_platt_scale[repetition]),
                    float(split_platt_shift[repetition]),
                )
                split_platt_certificate = stratified_brier_certificate(
                    split_certificate_frame,
                    "probability_split_platt",
                    args.bootstrap_repetitions,
                    frame_seed(
                        split_certificate_frame,
                        args.seed + dataset_index * 2000003 + cluster_budget * 2017,
                    ),
                    True,
                )
                split_platt_heldout = apply_platt(
                    identity_heldout,
                    float(split_platt_scale[repetition]),
                    float(split_platt_shift[repetition]),
                )
                direct_beta = apply_beta(
                    identity_heldout,
                    float(beta_a[repetition]),
                    float(beta_b[repetition]),
                    float(beta_shift[repetition]),
                )
                direct_temperature = apply_temperature(
                    identity_heldout, float(inverse_temperature[repetition])
                )
                split_certificate_frame["probability_split_beta"] = apply_beta(
                    split_certificate_frame.probability_identity.to_numpy(float),
                    float(split_beta_a[repetition]),
                    float(split_beta_b[repetition]),
                    float(split_beta_shift[repetition]),
                )
                split_beta_certificate = stratified_brier_certificate(
                    split_certificate_frame,
                    "probability_split_beta",
                    args.bootstrap_repetitions,
                    frame_seed(
                        split_certificate_frame,
                        args.seed + dataset_index * 3000017 + cluster_budget * 3011,
                    ),
                    True,
                )
                split_beta_heldout = apply_beta(
                    identity_heldout,
                    float(split_beta_a[repetition]),
                    float(split_beta_b[repetition]),
                    float(split_beta_shift[repetition]),
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
                        split_platt_heldout
                        if split_platt_certificate["certified"]
                        else identity_heldout,
                        bool(split_platt_certificate["certified"]),
                    ),
                    "target_beta": (direct_beta, True),
                    "target_temperature": (direct_temperature, True),
                    "split_certified_beta": (
                        split_beta_heldout
                        if split_beta_certificate["certified"]
                        else identity_heldout,
                        bool(split_beta_certificate["certified"]),
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
                            "certificate_configurations": len(split_certificate_frame),
                            **split_platt_certificate,
                        },
                        {
                            "dataset": dataset,
                            "budget_fraction": fraction,
                            "audit_repetition": repetition,
                            "method": "split_certified_beta",
                            "fit_configurations": len(split_fits[repetition]),
                            "certificate_configurations": len(split_certificate_frame),
                            **split_beta_certificate,
                        },
                    ]
                )
                parameter_rows.extend(
                    [
                        {
                            "dataset": dataset,
                            "budget_fraction": fraction,
                            "audit_repetition": repetition,
                            "method": "target_platt",
                            "parameter_a": float(platt_scale[repetition]),
                            "parameter_b": np.nan,
                            "shift": float(platt_shift[repetition]),
                        },
                        {
                            "dataset": dataset,
                            "budget_fraction": fraction,
                            "audit_repetition": repetition,
                            "method": "split_certified_platt",
                            "parameter_a": float(split_platt_scale[repetition]),
                            "parameter_b": np.nan,
                            "shift": float(split_platt_shift[repetition]),
                        },
                        {
                            "dataset": dataset,
                            "budget_fraction": fraction,
                            "audit_repetition": repetition,
                            "method": "target_beta",
                            "parameter_a": float(beta_a[repetition]),
                            "parameter_b": float(beta_b[repetition]),
                            "shift": float(beta_shift[repetition]),
                        },
                        {
                            "dataset": dataset,
                            "budget_fraction": fraction,
                            "audit_repetition": repetition,
                            "method": "split_certified_beta",
                            "parameter_a": float(split_beta_a[repetition]),
                            "parameter_b": float(split_beta_b[repetition]),
                            "shift": float(split_beta_shift[repetition]),
                        },
                        {
                            "dataset": dataset,
                            "budget_fraction": fraction,
                            "audit_repetition": repetition,
                            "method": "target_temperature",
                            "parameter_a": float(inverse_temperature[repetition]),
                            "parameter_b": np.nan,
                            "shift": 0.0,
                        },
                    ]
                )
    fit_progress.close()

    replicates = pd.DataFrame(replicate_rows)
    legacy_summary = summarize_methods(
        replicates,
        gain_matrices,
        heldout_strata,
        LEGACY_METHODS,
        args.bootstrap_repetitions,
        args.seed,
    )
    new_summary = summarize_methods(
        replicates,
        gain_matrices,
        heldout_strata,
        NEW_METHODS,
        args.bootstrap_repetitions,
        args.seed + 424242,
    )
    summary = pd.concat([legacy_summary, new_summary], ignore_index=True)
    dataset_order = list(prepared)
    summary["dataset"] = pd.Categorical(
        summary.dataset, categories=dataset_order, ordered=True
    )
    summary["method"] = pd.Categorical(
        summary.method, categories=METHOD_ORDER, ordered=True
    )
    summary = summary.sort_values(["dataset", "budget_fraction", "method"])
    summary[["dataset", "method"]] = summary[["dataset", "method"]].astype(str)
    summary_values = summary[
        ["mean_brier_gain", "brier_gain_ci_low", "brier_gain_ci_high"]
    ].to_numpy(float)
    if not np.isfinite(summary_values).all():
        raise RuntimeError("Non-finite strong-baseline summary detected")
    reproduction = (
        verify_legacy_reproduction(summary)
        if not args.smoke
        else {"status": "skipped_in_smoke_mode"}
    )

    args.output_root.mkdir(parents=True, exist_ok=False)
    replicates.to_csv(args.output_root / "decision_curve_replicates.csv", index=False)
    summary.to_csv(args.output_root / "decision_curve_summary.csv", index=False)
    pd.DataFrame(certificate_rows).to_csv(
        args.output_root / "certification_diagnostics.csv", index=False
    )
    pd.DataFrame(parameter_rows).to_csv(
        args.output_root / "calibration_parameters.csv", index=False
    )
    pd.DataFrame(budget_rows).drop_duplicates().to_csv(
        args.output_root / "budget_design.csv", index=False
    )
    paired_comparisons(replicates).to_csv(
        args.output_root / "paired_dcs_comparisons.csv", index=False
    )
    low_label_summary(summary).to_csv(
        args.output_root / "low_label_summary.csv", index=False
    )
    plot_direct(summary, args.output_root / "fig_strong_direct_calibration")
    plot_selective(summary, args.output_root / "fig_split_certified_calibration")
    (args.output_root / "legacy_reproduction.json").write_text(
        json.dumps(reproduction, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "retrospective_strong_calibration_baselines_complete",
        "date": "2026-09-08",
        "claim_boundary": (
            "Retrospective sensitivity analysis on repeated nested subsets of the "
            "frozen v11 audit half and the reused frozen held-out half. It tests "
            "matched-budget calibration comparators; it does not alter the frozen "
            "DCS-OPCT action, release gates, failures, or external-confirmation status."
        ),
        "frozen_v11_unchanged": True,
        "frozen_v11_sha256": freeze_hash,
        "beta_formula": "sigmoid(a*log(p)-b*log(1-p)+c), a>0, b>0",
        "temperature_formula": "sigmoid(s*logit(p)), s>0",
        "budget_fractions": list(BUDGET_FRACTIONS),
        "methods": list(METHOD_ORDER),
        "audit_repetitions": args.audit_repetitions,
        "calibration_epochs": args.calibration_epochs,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "material_negative_transfer_threshold": -0.001,
        "auc_noninferiority_margin": -0.02,
        "minimum_certificate_brier_gain_lcb": RULE.minimum_certified_gain,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "gpu_name": torch.cuda.get_device_name(0),
        "legacy_reproduction": reproduction,
        "input_sha256": {
            "assignments": sha256(ASSIGNMENTS),
            "locks": sha256(LOCKS),
            "freeze_manifest": freeze_hash,
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
