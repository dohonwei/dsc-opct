from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from amigos_v11_contract import DEFAULT_DATA_ROOT, ROOT, verify_preaccess_contract


DEFAULT_OUTPUT = ROOT / "outputs/amigos_v11_preaccess"
PREVALENCES = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
INTRACLUSTER_CORRELATIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
CLUSTERS_PER_HALF = 30
CONFIGURATIONS_PER_CLUSTER = 4
DEFAULT_REPETITIONS = 50_000
DEFAULT_SEED = 20260909


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate AMIGOS endpoint-eligibility operating characteristics "
            "without accessing participant data or changing the frozen gate."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def exact_mixed_probability(prevalence: float, independent_units: int) -> float:
    one_half = 1.0 - (1.0 - prevalence) ** independent_units - prevalence**independent_units
    return float(one_half**2)


def simulate_cell(
    prevalence: float,
    intracluster_correlation: float,
    repetitions: int,
    seed: int,
    batch_size: int = 1000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    eligible = 0
    at_least_five = 0
    total_events = np.empty(repetitions, dtype=np.int16)
    cursor = 0
    while cursor < repetitions:
        size = min(batch_size, repetitions - cursor)
        if intracluster_correlation == 0.0:
            events = rng.random(
                (
                    size,
                    2 * CLUSTERS_PER_HALF,
                    CONFIGURATIONS_PER_CLUSTER,
                )
            ) < prevalence
        elif intracluster_correlation == 1.0:
            cluster_event = rng.random((size, 2 * CLUSTERS_PER_HALF, 1)) < prevalence
            events = np.repeat(cluster_event, CONFIGURATIONS_PER_CLUSTER, axis=2)
        else:
            concentration = 1.0 / intracluster_correlation - 1.0
            alpha = prevalence * concentration
            beta = (1.0 - prevalence) * concentration
            cluster_probability = rng.beta(
                alpha, beta, size=(size, 2 * CLUSTERS_PER_HALF, 1)
            )
            events = rng.random(
                (
                    size,
                    2 * CLUSTERS_PER_HALF,
                    CONFIGURATIONS_PER_CLUSTER,
                )
            ) < cluster_probability

        audit_count = events[:, :CLUSTERS_PER_HALF].sum(axis=(1, 2))
        heldout_count = events[:, CLUSTERS_PER_HALF:].sum(axis=(1, 2))
        half_size = CLUSTERS_PER_HALF * CONFIGURATIONS_PER_CLUSTER
        mixed = (
            (audit_count > 0)
            & (audit_count < half_size)
            & (heldout_count > 0)
            & (heldout_count < half_size)
        )
        eligible += int(mixed.sum())
        at_least_five += int(((audit_count >= 5) & (heldout_count >= 5)).sum())
        total_events[cursor : cursor + size] = audit_count + heldout_count
        cursor += size

    return {
        "both_halves_mixed_probability": eligible / repetitions,
        "both_halves_at_least_five_events_probability": at_least_five / repetitions,
        "total_event_count_q05": float(np.quantile(total_events, 0.05)),
        "total_event_count_median": float(np.quantile(total_events, 0.50)),
        "total_event_count_q95": float(np.quantile(total_events, 0.95)),
    }


def make_figure(results: pd.DataFrame, output_root: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "legend.fontsize": 7.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        0.0: "#0072B2",
        0.25: "#009E73",
        0.50: "#E69F00",
        0.75: "#D55E00",
        1.0: "#7A5195",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.05), sharex=True)
    for rho, group in results.groupby("intracluster_correlation", sort=True):
        label = f"ICC = {rho:g}"
        axes[0].plot(
            100 * group.event_prevalence,
            group.both_halves_mixed_probability,
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            color=colors[float(rho)],
            label=label,
        )
        axes[1].plot(
            100 * group.event_prevalence,
            group.both_halves_at_least_five_events_probability,
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            color=colors[float(rho)],
            label=label,
        )
    axes[0].set_title("Registered endpoint eligibility")
    axes[0].set_ylabel("Probability")
    axes[1].set_title("Stability diagnostic only")
    for axis in axes:
        axis.set_xlabel("Material-event prevalence (%)")
        axis.set_ylim(-0.025, 1.025)
        axis.set_xscale("log")
        axis.set_xticks([0.25, 0.5, 1, 2, 5, 10, 20])
        axis.set_xticklabels(["0.25", "0.5", "1", "2", "5", "10", "20"])
        axis.grid(axis="y", color="#D8DEE6", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="lower right")
    fig.text(
        0.5,
        -0.01,
        "60 four-dose clusters; 30 audit and 30 held out. The five-event panel is not an acceptance gate.",
        ha="center",
        color="#4B5563",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_root / f"amigos_v11_endpoint_eligibility.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    reservation, _, freeze = verify_preaccess_contract()
    if DEFAULT_DATA_ROOT.exists():
        raise RuntimeError(
            "This planning analysis must be completed before the AMIGOS archive is present."
        )
    if args.repetitions < 10_000:
        raise ValueError("At least 10,000 repetitions are required")

    cells = [
        (prevalence, rho)
        for prevalence in PREVALENCES
        for rho in INTRACLUSTER_CORRELATIONS
    ]
    rows = []
    progress = tqdm(cells, desc="AMIGOS eligibility simulation", unit="cell", dynamic_ncols=True)
    for index, (prevalence, rho) in enumerate(progress):
        metrics = simulate_cell(
            prevalence,
            rho,
            args.repetitions,
            args.seed + index * 1009,
        )
        if rho == 0.0:
            exact = exact_mixed_probability(
                prevalence, CLUSTERS_PER_HALF * CONFIGURATIONS_PER_CLUSTER
            )
        elif rho == 1.0:
            exact = exact_mixed_probability(prevalence, CLUSTERS_PER_HALF)
        else:
            exact = np.nan
        rows.append(
            {
                "event_prevalence": prevalence,
                "intracluster_correlation": rho,
                "clusters_per_half": CLUSTERS_PER_HALF,
                "configurations_per_cluster": CONFIGURATIONS_PER_CLUSTER,
                "repetitions": args.repetitions,
                "exact_boundary_probability": exact,
                **metrics,
            }
        )
    progress.close()

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    csv_path = args.output_root / "endpoint_eligibility_operating_characteristics.csv"
    results.to_csv(csv_path, index=False)
    make_figure(results, args.output_root)

    boundary_error = results.loc[
        results.intracluster_correlation.isin([0.0, 1.0])
    ].copy()
    boundary_error["absolute_error"] = (
        boundary_error.both_halves_mixed_probability
        - boundary_error.exact_boundary_probability
    ).abs()
    report = {
        "status": "preaccess_planning_analysis_complete",
        "date": "2026-09-09",
        "dataset": "AMIGOS",
        "participant_values_accessed": False,
        "archive_present_during_execution": False,
        "registered_design": {
            "configuration_clusters": 60,
            "configurations_per_cluster": CONFIGURATIONS_PER_CLUSTER,
            "audit_clusters": CLUSTERS_PER_HALF,
            "heldout_clusters": CLUSTERS_PER_HALF,
            "material_event_threshold": reservation["counterfactual_contract"][
                "material_event_threshold"
            ],
            "endpoint_eligibility": reservation["acceptance_gate"][
                "endpoint_eligibility"
            ],
        },
        "simulation": {
            "model": "exchangeable beta-binomial four-dose clusters",
            "event_prevalences": list(PREVALENCES),
            "intracluster_correlations": list(INTRACLUSTER_CORRELATIONS),
            "repetitions_per_cell": args.repetitions,
            "seed": args.seed,
            "maximum_exact_boundary_absolute_error": float(
                boundary_error.absolute_error.max()
            ),
        },
        "interpretation_boundary": (
            "This analysis quantifies only the probability that both partitions contain "
            "event and non-event configurations under hypothetical prevalence and "
            "within-cluster dependence. It does not estimate AMIGOS prevalence, "
            "calibration power, applicability, effectiveness, non-harm, or safety."
        ),
        "non_gate_diagnostic": (
            "The probability that both halves contain at least five events is reported "
            "for planning context only and is not a registered acceptance criterion."
        ),
        "freeze_id": freeze["freeze_id"],
    }
    (args.output_root / "endpoint_eligibility_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "AMIGOS pre-access endpoint-eligibility analysis complete; "
        f"maximum exact-boundary error={report['simulation']['maximum_exact_boundary_absolute_error']:.4f}"
    )


if __name__ == "__main__":
    main()
