from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


PREVALENCES = (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
INTRACLUSTER_CORRELATIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
CONFIGURATIONS_PER_CLUSTER = 4


def exact_mixed_probability(prevalence: float, independent_units: int) -> float:
    if not 0.0 < prevalence < 1.0 or independent_units < 2:
        raise ValueError("Prevalence must lie in (0, 1) and units must be at least two")
    one_half = 1.0 - (1.0 - prevalence) ** independent_units - prevalence**independent_units
    return float(one_half**2)


def simulate_cell(
    prevalence: float,
    intracluster_correlation: float,
    clusters_per_half: int,
    repetitions: int,
    seed: int,
    *,
    configurations_per_cluster: int = CONFIGURATIONS_PER_CLUSTER,
    batch_size: int = 1000,
) -> dict[str, float]:
    if not 0.0 < prevalence < 1.0:
        raise ValueError("Event prevalence must lie strictly between zero and one")
    if not 0.0 <= intracluster_correlation <= 1.0:
        raise ValueError("Intracluster correlation must lie in [0, 1]")
    if clusters_per_half < 1 or configurations_per_cluster < 2:
        raise ValueError("The cluster design is too small")
    if repetitions < 1 or batch_size < 1:
        raise ValueError("Repetitions and batch size must be positive")

    rng = np.random.default_rng(seed)
    eligible = 0
    at_least_five = 0
    total_events = np.empty(repetitions, dtype=np.int16)
    cursor = 0
    total_clusters = 2 * clusters_per_half
    half_size = clusters_per_half * configurations_per_cluster
    while cursor < repetitions:
        size = min(batch_size, repetitions - cursor)
        shape = (size, total_clusters, configurations_per_cluster)
        if intracluster_correlation == 0.0:
            events = rng.random(shape) < prevalence
        elif intracluster_correlation == 1.0:
            cluster_event = rng.random((size, total_clusters, 1)) < prevalence
            events = np.repeat(cluster_event, configurations_per_cluster, axis=2)
        else:
            concentration = 1.0 / intracluster_correlation - 1.0
            alpha = prevalence * concentration
            beta = (1.0 - prevalence) * concentration
            cluster_probability = rng.beta(
                alpha, beta, size=(size, total_clusters, 1)
            )
            events = rng.random(shape) < cluster_probability

        audit_count = events[:, :clusters_per_half].sum(axis=(1, 2))
        heldout_count = events[:, clusters_per_half:].sum(axis=(1, 2))
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


def simulate_grid(
    *,
    dataset: str,
    clusters_per_half: int,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    cells = [
        (prevalence, rho)
        for prevalence in PREVALENCES
        for rho in INTRACLUSTER_CORRELATIONS
    ]
    rows = []
    progress = tqdm(
        cells,
        desc=f"{dataset} eligibility simulation",
        unit="cell",
        dynamic_ncols=True,
    )
    for index, (prevalence, rho) in enumerate(progress):
        metrics = simulate_cell(
            prevalence,
            rho,
            clusters_per_half,
            repetitions,
            seed + index * 1009,
        )
        if rho == 0.0:
            exact = exact_mixed_probability(
                prevalence, clusters_per_half * CONFIGURATIONS_PER_CLUSTER
            )
        elif rho == 1.0:
            exact = exact_mixed_probability(prevalence, clusters_per_half)
        else:
            exact = np.nan
        rows.append(
            {
                "event_prevalence": prevalence,
                "intracluster_correlation": rho,
                "clusters_per_half": clusters_per_half,
                "configurations_per_cluster": CONFIGURATIONS_PER_CLUSTER,
                "repetitions": repetitions,
                "exact_boundary_probability": exact,
                **metrics,
            }
        )
    progress.close()
    return pd.DataFrame(rows)


def maximum_boundary_error(results: pd.DataFrame) -> float:
    boundary = results.loc[
        results.intracluster_correlation.isin([0.0, 1.0])
    ].copy()
    error = (
        boundary.both_halves_mixed_probability
        - boundary.exact_boundary_probability
    ).abs()
    return float(error.max())


def make_figure(
    results: pd.DataFrame,
    output_root: Path,
    stem: str,
    footer: str,
) -> None:
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
    fig.text(0.5, -0.01, footer, ha="center", color="#4B5563", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"{stem}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)
