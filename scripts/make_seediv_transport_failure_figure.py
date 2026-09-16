from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "ink": "#253238",
    "muted": "#66757d",
    "identity": "#4c78a8",
    "selected": "#d1495b",
    "reference": "#7a7a7a",
    "support": "#e09f3e",
    "good": "#2a9d8f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the untouched SEED-IV transport failure.")
    parser.add_argument(
        "--external-root", type=Path, default=Path("outputs/seediv_transport_policy_external")
    )
    parser.add_argument(
        "--diagnostic-root", type=Path, default=Path("outputs/seediv_transport_policy_diagnostics")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/seediv_transport_policy_figures")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decision = pd.read_csv(args.external_root / "frozen_policy_decision.csv").iloc[0]
    metrics = pd.read_csv(args.external_root / "method_performance.csv")
    intervals = pd.read_csv(args.external_root / "paired_bootstrap_differences.csv")
    support = pd.read_csv(args.diagnostic_root / "signature_support_audit.csv")
    subgroups = pd.read_csv(args.diagnostic_root / "subgroup_failure_analysis.csv")
    selected = str(decision.selected_method)
    selected_metrics = metrics.loc[metrics.method == selected].iloc[0]
    identity_metrics = metrics.loc[metrics.method == "identity"].iloc[0]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#aab3b8",
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    forecast = [float(decision.predicted_brier_gain), float(decision.predicted_auc_delta)]
    realized = [
        float(identity_metrics.brier_score - selected_metrics.brier_score),
        float(selected_metrics.roc_auc - identity_metrics.roc_auc),
    ]
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, forecast, width, color=COLORS["support"], label="Policy forecast")
    ax.bar(x + width / 2, realized, width, color=COLORS["selected"], label="Realized")
    ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
    ax.axhspan(-1, 1, color="#eef2f3", zorder=-2, label="Metric-feasible range")
    ax.set_xticks(x, ["Brier gain", "AUROC change"])
    ax.set_ylabel("Predicted or realized change")
    ax.set_title("A  Out-of-range benefit forecasts")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[0, 1]
    display = metrics.loc[~metrics.oracle_only].sort_values("brier_score", ascending=True)
    labels = [name.replace("_", " ") for name in display.method]
    colors = [
        COLORS["selected"] if name == selected else COLORS["identity"] if name == "identity" else "#9aa6ac"
        for name in display.method
    ]
    ax.barh(np.arange(len(display)), display.brier_score, color=colors)
    ax.set_yticks(np.arange(len(display)), labels)
    ax.invert_yaxis()
    ax.axvline(float(identity_metrics.brier_score), color=COLORS["identity"], linestyle="--", linewidth=1)
    ax.set_xlabel("Brier score (lower is better)")
    ax.set_title("B  Frozen action increased probability error")

    ax = axes[1, 0]
    support_summary = support.loc[support.signature_feature == "__SUMMARY__"].copy()
    support_summary = support_summary.sort_values("fraction_features_outside_public_range")
    colors = [COLORS["selected"] if name == selected else COLORS["support"] for name in support_summary.method]
    ax.barh(
        np.arange(len(support_summary)),
        100 * support_summary.fraction_features_outside_public_range,
        color=colors,
    )
    ax.set_yticks(np.arange(len(support_summary)), [name.replace("_", " ") for name in support_summary.method])
    ax.set_xlim(0, 70)
    ax.set_xlabel("Signature features outside public range (%)")
    ax.set_title("C  Real target lay outside synthetic support")

    ax = axes[1, 1]
    heat = (
        subgroups.groupby(["task", "representation"]).brier_gain.mean().unstack("representation")
    )
    heat = heat.reindex(index=["arousal", "valence"], columns=["all", "relative_power", "normalized_asymmetry"])
    limit = max(abs(float(np.nanmin(heat.values))), abs(float(np.nanmax(heat.values))))
    image = ax.imshow(heat.values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(3), ["All", "Relative power", "Norm. asymmetry"])
    ax.set_yticks(np.arange(2), ["Arousal", "Valence"])
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            value = float(heat.iloc[row, column])
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center", color="white" if abs(value) > limit * 0.55 else COLORS["ink"])
    ax.set_title("D  Brier gain by task and representation")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Identity minus CORAL Brier")

    fig.suptitle(
        "Untouched SEED-IV validation: synthetic-policy safety did not transfer",
        fontsize=14,
        fontweight="bold",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    png = args.output_root / "fig_seediv_transport_failure.png"
    pdf = args.output_root / "fig_seediv_transport_failure.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
