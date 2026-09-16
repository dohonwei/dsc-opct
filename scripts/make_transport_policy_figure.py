from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = ROOT / "outputs" / "transport_policy_public_development"
OUTPUT_ROOT = ROOT / "outputs" / "transport_policy_figures"
METHOD_ORDER = [
    "identity",
    "mean_shift",
    "coral",
    "quantile_mapping",
    "support_clipping",
    "scmt",
]
METHOD_COLORS = {
    "identity": "#8A8F98",
    "mean_shift": "#2F6B7C",
    "coral": "#D1495B",
    "quantile_mapping": "#E9A23B",
    "support_clipping": "#3C8D70",
    "scmt": "#6F5AA8",
}


def main() -> None:
    summary = pd.read_csv(POLICY_ROOT / "losfo_policy_summary.csv")
    summary = summary.loc[summary.outer_held_family != "OVERALL"].copy()
    decisions = pd.read_csv(POLICY_ROOT / "losfo_policy_decisions.csv")
    utility = pd.read_csv(POLICY_ROOT / "decision_utility_curve.csv")
    intervals = pd.read_csv(POLICY_ROOT / "policy_cluster_bootstrap_intervals.csv")
    families = summary.outer_held_family.tolist()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)

    ax = axes[0, 0]
    positions = np.arange(len(families))
    width = 0.36
    ax.barh(
        positions - width / 2,
        summary.mean_brier_gain * 1000,
        height=width,
        color="#2F6B7C",
        label="Selective policy",
    )
    ax.barh(
        positions + width / 2,
        summary.mean_fixed_brier_gain * 1000,
        height=width,
        color="#D1495B",
        label="Best fixed adapter (training folds)",
    )
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xlim(-8, 8)
    label_row = summary.loc[summary.outer_held_family == "label_shift"].iloc[0]
    ax.annotate(
        f"fixed = {label_row.mean_fixed_brier_gain * 1000:.1f}",
        xy=(-7.8, families.index("label_shift") + width / 2),
        xytext=(-7.6, families.index("label_shift") + width / 2),
        va="center",
        ha="left",
        fontsize=7.5,
        color="#FFFFFF",
        fontweight="bold",
    )
    ax.set_yticks(positions, [name.replace("_", " ") for name in families])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Brier gain vs identity (x 10^-3)")
    ax.set_title("A  Held-out shift-family performance", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[0, 1]
    counts = (
        decisions.groupby(["outer_held_family", "selected_method"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=families, columns=METHOD_ORDER, fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    bottom = np.zeros(len(families))
    for method in METHOD_ORDER:
        values = proportions[method].to_numpy()
        ax.bar(
            np.arange(len(families)),
            values,
            bottom=bottom,
            color=METHOD_COLORS[method],
            label=method.replace("_", " "),
            width=0.72,
        )
        bottom += values
    short_labels = [
        "conditional", "covariance", "label", "mean",
        "mixed", "support", "tails", "variance",
    ]
    ax.set_xticks(np.arange(len(families)), short_labels, rotation=28, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Selection proportion")
    ax.set_title("B  Adapter selection and abstention", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))

    ax = axes[1, 0]
    subset = utility.loc[utility.negative_transfer_penalty == 1.0]
    strategy_style = {
        "policy": ("#2F6B7C", "o", "Selective policy"),
        "fixed": ("#D1495B", "s", "Fixed adapter"),
        "identity": ("#8A8F98", "^", "Always abstain"),
        "oracle": ("#3C8D70", "D", "Label-informed oracle"),
    }
    for strategy, (color, marker, label) in strategy_style.items():
        rows = subset.loc[subset.strategy == strategy].sort_values("adaptation_cost")
        ax.plot(
            rows.adaptation_cost * 1000,
            rows.mean_utility * 1000,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4,
            label=label,
        )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("Per-adaptation cost (x 10^-3)")
    ax.set_ylabel("Mean decision utility (x 10^-3)")
    ax.set_title("C  Benefit-cost sensitivity", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    metrics = [
        "mean_policy_brier_gain",
        "mean_fixed_brier_gain",
        "policy_minus_fixed_gain",
        "mean_oracle_brier_gain",
    ]
    labels = ["Selective policy", "Fixed adapter", "Policy - fixed", "Oracle ceiling"]
    colors = ["#2F6B7C", "#D1495B", "#E9A23B", "#3C8D70"]
    rows = intervals.set_index("metric").loc[metrics]
    y = np.arange(len(rows))
    estimate = rows.estimate.to_numpy() * 1000
    lower = rows.ci_low.to_numpy() * 1000
    upper = rows.ci_high.to_numpy() * 1000
    ax.errorbar(
        estimate,
        y,
        xerr=np.vstack([estimate - lower, upper - estimate]),
        fmt="none",
        ecolor="#50545A",
        elinewidth=1.2,
        capsize=3,
    )
    ax.scatter(estimate, y, c=colors, s=42, zorder=3)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Brier gain (x 10^-3), cluster-bootstrap 95% CI")
    ax.set_title("D  Aggregate uncertainty", loc="left", fontweight="bold")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_ROOT / "fig_transport_policy_evidence.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_ROOT / "fig_transport_policy_evidence.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
