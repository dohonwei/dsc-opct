from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "elsarticle" / "figures"

COLORS = {
    "blue": "#2F6B9A",
    "green": "#147D70",
    "amber": "#A56824",
    "red": "#A23D43",
    "ink": "#263238",
    "muted": "#66727E",
    "line": "#CDD5DD",
    "pale": "#F5F7F9",
    "audit": "#E9F3EF",
    "holdout": "#FCEEEF",
}


def box(ax, xy, width, height, title, lines, color, fill="white", title_size=10.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        linewidth=1.25, edgecolor=color, facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + 0.014, y + height - 0.032, title, fontsize=title_size,
            fontweight="bold", color=color, va="top")
    ax.text(x + 0.014, y + height - 0.082, lines, fontsize=8.8,
            color=COLORS["ink"], va="top", linespacing=1.25)


def arrow(ax, start, end, color=None, style="-|>", linewidth=1.45):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        linewidth=linewidth, color=color or COLORS["muted"],
        connectionstyle="arc3,rad=0.0",
    ))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "DCS-OPCT evidence chain and information-access boundaries",
            ha="center", va="center", fontsize=15, fontweight="bold", color=COLORS["ink"])

    box(ax, (0.025, 0.57), 0.29, 0.29, "STAGE I  Mechanism",
        "Source predictions + labels\nCounterfactual identity dose\nMatched training; fixed test rows\nConfiguration endpoint",
        COLORS["blue"], title_size=11.5)
    box(ax, (0.355, 0.57), 0.29, 0.29, "STAGE II  Risk",
        "Source configuration labels\nFive mechanism predictors\nFrozen L2-logistic probability\nOutput: audit-risk ranking",
        COLORS["green"], title_size=11.5)
    box(ax, (0.685, 0.57), 0.29, 0.29, "STAGE III  Release",
        "Unlabeled target geometry\nCORAL + quantile candidate\nLimited-label Brier certificate\nRelease or return original",
        COLORS["red"], title_size=11.5)
    for x1, x2 in [(0.315, 0.355), (0.645, 0.685)]:
        arrow(ax, (x1, 0.715), (x2, 0.715))

    ax.text(0.025, 0.505, "PRIMARY ANALYSIS", fontsize=10.2, fontweight="bold", color=COLORS["red"])
    ax.text(0.31, 0.505, "transductive, outcome-held-out", fontsize=10.2,
            fontweight="bold", color=COLORS["ink"])
    box(ax, (0.025, 0.29), 0.28, 0.17, "Full target unlabeled batch",
        "Audit + held-out covariates enter\ntransform fitting and assignment.",
        COLORS["amber"], COLORS["pale"], 10.5)
    box(ax, (0.36, 0.29), 0.25, 0.17, "Audit partition",
        "Audit covariates fit transforms.\nAudit labels certify release.",
        COLORS["green"], COLORS["audit"], 10.5)
    box(ax, (0.665, 0.29), 0.31, 0.17, "Outcome-held-out partition",
        "Outcomes hidden until final\nevaluation. Covariates are not unseen.",
        COLORS["red"], COLORS["holdout"], 10.5)
    arrow(ax, (0.305, 0.375), (0.36, 0.375))
    arrow(ax, (0.61, 0.375), (0.665, 0.375))

    ax.text(0.025, 0.235, "STRICTER SENSITIVITY", fontsize=10.2, fontweight="bold", color=COLORS["green"])
    ax.text(0.36, 0.235, "configuration-level inductive", fontsize=10.2,
            fontweight="bold", color=COLORS["ink"])
    box(ax, (0.025, 0.06), 0.28, 0.13, "Seed-only assignment",
        "Stratum + ordered seed IDs only.\nNo target feature or outcome.",
        COLORS["blue"], COLORS["pale"], 10.5)
    box(ax, (0.36, 0.06), 0.25, 0.13, "Audit-side fitting",
        "Audit covariates fit transforms/gates.\nAudit labels certify release.",
        COLORS["green"], COLORS["audit"], 10.5)
    box(ax, (0.665, 0.06), 0.31, 0.13, "Covariate/outcome-unseen",
        "Frozen action applied once.\nRaw-group reuse can remain.",
        COLORS["red"], COLORS["holdout"], 10.0)
    arrow(ax, (0.305, 0.125), (0.36, 0.125))
    arrow(ax, (0.61, 0.125), (0.665, 0.125))

    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_v11_three_stage_framework.{suffix}", dpi=320, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
