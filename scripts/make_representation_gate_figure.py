from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("outputs")
INPUT = ROOT / "representation_exposure_gate_crossed_valid" / "gate_summary.csv"
OUTPUT = ROOT / "identity_audit_figures"
COLORS = {
    "absolute_power": "#0072B2",
    "relative_power": "#D55E00",
    "normalized_asymmetry": "#009E73",
}
LABELS = {
    "absolute_power": "Absolute power",
    "relative_power": "Relative power",
    "normalized_asymmetry": "Normalized asymmetry",
}
INK = "#111827"
GRID = "#D1D5DB"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    configure()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    frame.to_csv(OUTPUT / "fig_representation_gate_data.csv", index=False)
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.45), sharey=True, gridspec_kw={"wspace": 0.14}
    )
    panels = (
        (
            "subject_probe_delta_vs_all",
            "A  Identity compression",
            "Subject-probe BAcc delta",
            -0.10,
            "less identity",
        ),
        (
            "mean_joint_bacc_delta_vs_all",
            "B  Mean generalization cost",
            "Joint-unseen BAcc delta",
            -0.01,
            "gate minimum",
        ),
        (
            "mean_subject_effect_reduction_sensitive_settings",
            "C  Exposure-effect reduction",
            "Subject-effect reduction",
            0.02,
            "gate minimum",
        ),
    )
    y = range(len(frame))
    for ax, (column, title, xlabel, threshold, _) in zip(axes, panels, strict=True):
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.axvline(threshold, color="#6B7280", linewidth=1.0, linestyle="--")
        for index, row in frame.reset_index(drop=True).iterrows():
            representation = row.representation
            ax.scatter(
                row[column],
                index,
                s=48,
                color=COLORS[representation],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        ax.set_title(title, loc="left", pad=8)
        ax.set_xlabel(xlabel)
        ax.set_yticks(list(y), [LABELS[value] for value in frame.representation])
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    axes[0].set_xlim(-0.34, 0.02)
    axes[1].set_xlim(-0.055, 0.006)
    axes[2].set_xlim(-0.002, 0.024)
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUTPUT / f"fig_representation_mitigation_gate.{suffix}",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"Figure written to {(OUTPUT / 'fig_representation_mitigation_gate.pdf').resolve()}")


if __name__ == "__main__":
    main()
