from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path("outputs")
MATCHED = ROOT / "paired_block_identity_exposure_benchmark_crossed_valid"
PROBES = ROOT / "crossed_identity_probes_crossed_valid"
THRESHOLD = ROOT / "paired_block_threshold_sensitivity"
FIGURES = ROOT / "identity_audit_figures"
DATASET_ORDER = ["DEAP", "MAHNOB-HCI"]
TASK_ORDER = ["arousal", "valence"]
IDENTITY_COLORS = {"subject": "#0072B2", "stimulus": "#D55E00"}
EFFECT_COLORS = {
    "subject_exposure_main_effect": "#0072B2",
    "stimulus_exposure_main_effect": "#D55E00",
    "subject_by_stimulus_interaction": "#6B7280",
}
INK = "#111827"
MUTED = "#6B7280"
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


def save(fig: mpl.figure.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def dataset_task_labels() -> list[str]:
    return [f"{dataset}\n{task.capitalize()}" for dataset in DATASET_ORDER for task in TASK_ORDER]


def opportunity_frame() -> pd.DataFrame:
    frame = pd.read_csv(MATCHED / "identity_opportunity_summary.csv")
    frame = frame.loc[frame.dataset.isin(DATASET_ORDER)].copy()
    frame["identity"] = frame.exposure.map(
        {"seen_subject": "subject", "seen_stimulus": "stimulus"}
    )
    frame["label"] = frame.dataset + "\n" + frame.task.str.capitalize()
    frame["identity_prior_above_chance"] = frame.identity_prior_balanced_accuracy - 0.5
    return frame


def encoding_frame() -> pd.DataFrame:
    frame = pd.read_csv(PROBES / "summary.csv")
    frame = frame.loc[frame.dataset.isin(DATASET_ORDER)].copy()
    frame["identity"] = frame.axis.map(
        {"subject_across_stimuli": "subject", "stimulus_across_subjects": "stimulus"}
    )
    return (
        frame.groupby(["dataset", "identity"], as_index=False)
        .agg(
            above_chance=("above_chance", "mean"),
            above_chance_min=("above_chance", "min"),
            above_chance_max=("above_chance", "max"),
        )
    )


def utilization_frame() -> pd.DataFrame:
    frame = pd.read_csv(THRESHOLD / "factorial_effects_all_thresholds.csv")
    frame = frame.loc[
        frame.dataset.isin(DATASET_ORDER)
        & (frame.strategy == "ge5_primary")
        & frame.contrast.isin(
            ["subject_exposure_main_effect", "stimulus_exposure_main_effect"]
        )
    ].copy()
    frame["identity"] = frame.contrast.map(
        {
            "subject_exposure_main_effect": "subject",
            "stimulus_exposure_main_effect": "stimulus",
        }
    )
    frame["mean_balanced_accuracy_delta"] = frame["balanced_accuracy_effect"]
    frame["label"] = frame.dataset + "\n" + frame.task.str.capitalize()
    return frame


def plot_grouped_points(
    ax: mpl.axes.Axes,
    frame: pd.DataFrame,
    categories: list[str],
    category_column: str,
    value_column: str,
    title: str,
    ylabel: str,
    ci_columns: tuple[str, str] | None = None,
) -> None:
    x = np.arange(len(categories), dtype=float)
    offsets = {"subject": -0.14, "stimulus": 0.14}
    markers = {"subject": "o", "stimulus": "s"}
    for identity in ("subject", "stimulus"):
        subset = frame.loc[frame.identity == identity].set_index(category_column).reindex(categories)
        values = subset[value_column].to_numpy(float)
        positions = x + offsets[identity]
        if ci_columns is not None:
            low = subset[ci_columns[0]].to_numpy(float)
            high = subset[ci_columns[1]].to_numpy(float)
            errors = np.vstack([values - low, high - values])
            ax.errorbar(
                positions,
                values,
                yerr=errors,
                fmt=markers[identity],
                ms=5,
                mfc=IDENTITY_COLORS[identity],
                mec="white",
                mew=0.5,
                color=IDENTITY_COLORS[identity],
                ecolor=IDENTITY_COLORS[identity],
                elinewidth=1.1,
                capsize=2,
                label=identity.capitalize(),
                zorder=3,
            )
        else:
            ax.scatter(
                positions,
                values,
                s=28,
                marker=markers[identity],
                color=IDENTITY_COLORS[identity],
                edgecolor="white",
                linewidth=0.5,
                label=identity.capitalize(),
                zorder=3,
            )
            if "above_chance_min" in subset:
                low = subset.above_chance_min.to_numpy(float)
                high = subset.above_chance_max.to_numpy(float)
                ax.vlines(positions, low, high, color=IDENTITY_COLORS[identity], linewidth=1.2, zorder=2)
    ax.axhline(0, color=INK, linewidth=0.8, zorder=1)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=INK, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)


def make_evidence_chain() -> None:
    opportunity = opportunity_frame()
    encoding = encoding_frame()
    utilization = utilization_frame()
    opportunity.to_csv(FIGURES / "fig_evidence_chain_opportunity.csv", index=False)
    encoding.to_csv(FIGURES / "fig_evidence_chain_encoding.csv", index=False)
    utilization.to_csv(FIGURES / "fig_evidence_chain_utilization.csv", index=False)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.55),
        gridspec_kw={"width_ratios": [1.45, 1.0, 1.45], "wspace": 0.40},
    )
    task_categories = dataset_task_labels()
    plot_grouped_points(
        axes[0],
        opportunity,
        task_categories,
        "label",
        "identity_prior_above_chance",
        "A  Label opportunity",
        "Identity-prior BAcc above 0.5",
    )
    plot_grouped_points(
        axes[1],
        encoding,
        DATASET_ORDER,
        "dataset",
        "above_chance",
        "B  Cross-axis decoding",
        "Probe BAcc above chance",
    )
    plot_grouped_points(
        axes[2],
        utilization,
        task_categories,
        "label",
        "mean_balanced_accuracy_delta",
        "C  Emotion-model utilization",
        "Matched exposure BAcc effect",
        ("ci_low", "ci_high"),
    )
    axes[0].set_ylim(-0.035, 0.39)
    axes[1].set_ylim(-0.035, 0.97)
    axes[2].set_ylim(-0.08, 0.14)
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(32 if ax is not axes[1] else 22)
            tick.set_ha("right")
    save(fig, "fig_identity_evidence_chain")


def make_factorial_effects() -> None:
    frame = pd.read_csv(THRESHOLD / "factorial_effects_all_thresholds.csv")
    frame = frame.loc[
        (frame.strategy == "ge5_primary") & frame.dataset.isin(DATASET_ORDER)
    ].copy()
    frame["label"] = frame.dataset + "\n" + frame.task.str.capitalize()
    frame.to_csv(FIGURES / "fig_factorial_effects_data.csv", index=False)
    contrast_order = list(EFFECT_COLORS)
    titles = {
        "subject_exposure_main_effect": "A  Subject exposure",
        "stimulus_exposure_main_effect": "B  Stimulus exposure",
        "subject_by_stimulus_interaction": "C  Exposure interaction",
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.55), sharey=True, gridspec_kw={"wspace": 0.12}
    )
    categories = dataset_task_labels()
    y = np.arange(len(categories))
    for ax, contrast in zip(axes, contrast_order, strict=True):
        subset = frame.loc[frame.contrast == contrast].set_index("label").reindex(categories)
        values = subset.balanced_accuracy_effect.to_numpy(float)
        low = subset.ci_low.to_numpy(float)
        high = subset.ci_high.to_numpy(float)
        significant = subset.p_holm_current_dataset_task_family.to_numpy(float) < 0.05
        ax.axvline(0, color=INK, linewidth=0.8)
        for index in range(len(categories)):
            color = EFFECT_COLORS[contrast]
            ax.plot([low[index], high[index]], [y[index], y[index]], color=color, linewidth=1.5)
            ax.scatter(
                values[index],
                y[index],
                s=40 if significant[index] else 30,
                color=color if significant[index] else "white",
                edgecolor=color,
                linewidth=1.2,
                zorder=3,
            )
        ax.set_title(titles[contrast], loc="left", pad=8)
        ax.set_xlabel("Balanced-accuracy effect")
        ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-0.085, 0.165)
    axes[0].set_yticks(y, categories)
    axes[0].invert_yaxis()
    save(fig, "fig_matched_factorial_effects")


def make_threshold_sensitivity() -> None:
    frame = pd.read_csv(THRESHOLD / "factorial_effect_threshold_stability.csv")
    frame = frame.loc[
        frame.dataset.isin(DATASET_ORDER)
        & frame.contrast.isin(
            ["subject_exposure_main_effect", "stimulus_exposure_main_effect"]
        )
    ].copy()
    frame.to_csv(FIGURES / "fig_threshold_sensitivity_data.csv", index=False)
    strategy_order = ["ge5_primary", "gt5", "exclude5"]
    strategy_labels = {
        "ge5_primary": "Score >= 5",
        "gt5": "Score > 5",
        "exclude5": "Exclude score = 5",
    }
    strategy_colors = {
        "ge5_primary": "#111827",
        "gt5": "#0072B2",
        "exclude5": "#009E73",
    }
    markers = {"ge5_primary": "o", "gt5": "s", "exclude5": "^"}
    titles = {
        "subject_exposure_main_effect": "A  Subject exposure",
        "stimulus_exposure_main_effect": "B  Stimulus exposure",
    }
    categories = dataset_task_labels()
    y = np.arange(len(categories), dtype=float)
    offsets = {"ge5_primary": -0.18, "gt5": 0.0, "exclude5": 0.18}
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.7), sharey=True, gridspec_kw={"wspace": 0.12}
    )
    for ax, contrast in zip(axes, titles, strict=True):
        subset = frame.loc[frame.contrast == contrast]
        ax.axvline(0, color=INK, linewidth=0.8)
        for strategy in strategy_order:
            values = (
                subset.loc[subset.strategy == strategy]
                .set_index(subset.loc[subset.strategy == strategy].dataset + "\n" + subset.loc[subset.strategy == strategy].task.str.capitalize())
                .reindex(categories)
            )
            estimates = values.balanced_accuracy_effect.to_numpy(float)
            low = values.ci_low.to_numpy(float)
            high = values.ci_high.to_numpy(float)
            positions = y + offsets[strategy]
            ax.errorbar(
                estimates,
                positions,
                xerr=np.vstack([estimates - low, high - estimates]),
                fmt=markers[strategy],
                ms=4.8,
                mfc=strategy_colors[strategy],
                mec="white",
                mew=0.5,
                color=strategy_colors[strategy],
                ecolor=strategy_colors[strategy],
                elinewidth=1.0,
                capsize=1.8,
                label=strategy_labels[strategy],
                zorder=3,
            )
        ax.set_title(titles[contrast], loc="left", pad=8)
        ax.set_xlabel("Balanced-accuracy effect")
        ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-0.055, 0.145)
    axes[0].set_yticks(y, categories)
    axes[0].invert_yaxis()
    save(fig, "fig_threshold_sensitivity")


def make_design_overview() -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.55),
        gridspec_kw={"width_ratios": [1.0, 1.45, 1.25], "wspace": 0.34},
    )

    # Panel A: one crossed test cell and its three candidate training quadrants.
    ax = axes[0]
    ax.set_title("A  Crossed test cell", loc="left", pad=5)
    cell_specs = (
        (0, 0, "#E5E7EB", "$\\mathcal{C}_{00}$\nneither"),
        (1, 0, "#D9ECF6", "$\\mathcal{C}_{10}$\nsubject"),
        (0, 1, "#FBE7D9", "$\\mathcal{C}_{01}$\nstimulus"),
        (1, 1, "#111827", "$\\mathcal{T}_{ab}$\ntest"),
    )
    for x, y, color, label in cell_specs:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.2))
        ax.text(
            x + 0.5,
            y + 0.5,
            label,
            ha="center",
            va="center",
            color="white" if color == "#111827" else INK,
            fontsize=7.1,
        )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], [r"$S\ne a$", r"$S=a$"])
    ax.set_yticks([0.5, 1.5], [r"$V\ne b$", r"$V=b$"])
    ax.tick_params(length=0)
    ax.set_xlabel("Subject fold")
    ax.set_ylabel("Stimulus fold")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Panel B: all four fits retain the same core and replace two matched slots.
    ax = axes[1]
    ax.set_title("B  Label-matched replacement", loc="left", pad=5)
    rows = (
        (r"$\mathcal{D}_{00}$", r"$\mathcal{Q}_{S}$", r"$\mathcal{Q}_{V}$", "#BAE0F2", "#F5C8A8"),
        (r"$\mathcal{D}_{10}$", r"$\mathcal{B}_{S}$", r"$\mathcal{Q}_{V}$", IDENTITY_COLORS["subject"], "#F5C8A8"),
        (r"$\mathcal{D}_{01}$", r"$\mathcal{Q}_{S}$", r"$\mathcal{B}_{V}$", "#BAE0F2", IDENTITY_COLORS["stimulus"]),
        (r"$\mathcal{D}_{11}$", r"$\mathcal{B}_{S}$", r"$\mathcal{B}_{V}$", IDENTITY_COLORS["subject"], IDENTITY_COLORS["stimulus"]),
    )
    for row, (condition, subject_slot, stimulus_slot, subject_color, stimulus_color) in enumerate(rows):
        y = 3.4 - row * 0.82
        ax.text(0.0, y + 0.19, condition, ha="right", va="center", fontsize=7.2)
        blocks = (
            (0.05, 0.50, "#D1D5DB", r"$\mathcal{K}$", INK),
            (0.56, 0.19, subject_color, subject_slot, INK if subject_slot == r"$\mathcal{Q}_{S}$" else "white"),
            (0.76, 0.19, stimulus_color, stimulus_slot, INK if stimulus_slot == r"$\mathcal{Q}_{V}$" else "white"),
        )
        for x, width, color, label, text_color in blocks:
            ax.add_patch(Rectangle((x, y), width, 0.38, facecolor=color, edgecolor="white", linewidth=0.7))
            ax.text(x + width / 2, y + 0.19, label, ha="center", va="center", color=text_color, fontsize=6.8)
    ax.text(0.50, 0.12, "Fixed size and class counts", ha="center", va="bottom", color=MUTED, fontsize=6.8)
    ax.set_xlim(-0.18, 1.0)
    ax.set_ylim(0, 4.15)
    ax.axis("off")

    # Panel C: the four matched fits identify two marginal exposure effects.
    ax = axes[2]
    ax.set_title("C  Factorial estimands", loc="left", pad=5)
    points = {(0, 0): r"$M_{00}$", (1, 0): r"$M_{01}$", (0, 1): r"$M_{10}$", (1, 1): r"$M_{11}$"}
    for (x, y), label in points.items():
        ax.scatter(x, y, s=44, facecolor="white", edgecolor=INK, linewidth=1.0, zorder=3)
        ax.text(x, y + 0.13, label, ha="center", va="bottom", fontsize=7.1)
    for y in (0, 1):
        ax.annotate("", xy=(0.90, y), xytext=(0.10, y), arrowprops={"arrowstyle": "->", "color": IDENTITY_COLORS["stimulus"], "lw": 1.3})
    for x in (0, 1):
        ax.annotate("", xy=(x, 0.90), xytext=(x, 0.10), arrowprops={"arrowstyle": "->", "color": IDENTITY_COLORS["subject"], "lw": 1.3})
    ax.text(0.50, -0.22, r"average horizontal change $=\Delta_V$", ha="center", color=IDENTITY_COLORS["stimulus"], fontsize=6.7)
    ax.text(-0.26, 0.50, r"average vertical change $=\Delta_S$", ha="center", va="center", rotation=90, color=IDENTITY_COLORS["subject"], fontsize=6.7)
    ax.text(0.50, -0.60, "Fixed across fits: test rows, training n,\nclass counts, and common core", ha="center", va="top", color=MUTED, fontsize=6.6)
    ax.set_xlim(-0.36, 1.20)
    ax.set_ylim(-0.82, 1.28)
    ax.axis("off")

    save(fig, "fig_paired_factorial_design")


def main() -> None:
    configure()
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_design_overview()
    make_evidence_chain()
    make_factorial_effects()
    make_threshold_sensitivity()
    print(f"Figures written to {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
