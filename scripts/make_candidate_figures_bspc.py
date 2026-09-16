from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virecon.statistics import ablation_comparisons, feature_group_comparisons  # noqa: E402


OUTPUT = ROOT / "outputs" / "virtual_reconstruction"
FIGURES = OUTPUT / "figures_bspc_candidate"
FIGURES.mkdir(parents=True, exist_ok=True)

DATASETS = {"deap": "DEAP", "hci": "MAHNOB-HCI"}
BUDGETS = ["M2", "M4", "M6", "M8"]
CHANNELS = {"M2": 2, "M4": 4, "M6": 6, "M8": 8}
METHOD_LABELS = {
    "interpolation": "Geometry",
    "ridge": "Ridge",
    "mlp": "MLP",
    "mc_dropout": "MC dropout",
    "regional": "ViRecon",
}
COLORS = {
    "geometry": "#6B7280",
    "ridge": "#009E73",
    "mlp": "#0072B2",
    "mc_dropout": "#7A5195",
    "regional": "#D55E00",
    "ink": "#111827",
    "muted": "#6B7280",
    "grid": "#D1D5DB",
    "paper": "#FFFFFF",
    "accent": "#0F766E",
}


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _ci(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def _save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _load_reconstruction() -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        frame = pd.read_csv(OUTPUT / f"{dataset}_loso_bspc_reconstruction_per_subject.csv")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["channels"] = out["budget"].map(CHANNELS)
    return out


def _load_paired() -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        frames.append(pd.read_csv(OUTPUT / f"{dataset}_loso_bspc_paired_statistics.csv"))
    return pd.concat(frames, ignore_index=True)


def _box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=color,
        facecolor="#FFFFFF",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h - 0.07, title, ha="center", va="top", weight="bold", color=color)
    ax.text(xy[0] + w / 2, xy[1] + h / 2 - 0.03, body, ha="center", va="center", color=COLORS["ink"], linespacing=1.25)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=COLORS["muted"],
        )
    )


def figure_protocol() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.55))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _box(ax, (0.03, 0.42), 0.20, 0.32, "Sparse input", "M2/M4/M6/M8\nnon-frontal EEG\npaper40 features", COLORS["mlp"])
    _box(ax, (0.30, 0.56), 0.18, 0.25, "Geometry prior", "inverse-square\nspatial estimate", COLORS["geometry"])
    _box(ax, (0.30, 0.24), 0.18, 0.25, "Neural base", "nonlinear MLP\nfeature mapping", COLORS["mlp"])
    _box(ax, (0.56, 0.42), 0.18, 0.32, "Validation gate", "inner-subject\nresidual acceptance\nand blend selection", COLORS["accent"])
    _box(ax, (0.80, 0.42), 0.17, 0.32, "Output", "virtual FP1/FP2\nfeature mean +\ncalibrated risk", COLORS["regional"])

    _arrow(ax, (0.23, 0.58), (0.30, 0.68))
    _arrow(ax, (0.23, 0.58), (0.30, 0.36))
    _arrow(ax, (0.48, 0.68), (0.56, 0.60))
    _arrow(ax, (0.48, 0.36), (0.56, 0.52))
    _arrow(ax, (0.74, 0.58), (0.80, 0.58))

    ax.text(0.03, 0.16, "Evaluation:", weight="bold", color=COLORS["ink"], ha="left")
    ax.text(
        0.18,
        0.16,
        "strict leave-one-subject-out on DEAP (n=32) and MAHNOB-HCI (n=27); FP1/FP2 are never used as inputs",
        color=COLORS["ink"],
        ha="left",
    )
    ax.text(0.03, 0.07, "Claim boundary:", weight="bold", color=COLORS["ink"], ha="left")
    ax.text(
        0.22,
        0.07,
        "feature-level interoperability and risk triage, not raw EEG recovery or physical electrode replacement",
        color=COLORS["ink"],
        ha="left",
    )
    _save(fig, "fig1_protocol_overview")


def figure_absolute_reconstruction() -> None:
    frame = _load_reconstruction()
    methods = ["interpolation", "ridge", "mlp", "mc_dropout", "regional"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True)
    metric_specs = [
        ("log_feature_correlation", "Feature correlation", "higher is better"),
        ("log_mae", "Signed-log MAE", "lower is better"),
    ]
    for col, (dataset, title) in enumerate(DATASETS.items()):
        for row, (metric, ylabel, subtitle) in enumerate(metric_specs):
            ax = axes[row, col]
            for method in methods:
                subset = frame[(frame["dataset"] == dataset) & (frame["method"] == method)]
                summary = subset.groupby("channels")[metric].agg(["mean", _ci]).reset_index()
                alpha = 1.0 if method in ("mlp", "mc_dropout", "regional") else 0.45
                lw = 1.8 if method in ("mlp", "mc_dropout", "regional") else 1.2
                ax.errorbar(
                    summary["channels"],
                    summary["mean"],
                    yerr=summary["_ci"],
                    marker="o",
                    linewidth=lw,
                    markersize=4,
                    capsize=2.2,
                    alpha=alpha,
                    color=COLORS.get(method, COLORS["muted"]),
                    label=METHOD_LABELS[method],
                )
            ax.set_xticks([2, 4, 6, 8])
            ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Measured channels")
            if row == 0:
                ax.set_title(title)
            ax.text(0.02, 0.92, subtitle, transform=ax.transAxes, color=COLORS["muted"], fontsize=7.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig2_reconstruction_absolute_clean")


def figure_paired_effects() -> None:
    paired = _load_paired()
    paired = paired[
        paired["family"].eq("reconstruction")
        &
        paired["contrast"].isin(["regional_vs_mlp", "regional_vs_mc_dropout"])
        & paired["metric"].isin(["log_feature_correlation", "log_mae"])
    ].copy()
    paired["dataset_label"] = paired["dataset"].map(DATASETS)
    paired["contrast_label"] = paired["contrast"].map({"regional_vs_mlp": "vs MLP", "regional_vs_mc_dropout": "vs MC dropout"})
    paired["budget_label"] = paired["budget"].str.replace("M", "", regex=False) + " ch"

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    metrics = [
        ("log_feature_correlation", "Feature-correlation gain\n(proposed - baseline)"),
        ("log_mae", "MAE gain\n(baseline - proposed)"),
    ]
    y_labels = []
    for dataset in DATASETS.values():
        for budget in BUDGETS:
            y_labels.append(f"{dataset} {budget}")
    y_pos = np.arange(len(y_labels))
    label_to_y = {label: idx for idx, label in enumerate(y_labels)}
    offsets = {"vs MLP": -0.13, "vs MC dropout": 0.13}
    colors = {"vs MLP": COLORS["mlp"], "vs MC dropout": COLORS["mc_dropout"]}

    for ax, (metric, title) in zip(axes, metrics):
        subset = paired[paired["metric"] == metric]
        for contrast in ["vs MLP", "vs MC dropout"]:
            part = subset[subset["contrast_label"] == contrast]
            ys = [label_to_y[f"{DATASETS[row.dataset]} {row.budget}"] + offsets[contrast] for row in part.itertuples()]
            ax.errorbar(
                part["mean_improvement"],
                ys,
                xerr=[part["mean_improvement"] - part["ci_low"], part["ci_high"] - part["mean_improvement"]],
                fmt="o",
                color=colors[contrast],
                markersize=4,
                linewidth=1.0,
                capsize=2,
                label=contrast,
            )
            for row, y in zip(part.itertuples(), ys):
                if row.p_holm < 0.05:
                    ax.text(row.ci_high + 0.004, y, "*", va="center", ha="left", color=colors[contrast], fontsize=9)
        ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Mean paired improvement")
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax + 0.01)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(y_labels)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.58, 0.02))
    fig.text(0.01, 0.01, "* Holm-corrected p < 0.05. Positive values favor ViRecon.", color=COLORS["muted"], fontsize=7.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    _save(fig, "fig3_paired_effects_forest")


def _heatmap(ax: plt.Axes, data: pd.DataFrame, value: str, title: str, vlim: float, cmap: str = "RdBu_r") -> None:
    matrix = data.pivot(index="target", columns="budget", values=value).reindex(columns=BUDGETS)
    im = ax.imshow(matrix.to_numpy(float), cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(BUDGETS)))
    ax.set_xticklabels(BUDGETS)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([g.upper().replace("_", "/") for g in matrix.index])
    for i, group in enumerate(matrix.index):
        for j, budget in enumerate(BUDGETS):
            row = data[(data["target"] == group) & (data["budget"] == budget)]
            if row.empty or pd.isna(matrix.loc[group, budget]):
                continue
            marker = "*" if float(row["p_holm"].iloc[0]) < 0.05 else ""
            ax.text(j, i, f"{matrix.loc[group, budget]:.3f}{marker}", ha="center", va="center", fontsize=6.4, color=COLORS["ink"])
    return im


def figure_feature_group_heatmap() -> None:
    frames = []
    for dataset in DATASETS:
        frames.append(pd.read_csv(OUTPUT / f"{dataset}_loso_bspc_feature_groups_per_subject.csv"))
    stats = feature_group_comparisons(pd.concat(frames, ignore_index=True))
    stats = stats[
        stats["metric"].eq("log_feature_correlation")
        & stats["contrast"].isin(["regional_vs_mlp", "regional_vs_mc_dropout"])
    ].copy()
    groups = ["de", "energy_pw", "psd", "asi_evi", "dasm_rasm_dcau"]
    stats["target"] = pd.Categorical(stats["target"], groups, ordered=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.15), sharex=True)
    titles = [
        ("deap", "regional_vs_mlp", "DEAP vs MLP"),
        ("deap", "regional_vs_mc_dropout", "DEAP vs MC dropout"),
        ("hci", "regional_vs_mlp", "MAHNOB-HCI vs MLP"),
        ("hci", "regional_vs_mc_dropout", "MAHNOB-HCI vs MC dropout"),
    ]
    im = None
    for ax, (dataset, contrast, title) in zip(axes.ravel(), titles):
        part = stats[(stats["dataset"] == dataset) & (stats["contrast"] == contrast)]
        im = _heatmap(ax, part, "mean_improvement", title, 0.12)
    fig.subplots_adjust(left=0.14, right=0.87, bottom=0.13, top=0.94, wspace=0.62, hspace=0.48)
    cax = fig.add_axes([0.90, 0.20, 0.018, 0.62])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Correlation gain")
    fig.text(0.01, 0.01, "* Holm-corrected p < 0.05. Values are proposed minus baseline.", color=COLORS["muted"], fontsize=7.5)
    _save(fig, "fig4_feature_group_heatmap")


def figure_ablation_heatmap() -> None:
    recon = _load_reconstruction()
    stats = ablation_comparisons(recon)
    primary = pd.concat(
        [
            stats[(stats["contrast"] == "regional_vs_regional_no_geo") & (stats["metric"] == "log_feature_correlation")].assign(endpoint="w/o geometry\ncorr."),
            stats[(stats["contrast"] == "regional_vs_regional_no_gate") & (stats["metric"] == "log_feature_correlation")].assign(endpoint="w/o gate\ncorr."),
            stats[(stats["contrast"] == "regional_vs_regional_no_uncertainty") & (stats["metric"] == "error_uncertainty_correlation")].assign(endpoint="w/o uncert.\nerr-uncert."),
            stats[(stats["contrast"] == "regional_vs_regional_no_uncertainty") & (stats["metric"] == "gaussian_nll")].assign(endpoint="w/o uncert.\nNLL"),
        ],
        ignore_index=True,
    )
    endpoints = [
        ("w/o geometry\ncorr.", "Geometry ablation: correlation gain"),
        ("w/o gate\ncorr.", "Gate ablation: correlation gain"),
        ("w/o uncert.\nerr-uncert.", "Uncertainty ablation: error-uncertainty association"),
        ("w/o uncert.\nNLL", "Uncertainty ablation: Gaussian NLL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.9), sharex=True)
    dataset_colors = {"deap": COLORS["mlp"], "hci": COLORS["regional"]}
    for ax, (endpoint, title) in zip(axes.ravel(), endpoints):
        for dataset, label in DATASETS.items():
            part = primary[(primary["dataset"] == dataset) & (primary["endpoint"] == endpoint)].copy()
            part["channels"] = part["budget"].map(CHANNELS)
            part = part.sort_values("channels")
            ax.plot(
                part["channels"],
                part["mean_improvement"],
                marker="o",
                linewidth=1.8,
                color=dataset_colors[dataset],
                label=label,
            )
            for row in part.itertuples():
                if row.p_holm < 0.05:
                    ax.text(row.channels, row.mean_improvement, "*", ha="center", va="bottom", color=dataset_colors[dataset], fontsize=9)
        ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks([2, 4, 6, 8])
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
        ax.set_ylabel("Full-model improvement")
    axes[1, 0].set_xlabel("Measured channels")
    axes[1, 1].set_xlabel("Measured channels")
    axes[0, 1].legend(frameon=False, loc="upper left")
    fig.text(0.01, 0.01, "* Holm-corrected p < 0.05. Positive values favor the full model.", color=COLORS["muted"], fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig5_ablation_evidence")


def figure_uncertainty_selective() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    methods = ["mc_dropout", "regional"]
    labels = {"mc_dropout": "MC dropout", "regional": "ViRecon"}
    colors = {"mc_dropout": COLORS["mc_dropout"], "regional": COLORS["regional"]}
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        frame = pd.read_csv(OUTPUT / f"{dataset}_loso_bspc_uncertainty_selectivity_per_subject.csv")
        frame["channels"] = frame["budget"].map(CHANNELS)
        for method in methods:
            summary = frame[frame["method"] == method].groupby("channels")["relative_gain_at_80"].agg(["mean", _ci]).reset_index()
            ax.errorbar(
                summary["channels"],
                100 * summary["mean"],
                yerr=100 * summary["_ci"],
                marker="o",
                linewidth=1.8,
                capsize=2.2,
                color=colors[method],
                label=labels[method],
            )
        ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
        ax.set_xticks([2, 4, 6, 8])
        ax.set_xlabel("Measured channels")
        ax.set_title(title)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel("MAE reduction at 80% retained (%)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig6_selective_reconstruction")


def figure_calibration_clean() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharex=True, sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        frame = pd.read_csv(OUTPUT / f"{dataset}_loso_bspc_calibration_per_subject.csv")
        ax.plot([0.5, 0.95], [0.5, 0.95], color=COLORS["ink"], linestyle="--", linewidth=1.0, label="Ideal")
        for method, color in [("mc_dropout", COLORS["mc_dropout"]), ("regional", COLORS["regional"])]:
            summary = frame[frame["method"] == method].groupby("nominal_coverage")["empirical_coverage"].agg(["mean", _ci]).reset_index()
            ax.errorbar(
                summary["nominal_coverage"],
                summary["mean"],
                yerr=summary["_ci"],
                marker="o",
                linewidth=1.8,
                markersize=3.8,
                capsize=2,
                color=color,
                label=METHOD_LABELS[method],
            )
        ax.set_title(title)
        ax.set_xlabel("Nominal coverage")
        ax.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig7_calibration_reliability_clean")


def main() -> None:
    _setup_style()
    figure_protocol()
    figure_absolute_reconstruction()
    figure_paired_effects()
    figure_feature_group_heatmap()
    figure_ablation_heatmap()
    figure_uncertainty_selective()
    figure_calibration_clean()
    print(f"Candidate figures: {FIGURES}")


if __name__ == "__main__":
    main()
