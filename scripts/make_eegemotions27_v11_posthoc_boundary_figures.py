from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


COLORS = {
    "all": "#2A6F97",
    "relative_power": "#2A9D8F",
    "normalized_asymmetry": "#D97745",
    "linear_logistic": "#264653",
    "gpu_mlp": "#E76F51",
}
REPRESENTATION_LABELS = {
    "all": "All features",
    "relative_power": "Relative power",
    "normalized_asymmetry": "Normalized asymmetry",
}
MODEL_LABELS = {"linear_logistic": "Logistic", "gpu_mlp": "GPU MLP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication figures for the EEGEmotions-27 v11 boundary analysis."
    )
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_external_robustness"),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_posthoc_boundary_analysis"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_posthoc_boundary_analysis/figures"),
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9DEE2", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)


def save_figure(figure: plt.Figure, root: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = root / f"{stem}.{suffix}"
        figure.savefig(path, facecolor="white")
        paths.append(path.as_posix())
    plt.close(figure)
    return paths


def applicability_figure(analysis_root: Path, output_root: Path) -> list[str]:
    table = pd.read_csv(analysis_root / "applicability_thresholds.csv")
    figure, axis = plt.subplots(figsize=(4.8, 3.4))
    labels = ["CORAL + OPCT", "Quantile mapping + OPCT"]
    values = table.absolute_probability_mean_shift.to_numpy(float)
    threshold = float(table.maximum_allowed_shift.iloc[0])
    bars = axis.bar(
        labels,
        values,
        width=0.58,
        color=["#2A9D8F", "#D97745"],
        edgecolor="#263238",
        linewidth=0.7,
    )
    axis.axhline(
        threshold,
        color="#B3261E",
        linewidth=1.3,
        linestyle="--",
        label=f"Frozen maximum = {threshold:.2f}",
    )
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.3f}\n(+{value - threshold:.3f})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_ylabel("Absolute mean probability shift")
    axis.set_ylim(0, max(values) + 0.035)
    axis.set_title("Both unlabeled projections crossed the release boundary", loc="left")
    axis.legend(frameon=False, loc="upper left")
    clean_axis(axis)
    return save_figure(figure, output_root, "fig_eegemotions27_applicability_boundary")


def generalization_figure(primary_root: Path, output_root: Path) -> list[str]:
    summary = pd.read_csv(primary_root / "model_outcomes/summary.csv")
    baseline = summary.loc[summary.nominal_dose.eq(0.0)].copy()
    order = ["all", "relative_power", "normalized_asymmetry"]
    models = ["linear_logistic", "gpu_mlp"]
    figure, axis = plt.subplots(figsize=(6.7, 3.8))
    centers = np.arange(len(order), dtype=float)
    offsets = {"linear_logistic": -0.16, "gpu_mlp": 0.16}
    markers = {"linear_logistic": "o", "gpu_mlp": "s"}
    for model in models:
        for index, representation in enumerate(order):
            values = baseline.loc[
                baseline.model.eq(model)
                & baseline.representation.eq(representation),
                "unseen_balanced_accuracy",
            ].to_numpy(float)
            x = centers[index] + offsets[model]
            jitter = np.linspace(-0.045, 0.045, len(values))
            axis.scatter(
                x + jitter,
                values,
                s=22,
                marker=markers[model],
                color=COLORS[model],
                alpha=0.62,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            axis.errorbar(
                x,
                float(values.mean()),
                yerr=float(values.std(ddof=1)),
                fmt=markers[model],
                color=COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=7,
                capsize=3,
                linewidth=1.1,
                zorder=4,
            )
    axis.axhline(0.5, color="#5F6368", linestyle="--", linewidth=1.1)
    axis.text(2.46, 0.505, "Chance reference", color="#5F6368", ha="right", fontsize=8)
    axis.set_xticks(centers, [REPRESENTATION_LABELS[item] for item in order])
    axis.set_ylabel("Unseen-subject and unseen-stimulus\nbalanced accuracy")
    axis.set_ylim(0.25, 0.57)
    axis.set_title("Category polarity did not generalize across both held-out axes", loc="left")
    handles = [
        Line2D(
            [0],
            [0],
            marker=markers[model],
            color="none",
            markerfacecolor=COLORS[model],
            markeredgecolor="white",
            markersize=7,
            label=MODEL_LABELS[model],
        )
        for model in models
    ]
    axis.legend(handles=handles, frameon=False, ncol=2, loc="lower right")
    clean_axis(axis)
    return save_figure(figure, output_root, "fig_eegemotions27_dual_generalization")


def identity_affect_figure(primary_root: Path, output_root: Path) -> list[str]:
    summary = pd.read_csv(primary_root / "model_outcomes/summary.csv")
    baseline = summary.loc[summary.nominal_dose.eq(0.0)].copy()
    encoding = pd.read_csv(primary_root / "model_outcomes/identity_encoding_margin.csv")
    linear_encoding = encoding.loc[encoding.model.eq("linear"), [
        "representation",
        "split_seed",
        "encoding_margin",
    ]]
    plot = baseline.merge(
        linear_encoding,
        on=["representation", "split_seed"],
        validate="many_to_one",
    )
    figure, axis = plt.subplots(figsize=(5.4, 4.0))
    markers = {"linear_logistic": "o", "gpu_mlp": "s"}
    for (representation, model), group in plot.groupby(
        ["representation", "model"], sort=True
    ):
        axis.scatter(
            group.encoding_margin,
            group.unseen_balanced_accuracy,
            s=48,
            marker=markers[model],
            color=COLORS[representation],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.86,
        )
    axis.axhline(0.5, color="#5F6368", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Subject-identity encoding margin (linear probe)")
    axis.set_ylabel("Category-polarity balanced accuracy")
    axis.set_xlim(0.79, 0.955)
    axis.set_ylim(0.25, 0.57)
    axis.set_title("Strong identity structure coexisted with weak affect transfer", loc="left")
    color_handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=COLORS[item],
            markeredgecolor="white", markersize=7, label=REPRESENTATION_LABELS[item]
        )
        for item in ("all", "relative_power", "normalized_asymmetry")
    ]
    shape_handles = [
        Line2D(
            [0], [0], marker=markers[item], color="#4A4A4A", linestyle="none",
            markersize=6, label=MODEL_LABELS[item]
        )
        for item in ("linear_logistic", "gpu_mlp")
    ]
    first = axis.legend(handles=color_handles, frameon=False, loc="lower left")
    axis.add_artist(first)
    axis.legend(handles=shape_handles, frameon=False, loc="upper right")
    clean_axis(axis)
    return save_figure(figure, output_root, "fig_eegemotions27_identity_vs_affect")


def dose_figure(analysis_root: Path, output_root: Path) -> list[str]:
    table = pd.read_csv(analysis_root / "dose_trajectory_summary.csv")
    order = ["all", "relative_power", "normalized_asymmetry"]
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.25), sharex=True, sharey=True)
    for axis, representation in zip(axes, order, strict=True):
        group = table.loc[table.representation.eq(representation)]
        for model in ("linear_logistic", "gpu_mlp"):
            line = group.loc[group.model.eq(model)].sort_values("nominal_dose")
            x = line.nominal_dose.to_numpy(float)
            mean = line.exposure_effect_mean.to_numpy(float)
            sd = line.exposure_effect_sd.to_numpy(float)
            axis.plot(
                x,
                mean,
                color=COLORS[model],
                marker="o" if model == "linear_logistic" else "s",
                linewidth=1.5,
                markersize=4,
                label=MODEL_LABELS[model],
            )
            axis.fill_between(x, mean - sd, mean + sd, color=COLORS[model], alpha=0.13)
        axis.axhline(0.0, color="#5F6368", linewidth=0.9, linestyle="--")
        axis.set_title(REPRESENTATION_LABELS[representation])
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xlabel("Nominal identity dose")
        clean_axis(axis)
    axes[0].set_ylabel("Exposure effect (mean +/- seed SD)")
    axes[-1].legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Dose responses were heterogeneous rather than globally monotonic",
        x=0.08,
        ha="left",
        y=1.02,
        fontsize=10,
    )
    return save_figure(figure, output_root, "fig_eegemotions27_dose_heterogeneity")


def gate_flow_figure(output_root: Path) -> list[str]:
    figure, axis = plt.subplots(figsize=(10.5, 3.0))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3)
    axis.axis("off")
    boxes = [
        (0.20, 0.90, 1.75, 1.20, "Pre-signal lock", "Frozen v11\nFixed analysis grid", "#E8F1F5"),
        (2.30, 0.90, 2.05, 1.20, "Unlabeled action", "CORAL and quantile\nprojections", "#E8F4F1"),
        (4.70, 0.68, 2.45, 1.64, "Release boundary failed", "Mean shifts: 0.107, 0.128\nFrozen maximum: 0.100", "#FCE8E6"),
        (7.50, 0.90, 1.55, 1.20, "Fallback", "Identity only", "#F4EFE6"),
        (9.40, 0.68, 2.35, 1.64, "Final gate", "No action released\nEffectiveness: no\nSafety success: no", "#ECEFF1"),
    ]
    for x, y, width, height, title, detail, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            linewidth=0.9,
            edgecolor="#37474F",
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(x + 0.16, y + height - 0.27, title, fontsize=8.5, weight="bold", va="top")
        axis.text(x + 0.16, y + height - 0.60, detail, fontsize=7.6, va="top", linespacing=1.30)
    for start, end in [
        ((1.98, 1.5), (2.27, 1.5)),
        ((4.38, 1.5), (4.67, 1.5)),
        ((7.18, 1.5), (7.47, 1.5)),
        ((9.08, 1.5), (9.37, 1.5)),
    ]:
        axis.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#455A64")
        )
    axis.text(
        5.95,
        0.27,
        "Failure occurred before target labels entered the action decision",
        ha="center",
        fontsize=8.5,
        color="#B3261E",
        weight="bold",
    )
    return save_figure(figure, output_root, "fig_eegemotions27_gate_flow")


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite figure directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    setup_style()
    builders = [
        ("applicability_boundary", lambda: applicability_figure(args.analysis_root, args.output_root)),
        ("dual_generalization", lambda: generalization_figure(args.primary_root, args.output_root)),
        ("identity_vs_affect", lambda: identity_affect_figure(args.primary_root, args.output_root)),
        ("dose_heterogeneity", lambda: dose_figure(args.analysis_root, args.output_root)),
        ("gate_flow", lambda: gate_flow_figure(args.output_root)),
    ]
    manifest = {
        "status": "post_hoc_publication_figures_complete",
        "date": "2026-09-09",
        "primary_gate_changed": False,
        "figures": {},
        "claim_boundary": (
            "Figures describe a post-hoc boundary analysis and do not alter the "
            "frozen one-shot result or establish effective safe transfer."
        ),
    }
    for name, builder in tqdm(builders, desc="EEGEmotions-27 publication figures", unit="figure"):
        manifest["figures"][name] = builder()
    (args.output_root / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    captions = """# EEGEmotions-27 v11 post-hoc figure captions

All five figures are exploratory boundary analyses. They do not modify the frozen one-shot gate.

1. **Applicability boundary.** Absolute probability-mean displacement induced by each unlabeled projection after order-preserving calibration. Both exceeded the frozen 0.10 maximum, causing pre-outcome abstention.
2. **Dual generalization.** Balanced accuracy for category-derived polarity when both participants and physical video stimuli were held out. Points are the five frozen split seeds; symbols with error bars show mean and one seed-level standard deviation for description only.
3. **Identity versus affect.** Linear subject-identity encoding margin versus category-polarity balanced accuracy at dose zero. Strong identity decodability coexisted with weak dual-axis affect generalization.
4. **Dose heterogeneity.** Mean exposure effect across the five frozen seeds as identity dose increased. Shading denotes one seed-level standard deviation and is descriptive, not an independent-sample confidence interval.
5. **Gate flow.** The frozen decision path. Excessive unlabeled probability displacement triggered identity fallback before target labels entered action selection; no intervention was released, so neither effectiveness nor safety success was claimed.
"""
    (args.output_root / "figure_captions.md").write_text(captions, encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
