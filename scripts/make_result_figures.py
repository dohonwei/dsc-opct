from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "virtual_reconstruction"
FIGURES = OUTPUT / "figures_bspc"
FIGURES.mkdir(parents=True, exist_ok=True)
DATASETS = {"deap": "DEAP", "hci": "MAHNOB-HCI"}
CHANNELS = {"M2": 2, "M4": 4, "M6": 6, "M8": 8}
METHODS = ("interpolation", "ridge", "mlp", "mc_dropout", "regional")
LABELS = {
    "interpolation": "Geometry",
    "ridge": "Ridge",
    "mlp": "MLP",
    "mc_dropout": "MC Dropout",
    "regional": "ViRecon",
}
COLORS = {
    "interpolation": "#6B7280",
    "ridge": "#009E73",
    "mlp": "#0072B2",
    "mc_dropout": "#7A5195",
    "regional": "#D55E00",
}
MARKERS = {"interpolation": "D", "ridge": "s", "mlp": "o", "mc_dropout": "x", "regional": "^"}


def _ci(values: pd.Series) -> float:
    return float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))


def reconstruction_figure() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), sharex=True)
    for column, (dataset, title) in enumerate(DATASETS.items()):
        frame = pd.read_csv(OUTPUT / f"{dataset}_loso_bspc_reconstruction_per_subject.csv")
        frame["channels"] = frame["budget"].map(CHANNELS)
        for row, (metric, ylabel) in enumerate(
            (("log_mae", "Signed-log MAE"), ("log_feature_correlation", "Feature correlation"))
        ):
            axis = axes[row, column]
            for method in METHODS:
                subset = frame[frame["method"] == method]
                summary = subset.groupby("channels")[metric].agg(["mean", _ci]).reset_index()
                axis.errorbar(
                    summary["channels"],
                    summary["mean"],
                    yerr=summary["_ci"],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linewidth=1.6,
                    markersize=5,
                    capsize=2.5,
                    label=LABELS[method],
                )
            axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xticks([2, 4, 6, 8])
            if column == 0:
                axis.set_ylabel(ylabel)
            if row == 0:
                axis.set_title(title, fontsize=11)
            if row == 1:
                axis.set_xlabel("Measured EEG channels")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save(figure, "reconstruction_metrics")


def uncertainty_summary_figure() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), sharex="row")
    width = 0.72
    offsets = {"mc_dropout": -0.38, "regional": 0.38}
    for column, (dataset, title) in enumerate(DATASETS.items()):
        selectivity = pd.read_csv(OUTPUT / f"{dataset}_loso_bspc_uncertainty_selectivity_per_subject.csv")
        selectivity["channels"] = selectivity["budget"].map(CHANNELS)
        calibration = pd.read_csv(OUTPUT / f"{dataset}_loso_bspc_calibration_error_per_subject.csv")
        calibration["channels"] = calibration["budget"].map(CHANNELS)
        for method in ("mc_dropout", "regional"):
            risk = selectivity[selectivity["method"] == method].groupby("channels")["relative_gain_at_80"].agg(
                ["mean", _ci]
            ).reset_index()
            ece = calibration[calibration["method"] == method].groupby("channels")["coverage_ece"].agg(
                ["mean", _ci]
            ).reset_index()
            x = risk["channels"].to_numpy() + offsets[method]
            axes[0, column].bar(
                x, 100 * risk["mean"], width=width, yerr=100 * risk["_ci"], capsize=2.5,
                color=COLORS[method], label=LABELS[method], alpha=0.9,
            )
            axes[1, column].bar(
                x, ece["mean"], width=width, yerr=ece["_ci"], capsize=2.5,
                color=COLORS[method], alpha=0.9,
            )
        axes[0, column].set_title(title, fontsize=11)
        axes[1, column].set_xlabel("Measured EEG channels")
        for row in range(2):
            axes[row, column].set_xticks([2, 4, 6, 8])
            axes[row, column].grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
            axes[row, column].set_axisbelow(True)
            axes[row, column].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("MAE reduction at 80% retained (%)")
    axes[1, 0].set_ylabel("Coverage ECE (lower is better)")
    axes[0, 1].legend(frameon=False, ncol=2, loc="upper right")
    figure.tight_layout()
    _save(figure, "uncertainty_summary")


def calibration_figure() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True, sharey=True)
    for axis, (dataset, title) in zip(axes, DATASETS.items()):
        frame = pd.read_csv(OUTPUT / f"{dataset}_loso_bspc_calibration_per_subject.csv")
        axis.plot([0.48, 0.97], [0.48, 0.97], color="#111827", linestyle="--", linewidth=1, label="Ideal")
        for method in ("mc_dropout", "regional"):
            summary = frame[frame["method"] == method].groupby("nominal_coverage")["empirical_coverage"].agg(
                ["mean", _ci]
            ).reset_index()
            axis.errorbar(
                summary["nominal_coverage"], summary["mean"], yerr=summary["_ci"],
                color=COLORS[method], marker=MARKERS[method], linewidth=1.8,
                markersize=5, capsize=2.5, label=LABELS[method],
            )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Nominal coverage")
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].legend(frameon=False, loc="lower right")
    figure.tight_layout()
    _save(figure, "calibration_reliability")


def _save(figure: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    reconstruction_figure()
    uncertainty_summary_figure()
    calibration_figure()
    print(f"Figures: {FIGURES}")
