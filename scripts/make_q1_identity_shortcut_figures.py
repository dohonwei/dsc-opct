from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


OUTPUTS = Path("outputs")
COUNTERFACTUAL = OUTPUTS / "counterfactual_identity_dose_crossed_valid"
RANKING = OUTPUTS / "model_ranking_consequences_crossed_valid"
TRANSPORT = OUTPUTS / "eppvr_risk_transport_diagnostics"
FIGURES = OUTPUTS / "q1_identity_shortcut_figures"

DATASETS = ["DEAP", "MAHNOB-HCI"]
TASKS = ["arousal", "valence"]
AXES = ["subject", "stimulus"]
EXPOSURES = ["seen_subject", "seen_stimulus", "seen_both"]
COLORS = {
    "DEAP": "#0072B2",
    "MAHNOB-HCI": "#D55E00",
    "arousal": "#0072B2",
    "valence": "#009E73",
    "seen_subject": "#0072B2",
    "seen_stimulus": "#D55E00",
    "seen_both": "#009E73",
    "observed": "#0072B2",
    "predicted": "#D55E00",
}
INK = "#111827"
MUTED = "#6B7280"
GRID = "#D1D5DB"
LIGHT = "#E5E7EB"


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


def strip_axes(ax: mpl.axes.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)


def counterfactual_amplification() -> pd.DataFrame:
    frame = pd.read_csv(COUNTERFACTUAL / "summary.csv")
    keys = ["dataset", "task", "axis", "representation", "model", "split_seed"]
    anchors = frame.loc[
        frame.nominal_dose == 0,
        keys + ["exposure_effect"],
    ].rename(columns={"exposure_effect": "dose_zero_exposure_effect"})
    frame = frame.merge(anchors, on=keys, validate="many_to_one")
    frame["dose_induced_amplification"] = (
        frame.exposure_effect - frame.dose_zero_exposure_effect
    )
    return frame


def make_counterfactual_dose_figure() -> None:
    frame = counterfactual_amplification()
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FIGURES / "fig_counterfactual_dose_data.csv", index=False)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.35),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.34, "wspace": 0.18},
    )
    panel = 0
    for row, axis_name in enumerate(AXES):
        for column, dataset in enumerate(DATASETS):
            ax = axes[row, column]
            subset = frame.loc[
                (frame.dataset == dataset) & (frame.axis == axis_name)
            ].copy()
            trajectory_keys = ["task", "representation", "model", "split_seed"]
            for _, trajectory in subset.groupby(trajectory_keys):
                trajectory = trajectory.sort_values("nominal_dose")
                ax.plot(
                    trajectory.nominal_dose,
                    trajectory.dose_induced_amplification,
                    color=LIGHT,
                    linewidth=0.55,
                    alpha=0.55,
                    zorder=1,
                )
            for task in TASKS:
                task_frame = subset.loc[subset.task == task]
                summary = (
                    task_frame.groupby("nominal_dose")
                    .dose_induced_amplification.mean()
                    .reset_index()
                )
                ax.plot(
                    summary.nominal_dose,
                    summary.dose_induced_amplification,
                    color=COLORS[task],
                    marker="o" if task == "arousal" else "s",
                    markersize=4.5,
                    linewidth=1.6,
                    label=task.capitalize(),
                    zorder=3,
                )
            ax.axhline(0, color=INK, linewidth=0.8, zorder=2)
            ax.axhline(0.02, color=MUTED, linewidth=0.8, linestyle="--", zorder=2)
            ax.set_title(
                f"{chr(65 + panel)}  {dataset}: {axis_name} exposure",
                loc="left",
                pad=7,
            )
            panel += 1
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_ylim(-0.065, 0.085)
            strip_axes(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel("Nominal identity-opportunity dose")
    for ax in axes[:, 0]:
        ax.set_ylabel("Dose-induced BAcc optimism")
    axes[0, 0].legend(frameon=False, loc="upper left", ncol=2)
    axes[0, 1].text(
        0.98,
        0.02,
        "Dashed line: material threshold",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=6.6,
    )
    save(fig, "fig_counterfactual_dose_response")


def make_model_ranking_figure() -> None:
    frame = pd.read_csv(RANKING / "ranking_consequences_summary.csv")
    frame.to_csv(FIGURES / "fig_model_ranking_data.csv", index=False)
    dataset_labels = {"DEAP": "DEAP", "MAHNOB-HCI": "MAHNOB"}
    categories = [
        f"{dataset_labels[dataset]}\n{task.capitalize()}"
        for dataset in DATASETS
        for task in TASKS
    ]
    frame["category"] = frame.dataset.map(dataset_labels) + "\n" + frame.task.str.capitalize()
    offsets = {"seen_subject": -0.22, "seen_stimulus": 0.0, "seen_both": 0.22}
    labels = {
        "seen_subject": "Subject exposed",
        "seen_stimulus": "Stimulus exposed",
        "seen_both": "Both exposed",
    }
    markers = {"seen_subject": "o", "seen_stimulus": "s", "seen_both": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.30})
    x = np.arange(len(categories), dtype=float)
    specifications = [
        (
            "mean_material_winner_reversal",
            "material_winner_reversal_ci_low",
            "material_winner_reversal_ci_high",
            "A  Material winner reversal",
            "Reversal probability",
        ),
        (
            "mean_deployment_regret",
            "deployment_regret_ci_low",
            "deployment_regret_ci_high",
            "B  Joint-unseen deployment regret",
            "Balanced-accuracy regret",
        ),
    ]
    for ax, (value, low, high, title, ylabel) in zip(axes, specifications, strict=True):
        for exposure in EXPOSURES:
            subset = (
                frame.loc[frame.exposure == exposure]
                .set_index("category")
                .reindex(categories)
            )
            estimates = subset[value].to_numpy(float)
            errors = np.vstack(
                [
                    estimates - subset[low].to_numpy(float),
                    subset[high].to_numpy(float) - estimates,
                ]
            )
            ax.errorbar(
                x + offsets[exposure],
                estimates,
                yerr=errors,
                fmt=markers[exposure],
                color=COLORS[exposure],
                markersize=5,
                linewidth=0,
                elinewidth=1.1,
                capsize=2,
                label=labels[exposure],
            )
        ax.set_xticks(x, categories)
        ax.set_title(title, loc="left", pad=7)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", length=0)
        strip_axes(ax)
    axes[0].set_ylim(-0.04, 1.06)
    axes[1].axhline(0.02, color=MUTED, linewidth=0.8, linestyle="--")
    axes[1].set_ylim(-0.003, 0.045)
    axes[0].legend(frameon=False, loc="upper right")
    save(fig, "fig_model_ranking_consequences")


def make_external_transport_figure() -> None:
    calibration = pd.read_csv(TRANSPORT / "calibration_bins.csv")
    shift = pd.read_csv(TRANSPORT / "predictor_shift.csv")
    calibration.to_csv(FIGURES / "fig_external_transport_calibration_data.csv", index=False)
    shift.to_csv(FIGURES / "fig_external_transport_shift_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), gridspec_kw={"wspace": 0.32})

    x = calibration.risk_bin.to_numpy(float)
    observed = calibration.observed_event_rate.to_numpy(float)
    observed_errors = np.vstack(
        [
            observed - calibration.observed_event_rate_ci_low.to_numpy(float),
            calibration.observed_event_rate_ci_high.to_numpy(float) - observed,
        ]
    )
    predicted = calibration.mean_predicted_probability.to_numpy(float)
    axes[0].errorbar(
        x,
        observed,
        yerr=observed_errors,
        color=COLORS["observed"],
        marker="o",
        markersize=4.8,
        linewidth=1.5,
        capsize=2,
        label="Observed event rate",
    )
    axes[0].plot(
        x,
        predicted,
        color=COLORS["predicted"],
        marker="s",
        markersize=4.3,
        linewidth=1.3,
        linestyle="--",
        label="Mean predicted risk",
    )
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Frozen-score quintile")
    axes[0].set_ylabel("Material optimism probability")
    axes[0].set_ylim(0, 0.75)
    axes[0].set_title("A  Ranking and enrichment", loc="left", pad=7)
    axes[0].legend(frameon=False, loc="upper left")
    strip_axes(axes[0])

    predicted_low = calibration.mean_predicted_probability_ci_low.to_numpy(float)
    predicted_high = calibration.mean_predicted_probability_ci_high.to_numpy(float)
    axes[1].plot([0, 0.75], [0, 0.75], color=MUTED, linewidth=0.8, linestyle="--")
    axes[1].errorbar(
        predicted,
        observed,
        xerr=np.vstack([predicted - predicted_low, predicted_high - predicted]),
        yerr=observed_errors,
        fmt="o",
        color=COLORS["observed"],
        markersize=5,
        elinewidth=1.0,
        capsize=2,
    )
    for row in calibration.itertuples(index=False):
        axes[1].annotate(
            str(row.risk_bin),
            (row.mean_predicted_probability, row.observed_event_rate),
            xytext=(4, 3),
            textcoords="offset points",
            color=INK,
            fontsize=6.5,
        )
    axes[1].set_xlim(0, 0.75)
    axes[1].set_ylim(0, 0.75)
    axes[1].set_xlabel("Mean predicted risk")
    axes[1].set_ylabel("Observed event rate")
    axes[1].set_title("B  Absolute calibration", loc="left", pad=7)
    strip_axes(axes[1], grid_axis="both")

    save(fig, "fig_external_risk_transport")
    make_predictor_support_figure(shift)


def make_predictor_support_figure(shift: pd.DataFrame) -> None:
    labels = {
        "opportunity_delta": "Opportunity increase",
        "metadata_prior_opportunity": "Metadata-only prior",
        "encoding_margin": "Identity-encoding margin",
        "dose_encoding_interaction": "Dose x encoding",
        "model_capacity": "Ordinal model capacity",
    }
    y = np.arange(len(shift))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    for position, row in zip(y, shift.itertuples(index=False), strict=True):
        denominator = row.public_max - row.public_min
        q025 = (row.eppvr_q025 - row.public_min) / denominator
        median = (row.eppvr_median - row.public_min) / denominator
        q975 = (row.eppvr_q975 - row.public_min) / denominator
        ax.add_patch(
            Rectangle(
                (0, position - 0.18),
                1,
                0.36,
                facecolor=LIGHT,
                edgecolor="none",
                zorder=1,
            )
        )
        ax.plot([q025, q975], [position, position], color=INK, linewidth=1.5, zorder=2)
        ax.scatter(
            median,
            position,
            s=31,
            color="#009E73",
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.axvline(1, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, [labels[value] for value in shift.feature])
    ax.set_xlim(-0.45, 1.9)
    ax.set_xlabel(
        "EPPVR quantiles normalized to public min-max support "
        "(gray: public range; line/dot: 2.5-97.5%/median)"
    )
    ax.set_title("Predictor support shift in the locked EPPVR validation", loc="left", pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    save(fig, "fig_external_predictor_support")


def make_subgroup_forest() -> None:
    frame = pd.read_csv(TRANSPORT / "subgroup_performance.csv")
    frame.to_csv(FIGURES / "fig_external_subgroup_data.csv", index=False)
    names = {
        "arousal": "Task: arousal",
        "valence": "Task: valence",
        "all": "Representation: all features",
        "baseline_delta": "Representation: baseline delta",
        "normalized_asymmetry": "Representation: asymmetry",
        "relative_power": "Representation: relative power",
        "gpu_mlp": "Model: GPU MLP",
        "linear_logistic": "Model: linear logistic",
        "0.25": "Nominal dose: 0.25",
        "0.5": "Nominal dose: 0.50",
        "0.75": "Nominal dose: 0.75",
        "1.0": "Nominal dose: 1.00",
    }
    family_colors = {
        "task": "#0072B2",
        "representation": "#009E73",
        "model": "#CC79A7",
        "nominal_dose": "#D55E00",
    }
    frame["label"] = frame.subgroup.map(names)
    frame = frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(7.2, 4.15))
    for index, row in frame.iterrows():
        color = family_colors[row.subgroup_family]
        ax.plot(
            [row.roc_auc_ci_low, row.roc_auc_ci_high],
            [y[index], y[index]],
            color=color,
            linewidth=1.4,
        )
        ax.scatter(
            row.roc_auc,
            y[index],
            s=32,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.text(
            0.97,
            y[index],
            f"{row.roc_auc:.2f}  ({row.n_events:.0f} events)",
            ha="right",
            va="center",
            fontsize=6.8,
            color=INK,
        )
    ax.axvline(0.5, color=INK, linewidth=0.8)
    ax.axvline(0.65, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yticks(y, frame.label)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("External AUROC with configuration-cluster 95% CI")
    ax.set_title("EPPVR subgroup sensitivity of the frozen audit-priority score", loc="left", pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    save(fig, "fig_external_risk_subgroups")


def main() -> None:
    configure()
    FIGURES.mkdir(parents=True, exist_ok=True)
    make_counterfactual_dose_figure()
    make_model_ranking_figure()
    make_external_transport_figure()
    make_subgroup_forest()
    print(f"Wrote Q1 identity-shortcut figures to {FIGURES.resolve()}")


if __name__ == "__main__":
    main()
