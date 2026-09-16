from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dcs_opct_v11_submission_artifacts"
DOSE = ROOT / "outputs" / "counterfactual_identity_dose_crossed_valid" / "summary.csv"
RANK = ROOT / "outputs" / "model_ranking_consequences_crossed_valid" / "ranking_consequences_summary.csv"

COLORS = {
    "mechanism": "#2F6B9A",
    "risk": "#15806F",
    "safety": "#B7791F",
    "ink": "#263238",
    "muted": "#64748B",
    "grid": "#D8DEE6",
    "danger": "#A63D40",
}


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def story_framework() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (0.04, COLORS["mechanism"], "1  MECHANISM", "Counterfactual identity dose", [
            "Fixed test rows and\ntraining constraints",
            "Controlled identity-label\nopportunity",
            "Utilization and ranking\nregret",
        ]),
        (0.365, COLORS["risk"], "2  RISK", "Frozen audit-risk model", [
            "Material optimism risk\n(>= 0.02)",
            "Mechanism-derived\npredictors only",
            "Ranking separated from\ncalibration",
        ]),
        (0.69, COLORS["safety"], "3  SAFETY", "Selective probability transport", [
            "CORAL action with\nquantile witness",
            "Distribution-covered\nhalf-audit",
            "Release correction or\nreturn identity",
        ]),
    ]
    width, height = 0.27, 0.59
    for x, color, eyebrow, title, bullets in stages:
        box = FancyBboxPatch(
            (x, 0.28), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.5, edgecolor=color, facecolor="white",
        )
        ax.add_patch(box)
        ax.add_patch(plt.Rectangle((x, 0.79), width, 0.08, color=color, linewidth=0))
        ax.text(x + 0.018, 0.83, eyebrow, color="white", fontsize=11, fontweight="bold", va="center")
        ax.text(x + 0.018, 0.735, title, color=COLORS["ink"], fontsize=11.2, fontweight="bold", va="top")
        y = 0.615
        for bullet in bullets:
            ax.text(x + 0.025, y, "-", color=color, fontsize=15, va="center")
            ax.text(x + 0.048, y, bullet, color=COLORS["ink"], fontsize=9.0, va="center", linespacing=1.12)
            y -= 0.135

    for x1, x2 in [(0.31, 0.365), (0.635, 0.69)]:
        ax.add_patch(FancyArrowPatch((x1, 0.605), (x2, 0.605), arrowstyle="-|>", mutation_scale=17, linewidth=1.8, color=COLORS["muted"]))

    ax.text(0.5, 0.955, "From evaluation shortcut to a bounded cross-domain audit decision", ha="center", va="center", fontsize=17, fontweight="bold", color=COLORS["ink"])
    ax.add_patch(FancyBboxPatch((0.04, 0.08), 0.92, 0.14, boxstyle="round,pad=0.01,rounding_size=0.008", facecolor="#F5F7FA", edgecolor=COLORS["grid"], linewidth=1.2))
    ax.text(0.065, 0.165, "CLAIM BOUNDARY", fontsize=9.5, fontweight="bold", color=COLORS["danger"], va="center")
    ax.text(0.225, 0.165, "Transported endpoint: probability of a configuration-level material identity-optimism event", fontsize=9.6, color=COLORS["ink"], va="center")
    ax.text(0.225, 0.115, "Not transported: trial-level emotion predictions or EEG classifier parameters", fontsize=9.6, color=COLORS["ink"], va="center")
    save(fig, "fig_v11_three_stage_framework")


def dose_and_ranking() -> None:
    dose = pd.read_csv(DOSE)
    keys = ["dataset", "task", "axis", "representation", "model", "split_seed"]
    anchors = (
        dose.loc[dose.nominal_dose.eq(0), keys + ["exposure_effect"]]
        .rename(columns={"exposure_effect": "anchor"})
    )
    work = dose.merge(anchors, on=keys, validate="many_to_one")
    work["amplification"] = work.exposure_effect - work.anchor
    work["material"] = work.amplification.ge(0.02).astype(float)
    work = work.loc[work.nominal_dose.gt(0)].copy()

    rank = pd.read_csv(RANK)
    rank = rank.loc[rank.exposure.eq("seen_subject")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), gridspec_kw={"width_ratios": [1.25, 1.05, 1.15]})
    ax0, ax1, ax2 = axes
    styles = {
        ("DEAP", "subject"): (COLORS["mechanism"], "o", "DEAP subject"),
        ("DEAP", "stimulus"): ("#6C8EBF", "s", "DEAP stimulus"),
        ("MAHNOB-HCI", "subject"): (COLORS["risk"], "^", "MAHNOB subject"),
        ("MAHNOB-HCI", "stimulus"): ("#66A89C", "D", "MAHNOB stimulus"),
    }
    for key, group in work.groupby(["dataset", "axis"], sort=True):
        color, marker, label = styles[key]
        summary = group.groupby("nominal_dose").amplification.agg(["mean", "sem"]).reset_index()
        x = summary.nominal_dose.to_numpy(float)
        y = summary["mean"].to_numpy(float)
        sem = summary["sem"].to_numpy(float)
        ax0.plot(x, y, color=color, marker=marker, linewidth=2, markersize=5, label=label)
        ax0.fill_between(x, y - sem, y + sem, color=color, alpha=0.14, linewidth=0)
    ax0.axhline(0, color=COLORS["muted"], linewidth=1)
    ax0.axhline(0.02, color=COLORS["danger"], linewidth=1.2, linestyle="--")
    ax0.set_title("A  Dose-induced amplification", loc="left", fontweight="bold")
    ax0.set_xlabel("Nominal identity-opportunity dose")
    ax0.set_ylabel("Balanced-accuracy amplification")
    ax0.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    subject = work.loc[work.axis.eq("subject")]
    for dataset, group in subject.groupby("dataset", sort=True):
        color = COLORS["mechanism"] if dataset == "DEAP" else COLORS["risk"]
        summary = group.groupby("nominal_dose").material.agg(["mean", "sem"]).reset_index()
        ax1.errorbar(summary.nominal_dose, summary["mean"], yerr=summary["sem"], color=color, marker="o", linewidth=2, capsize=3, label=dataset)
    ax1.set_ylim(-0.03, 0.65)
    ax1.set_title("B  Material-risk prevalence", loc="left", fontweight="bold")
    ax1.set_xlabel("Nominal subject dose")
    ax1.set_ylabel("Fraction with amplification >= 0.02")
    ax1.legend(frameon=False)

    labels = (rank.dataset.str.replace("MAHNOB-HCI", "MAHNOB") + "\n" + rank.task.str.capitalize()).tolist()
    x = np.arange(len(rank))
    reversal = rank.mean_material_winner_reversal.to_numpy(float)
    regret = rank.mean_deployment_regret.to_numpy(float)
    ax2.bar(x - 0.18, reversal, width=0.36, color=COLORS["safety"], label="Material reversal")
    ax2.set_ylabel("Material winner-reversal rate")
    ax2.set_ylim(0, 0.9)
    ax2.set_xticks(x, labels, fontsize=8)
    ax2b = ax2.twinx()
    ax2b.plot(x + 0.18, regret, color=COLORS["danger"], marker="o", linewidth=2, label="Joint-unseen regret")
    ax2b.set_ylabel("Deployment regret")
    ax2b.set_ylim(0, max(0.03, regret.max() * 1.35))
    ax2.set_title("C  Model-selection consequence", loc="left", fontweight="bold")
    handles = [ax2.patches[0], ax2b.lines[0]]
    ax2.legend(handles, ["Material reversal", "Joint-unseen regret"], frameon=False, fontsize=8, loc="upper right")

    for ax in axes:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax2b.spines["top"].set_visible(False)
    fig.suptitle("Counterfactual identity opportunity connects mechanism to model-selection risk", fontsize=15, fontweight="bold", color=COLORS["ink"], y=1.02)
    fig.tight_layout()
    save(fig, "fig_v11_dose_and_ranking_consequences")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    story_framework()
    dose_and_ranking()
    print("Generated two DCS-OPCT v11 story figures.")


if __name__ == "__main__":
    main()
