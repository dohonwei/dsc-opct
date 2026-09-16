from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DOSE_ROOT = ROOT / "outputs/faced_v11_post_access_common_montage_dose"
DCS_ROOT = ROOT / "outputs/faced_v11_post_access_common_montage_dcs_opct"
OUTPUT_ROOT = ROOT / "outputs/faced_v11_post_access_common_montage_artifacts_v2"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
MATERIAL_THRESHOLD = 0.02
KEYS = ["representation", "model", "split_seed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def identity_summary(identity: pd.DataFrame) -> pd.DataFrame:
    return (
        identity.groupby(["representation", "model"], as_index=False)
        .agg(
            n_features=("n_features", "first"),
            chance=("chance", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sd=("balanced_accuracy", "std"),
            encoding_margin_mean=("encoding_margin", "mean"),
        )
        .sort_values(["representation", "model"])
    )


def dose_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = summary.loc[
        summary.nominal_dose.eq(0), KEYS + ["exposure_effect"]
    ].rename(columns={"exposure_effect": "dose_zero_exposure_effect"})
    nonzero = summary.loc[summary.nominal_dose.gt(0)].merge(
        anchors, on=KEYS, validate="many_to_one"
    )
    nonzero["dose_induced_amplification"] = (
        nonzero.exposure_effect - nonzero.dose_zero_exposure_effect
    )
    nonzero["material_optimism_event"] = (
        nonzero.dose_induced_amplification >= MATERIAL_THRESHOLD
    ).astype(int)
    rows = []
    for key, block in summary.groupby(KEYS, sort=True):
        rho, _ = spearmanr(block.nominal_dose, block.exposure_effect)
        rows.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "dose_response_spearman": float(rho),
                "dose_response_monotone_increasing": bool(rho >= 0.9),
                "dose_response_monotone_decreasing": bool(rho <= -0.9),
            }
        )
    monotonicity = pd.DataFrame(rows)
    grouped = (
        nonzero.groupby(["representation", "model"], as_index=False)
        .agg(
            mean_amplification=("dose_induced_amplification", "mean"),
            sd_amplification=("dose_induced_amplification", "std"),
            min_amplification=("dose_induced_amplification", "min"),
            max_amplification=("dose_induced_amplification", "max"),
            material_events=("material_optimism_event", "sum"),
            configurations=("material_optimism_event", "size"),
        )
        .merge(
            monotonicity.groupby(["representation", "model"], as_index=False).agg(
                mean_spearman=("dose_response_spearman", "mean"),
                monotone_increasing_seeds=("dose_response_monotone_increasing", "sum"),
                monotone_decreasing_seeds=("dose_response_monotone_decreasing", "sum"),
            ),
            on=["representation", "model"],
            validate="one_to_one",
        )
    )
    return nonzero, grouped


def make_figure(
    identity: pd.DataFrame,
    summary: pd.DataFrame,
    nonzero: pd.DataFrame,
    components: pd.DataFrame,
    output_root: Path,
) -> None:
    style()
    rep_order = ["all", "relative_power", "normalized_asymmetry"]
    rep_labels = ["All", "Relative power", "Norm. asymmetry"]
    colors = {"linear": "#2673A8", "rbf": "#D65F4C"}
    model_colors = {"linear_logistic": "#2673A8", "gpu_mlp": "#D65F4C"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    ax = axes[0, 0]
    x = np.arange(len(rep_order))
    for offset, model in zip((-0.16, 0.16), ("linear", "rbf"), strict=True):
        block = (
            identity.loc[identity.model.eq(model)]
            .groupby("representation")
            .balanced_accuracy.agg(["mean", "std"])
            .reindex(rep_order)
        )
        ax.errorbar(
            x + offset,
            block["mean"],
            yerr=block["std"],
            fmt="o",
            capsize=2.5,
            color=colors[model],
            label=model.upper(),
        )
    ax.axhline(identity.chance.mean(), color="#444444", ls="--", lw=1, label="Chance")
    ax.set_xticks(x, rep_labels)
    ax.set_ylabel("Identity-probe balanced accuracy")
    ax.set_ylim(0, 1.04)
    ax.set_title("A  Participant identity is strongly decodable", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower left")

    ax = axes[0, 1]
    for (representation, model), block in summary.groupby(
        ["representation", "model"], sort=True
    ):
        line = block.groupby("nominal_dose").exposure_effect.mean()
        ax.plot(
            line.index,
            line.values,
            marker="o",
            ms=3,
            lw=1.2,
            color=model_colors[model],
            alpha=0.9 if representation == "all" else 0.45,
            ls="-" if representation == "all" else ("--" if representation == "relative_power" else ":"),
        )
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Nominal identity-opportunity dose")
    ax.set_ylabel("Exposure effect")
    ax.set_title("B  Dose utilization is heterogeneous", loc="left", weight="bold")
    model_legend = ax.legend(
        handles=[
            Line2D([0], [0], color=model_colors["linear_logistic"], lw=1.5, label="Logistic"),
            Line2D([0], [0], color=model_colors["gpu_mlp"], lw=1.5, label="GPU MLP"),
        ],
        frameon=False,
        loc="upper right",
        title="Classifier",
        title_fontsize=7.5,
    )
    ax.add_artist(model_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#555555", lw=1.3, ls="-", label="All"),
            Line2D([0], [0], color="#555555", lw=1.3, ls="--", label="Relative"),
            Line2D([0], [0], color="#555555", lw=1.3, ls=":", label="Asymmetry"),
        ],
        frameon=False,
        loc="lower left",
        title="Representation",
        title_fontsize=7.5,
    )

    ax = axes[1, 0]
    bins = np.linspace(-0.02, 0.022, 18)
    ax.hist(
        nonzero.dose_induced_amplification,
        bins=bins,
        color="#4C956C",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0, color="#555555", lw=0.8)
    ax.axvline(MATERIAL_THRESHOLD, color="#B33A3A", ls="--", lw=1.2)
    ax.set_xlabel("Dose-induced amplification vs dose 0")
    ax.set_ylabel("Configuration count")
    ax.set_title("C  No material event at the frozen threshold", loc="left", weight="bold")
    ax.text(
        MATERIAL_THRESHOLD - 0.0008,
        ax.get_ylim()[1] * 0.94,
        "0.02 threshold",
        ha="right",
        va="top",
        color="#B33A3A",
        fontsize=7,
    )

    ax = axes[1, 1]
    shifts = components.set_index("method").probability_mean_shift.abs()
    labels = ["CORAL", "Quantile"]
    values = [shifts["coral"], shifts["quantile_mapping"]]
    bars = ax.bar(labels, values, color=["#2673A8", "#D65F4C"], width=0.58)
    ax.axhline(0.10, color="#B33A3A", ls="--", lw=1.2)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center")
    ax.set_ylim(0, max(values) * 1.28)
    ax.set_ylabel("Absolute mean probability shift")
    ax.set_title("D  Frozen applicability gate rejects action", loc="left", weight="bold")
    ax.text(0.98, 0.96, "limit = 0.10", transform=ax.transAxes, ha="right", va="top", color="#B33A3A")

    fig.savefig(output_root / "fig_faced_post_access_external_boundary.pdf", bbox_inches="tight")
    fig.savefig(output_root / "fig_faced_post_access_external_boundary.png", bbox_inches="tight")
    plt.close(fig)


def make_latex_table(dose_summary: pd.DataFrame, output_root: Path) -> None:
    labels = {
        "all": "All",
        "relative_power": "Relative power",
        "normalized_asymmetry": "Normalized asymmetry",
        "linear_logistic": "Logistic",
        "gpu_mlp": "GPU MLP",
    }
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Representation & Classifier & Mean $\Delta$ & Min $\Delta$ & Max $\Delta$ & Events \\",
        r"\midrule",
    ]
    for row in dose_summary.itertuples(index=False):
        lines.append(
            f"{labels[row.representation]} & {labels[row.model]} & "
            f"{row.mean_amplification:.4f} & {row.min_amplification:.4f} & "
            f"{row.max_amplification:.4f} & {int(row.material_events)}/{int(row.configurations)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_root / "table_faced_post_access_dose.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    identity_path = DOSE_ROOT / "identity_encoding_margin.csv"
    summary_path = DOSE_ROOT / "summary.csv"
    component_path = DCS_ROOT / "component_projection_diagnostics.csv"
    gate_path = DCS_ROOT / "external_confirmation_gate.json"
    identity = pd.read_csv(identity_path)
    summary = pd.read_csv(summary_path)
    components = pd.read_csv(component_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    identity_table = identity_summary(identity)
    nonzero, dose_summary = dose_tables(summary)
    identity_table.to_csv(OUTPUT_ROOT / "faced_identity_encoding_summary.csv", index=False)
    dose_summary.to_csv(OUTPUT_ROOT / "faced_dose_amplification_summary.csv", index=False)
    nonzero.to_csv(OUTPUT_ROOT / "faced_configuration_outcomes.csv", index=False)
    components.to_csv(OUTPUT_ROOT / "faced_dcs_applicability.csv", index=False)
    make_figure(identity, summary, nonzero, components, OUTPUT_ROOT)
    make_latex_table(dose_summary, OUTPUT_ROOT)

    report = {
        "status": "faced_post_access_exploratory_artifacts_complete",
        "evidence_role": "post_access_exploratory_external_stress_test_only",
        "confirmatory_claim_permitted": False,
        "participants": 123,
        "trials": 2952,
        "nonzero_dose_configurations": int(len(nonzero)),
        "material_threshold": MATERIAL_THRESHOLD,
        "material_events": int(nonzero.material_optimism_event.sum()),
        "maximum_dose_induced_amplification": float(
            nonzero.dose_induced_amplification.max()
        ),
        "minimum_dose_induced_amplification": float(
            nonzero.dose_induced_amplification.min()
        ),
        "identity_probe_balanced_accuracy_range": [
            float(identity.balanced_accuracy.min()),
            float(identity.balanced_accuracy.max()),
        ],
        "chance_identity_accuracy": float(identity.chance.mean()),
        "dcs_candidate_method": gate["candidate_method"],
        "dcs_selected_method": gate["selected_method"],
        "dcs_claim_supported": bool(gate["claim_supported"]),
        "interpretation": (
            "FACED shows strong participant decodability without material "
            "dose-induced optimism at the frozen endpoint. The frozen DCS-OPCT "
            "applicability gate rejected both transported components; this is "
            "an exploratory boundary result, not external confirmation."
        ),
        "source_sha256": {
            str(identity_path.relative_to(ROOT)): sha256(identity_path),
            str(summary_path.relative_to(ROOT)): sha256(summary_path),
            str(component_path.relative_to(ROOT)): sha256(component_path),
            str(gate_path.relative_to(ROOT)): sha256(gate_path),
            str(FREEZE.relative_to(ROOT)): sha256(FREEZE),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT_ROOT / "faced_post_access_evidence_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
