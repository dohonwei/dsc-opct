from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "outputs" / "distribution_covered_stratified_opct_v11_development"
AVDOS_GATE = ROOT / "outputs" / "avdos_v11_exploratory_ppg"
AVDOS_DOSE = ROOT / "outputs" / "avdos_exploratory_ppg_dose"
FACED = ROOT / "outputs" / "faced_v11_post_access_common_montage_artifacts_v2"
FREEZE = ROOT / "docs" / "distribution_covered_stratified_opct_v11_final_freeze.json"
OUT = ROOT / "outputs" / "dcs_opct_v11_submission_artifacts_crossfit_v11"
STRONG = ROOT / "outputs" / "dcs_opct_v11_strong_calibration_baselines"
CROSSFIT = ROOT / "outputs" / "dcs_opct_v11_crossfit_certified_baselines"
PAIRED = ROOT / "outputs" / "dcs_opct_v11_paired_inference_crossfit"
ENGINEERING = ROOT / "outputs" / "material_threshold_engineering_anchor"
COMPONENT = ROOT / "outputs" / "dcs_opct_v11_leave_one_component"

DATASET_ORDER = ["EPPVR", "CASE", "CEAP", "SEED-IV", "DREAMER", "AVDOS-VR"]
ADAPT_COLOR = "#0072B2"
ABSTAIN_COLOR = "#7A7F87"
EXPLORATORY_COLOR = "#D55E00"
PASS_COLOR = "#009E73"
FAIL_COLOR = "#C44E52"
INK = "#17202A"
MUTED = "#667085"
GRID = "#D8DEE6"
LIGHT = "#EEF2F6"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_action_matrix() -> pd.DataFrame:
    locks = pd.read_csv(DEV / "unlabeled_action_locks.csv").set_index("dataset")
    certs = pd.read_csv(DEV / "primary_certificates.csv").set_index("dataset")
    heldout = pd.read_csv(DEV / "primary_heldout_results.csv").set_index("dataset")
    rows: list[dict] = []
    for dataset in DATASET_ORDER[:-1]:
        lock = locks.loc[dataset]
        cert = certs.loc[dataset]
        test = heldout.loc[dataset]
        adapted = bool(test.adapted)
        rows.append(
            {
                "dataset": dataset,
                "evidence_role": "retrospective development",
                "selected_action": "CORAL + OPCT" if adapted else "identity",
                "applicable": bool(lock.applicable),
                "audit_certificate": bool(cert.certified),
                "audit_brier_gain": cert.brier_gain,
                "audit_brier_gain_lcb": cert.brier_gain_lcb,
                "heldout_brier_gain": float(test.brier_gain),
                "auc_delta": float(test.auc_delta),
                "safety_pass": bool(
                    not test.material_negative_transfer and test.auc_noninferior
                ),
                "effectiveness_pass": bool(adapted and test.brier_gain > 0.001),
                "outcome": (
                    "certified adaptation" if adapted else "rule-based non-intervention"
                ),
                "failure_reason": "" if adapted else str(cert.failure_reason),
            }
        )

    avdos_gate = read_json(AVDOS_GATE / "external_confirmation_gate.json")
    rows.append(
        {
            "dataset": "AVDOS-VR",
            "evidence_role": "post-access exploratory",
            "selected_action": "identity",
            "applicable": False,
            "audit_certificate": False,
            "audit_brier_gain": np.nan,
            "audit_brier_gain_lcb": np.nan,
            "heldout_brier_gain": float(avdos_gate["heldout_metrics"]["brier_gain"]),
            "auc_delta": float(avdos_gate["heldout_metrics"]["auc_delta"]),
            "safety_pass": bool(avdos_gate["safety_pass"]),
            "effectiveness_pass": bool(avdos_gate["effectiveness_pass"]),
            "outcome": "exploratory non-intervention",
            "failure_reason": "unlabeled_action_inapplicable",
        }
    )
    return pd.DataFrame(rows)


def build_gate_diagnostics() -> pd.DataFrame:
    components = pd.read_csv(DEV / "component_projection_diagnostics.csv")
    locks = pd.read_csv(DEV / "unlabeled_action_locks.csv")
    frame = components.merge(
        locks[
            [
                "dataset",
                "directional_agreement",
                "displacement_cosine",
                "normalized_disagreement",
                "witness_pass",
            ]
        ],
        on="dataset",
        validate="many_to_one",
    )
    avdos = read_json(AVDOS_GATE / "primary_action_and_audit_lock.json")
    avdos_rows = []
    for item in avdos["component_diagnostics"]:
        avdos_rows.append(
            {
                "dataset": "AVDOS-VR",
                "method": item["method"],
                "available": True,
                "scale": item["scale"],
                "shift": item["shift"],
                "final_loss": item["final_loss"],
                "probability_mean_shift": item["probability_mean_shift"],
                "probability_rank": item["probability_rank"],
                "decision_flip_rate": item["decision_flip_rate"],
                "order_inversions": item["order_inversions"],
                "applicable": False,
                "directional_agreement": np.nan,
                "displacement_cosine": np.nan,
                "normalized_disagreement": np.nan,
                "witness_pass": False,
            }
        )
    return pd.concat([frame, pd.DataFrame(avdos_rows)], ignore_index=True)


def build_avdos_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(AVDOS_DOSE / "summary.csv")
    keys = ["dataset", "task", "axis", "representation", "model", "split_seed"]
    anchors = summary.loc[
        summary.nominal_dose == 0, keys + ["exposure_effect"]
    ].rename(columns={"exposure_effect": "dose_zero_exposure_effect"})
    summary = summary.merge(anchors, on=keys, validate="many_to_one")
    summary["dose_induced_amplification"] = (
        summary.exposure_effect - summary.dose_zero_exposure_effect
    )
    dose = (
        summary.groupby(["representation", "model", "nominal_dose"], as_index=False)
        .dose_induced_amplification.agg(["mean", "std", "count"])
        .reset_index()
    )
    dose["sem"] = dose["std"] / np.sqrt(dose["count"])
    encoding = pd.read_csv(AVDOS_DOSE / "identity_encoding_margin.csv")
    encoding = (
        encoding.groupby(["representation", "model"], as_index=False)
        .encoding_margin.agg(["mean", "std", "count"])
        .reset_index()
    )
    encoding["sem"] = encoding["std"] / np.sqrt(encoding["count"])
    return dose, encoding


def make_action_outcome_figure(matrix: pd.DataFrame) -> None:
    columns = ["Applicable", "Certified", "Effective", "Released action\nnon-harm"]
    value_columns = [
        "applicable",
        "audit_certificate",
        "effectiveness_pass",
        "safety_pass",
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.15))
    ax.set_xlim(-0.65, len(columns) + 3.45)
    ax.set_ylim(-0.65, len(matrix) - 0.35)
    for row in range(len(matrix)):
        if row % 2 == 0:
            ax.add_patch(
                Rectangle(
                    (-0.65, row - 0.5),
                    len(columns) + 4.1,
                    1,
                    facecolor="#F8FAFC",
                    edgecolor="none",
                    zorder=0,
                )
            )
        for col, value_column in enumerate(value_columns):
            released = matrix.iloc[row].selected_action != "identity"
            evaluable = col < 3 or released
            passed = bool(matrix.iloc[row][value_column]) if evaluable else False
            ax.scatter(
                col,
                row,
                s=260,
                marker="s",
                facecolor=PASS_COLOR if passed else LIGHT,
                edgecolor=PASS_COLOR if passed else "#AAB2BD",
                linewidth=1.0,
                zorder=2,
            )
            ax.text(
                col,
                row,
                "PASS" if passed else "N/A",
                ha="center",
                va="center",
                color="white" if passed else MUTED,
                fontsize=6.3,
                fontweight="bold",
                zorder=3,
            )
        outcome = matrix.iloc[row].outcome
        outcome_color = (
            ADAPT_COLOR
            if outcome == "certified adaptation"
            else EXPLORATORY_COLOR
            if "exploratory" in outcome
            else ABSTAIN_COLOR
        )
        ax.text(4.15, row, outcome, ha="left", va="center", color=outcome_color)
        gain = matrix.iloc[row].heldout_brier_gain
        ax.text(7.00, row, f"{gain:+.3f}", ha="center", va="center", color=INK)
    ax.set_xticks(list(range(len(columns))) + [4.15, 7.00])
    ax.set_xticklabels(columns + ["Decision", "Held-out\nBrier gain"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks(range(len(matrix)), matrix.dataset)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Cross-domain audit-risk decisions across six datasets",
        loc="left",
        pad=30,
        fontweight="bold",
    )
    ax.text(
        0,
        -0.11,
        "Non-harm is shown only for released corrections; identity fallback is non-intervention, not a safety success.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=7.0,
    )
    save_figure(fig, "fig_v11_six_dataset_action_matrix")


def make_gain_figure(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.axvspan(-0.002, 0.001, color="#F5E6E8", alpha=0.75, zorder=0)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.axvline(0.001, color=FAIL_COLOR, linewidth=0.9, linestyle="--")
    for index, row in matrix.iterrows():
        if np.isfinite(row.audit_brier_gain):
            ax.plot(
                [row.audit_brier_gain_lcb, row.audit_brier_gain],
                [index, index],
                color=ADAPT_COLOR,
                linewidth=1.8,
                solid_capstyle="round",
            )
            ax.scatter(row.audit_brier_gain, index, color=ADAPT_COLOR, s=38, zorder=3)
            ax.plot(
                row.audit_brier_gain_lcb,
                index,
                marker="|",
                color=ADAPT_COLOR,
                markersize=9,
                markeredgewidth=1.4,
                zorder=3,
            )
        held_color = ADAPT_COLOR if row.effectiveness_pass else ABSTAIN_COLOR
        if row.dataset == "AVDOS-VR":
            held_color = EXPLORATORY_COLOR
        ax.scatter(row.heldout_brier_gain, index, color=held_color, marker="D", s=38, zorder=4)
        label_y = index + (0.16 if row.dataset == "CEAP" else 0.0)
        ax.text(
            row.heldout_brier_gain + 0.0015,
            label_y,
            f"{row.heldout_brier_gain:+.3f}",
            va="center",
            fontsize=7.0,
            color=held_color,
        )
    ax.set_yticks(np.arange(len(matrix)), matrix.dataset)
    ax.invert_yaxis()
    ax.set_xlim(-0.003, 0.054)
    ax.set_xlabel("Brier gain for audit-risk probabilities vs identity input")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(
        "Audit-risk calibration certificate and held-out benefit",
        loc="left",
        fontweight="bold",
    )
    handles = [
        Line2D([0], [0], color=ADAPT_COLOR, marker="o", label="Audit gain; line begins at one-sided LCB"),
        Line2D([0], [0], color=INK, marker="D", linewidth=0, label="Held-out gain"),
        Line2D([0], [0], color=FAIL_COLOR, linestyle="--", label="Certification threshold = 0.001"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    save_figure(fig, "fig_v11_audit_and_heldout_gain")


def make_support_gate_figure(diagnostics: pd.DataFrame) -> None:
    locks = diagnostics.drop_duplicates("dataset").set_index("dataset")
    components = diagnostics.pivot(
        index="dataset", columns="method", values="probability_mean_shift"
    ).reindex(DATASET_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"wspace": 0.36})

    ax = axes[0]
    ax.add_patch(
        Rectangle(
            (0.9, 0),
            0.11,
            0.35,
            facecolor="#E5F4EE",
            edgecolor=PASS_COLOR,
            linewidth=0.8,
            zorder=0,
        )
    )
    label_offsets = {
        "EPPVR": (5, 8),
        "CASE": (5, 7),
        "CEAP": (-25, 8),
        "DREAMER": (5, 7),
    }
    for dataset in ["EPPVR", "CASE", "CEAP", "DREAMER"]:
        row = locks.loc[dataset]
        color = ADAPT_COLOR if bool(row.applicable) else ABSTAIN_COLOR
        ax.scatter(
            row.directional_agreement,
            row.normalized_disagreement,
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            dataset,
            (row.directional_agreement, row.normalized_disagreement),
            xytext=label_offsets[dataset],
            textcoords="offset points",
            fontsize=7.0,
        )
    ax.axvline(0.9, color=PASS_COLOR, linewidth=0.8, linestyle="--")
    ax.axhline(0.35, color=PASS_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xlim(0.80, 1.01)
    ax.set_ylim(0, 0.43)
    ax.set_xlabel("Directional agreement")
    ax.set_ylabel("Normalized disagreement")
    ax.set_title("A  Witness-consensus geometry", loc="left", fontweight="bold")
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.905, 0.025, "admissible", color=PASS_COLOR, fontsize=6.8, fontweight="bold")

    ax = axes[1]
    y = np.arange(len(DATASET_ORDER))
    width = 0.31
    coral = components["coral"].abs().to_numpy(float)
    quantile = components["quantile_mapping"].abs().to_numpy(float)
    ax.barh(y - width / 2, coral, height=width, color=ADAPT_COLOR, label="CORAL primary")
    ax.barh(y + width / 2, quantile, height=width, color="#E69F00", label="Quantile witness")
    ax.axvline(0.10, color=FAIL_COLOR, linewidth=0.9, linestyle="--")
    ax.set_yticks(y, DATASET_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.40)
    ax.set_xlabel("Absolute audit-risk probability mean shift")
    ax.set_title("B  Component displacement budget", loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.105, -0.52, "maximum = 0.10", color=FAIL_COLOR, fontsize=6.7)
    save_figure(fig, "fig_v11_support_gate_geometry")


def make_avdos_figure(dose: pd.DataFrame, encoding: pd.DataFrame) -> None:
    representation_order = ["raw_ppg", "heart_ppg", "minimal_ppg", "heart_rate_signal"]
    labels = ["Raw PPG", "Heart + PPG", "Minimal PPG", "Heart-rate signal"]
    model_order = ["linear", "rbf"]
    model_colors = {"linear": ADAPT_COLOR, "rbf": EXPLORATORY_COLOR}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    x = np.arange(len(representation_order))
    offsets = {"linear": -0.15, "rbf": 0.15}
    for model in model_order:
        subset = encoding.loc[encoding.model == model].set_index("representation").reindex(representation_order)
        ax.errorbar(
            x + offsets[model],
            subset["mean"],
            yerr=1.96 * subset["sem"],
            fmt="o" if model == "linear" else "s",
            color=model_colors[model],
            capsize=2.5,
            linewidth=1.1,
            markersize=4.5,
            label=model.upper(),
        )
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, labels, rotation=24, ha="right")
    ax.set_ylabel("Identity-encoding margin above chance")
    ax.set_title("A  Subject identity remains encoded", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    ax = axes[1]
    colors = ["#0072B2", "#009E73", "#CC79A7", "#E69F00"]
    for representation, label, color in zip(representation_order, labels, colors, strict=True):
        subset = (
            dose.loc[dose.representation == representation]
            .groupby("nominal_dose", as_index=False)
            .agg(mean=("mean", "mean"), sem=("mean", "sem"))
            .sort_values("nominal_dose")
        )
        ax.errorbar(
            subset.nominal_dose,
            subset["mean"],
            yerr=1.96 * subset["sem"],
            color=color,
            marker="o",
            markersize=3.6,
            linewidth=1.3,
            capsize=2,
            label=label,
        )
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.axhline(0.02, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Nominal subject-opportunity dose")
    ax.set_ylabel("Dose-induced BAcc amplification")
    ax.set_title("B  No monotonic utilization response", loc="left", fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.6, loc="lower left")
    fig.text(
        0.5,
        -0.04,
        "Post-access exploratory AVDOS-VR analysis; error bars are 1.96 SEM across five split seeds.",
        ha="center",
        color=MUTED,
        fontsize=7.0,
    )
    save_figure(fig, "fig_avdos_exploratory_identity_and_dose")


def make_failure_timeline() -> pd.DataFrame:
    rows = [
        ("v6", "CEAP external", "effectiveness failure"),
        ("v7", "Consensus OPCT", "external failure retained"),
        ("v8", "Geometry OPCT", "development gate failed"),
        ("v9", "Witness-gated OPCT", "no observed harm; CEAP release too low"),
        ("v10", "Budget-saturating", "negative transfer"),
        ("v11", "Distribution-covered", "development gate passed"),
        ("AVDOS", "Registered confirmation", "stopped before training"),
        ("AVDOS-E", "Exploratory PPG", "identity fallback"),
    ]
    frame = pd.DataFrame(rows, columns=["stage", "method_or_test", "outcome"])
    frame["status"] = ["failed", "failed", "failed", "limited", "failed", "passed", "stopped", "abstained"]
    fig, ax = plt.subplots(figsize=(7.2, 2.45))
    color_map = {
        "failed": FAIL_COLOR,
        "limited": "#E69F00",
        "passed": PASS_COLOR,
        "stopped": "#8C6BB1",
        "abstained": ABSTAIN_COLOR,
    }
    ax.plot([0, len(frame) - 1], [0, 0], color=GRID, linewidth=2.0, zorder=0)
    for i, row in frame.iterrows():
        color = color_map[row.status]
        ax.scatter(i, 0, s=92, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        top = i % 2 == 0
        y_text = 0.38 if top else -0.42
        ax.plot([i, i], [0.05 if top else -0.05, y_text * 0.72], color=GRID, linewidth=0.8)
        ax.text(i, y_text, row.stage, ha="center", va="bottom" if top else "top", fontweight="bold", color=color)
        ax.text(
            i,
            y_text + (0.10 if top else -0.10),
            row.outcome,
            ha="center",
            va="bottom" if top else "top",
            fontsize=6.4,
            color=INK,
            rotation=18 if top else -18,
        )
    ax.set_xlim(-0.45, len(frame) - 0.55)
    ax.set_ylim(-0.82, 0.80)
    ax.axis("off")
    ax.set_title("Preserved method-development and external-test chronology", loc="left", fontweight="bold")
    save_figure(fig, "fig_v11_failure_and_protocol_timeline")
    return frame


def write_latex_tables(matrix: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    rows = []
    for _, row in matrix.iterrows():
        audit = "--" if not np.isfinite(row.audit_brier_gain) else f"{row.audit_brier_gain:.4f}"
        lcb = "--" if not np.isfinite(row.audit_brier_gain_lcb) else f"{row.audit_brier_gain_lcb:.4f}"
        role = "Development" if row.evidence_role == "retrospective development" else "Exploratory"
        rows.append(
            f"{row.dataset} & {role} & {row.selected_action} & {audit} & {lcb} & "
            f"{row.heldout_brier_gain:.4f} & {row.auc_delta:.4f} & {row.outcome} \\\\"
        )
    action_table = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{DCS-OPCT v11 audit-risk calibration actions and outcomes. AVDOS-VR is post-access exploratory; identity fallback is non-intervention and cannot support confirmatory effectiveness.}",
            r"\label{tab:v11_actions}",
            r"\small",
            r"\begin{tabular}{lllrrrrl}",
            r"\toprule",
            r"Dataset & Evidence role & Action & Audit gain & LCB & Held-out gain & $\Delta$AUROC & Outcome \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (OUT / "table_v11_action_outcome.tex").write_text(action_table, encoding="utf-8")

    pivot = diagnostics.pivot(index="dataset", columns="method", values="probability_mean_shift")
    locks = diagnostics.drop_duplicates("dataset").set_index("dataset")
    gate_rows = []
    for dataset in DATASET_ORDER:
        row = locks.loc[dataset]
        agreement = "--" if not np.isfinite(row.directional_agreement) else f"{row.directional_agreement:.3f}"
        cosine = "--" if not np.isfinite(row.displacement_cosine) else f"{row.displacement_cosine:.3f}"
        disagreement = "--" if not np.isfinite(row.normalized_disagreement) else f"{row.normalized_disagreement:.3f}"
        gate_rows.append(
            f"{dataset} & {abs(pivot.loc[dataset, 'coral']):.3f} & "
            f"{abs(pivot.loc[dataset, 'quantile_mapping']):.3f} & {agreement} & "
            f"{cosine} & {disagreement} & {'Pass' if bool(row.witness_pass) else 'Fail'} \\\\"
        )
    gate_table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Unlabeled applicability diagnostics for audit-risk probability transport. Mean shifts are absolute probability changes.}",
            r"\label{tab:v11_applicability}",
            r"\small",
            r"\begin{tabular}{lrrrrrl}",
            r"\toprule",
            r"Dataset & CORAL shift & QM shift & Agree. & Cosine & Disagree. & Gate \\",
            r"\midrule",
            *gate_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (OUT / "table_v11_applicability.tex").write_text(gate_table, encoding="utf-8")


def add_strong_calibration_artifacts() -> None:
    copies = {
        "fig_strong_direct_calibration.pdf": "fig_strong_direct_calibration.pdf",
        "fig_strong_direct_calibration.png": "fig_strong_direct_calibration.png",
        "fig_split_certified_calibration.pdf": "fig_split_certified_calibration.pdf",
        "fig_split_certified_calibration.png": "fig_split_certified_calibration.png",
        "decision_curve_summary.csv": "strong_calibration_decision_curve_summary.csv",
        "low_label_summary.csv": "strong_calibration_low_label_summary.csv",
        "paired_dcs_comparisons.csv": "strong_calibration_paired_dcs_comparisons.csv",
        "legacy_reproduction.json": "strong_calibration_legacy_reproduction.json",
        "manifest.json": "strong_calibration_manifest.json",
    }
    for source_name, destination_name in copies.items():
        shutil.copy2(STRONG / source_name, OUT / destination_name)

    crossfit_copies = {
        "decision_curve_replicates.csv": "crossfit_certified_decision_curve_replicates.csv",
        "decision_curve_summary.csv": "crossfit_certified_decision_curve_summary.csv",
        "certification_diagnostics.csv": "crossfit_certified_diagnostics.csv",
        "paired_vs_dcs.csv": "crossfit_certified_descriptive_pairs.csv",
        "manifest.json": "crossfit_certified_manifest.json",
    }
    for source_name, destination_name in crossfit_copies.items():
        shutil.copy2(CROSSFIT / source_name, OUT / destination_name)

    paired_copies = {
        "combined_decision_curve_replicates.csv": "combined_decision_curve_replicates.csv",
        "paired_gain_inference.csv": "paired_gain_inference.csv",
        "paired_safety_inference.csv": "paired_safety_inference.csv",
        "domain_budget_safety_upper_bounds.csv": "domain_budget_safety_upper_bounds.csv",
        "low_label_paired_inference.csv": "low_label_paired_inference.csv",
        "manifest.json": "paired_inference_manifest.json",
    }
    for source_name, destination_name in paired_copies.items():
        shutil.copy2(PAIRED / source_name, OUT / destination_name)

    summary = pd.read_csv(STRONG / "low_label_summary.csv")
    crossfit_summary = pd.read_csv(CROSSFIT / "decision_curve_summary.csv")
    crossfit_low = (
        crossfit_summary.loc[crossfit_summary.budget_fraction.isin((0.2, 0.4))]
        .groupby(["dataset", "method"], as_index=False)
        .agg(
            mean_release_rate=("release_rate", "mean"),
            mean_brier_gain=("mean_brier_gain", "mean"),
            mean_material_negative_transfer_frequency=(
                "material_negative_transfer_frequency", "mean"
            ),
            mean_auc_noninferiority_failure_frequency=(
                "auc_noninferiority_failure_frequency", "mean"
            ),
        )
    )
    summary = pd.concat([summary, crossfit_low], ignore_index=True)
    macro = (
        summary.groupby("method", as_index=True)[
            [
                "mean_release_rate",
                "mean_brier_gain",
                "mean_material_negative_transfer_frequency",
                "mean_auc_noninferiority_failure_frequency",
            ]
        ]
        .mean()
    )
    method_order = [
        ("dcs_selective", "DCS selective"),
        ("dcs_unconditional", "DCS unconditional diagnostic"),
        ("target_platt", "Target Platt"),
        ("target_beta", "Target beta"),
        ("target_temperature", "Target temperature"),
        ("target_isotonic", "Target isotonic"),
        ("split_certified_platt", "Split-certified Platt"),
        ("split_certified_beta", "Split-certified beta"),
        ("crossfit_certified_platt", "Cross-fit-certified Platt"),
        ("crossfit_certified_beta", "Cross-fit-certified beta"),
    ]
    rows = []
    for method, label in method_order:
        row = macro.loc[method]
        rows.append(
            f"{label} & {row.mean_release_rate:.3f} & {row.mean_brier_gain:.4f} & "
            f"{row.mean_material_negative_transfer_frequency:.3f} & "
            f"{row.mean_auc_noninferiority_failure_frequency:.3f} \\\\"
        )
    table = "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Method & Release rate & Mean gain & Material negative & AUROC failure \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    (OUT / "table_v11_strong_calibration_low_label.tex").write_text(
        table, encoding="utf-8"
    )


def make_minimum_budget_certification_figure() -> None:
    strong = pd.read_csv(STRONG / "decision_curve_summary.csv")
    crossfit = pd.read_csv(CROSSFIT / "decision_curve_summary.csv")
    summary = pd.concat([strong, crossfit], ignore_index=True)
    methods = [
        "dcs_selective", "target_platt", "target_beta",
        "split_certified_platt", "split_certified_beta",
        "crossfit_certified_platt", "crossfit_certified_beta",
    ]
    labels = [
        "DCS", "Direct Platt", "Direct beta", "Split Platt", "Split beta",
        "Cross-fit Platt", "Cross-fit beta",
    ]
    datasets = ["EPPVR", "SEED-IV", "DREAMER", "CASE", "CEAP"]
    minimum = summary.loc[summary.budget_fraction.eq(0.2)].set_index(["dataset", "method"])
    metrics = [
        ("mean_brier_gain", "A  Held-out Brier gain", "RdBu", TwoSlopeNorm(vmin=-0.01, vcenter=0, vmax=0.05), "+.3f"),
        ("material_negative_transfer_frequency", "B  Material-negative frequency", "Reds", None, ".2f"),
        ("release_rate", "C  Release rate", "Blues", None, ".2f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 4.25), gridspec_kw={"wspace": 0.28})
    for axis_index, (column, title, cmap, norm, number_format) in enumerate(metrics):
        ax = axes[axis_index]
        values = np.array([
            [float(minimum.loc[(dataset, method), column]) for dataset in datasets]
            for method in methods
        ])
        image = ax.imshow(
            values,
            cmap=cmap,
            norm=norm,
            vmin=None if norm is not None else 0,
            vmax=None if norm is not None else 1,
            aspect="auto",
        )
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if column == "mean_brier_gain":
                    dark = value < -0.004 or value > 0.033
                else:
                    dark = value > 0.52
                ax.text(
                    col, row, format(value, number_format), ha="center", va="center",
                    fontsize=5.6, color="white" if dark else INK,
                )
        ax.set_xticks(np.arange(len(datasets)), datasets, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(methods)), labels if axis_index == 0 else [])
        ax.tick_params(length=0, pad=3)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
        colorbar.ax.tick_params(labelsize=6, length=2)
    fig.text(
        0.5, -0.035,
        "Twenty percent of the frozen audit pool; identity fallback counts as no release, not effective calibration.",
        ha="center", color=MUTED, fontsize=7.0,
    )
    save_figure(fig, "fig_v11_crossfit_certification_stress_test")


def write_crossfit_inference_table() -> None:
    gains = pd.read_csv(PAIRED / "paired_gain_inference.csv")
    safety = pd.read_csv(PAIRED / "paired_safety_inference.csv")
    keys = ["dataset", "budget_fraction", "audit_configurations", "comparator_method", "inference_scope"]
    frame = gains.merge(safety, on=keys, validate="one_to_one")
    frame = frame.loc[
        frame.budget_fraction.eq(0.2)
        & frame.comparator_method.isin(("crossfit_certified_platt", "crossfit_certified_beta"))
    ]
    dataset_order = {name: index for index, name in enumerate(["EPPVR", "SEED-IV", "DREAMER", "CASE", "CEAP"])}
    method_order = {"crossfit_certified_platt": 0, "crossfit_certified_beta": 1}
    frame = frame.assign(
        dataset_order=frame.dataset.map(dataset_order),
        method_order=frame.comparator_method.map(method_order),
    ).sort_values(["dataset_order", "method_order"])
    rows = []
    for _, row in frame.iterrows():
        label = "Platt" if row.comparator_method.endswith("platt") else "Beta"
        rows.append(
            f"{row.dataset} & {label} & {row.mean_gain_difference_dcs_minus_comparator:+.4f} "
            f"[{row.mean_gain_difference_ci_low:+.4f}, {row.mean_gain_difference_ci_high:+.4f}] & "
            f"{row.paired_randomization_p_holm:.4f} & "
            f"{int(row.comparator_material_negative_count)}/100 & {row.exact_mcnemar_p_holm:.4f} \\\\"
        )
    table = "\n".join([
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Cross-fit map & $\Delta$ gain [95\% CI] & $p_{\mathrm{Holm}}$ & Harm & McNemar $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    (OUT / "table_v11_crossfit_paired_inference.tex").write_text(table, encoding="utf-8")


def validate_inputs_and_outputs(matrix: pd.DataFrame, diagnostics: pd.DataFrame) -> dict:
    freeze = read_json(FREEZE)
    faced_validation = read_json(FACED / "independent_validation_report.json")
    component_validation = read_json(COMPONENT / "independent_validation_report.json")
    checks = {
        "six_datasets_present": matrix.dataset.tolist() == DATASET_ORDER,
        "development_gate_passed": read_json(DEV / "development_acceptance_gate.json")["all_passed"],
        "three_development_adaptations": int(matrix.effectiveness_pass.sum()) == 3,
        "all_released_actions_pass_recorded_safety_criteria": bool(
            matrix.loc[matrix.selected_action != "identity", "safety_pass"].all()
        ),
        "avdos_exploratory_only": matrix.iloc[-1].evidence_role == "post-access exploratory",
        "avdos_effectiveness_not_claimed": not bool(matrix.iloc[-1].effectiveness_pass),
        "faced_exploratory_validation_passed": (
            faced_validation["status"] == "passed"
            and faced_validation["passed_checks"] == 18
            and faced_validation["confirmatory_claim_permitted"] is False
        ),
        "leave_one_component_validation_passed": (
            component_validation["status"] == "passed"
            and component_validation["checks_passed"] == 17
            and component_validation["checks_total"] == 17
        ),
        "zero_order_inversions": bool((diagnostics.order_inversions == 0).all()),
        "freeze_method_is_v11": freeze["freeze_id"] == "dcs-opct-v11-final-20260908",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Submission artifact validation failed: {failed}")
    return checks


def main() -> None:
    configure()
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    matrix = build_action_matrix()
    diagnostics = build_gate_diagnostics()
    dose, encoding = build_avdos_summaries()
    matrix.to_csv(OUT / "six_dataset_action_outcome.csv", index=False)
    diagnostics.to_csv(OUT / "support_gate_diagnostics.csv", index=False)
    dose.to_csv(OUT / "avdos_dose_summary.csv", index=False)
    encoding.to_csv(OUT / "avdos_identity_encoding_summary.csv", index=False)
    make_action_outcome_figure(matrix)
    make_gain_figure(matrix)
    make_support_gate_figure(diagnostics)
    make_avdos_figure(dose, encoding)
    timeline = make_failure_timeline()
    timeline.to_csv(OUT / "failure_and_protocol_timeline.csv", index=False)
    write_latex_tables(matrix, diagnostics)
    add_strong_calibration_artifacts()
    make_minimum_budget_certification_figure()
    write_crossfit_inference_table()
    engineering_files = [
        "ranking_unit_margins.csv",
        "candidate_pair_margins.csv",
        "margin_vulnerability_summary.csv",
        "audit_budget_utility.csv",
        "configuration_decision_curve.csv",
        "fig_material_threshold_engineering_anchor.pdf",
        "fig_material_threshold_engineering_anchor.png",
        "manifest.json",
        "independent_validation_report.json",
    ]
    for name in engineering_files:
        shutil.copy2(ENGINEERING / name, OUT / f"engineering_anchor_{name}")
    faced_files = [
        "faced_configuration_outcomes.csv",
        "faced_dcs_applicability.csv",
        "faced_dose_amplification_summary.csv",
        "faced_identity_encoding_summary.csv",
        "faced_post_access_evidence_report.json",
        "fig_faced_post_access_external_boundary.pdf",
        "fig_faced_post_access_external_boundary.png",
        "independent_validation_report.json",
        "table_faced_post_access_dose.tex",
    ]
    for name in faced_files:
        destination_name = name if name.startswith("faced_") else f"faced_{name}"
        shutil.copy2(FACED / name, OUT / destination_name)
    component_files = [
        "same_assignment_leave_one_component.csv",
        "random_half_allocation_repetitions.csv",
        "distribution_coverage_summary.csv",
        "table_leave_one_component.tex",
        "table_distribution_coverage_diagnostic.tex",
        "manifest.json",
        "independent_validation_report.json",
    ]
    for name in component_files:
        shutil.copy2(COMPONENT / name, OUT / f"component_{name}")
    checks = validate_inputs_and_outputs(matrix, diagnostics)
    input_paths = [
        FREEZE,
        DEV / "development_acceptance_gate.json",
        DEV / "primary_certificates.csv",
        DEV / "primary_heldout_results.csv",
        DEV / "component_projection_diagnostics.csv",
        DEV / "unlabeled_action_locks.csv",
        AVDOS_GATE / "primary_action_and_audit_lock.json",
        AVDOS_GATE / "external_confirmation_gate.json",
        AVDOS_DOSE / "summary.csv",
        AVDOS_DOSE / "identity_encoding_margin.csv",
        STRONG / "decision_curve_summary.csv",
        STRONG / "low_label_summary.csv",
        STRONG / "paired_dcs_comparisons.csv",
        STRONG / "legacy_reproduction.json",
        STRONG / "manifest.json",
        CROSSFIT / "decision_curve_replicates.csv",
        CROSSFIT / "decision_curve_summary.csv",
        CROSSFIT / "certification_diagnostics.csv",
        CROSSFIT / "paired_vs_dcs.csv",
        CROSSFIT / "manifest.json",
        PAIRED / "combined_decision_curve_replicates.csv",
        PAIRED / "paired_gain_inference.csv",
        PAIRED / "paired_safety_inference.csv",
        PAIRED / "domain_budget_safety_upper_bounds.csv",
        PAIRED / "low_label_paired_inference.csv",
        PAIRED / "manifest.json",
        ENGINEERING / "margin_vulnerability_summary.csv",
        ENGINEERING / "audit_budget_utility.csv",
        ENGINEERING / "configuration_decision_curve.csv",
        ENGINEERING / "manifest.json",
        ENGINEERING / "independent_validation_report.json",
        FACED / "faced_configuration_outcomes.csv",
        FACED / "faced_dcs_applicability.csv",
        FACED / "faced_dose_amplification_summary.csv",
        FACED / "faced_identity_encoding_summary.csv",
        FACED / "faced_post_access_evidence_report.json",
        FACED / "independent_validation_report.json",
        COMPONENT / "same_assignment_leave_one_component.csv",
        COMPONENT / "random_half_allocation_repetitions.csv",
        COMPONENT / "distribution_coverage_summary.csv",
        COMPONENT / "table_leave_one_component.tex",
        COMPONENT / "table_distribution_coverage_diagnostic.tex",
        COMPONENT / "manifest.json",
        COMPONENT / "independent_validation_report.json",
        ROOT / "docs" / "faced_v11_execution_failure_001.json",
        ROOT / "docs" / "faced_v11_execution_failure_002.json",
        ROOT / "docs" / "faced_v11_structural_eligibility_audit.json",
        ROOT / "docs" / "faced_v11_post_access_common_montage_protocol_amendment_001.json",
        ROOT / "docs" / "faced_v11_post_access_common_montage_implementation_lock.json",
        ROOT / "docs" / "faced_v11_post_access_common_montage_analysis_lock.json",
    ]
    outputs = sorted(path for path in OUT.iterdir() if path.is_file())
    manifest = {
        "status": "submission_artifacts_generated_and_validated",
        "date": "2026-09-09",
        "method": "DCS-OPCT v11",
        "claim_boundary": (
            "Selective audit-risk probability calibration is effective within the evaluated "
            "development support region; outside it, the method returns identity. AVDOS-VR and "
            "FACED are post-access exploratory boundary analyses. Identity fallback is "
            "non-intervention, not evidence of effectiveness or universal safety."
        ),
        "checks": checks,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths},
        "output_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in outputs
            if path.name
            not in {"submission_artifact_manifest.json", "independent_validation_report.json"}
        },
    }
    (OUT / "submission_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Generated {len(outputs) + 1} validated artifacts in {OUT}")


if __name__ == "__main__":
    main()
