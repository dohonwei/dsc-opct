from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/eegemotions27_v11_manuscript_integration_preview"
OUT = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2"
T2 = ROOT / "outputs/dcs_opct_v11_end_to_end_simplified_comparators_20260910_v2"
T3 = ROOT / "outputs/dcs_opct_v11_frozen_action_stability_20260910_v2"
T5 = ROOT / "outputs/dcs_opct_v11_worked_audit_example_20260910_v4"
STRONG = ROOT / "outputs/dcs_opct_v11_strong_calibration_baselines"
T6 = ROOT / "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910"
T7 = ROOT / "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910"

DATASETS = ["EPPVR", "SEED-IV", "DREAMER", "CASE", "CEAP"]
DIRECT = {
    "dcs_selective": ("DCS-OPCT", "#087f8c"),
    "target_platt": ("Target Platt", "#3568a8"),
    "target_isotonic": ("Target isotonic", "#d17a18"),
    "target_beta": ("Target beta", "#b13f4a"),
    "target_temperature": ("Temperature", "#60752b"),
}
SPLIT = {
    "dcs_selective": ("DCS-OPCT", "#087f8c"),
    "split_certified_platt": ("Split-certified Platt", "#96508d"),
    "split_certified_beta": ("Split-certified beta", "#604b9f"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {name} anchor, found {count}")
    return text.replace(old, new, 1)


def replace_between(text: str, start: str, end: str, new: str, name: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(f"Expected one {name} section boundary")
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + new + text[end_index:]


def prepare_tree() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(BASE / "figures", OUT / "figures")
    shutil.copytree(BASE / "tables", OUT / "tables")
    for name in ["elsarticle.cls", "elsarticle-num.bst", "dcs_opct_v11_bspc_refs.bib"]:
        shutil.copy2(BASE / name, OUT / name)
    for source in [
        T2 / "fig_end_to_end_simplified_comparators.pdf",
        T2 / "fig_end_to_end_simplified_comparators.png",
        T3 / "fig_frozen_action_stability.pdf",
        T3 / "fig_frozen_action_stability.png",
        T5 / "fig_worked_audit_trace.pdf",
        T5 / "fig_worked_audit_trace.png",
        T6 / "fig_cluster_dependence_audit_yield.pdf",
        T6 / "fig_cluster_dependence_audit_yield.png",
    ]:
        shutil.copy2(source, OUT / "figures" / source.name)


def plot_metric_panels(frame, methods, metric, ylabel, title, stem) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    })
    # A portrait grid uses the journal page more efficiently than a wide 2x3
    # panel after LaTeX scales the figure to the text width.
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.2), sharey=True)
    axes_flat = axes.ravel()
    for axis, dataset in zip(axes_flat, DATASETS, strict=False):
        subset = frame[frame["dataset"] == dataset]
        for method, (label, color) in methods.items():
            rows = subset[subset["method"] == method].sort_values("budget_fraction")
            if rows.empty:
                continue
            axis.plot(100 * rows["budget_fraction"], rows[metric], marker="o",
                      linewidth=1.7, markersize=3.8, label=label, color=color)
        axis.axhline(0, color="#343a40", linewidth=0.8)
        axis.set_title(dataset, fontweight="bold")
        axis.set_xlabel("Audit pool used (%)")
        axis.set_xticks([20, 60, 100])
        axis.tick_params(labelsize=9)
    axes_flat[-1].axis("off")
    axes_flat[0].set_ylabel(ylabel)
    axes_flat[2].set_ylabel(ylabel)
    axes_flat[4].set_ylabel(ylabel)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), frameon=False)
    fig.suptitle(title, y=0.985, fontweight="bold")
    fig.tight_layout(rect=(0, 0.085, 1, 0.95), h_pad=1.5, w_pad=1.4)
    for suffix in ["pdf", "png"]:
        fig.savefig(OUT / "figures" / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_split_comparator_figures() -> None:
    summary = pd.read_csv(STRONG / "decision_curve_summary.csv")
    specs = [
        (DIRECT, "mean_brier_gain", "Mean held-out Brier gain", "Direct target calibration: held-out gain", "fig_direct_calibration_gain_readable"),
        (DIRECT, "material_negative_transfer_frequency", "Material-negative frequency", "Direct target calibration: intervention risk", "fig_direct_calibration_risk_readable"),
        (SPLIT, "release_rate", "Release rate", "Split-certified calibration: release frequency", "fig_split_certified_release_readable"),
        (SPLIT, "mean_brier_gain", "Mean held-out Brier gain", "Split-certified calibration: held-out gain", "fig_split_certified_gain_readable"),
        (SPLIT, "material_negative_transfer_frequency", "Material-negative frequency", "Split-certified calibration: intervention risk", "fig_split_certified_risk_readable"),
    ]
    for args in specs:
        plot_metric_panels(summary, *args)


def write_tables() -> None:
    comparator = pd.read_csv(T2 / "end_to_end_comparator_summary.csv")
    labels = {
        "risk_ranking_only": "Risk ranking only",
        "certificate_only": "Certificate only",
        "single_unlabeled_gate": "Single unlabeled gate",
        "full_dcs_opct": "Full DCS-OPCT",
    }
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Variant & Action labels & Fraction & Certificates & Released domains \\",
        r"\midrule",
    ]
    for key, label in labels.items():
        row = comparator[comparator["variant"] == key].iloc[0]
        value = (
            f"{label} & {int(row['action_labels_required'])} & "
            f"{float(row['action_label_fraction_of_ceiling']):.3f} & "
            f"{int(row['certificate_attempts'])} & {int(row['released_domains'])}"
        )
        lines.append(value + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tables/table_simplified_comparators.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    summary = json.loads((T3 / "summary.json").read_text(encoding="utf-8"))
    stability = [
        r"\begin{tabular}{lr}", r"\toprule", r"Diagnostic & Result \\",
        r"\midrule", r"One-at-a-time domain--action rows & 320 \\",
        f"Rows with an action flip & {summary['one_at_a_time_action_flip_count']}" + r" \\",
        r"Domains affected by any local flip & EPPVR; CEAP \\",
        f"Joint local scenarios & {summary['joint_scenario_count']:,}" + r" \\",
        f"Distinct release patterns & {summary['joint_distinct_release_patterns']}" + r" \\",
        r"External domains released after relaxation & 0 \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    (OUT / "tables/table_frozen_action_stability.tex").write_text(
        "\n".join(stability) + "\n", encoding="utf-8"
    )

    audit = pd.read_csv(T6 / "audit_yield_summary.csv")
    primary = audit[
        audit["material_effect_threshold"].round(6).eq(0.02)
        & audit["audit_budget_fraction"].round(6).eq(0.20)
    ].iloc[0]
    dependence = pd.read_csv(T6 / "cluster_dependence_summary.csv")
    primary_dependence = dependence[
        dependence["material_effect_threshold"].round(6).eq(0.02)
        & dependence["scope"].eq("overall")
    ].set_index("endpoint")
    cluster_table = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Quantity & Estimate & Comparator or consequence \\",
        r"\midrule",
        (
            r"Event-bearing clusters captured & "
            f"{primary['risk_cluster_event_sensitivity']:.3f} & "
            f"Random mean {primary['random_cluster_event_sensitivity_mean']:.3f} "
            + r"\\"
        ).rstrip(),
        (
            r"Exact stratified random-allocation test & "
            f"$p={primary['cluster_event_exact_random_p_one_sided']:.4f}$ & "
            r"One-sided, post-hoc designated cell \\"
        ),
        (
            r"Event configurations captured & "
            f"{primary['risk_row_event_sensitivity']:.3f} & 43/114 configurations "
            + r"\\"
        ).rstrip(),
        (
            r"Material excess captured & "
            f"{primary['risk_material_excess_captured']:.3f} & Above the 0.02 endpoint "
            + r"\\"
        ).rstrip(),
        (
            r"Material-event ICC(1,1) & "
            f"{primary_dependence.loc['material_event', 'icc_1_1']:.3f} & "
            f"Effective $n={primary_dependence.loc['material_event', 'effective_configuration_rows']:.0f}$ of 480 "
            + r"\\"
        ).rstrip(),
        (
            r"Amplification ICC(1,1) & "
            f"{primary_dependence.loc['amplification', 'icc_1_1']:.3f} & "
            f"Effective $n={primary_dependence.loc['amplification', 'effective_configuration_rows']:.0f}$ of 480 "
            + r"\\"
        ).rstrip(),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (OUT / "tables/table_complete_cluster_audit_yield.tex").write_text(
        "\n".join(cluster_table) + "\n", encoding="utf-8"
    )

    evidence = [
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.15\textwidth}>{\raggedright\arraybackslash}p{0.19\textwidth}>{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.29\textwidth}@{}}",
        r"\toprule", r"Evidence layer & Datasets & Empirical contribution & Claim boundary \\", r"\midrule",
        r"Mechanism sources & DEAP; MAHNOB-HCI & Dose experiment and frozen risk-model training & Source-domain development, not external confirmation \\",
        r"Target development & EPPVR; CASE; CEAP; SEED-IV; DREAMER & Calibration release or identity fallback under the frozen action & Same-study retrospective evidence; no prospective effectiveness claim \\",
        r"Post-access tests & AVDOS-VR; FACED & Cross-physiology and common-montage failure boundaries & Non-intervention or non-estimability, not effectiveness or safety success \\",
        r"Pre-signal test & EEGEmotions-27 & Applicability failure before action-stage label access & Outside the prospective family; category polarity is not experienced affect \\",
        r"Reserved tests & AMIGOS; Emognition & Locked access, schema, GPU, and one-shot execution stacks & No result yet; both independent tests are still required \\",
        r"\bottomrule", r"\end{tabular}",
    ]
    (OUT / "tables/table_evidence_role_matrix.tex").write_text(
        "\n".join(evidence) + "\n", encoding="utf-8"
    )


def build_manuscript() -> Path:
    text = (BASE / "manuscript_preview.tex").read_text(encoding="utf-8")
    methods_anchor = (
        "These analyses quantify retrospective design sensitivity on reused held-out halves, "
        "not prospective external-validation uncertainty or future-domain risk."
    )
    methods_addition = methods_anchor + r"""

Three simplified end-to-end variants were reconstructed from the same frozen actions, assignments, and 720-configuration action-stage budget ceiling. Certificate only removed both unlabeled gates, single unlabeled gate retained only the primary component screen, and full DCS-OPCT retained the component screen and independent witness. This same-assignment diagnostic counted labels entering action selection; it did not refit candidates, change thresholds, or estimate prospective sample complexity. We also performed a post-hoc local action-stability analysis over eight frozen decision parameters. Each parameter was perturbed one at a time over two more permissive and two stricter settings, followed by a three-level joint cube over all eight parameters. These perturbations diagnose local brittleness of archived decisions and were not used for threshold selection."""
    text = replace_once(text, methods_anchor, methods_addition, "new diagnostic methods")

    old_audit_methods = (
        "To anchor the endpoint to an engineering consequence, we also measured absolute "
        "balanced-accuracy margins among the four candidate models under joint-unseen "
        "evaluation. For each dataset--task--seed unit, we recorded the top-two margin and "
        "all six pairwise margins. Separately, leave-dataset-out risk probabilities ranked "
        "configurations for audit at budgets from 10\\% to 50\\%. We reported material-event "
        "sensitivity, enrichment over prevalence, and the fraction of positive material "
        "amplification captured. Confidence intervals used 10,000 bootstrap repetitions over "
        "complete split-seed clusters. A dimensionless decision curve assigned a missed "
        "material event unit cost and varied the relative cost of a false audit; it was a "
        "scenario analysis, not a measured monetary utility function."
    )
    new_audit_methods = (
        "To anchor the endpoint to an engineering consequence, we also measured absolute "
        "balanced-accuracy margins among the four candidate models under joint-unseen "
        "evaluation. For each dataset--task--seed unit, we recorded the top-two margin and "
        "all six pairwise margins. The audit-yield analysis treated the four nonzero doses "
        "for each dataset--task--representation--model--split-seed combination as one "
        "indivisible cluster. Clusters were ranked within each held-out source dataset by the "
        "maximum frozen leave-dataset-out risk probability and selected at 10\\%--50\\% "
        "budgets. The primary reporting cell was designated at the frozen 0.02 event threshold "
        "and the existing 20\\% minimum-budget condition; "
        "budget. A stratified hypergeometric test compared captured event-bearing clusters "
        "with same-budget random allocation; 10,000 random allocations characterized "
        "secondary yield metrics. ICC(1,1), design effects, effective row counts, and naive "
        "versus complete-cluster bootstrap intervals quantified within-configuration "
        "dependence. This post-hoc analysis did not alter the risk model, v11 actions, or "
        "release thresholds."
    )
    text = replace_once(
        text, old_audit_methods, new_audit_methods, "complete-cluster audit methods"
    )

    results_anchor = (
        "No threshold, seed, candidate probability, or held-out row was changed for this "
        "diagnostic (Supplementary Material)."
    )
    results_addition = results_anchor + r"""

Under the common 720-configuration ceiling, certificate-only processing sent all 720 labels to action-stage certification, the single-gate variant used 600, and full DCS-OPCT used 480. The three calibration variants preserved the same observed release set---EPPVR, CASE, and CEAP---and the same equal-domain mean held-out Brier gain of 0.01665. Thus, the complete unlabeled sequence reduced retrospective action-stage label demand by 33.3\% relative to certificate-only processing without changing the archived release set (Fig.~\ref{fig:simplified-comparators}). This is a same-assignment efficiency diagnostic, not proof that either gate is prospectively necessary.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_end_to_end_simplified_comparators.pdf}
\caption{Same-ceiling simplified end-to-end comparators. The common action-stage ceiling was 720 configuration labels across five retrospective target-development domains. Unlabeled screening reduced labels reaching certification while preserving the observed three-domain release set and held-out gain. Risk ranking alone is non-interventional. The comparison reuses frozen assignments and outcomes and does not establish prospective component necessity.}
\label{fig:simplified-comparators}
\end{figure*}

Every tested one-at-a-time relaxation preserved identity in the unsupported development and external domains. Only the strictest settings changed an action, affecting EPPVR through directional agreement and CEAP through directional agreement, displacement cosine, or the Brier lower-bound certificate. Across 6,561 joint local scenarios, only three release patterns occurred, all subsets of the frozen EPPVR--CASE--CEAP set. Full margins and a worked EPPVR/EEGEmotions-27 trace are reported in the Supplementary Material."""
    text = replace_once(text, results_anchor, results_addition, "new diagnostic results")

    old_audit_results = (
        "The engineering-anchor analysis gave the 0.02 magnitude a concrete operational "
        "interpretation. Seventeen of 20 joint-unseen model-selection units (85\\%, "
        "cluster-bootstrap 95\\% CI 70--100\\%) had a top-two candidate margin no greater "
        "than 0.02, and 65 of 120 model pairs (54.2\\%, 44.2--65.0\\%) were separated by no "
        "more than this amount. At a 20\\% audit budget, frozen-risk ordering captured "
        "36.0\\% (28.7--45.6\\%) of material events and 37.5\\% (28.9--48.6\\%) of their "
        "positive amplification, an event enrichment of 1.80 (1.43--2.28) over prevalence. "
        "The fixed binary screen outperformed both audit-all and audit-none only for assumed "
        "false-audit/missed-event cost ratios from 0.25 to 0.45. Thus, 0.02 is relevant to "
        "model selection and risk ranking can concentrate audit effort, but neither result "
        "establishes endpoint optimality or universal decision utility."
    )
    new_audit_results = (
        "The engineering-anchor analysis gave the 0.02 magnitude a concrete operational "
        "interpretation. Seventeen of 20 joint-unseen model-selection units (85\\%, "
        "cluster-bootstrap 95\\% CI 70--100\\%) had a top-two candidate margin no greater "
        "than 0.02, and 65 of 120 model pairs (54.2\\%, 44.2--65.0\\%) were separated by no "
        "more than this amount. In the corrected complete-cluster analysis, a 20\\% budget "
        "selected 24 clusters, corresponding to 96 configuration labels, and captured 19 of "
        "56 event-bearing clusters (33.9\\%) versus a random mean of 20.0\\% (stratified exact "
        "$p=0.000345$). The same allocation captured 43 of 114 event configurations "
        "(37.7\\%) and 37.4\\% of amplification exceeding the material threshold. Maximum, "
        "mean, and median aggregation of the four frozen risk scores selected the same 24 "
        "clusters and produced identical yield. Dependence "
        "was substantial: material-event ICC was 0.420 and reduced the design-effect-based "
        "effective row count from 480 to approximately 212; complete-cluster intervals were "
        "1.30--1.75 times as wide as naive row-bootstrap intervals across the reported "
        "endpoints and datasets. These retrospective source-domain results support Stage II "
        "as an audit-prioritization mechanism, not prospective workflow utility, external "
        "effectiveness, or a transferable operating-point guarantee (Supplementary Material)."
    )
    text = replace_once(
        text, old_audit_results, new_audit_results, "complete-cluster audit results"
    )

    discussion_anchor = (
        "These are retrospective observations from the frozen domains, not proof that every "
        "component is necessary in every future domain."
    )
    discussion_addition = discussion_anchor + (
        " The same-ceiling reconstruction sharpened this interpretation: the unlabeled "
        "sequence saved 240 action-stage labels relative to certificate-only processing while "
        "preserving the archived release set, but it does not identify a prospective causal "
        "contribution or guarantee the same saving under a new target distribution."
    )
    text = replace_once(text, discussion_anchor, discussion_addition, "discussion efficiency")

    limitation_anchor = (
        "They therefore support operational interpretation only and cannot establish a "
        "clinically meaningful cutoff, measured cost effectiveness, or threshold optimality."
    )
    limitation_addition = limitation_anchor + (
        " The local action-stability grid likewise describes the neighborhood of the frozen "
        "decisions; because it was conducted after observing the development results, it "
        "cannot validate or optimize any threshold."
    )
    text = replace_once(text, limitation_anchor, limitation_addition, "stability limitation")

    validation_anchor = "six post-hoc artifact checks, and 78 submission-artifact checks."
    validation_replacement = (
        "six post-hoc artifact checks, thirteen simplified-comparator checks, fourteen frozen-"
        "action-stability checks, twelve worked-example checks, sixteen complete-cluster audit-"
        "yield checks, eight cluster-score aggregation checks, and 78 submission-artifact checks."
    )
    text = replace_once(text, validation_anchor, validation_replacement, "validation counts")
    output = OUT / "manuscript_preview.tex"
    output.write_text(text, encoding="utf-8")
    return output


def build_supplement() -> Path:
    text = (BASE / "supplementary_preview.tex").read_text(encoding="utf-8")
    text = replace_once(
        text,
        r"\usepackage{graphicx}",
        "\\usepackage{graphicx}\n\\usepackage{array}",
        "array package",
    )
    cluster_section = r"""\section{Post-hoc engineering anchor, cluster dependence, and audit yield}

We first evaluated whether a distortion of 0.02 balanced accuracy is large relative to model-selection margins. Under the joint-unseen reference protocol, each dataset--task--seed unit contained four candidate models. We computed the top-two margin and all six absolute pairwise margins. Seventeen of 20 top-two margins (85\%, cluster-bootstrap 95\% CI 70--100\%) and 65 of 120 pairwise margins (54.2\%, 44.2--65.0\%) were no greater than 0.02. This anchors the endpoint to model-selection sensitivity without assigning it physiological or clinical meaning.

The earlier row-ranked audit diagnostic could split the four nonzero doses belonging to one configuration cluster. We therefore repeated the allocation at the unit used by the v11 certificate: dataset--task--representation--model--split seed, with all four dose configurations selected together. Within each held-out source dataset, the frozen cluster score was the maximum of its four leave-dataset-out risk probabilities. Budgets of 10--50\% selected the same fraction of complete clusters from each dataset. The primary reporting cell was designated post hoc at the frozen 0.02 event threshold and the existing 20\% minimum-budget condition; other cells were sensitivity analyses. Random allocation retained the same per-dataset cluster counts. An outcome-informed oracle is shown only as a descriptive ceiling.

At the designated cell, 24 of 120 clusters were selected, requiring 96 configuration labels. Frozen ranking captured 19 of 56 event-bearing clusters (33.9\%), compared with a random mean of 20.0\% and a random 97.5th percentile of 28.6\%. The descriptive one-sided stratified hypergeometric probability of capturing at least 19 event-bearing clusters was $p=0.000345$. The selected clusters contained 43 of 114 material-event configurations (37.7\%) and 37.4\% of total amplification above the 0.02 threshold. Maximum, mean, and median aggregation of the four frozen per-dose probabilities selected exactly the same 24 clusters (all pairwise Jaccard indices 1.0) and therefore produced identical yield estimates.

The four dose rows were not statistically independent. At the primary threshold, overall ICC(1,1) was 0.420 for the binary material event, 0.591 for amplification, and 0.753 for the frozen risk probability. The corresponding design-effect-based effective row counts were approximately 212, 173, and 147 rather than 480. Complete-cluster bootstrap intervals were 1.30--1.75 times as wide as naive row-bootstrap intervals across event prevalence and mean amplification in the pooled and dataset-specific analyses. All inferential summaries therefore retain the complete cluster as the resampling unit.

\begin{table}[!htbp]
\centering
\caption{Designated complete-cluster audit-yield and dependence results. The post-hoc reporting cell used the frozen 0.02 event threshold and the existing 20\% minimum-budget condition.}
\label{tab:complete-cluster-audit-yield}
\small
\resizebox{\textwidth}{!}{\input{tables/table_complete_cluster_audit_yield.tex}}
\end{table}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_cluster_dependence_audit_yield.pdf}
\caption{Complete-cluster audit allocation and dependence sensitivity. (A) Frozen risk ranking, same-budget random allocation, and an outcome-informed descriptive oracle for event-bearing-cluster capture. Shading is the central 95\% random-allocation interval. (B) Event configurations and material excess captured by the frozen ranking. (C) Enrichment of selected event-bearing clusters over prevalence. (D) ICC(1,1) and design-effect-based effective row counts at the frozen 0.02 threshold. These are retrospective source-domain diagnostics, not prospective workflow-utility or external-effectiveness results.}
\label{fig:complete-cluster-audit-yield}
\end{figure*}

The exact primary comparison shows that the frozen Stage II ranking concentrated audit yield after respecting the complete-cluster label unit. It does not establish causal component necessity, measured cost effectiveness, external effectiveness, or a transferable operating-point guarantee. The binary decision-curve results from the earlier row-level diagnostic are retained as exploratory provenance but are not used for the complete-cluster primary claim.

"""
    text = replace_between(
        text,
        "\\section{Post-hoc engineering anchor and audit utility}\n",
        "\\section{Model-ranking consequences}\n",
        cluster_section,
        "complete-cluster engineering anchor",
    )
    insertion_anchor = "\\section{Post-hoc seven-domain risk robustness}\n"
    new_sections = r"""
\section{End-to-end component efficiency under a common label ceiling}

The simplified variants reused the frozen candidate probabilities, distribution-covered assignments, held-out rows, certificates, and 720-configuration action-stage budget ceiling. Risk ranking alone did not define a calibration action. Certificate-only processing sent all 720 labels to the five target-domain certificates. The primary unlabeled component screen reduced this requirement to 600 by stopping SEED-IV, and the independent witness reduced it to 480 by additionally stopping DREAMER. All three calibration variants released EPPVR, CASE, and CEAP and retained the same summed held-out Brier gain. The resulting 33.3\% reduction is retrospective label-demand accounting under the archived assignment, not a prospective sample-complexity theorem or proof of universal component necessity.

\begin{table}[!htbp]
\centering
\caption{Same-ceiling simplified end-to-end comparator summary. Fraction is the proportion of the 720-configuration ceiling entering action-stage processing.}
\label{tab:simplified-comparators}
\small
\resizebox{\textwidth}{!}{\input{tables/table_simplified_comparators.tex}}
\end{table}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_end_to_end_simplified_comparators.pdf}
\caption{Label demand, held-out outcome, and released actions for the simplified frozen variants. Identity in the risk-ranking-only column denotes that ranking does not itself define an intervention.}
\label{fig:supp-simplified-comparators}
\end{figure*}

\section{Post-hoc stability of frozen actions}

Eight frozen parameters were varied one at a time over two more permissive and two stricter local settings, yielding 320 domain--action rows. Every local relaxation preserved identity in DREAMER, SEED-IV, AVDOS-VR, FACED, and EEGEmotions-27. Four strict-setting rows changed a release: EPPVR failed only at directional agreement 0.95, whereas CEAP failed at directional agreement 0.95, displacement cosine 0.95, or Brier-gain lower bound 0.002. CASE did not flip. The complete three-level joint cube contained 6,561 scenarios and only three release patterns: EPPVR--CASE--CEAP, EPPVR--CASE, or CASE alone. This analysis maps local decision margins after freezing; it did not select thresholds or provide prospective effectiveness evidence.

\begin{table}[!htbp]
\centering
\caption{Frozen-action local stability summary. External-domain relaxation refers to AVDOS-VR, FACED, and EEGEmotions-27.}
\label{tab:frozen-action-stability}
\small
\input{tables/table_frozen_action_stability.tex}
\end{table}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=0.88\textwidth]{figures/fig_frozen_action_stability.pdf}
\caption{Post-hoc local stability of the frozen DCS-OPCT v11 decisions. Positive signed margins pass a frozen gate and negative margins fail it. Rank and inversion margins are shown at their numerical boundary and should not be interpreted as substantive slack.}
\label{fig:frozen-action-stability}
\end{figure*}

\section{Worked frozen audit trace}

The EPPVR trace illustrates a released correction: the largest component shift was 0.055, directional agreement was 0.938, displacement cosine was 0.997, and normalized disagreement was 0.209. After both unlabeled stages passed, 160 action-stage labels entered the fixed audit. The one-sided Brier-gain lower bound was 0.0096, above the frozen 0.001 requirement, and the held-out gain was 0.0263. EEGEmotions-27 illustrates a pre-label stop: its largest component shift was 0.128, above 0.10, so the witness and labeled certificate were not reached and zero action-stage labels were accessed. Stages after a stop are marked N/A rather than shown as hypothetical passes. Identity is non-intervention and is not counted as effectiveness, non-harm, or successful external calibration.

\begin{figure*}[!htbp]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_worked_audit_trace.pdf}
\caption{Worked frozen action paths for a released development action (EPPVR) and a separate pre-signal external applicability failure (EEGEmotions-27).}
\label{fig:worked-audit-trace}
\end{figure*}

"""
    text = replace_once(text, insertion_anchor, new_sections + insertion_anchor, "supplement diagnostics")

    evidence_anchor = "\\section{Reproducibility and claim boundary}\n"
    evidence_section = r"""
\section{Evidence-role matrix}

Table~\ref{tab:evidence-role-matrix} separates empirical evidence from protocol readiness. AMIGOS and Emognition are not counted as completed experiments, and EEGEmotions-27 is not promoted into that prospective family. This organization prevents dataset count from being used as a surrogate for evidential independence.

\begin{table*}[!htbp]
\centering
\caption{Evidence roles and inferential boundaries. Original dataset licenses, approvals, and participant-consent conditions govern data release.}
\label{tab:evidence-role-matrix}
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.08}
\input{tables/table_evidence_role_matrix.tex}
\end{table*}

"""
    text = replace_once(text, evidence_anchor, evidence_section + evidence_anchor, "evidence matrix")

    old_figures = r"""\begin{figure}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_strong_direct_calibration.pdf}
\caption{Dataset-specific held-out Brier gain and material-negative-transfer frequency for selective DCS-OPCT and direct target-label calibrators. Beta calibration closely tracked positive-slope Platt scaling. Temperature scaling is a restricted no-intercept control. Curves summarize repeated audit ordering on the same fixed held-out halves and are retrospective sensitivity results.}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_split_certified_calibration.pdf}
\caption{Release rate, held-out Brier gain, and material-negative-transfer frequency for selective DCS-OPCT and split-fit/certified Platt or monotone beta calibration. DCS-OPCT's zero release in SEED-IV and DREAMER follows the label-free applicability gates; it denotes non-intervention rather than successful calibration. Corresponding 20\% budget numerical values, paired confidence intervals, sign-randomization tests, exact McNemar tests, and multiplicity-adjusted results are reported in Table~\ref{tab:crossfit-paired}.}
\end{figure}"""
    new_figures = r"""\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_direct_calibration_gain_readable.pdf}
\caption{Dataset-specific held-out Brier gain for selective DCS-OPCT and direct target-label calibrators.}
\label{fig:direct-gain-readable}
\end{figure*}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_direct_calibration_risk_readable.pdf}
\caption{Material-negative-transfer frequency for direct target calibration. Identity fallback in DCS-OPCT is non-intervention, not effectiveness. Curves are retrospective sensitivity results on fixed held-out halves.}
\label{fig:direct-risk-readable}
\end{figure*}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_split_certified_release_readable.pdf}
\caption{Release frequency for selective DCS-OPCT and split-fit/certified Platt or monotone beta calibration.}
\label{fig:split-release-readable}
\end{figure*}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_split_certified_gain_readable.pdf}
\caption{Held-out Brier gain for selective DCS-OPCT and split-fit/certified calibration. Rejected repetitions contribute identity gain zero.}
\label{fig:split-gain-readable}
\end{figure*}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_split_certified_risk_readable.pdf}
\caption{Material-negative-transfer frequency for selective DCS-OPCT and split-fit/certified calibration. Corresponding 20\% budget paired inference is reported in Table~\ref{tab:crossfit-paired}.}
\label{fig:split-risk-readable}
\end{figure*}"""
    text = replace_once(text, old_figures, new_figures, "dense supplementary figures")

    validation_anchor = "6 of 6 EEGEmotions-27 post-hoc artifact checks, and 78 of 78 submission-artifact checks."
    validation_replacement = (
        "6 of 6 EEGEmotions-27 post-hoc artifact checks, 13 of 13 simplified-comparator "
        "checks, 14 of 14 frozen-action-stability checks, 12 of 12 worked-example checks, "
        "16 of 16 complete-cluster audit-yield checks, 8 of 8 cluster-score aggregation "
        "checks, and 78 of 78 submission-artifact checks."
    )
    text = replace_once(text, validation_anchor, validation_replacement, "supplement validation counts")
    output = OUT / "supplementary_preview.tex"
    output.write_text(text, encoding="utf-8")
    return output


def write_manifest(manuscript: Path, supplement: Path) -> None:
    tracked = [manuscript, supplement]
    tracked += sorted((OUT / "figures").glob("fig_*readable.*"))
    for stem in [
        "fig_end_to_end_simplified_comparators",
        "fig_frozen_action_stability",
        "fig_worked_audit_trace",
        "fig_cluster_dependence_audit_yield",
    ]:
        tracked += sorted((OUT / "figures").glob(f"{stem}.*"))
    for name in [
        "table_simplified_comparators.tex",
        "table_frozen_action_stability.tex",
        "table_evidence_role_matrix.tex",
        "table_complete_cluster_audit_yield.tex",
    ]:
        tracked.append(OUT / "tables" / name)
    manifest = {
        "status": "staged_preview_not_merged_into_authoritative_manuscript",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_preview": str(BASE.relative_to(ROOT)).replace("\\", "/"),
        "authoritative_sources_unchanged": True,
        "freeze_sha256": sha256(
            ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
        ),
        "eegemotions_lock_sha256": sha256(
            ROOT / "docs/eegemotions27_v11_external_robustness_implementation_lock.json"
        ),
        "cluster_score_aggregation_manifest_sha256": sha256(T7 / "manifest.json"),
        "files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in tracked
        },
        "claim_boundary": (
            "T2, T3, T5, and the complete-cluster audit-yield analysis are retrospective "
            "frozen-artifact diagnostics. They support label-demand accounting, local "
            "action stability, procedural traceability, dependence-aware uncertainty, and "
            "source-domain audit prioritization, not prospective effectiveness, universal "
            "safety, causal necessity, or threshold optimality."
        ),
    }
    (OUT / "preview_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    prepare_tree()
    make_split_comparator_figures()
    write_tables()
    manuscript = build_manuscript()
    supplement = build_supplement()
    write_manifest(manuscript, supplement)
    print(f"Staged evidence-integration preview: {OUT}")


if __name__ == "__main__":
    main()
