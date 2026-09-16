from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3"
HIGHER = ROOT / "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910"
OUT = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4"
INVENTORY = ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/inventory.json"
INVENTORY_VALIDATION = ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


def prepare_tree() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(BASE, OUT)
    for pattern in ["*.pdf", "*.log", "independent_validation_report.json"]:
        for path in OUT.glob(pattern):
            path.unlink()
    rendered = OUT / "rendered"
    if rendered.exists():
        shutil.rmtree(rendered)
    for suffix in ["pdf", "png"]:
        shutil.copy2(
            HIGHER / f"fig_higher_level_cluster_sensitivity.{suffix}",
            OUT / "figures" / f"fig_higher_level_cluster_sensitivity.{suffix}",
        )


def write_table() -> None:
    summary = pd.read_csv(HIGHER / "higher_level_audit_yield.csv")
    row = summary.loc[summary.score_aggregation.eq("max_risk")].iloc[0]
    intervals = pd.read_csv(HIGHER / "higher_level_bootstrap_intervals.csv").set_index("endpoint")
    lines = [
        r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Quantity & Estimate \\", r"\midrule",
        r"Higher-level units & 20 dataset--task--seed blocks \\",
        f"Selected units at 20\\% budget & {int(row.selected_blocks)} blocks / {int(row.labeled_rows)} rows " + r"\\",
        f"Event rows captured & {row.event_row_sensitivity:.3f} " + r"\\",
        f"Material excess captured & {row.material_excess_captured:.3f} " + r"\\",
        f"Event enrichment & {row.event_enrichment:.3f} " + r"\\",
        f"Exact random-allocation $p$, event capture & {row.exact_random_p_event_row_sensitivity:.3f} " + r"\\",
        f"Exact random-allocation $p$, excess capture & {row.exact_random_p_material_excess:.3f} " + r"\\",
        (
            "Event prevalence, 95\\% block-bootstrap interval & "
            f"{intervals.loc['event_prevalence', 'estimate']:.3f} "
            f"[{intervals.loc['event_prevalence', 'block_bootstrap_q025']:.3f}, "
            f"{intervals.loc['event_prevalence', 'block_bootstrap_q975']:.3f}] " + r"\\"
        ),
        (
            "Mean amplification, 95\\% block-bootstrap interval & "
            f"{intervals.loc['mean_amplification', 'estimate']:.4f} "
            f"[{intervals.loc['mean_amplification', 'block_bootstrap_q025']:.4f}, "
            f"{intervals.loc['mean_amplification', 'block_bootstrap_q975']:.4f}] " + r"\\"
        ),
        r"\bottomrule", r"\end{tabular}",
    ]
    (OUT / "tables/table_higher_level_cluster_sensitivity.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def sync_counts(text: str) -> str:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    validation = json.loads(INVENTORY_VALIDATION.read_text(encoding="utf-8"))
    replacement = (
        f"The current immutable reproducibility inventory contains {inventory['unique_file_count']} files "
        f"and passed {validation['checks_passed']}/{validation['checks_total']} checks."
    )
    return re.sub(
        r"The current immutable reproducibility inventory contains \d+ files and passed \d+/\d+ checks\.",
        replacement,
        text,
    )


def build_manuscript() -> Path:
    text = (BASE / "manuscript_preview.tex").read_text(encoding="utf-8")
    methods_anchor = (
        "Dependence was substantial: material-event ICC was 0.420 and reduced the design-effect-based "
        "effective row count from 480 to approximately 212; complete-cluster intervals were "
        "1.30--1.75 times as wide as naive row-bootstrap intervals across the reported endpoints and datasets."
    )
    methods_addition = methods_anchor + (
        " We therefore added a more conservative post-hoc sensitivity that treated each "
        "dataset--task--split-seed family as one indivisible unit. This produced 20 conditional "
        "higher-level blocks, each containing all representations, models, and nonzero doses. "
        "At the 20\\% budget, two whole blocks per source dataset were selected. All 2,025 valid "
        "same-budget allocations were enumerated exactly, and uncertainty for event prevalence and "
        "mean amplification was estimated by stratified resampling of these blocks."
    )
    text = replace_once(text, methods_anchor, methods_addition, "higher-level methods")

    results_anchor = (
        "These retrospective source-domain results support Stage II as an audit-prioritization mechanism, "
        "not prospective workflow utility, external effectiveness, or a transferable operating-point guarantee "
        "(Supplementary Material)."
    )
    results_addition = (
        "At the higher dataset--task--seed level, the same 20\\% label budget selected four blocks "
        "and captured 26.3\\% of event rows and 25.9\\% of material excess, corresponding to 1.316-fold "
        "event enrichment. However, neither quantity exceeded the exact whole-block random-allocation "
        "distribution ($p=0.207$ and $p=0.260$, respectively). Maximum, mean, and median block-risk "
        "aggregation again selected the same blocks. Thus, the significant complete-configuration result "
        "does not persist as statistically persuasive workflow-level yield under the coarser dependence unit. "
        "The combined evidence supports configuration prioritization within the fixed development sources, "
        "not a general audit-allocation benefit across tasks, seeds, or future domains (Supplementary Material)."
    )
    text = replace_once(text, results_anchor, results_addition, "higher-level results")
    text = sync_counts(text)
    path = OUT / "manuscript_preview.tex"
    path.write_text(text, encoding="utf-8")
    return path


def build_supplement() -> Path:
    text = (BASE / "supplementary_preview.tex").read_text(encoding="utf-8")
    anchor = "\\section{Model-ranking consequences}\n"
    section = r"""\section{Higher-level dataset--task--seed sensitivity}

The complete-configuration analysis retains all four doses together but representations and models within a dataset--task--seed family still share source trials, labels, and data partitions. We therefore repeated the primary 0.02-threshold, 20\% budget analysis using dataset--task--seed as the indivisible unit. The two source datasets contributed ten blocks each (two tasks by five split seeds), and every block contained 24 rows spanning three representations, two models, and four nonzero doses. Two whole blocks were selected per dataset, preserving the same 96-row label budget as the complete-configuration analysis.

Maximum, mean, and median aggregation of the frozen risk probability selected the same four blocks. They captured 26.3\% of all material-event rows and 25.9\% of material excess, with event prevalence 1.316 times the source-domain average. We enumerated all $\binom{10}{2}^2=2{,}025$ valid stratified same-budget allocations. The exact one-sided probabilities were $p=0.207$ for event-row sensitivity and $p=0.260$ for material-excess capture. The higher-level result therefore did not provide statistically persuasive evidence that the frozen ordering improves whole-block audit yield.

Stratified bootstrap resampling of the 20 blocks gave event prevalence 0.238 (95\% interval 0.165--0.310) and mean amplification 0.0107 (0.0071--0.0140). These intervals account for shared representation, model, dose, trial, and split structure within each dataset--task--seed family, but remain conditional on only two source datasets. They neither create domain-level replication nor support prospective workflow utility.

\begin{table}[!htbp]
\centering
\caption{Post-hoc higher-level dependence sensitivity at the frozen 0.02 endpoint and 20\% budget. Exact probabilities enumerate all stratified whole-block allocations.}
\label{tab:higher-level-cluster}
\small
\resizebox{\textwidth}{!}{\input{tables/table_higher_level_cluster_sensitivity.tex}}
\end{table}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=0.96\textwidth]{figures/fig_higher_level_cluster_sensitivity.pdf}
\caption{Dataset--task--seed sensitivity. (A) Frozen block risk versus material-event prevalence for the 20 higher-level units. (B) Whole-block frozen-risk allocation compared with the exact random-allocation mean. (C) Exact null distribution for event-row sensitivity; the vertical line is the frozen allocation. The result is conditional post-hoc sensitivity, not independent-domain validation.}
\label{fig:higher-level-cluster}
\end{figure*}

"""
    text = replace_once(text, anchor, section + anchor, "supplement higher-level section")
    text = sync_counts(text)
    path = OUT / "supplementary_preview.tex"
    path.write_text(text, encoding="utf-8")
    return path


def write_manifest(manuscript: Path, supplement: Path) -> None:
    tracked = [
        manuscript,
        supplement,
        OUT / "tables/table_higher_level_cluster_sensitivity.tex",
        OUT / "figures/fig_higher_level_cluster_sensitivity.pdf",
        OUT / "figures/fig_higher_level_cluster_sensitivity.png",
    ]
    manifest = {
        "status": "staged_preview_not_merged_into_authoritative_manuscript",
        "preview_version": "v4_higher_level_dependence_integration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_preview": str(BASE.relative_to(ROOT)).replace("\\", "/"),
        "authoritative_sources_unchanged": True,
        "freeze_sha256": sha256(ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"),
        "higher_level_manifest_sha256": sha256(HIGHER / "manifest.json"),
        "files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked},
        "claim_boundary": (
            "The higher-level analysis preserves the non-significant exact whole-block result. It limits "
            "the audit-yield claim to configuration prioritization within two fixed development sources "
            "and does not create prospective workflow utility, independent-domain replication, external "
            "effectiveness, or future-domain non-harm evidence."
        ),
    }
    (OUT / "preview_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    prepare_tree()
    write_table()
    manuscript = build_manuscript()
    supplement = build_supplement()
    write_manifest(manuscript, supplement)
    print(f"Staged evidence-integration preview v4: {OUT}")


if __name__ == "__main__":
    main()
