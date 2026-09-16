from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


REPRESENTATION_LABELS = {
    "all": "All features",
    "relative_power": "Relative power",
    "normalized_asymmetry": "Normalized asymmetry",
}
MODEL_LABELS = {"linear_logistic": "Logistic", "gpu_mlp": "GPU MLP"}
PROJECTION_LABELS = {
    "coral": "CORAL + OPCT",
    "quantile_mapping": "Quantile mapping + OPCT",
}
PROBE_LABELS = {"linear": "Linear", "rbf": "RBF"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create staged LaTeX tables for the EEGEmotions-27 v11 boundary."
    )
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_external_robustness"),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_posthoc_boundary_analysis_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_posthoc_boundary_analysis_v2/latex_tables"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def yes_no(value: bool) -> str:
    return "Yes" if bool(value) else "No"


def write_table(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def gate_table(primary_root: Path) -> str:
    gate = json.loads(
        (primary_root / "outcome/external_confirmation_gate.json").read_text(
            encoding="utf-8"
        )
    )
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Frozen EEGEmotions-27 external-robustness gate. Identity denotes non-intervention and is not counted as effectiveness or safety success.}}
\label{{tab:eegemotions27-gate}}
\begin{{tabular}}{{ll}}
\toprule
Gate item & Frozen result \\
\midrule
Endpoint estimable & {yes_no(gate['endpoint_estimable'])} \\
Unlabeled candidate & {gate['candidate_method']} \\
Released action & None (identity fallback) \\
Certificate passed & {yes_no(gate['certificate_pass'])} \\
Held-out effectiveness passed & {yes_no(gate['effectiveness_pass'])} \\
Released-action non-harm & Not applicable \\
External claim supported & {yes_no(gate['claim_supported'])} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def applicability_table(analysis_root: Path) -> str:
    table = pd.read_csv(analysis_root / "applicability_thresholds.csv")
    rows = []
    for item in table.itertuples(index=False):
        rows.append(
            f"{PROJECTION_LABELS[item.method]} & "
            f"{item.absolute_probability_mean_shift:.4f} & "
            f"{item.maximum_allowed_shift:.2f} & "
            f"{item.threshold_excess:.4f} & "
            f"{item.probability_rank:.6f} & "
            f"{int(item.order_inversions)} & {yes_no(item.applicable)} \\\\" 
        )
    body = "\n".join(rows)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Post-outcome description of the frozen unlabeled applicability decision. Applicability itself was locked before target outcome labels were attached.}}
\label{{tab:eegemotions27-applicability}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Projection & $|\Delta\bar p|$ & Limit & Excess & Rank & Inversions & Applicable \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def generalization_table(analysis_root: Path) -> str:
    table = pd.read_csv(analysis_root / "baseline_dual_generalization.csv")
    table["representation_order"] = table.representation.map(
        {"all": 0, "relative_power": 1, "normalized_asymmetry": 2}
    )
    table["model_order"] = table.model.map({"linear_logistic": 0, "gpu_mlp": 1})
    table = table.sort_values(["representation_order", "model_order"])
    rows = []
    for item in table.itertuples(index=False):
        rows.append(
            f"{REPRESENTATION_LABELS[item.representation]} & "
            f"{MODEL_LABELS[item.model]} & {int(item.n)} & "
            f"{item.balanced_accuracy_mean:.3f} $\pm$ {item.balanced_accuracy_sd:.3f} & "
            f"{item.balanced_accuracy_min:.3f}--{item.balanced_accuracy_max:.3f} & "
            f"{item.mean_minus_chance:.3f} \\\\" 
        )
    body = "\n".join(rows)
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Descriptive dual-unseen category-polarity performance at zero identity dose. Mean $\pm$ SD summarizes five correlated frozen split seeds and is not an independent-sample confidence interval.}}
\label{{tab:eegemotions27-dual-generalization}}
\begin{{tabular}}{{llrrrr}}
\toprule
Representation & Classifier & Seeds & Balanced accuracy & Range & Mean $-0.5$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def encoding_boundary_table(analysis_root: Path) -> str:
    encoding = pd.read_csv(analysis_root / "identity_encoding_summary.csv")
    events = pd.read_csv(analysis_root / "material_event_summary.csv")
    model_reversals = pd.read_csv(analysis_root / "model_ranking_reversals.csv")
    representation_reversals = pd.read_csv(
        analysis_root / "representation_ranking_reversals.csv"
    )
    monotonicity = pd.read_csv(
        analysis_root / "dose_monotonicity_by_configuration.csv"
    )
    encoding["representation_order"] = encoding.representation.map(
        {"all": 0, "relative_power": 1, "normalized_asymmetry": 2}
    )
    encoding["probe_order"] = encoding.identity_probe.map({"linear": 0, "rbf": 1})
    encoding = encoding.sort_values(["representation_order", "probe_order"])
    encoding_rows = []
    for item in encoding.itertuples(index=False):
        encoding_rows.append(
            f"{REPRESENTATION_LABELS[item.representation]} & {PROBE_LABELS[item.identity_probe]} & "
            f"{item.encoding_margin_mean:.3f} $\pm$ {item.encoding_margin_sd:.3f} & "
            f"{item.encoding_margin_min:.3f}--{item.encoding_margin_max:.3f} \\\\" 
        )
    overall = events.loc[events.grouping.eq("overall")].set_index("partition")
    summary_rows = [
        "Audit material events & "
        f"{int(overall.loc['audit', 'positive_material_events'])} positive, "
        f"{int(overall.loc['audit', 'negative_material_events'])} negative \\\\ ",
        "Held-out material events & "
        f"{int(overall.loc['heldout', 'positive_material_events'])} positive, "
        f"{int(overall.loc['heldout', 'negative_material_events'])} negative \\\\ ",
        f"Strong monotonic dose patterns & {int(monotonicity.monotonic_positive.sum() + monotonicity.monotonic_negative.sum())}/{len(monotonicity)} \\\\ ",
        f"Model-ranking reversals & {int(model_reversals.ranking_reversal.sum())}/{len(model_reversals)} \\\\ ",
        f"Representation-ranking reversals & {int(representation_reversals.ranking_reversal.sum())}/{len(representation_reversals)} \\\\ ",
    ]
    return rf"""
\begin{{table*}}[t]
\centering
\caption{{Post-hoc encoding--utilization boundary. Identity margins are descriptive five-seed summaries. Event and ranking analyses cannot alter the frozen external gate.}}
\label{{tab:eegemotions27-boundary}}
\begin{{tabular}}{{llrr}}
\toprule
Representation & Probe & Encoding margin & Range \\
\midrule
{chr(10).join(encoding_rows)}
\bottomrule
\end{{tabular}}
\par\vspace{{1.0ex}}
\begin{{tabular}}{{lr}}
\toprule
Boundary diagnostic & Observed \\
\midrule
{chr(10).join(summary_rows)}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite LaTeX table directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    builders = {
        "table_eegemotions27_gate.tex": lambda: gate_table(args.primary_root),
        "table_eegemotions27_applicability.tex": lambda: applicability_table(
            args.analysis_root
        ),
        "table_eegemotions27_dual_generalization.tex": lambda: generalization_table(
            args.analysis_root
        ),
        "table_eegemotions27_encoding_boundary.tex": lambda: encoding_boundary_table(
            args.analysis_root
        ),
    }
    paths = []
    for filename, builder in tqdm(
        builders.items(), desc="EEGEmotions-27 LaTeX tables", unit="table"
    ):
        path = args.output_root / filename
        write_table(path, builder())
        paths.append(path)
    preview_path = args.output_root / "eegemotions27_tables_preview.tex"
    preview_inputs = "\n\\clearpage\n".join(
        f"\\input{{{path.name}}}" for path in paths
    )
    preview_text = rf"""\documentclass[10pt]{{article}}
\usepackage[margin=18mm]{{geometry}}
\IfFileExists{{booktabs.sty}}{{\usepackage{{booktabs}}}}{{\newcommand{{\toprule}}{{\hline}}\newcommand{{\midrule}}{{\hline}}\newcommand{{\bottomrule}}{{\hline}}}}
\usepackage{{graphicx}}
\begin{{document}}
{preview_inputs}
\end{{document}}
"""
    write_table(preview_path, preview_text)
    paths.append(preview_path)
    manifest = {
        "status": "staged_latex_tables_complete",
        "date": "2026-09-09",
        "primary_gate_changed": False,
        "tables": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in paths
        },
        "integration_status": "awaiting_author_figure_and_table_review",
        "claim_boundary": (
            "These tables are staged artifacts. Post-hoc tables cannot alter or "
            "rescue the frozen one-shot gate."
        ),
    }
    (args.output_root / "table_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
