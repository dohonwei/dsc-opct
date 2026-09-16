from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/dcs_opct_v12_multidomain_risk_robustness_v3"
OUT = ROOT / "outputs/dcs_opct_v12_multidomain_artifacts_v3"
DATASETS = ["DEAP", "MAHNOB-HCI", "EPPVR", "CASE", "CEAP", "SEED-IV", "DREAMER"]
METHODS = [
    "frozen_two_source_v11",
    "pooled_gpu",
    "domain_balanced_gpu",
    "smooth_worst_domain_gpu",
]
LABELS = {
    "frozen_two_source_v11": "Frozen two-source v11",
    "pooled_gpu": "Seven-domain pooled",
    "domain_balanced_gpu": "Seven-domain balanced",
    "smooth_worst_domain_gpu": "Smooth worst-domain",
}
COLORS = {
    "frozen_two_source_v11": "#7A7F87",
    "pooled_gpu": "#0072B2",
    "domain_balanced_gpu": "#009E73",
    "smooth_worst_domain_gpu": "#D55E00",
}
MARKERS = {
    "frozen_two_source_v11": "o",
    "pooled_gpu": "s",
    "domain_balanced_gpu": "^",
    "smooth_worst_domain_gpu": "D",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "axes.edgecolor": "#17202A",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_lines(ax: plt.Axes, metrics: pd.DataFrame, column: str, title: str,
                reference: float | None = None) -> None:
    x = np.arange(len(DATASETS))
    for method in METHODS:
        values = metrics.loc[metrics.method.eq(method)].set_index("dataset").loc[DATASETS, column]
        primary = method == "domain_balanced_gpu"
        ax.plot(
            x,
            values,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=4.4 if primary else 3.6,
            linewidth=1.8 if primary else 1.0,
            alpha=1.0 if primary else 0.75,
            label=LABELS[method],
        )
    if reference is not None:
        ax.axhline(reference, color="#555555", linestyle="--", linewidth=0.9)
    ax.set_xticks(x, DATASETS, rotation=30, ha="right")
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(axis="y", color="#D8DEE6", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def write_table(metrics: pd.DataFrame) -> None:
    indexed = metrics.set_index(["dataset", "method"])
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Prev. & v11 AUROC & DB AUROC & v11 BS & DB BS & DB BAcc \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        frozen = indexed.loc[(dataset, "frozen_two_source_v11")]
        balanced = indexed.loc[(dataset, "domain_balanced_gpu")]
        lines.append(
            f"{dataset} & {balanced.prevalence:.3f} & {frozen.auroc:.3f} & "
            f"{balanced.auroc:.3f} & {frozen.brier_skill:.3f} & "
            f"{balanced.brier_skill:.3f} & {balanced.balanced_accuracy:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / "table_v12_multidomain_robustness.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=False)
    configure()
    metrics = pd.read_csv(SOURCE / "domain_metrics.csv")
    gate = json.loads((SOURCE / "retrospective_robustness_gate.json").read_text(encoding="utf-8"))
    coefficients = pd.read_csv(SOURCE / "fold_coefficients.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.35), constrained_layout=True)
    panel_lines(axes[0, 0], metrics, "auroc", "A  Leave-one-dataset-out discrimination", 0.60)
    axes[0, 0].set_ylabel("AUROC")
    axes[0, 0].set_ylim(0.40, 0.78)
    panel_lines(axes[0, 1], metrics, "brier_skill", "B  Probability skill", 0.0)
    axes[0, 1].set_ylabel("Brier skill vs training prevalence")
    panel_lines(axes[1, 0], metrics, "balanced_accuracy", "C  Nested operating point", 0.60)
    axes[1, 0].set_ylabel("Balanced accuracy")
    axes[1, 0].set_ylim(0.40, 0.72)

    pivot = metrics.pivot(index="dataset", columns="method", values="brier").loc[DATASETS]
    delta = pivot.frozen_two_source_v11 - pivot.domain_balanced_gpu
    colors = np.where(delta >= 0, "#009E73", "#C44E52")
    axes[1, 1].bar(np.arange(len(DATASETS)), delta, color=colors, width=0.68)
    axes[1, 1].axhline(0, color="#17202A", linewidth=0.8)
    axes[1, 1].set_xticks(np.arange(len(DATASETS)), DATASETS, rotation=30, ha="right")
    axes[1, 1].set_ylabel("Brier improvement over frozen v11")
    axes[1, 1].set_title("D  Broader training did not ensure transfer", loc="left", weight="bold")
    axes[1, 1].grid(axis="y", color="#D8DEE6", linewidth=0.6)
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"fig_v12_multidomain_robustness.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    write_table(metrics)

    selected = metrics.loc[metrics.method.eq("domain_balanced_gpu")]
    frozen = metrics.loc[metrics.method.eq("frozen_two_source_v11")]
    weighted_brier_selected = float(np.average(selected.brier, weights=selected.n))
    weighted_brier_frozen = float(np.average(frozen.brier, weights=frozen.n))
    selected_coefficients = coefficients.loc[coefficients.method.eq("domain_balanced_gpu")]
    summary = {
        "status": "prespecified_retrospective_gate_failed",
        "datasets": len(DATASETS),
        "configurations": int(selected.n.sum()),
        "median_auroc": float(selected.auroc.median()),
        "median_brier_skill": float(selected.brier_skill.median()),
        "median_balanced_accuracy": float(selected.balanced_accuracy.median()),
        "domains_auroc_at_least_060": int((selected.auroc >= 0.60).sum()),
        "domain_balanced_weighted_brier": weighted_brier_selected,
        "frozen_v11_weighted_brier": weighted_brier_frozen,
        "weighted_brier_improvement": weighted_brier_frozen - weighted_brier_selected,
        "domains_with_brier_improvement": int((
            frozen.set_index("dataset").brier - selected.set_index("dataset").brier
        ).gt(0).sum()),
        "positive_domain_balanced_coefficients": int(selected_coefficients.coefficient.gt(0).sum()),
        "domain_balanced_coefficients": len(selected_coefficients),
        "failed_gate_checks": [name for name, passed in gate["checks"].items() if not passed],
        "interpretation": (
            "Broader retrospective training improved pooled probability error but did not "
            "establish reliable leave-dataset-out discrimination or operating-point transfer."
        ),
        "claim_boundary": "Negative retrospective robustness evidence; no prospective safety or effectiveness claim.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs = sorted(OUT.iterdir())
    manifest = {
        "status": "complete_preserve_failed_gate",
        "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
        "source_validation_sha256": sha256(SOURCE / "independent_validation_report.json"),
        "script_sha256": sha256(Path(__file__)),
        "output_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
