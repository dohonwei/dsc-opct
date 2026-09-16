from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("dcs_selective", "target_platt", "target_isotonic", "split_certified_platt")
LABELS = {
    "dcs_selective": "DCS selective",
    "target_platt": "Target Platt",
    "target_isotonic": "Target isotonic",
    "split_certified_platt": "Split-certified Platt",
}
COLORS = {
    "dcs_selective": "#087f8c",
    "target_platt": "#3266a8",
    "target_isotonic": "#c17c24",
    "split_certified_platt": "#8f4c8a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready audit-budget artifacts.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/dcs_opct_v11_audit_budget_decision_curves/decision_curve_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dcs_opct_v11_audit_budget_submission_artifacts"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_tradeoff(summary: pd.DataFrame, output: Path) -> None:
    datasets = summary.dataset.drop_duplicates().tolist()
    low = summary.loc[summary.budget_fraction.eq(0.2)]
    pooled = (
        summary.groupby(["budget_fraction", "method"], sort=False)
        .agg(
            mean_gain=("mean_brier_gain", "mean"),
            material_negative=("material_negative_transfer_frequency", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3), constrained_layout=True)
    x = np.arange(len(datasets), dtype=float)
    width = 0.19
    for method_index, method in enumerate(METHODS):
        part = low.set_index(["dataset", "method"]).loc[
            [(dataset, method) for dataset in datasets]
        ]
        offset = (method_index - 1.5) * width
        axes[0].bar(
            x + offset,
            part.mean_brier_gain,
            width,
            color=COLORS[method],
            label=LABELS[method],
        )
        axes[1].bar(
            x + offset,
            part.material_negative_transfer_frequency,
            width,
            color=COLORS[method],
            label=LABELS[method],
        )
        trajectory = pooled.loc[pooled.method.eq(method)].sort_values("budget_fraction")
        axes[2].plot(
            trajectory.material_negative,
            trajectory.mean_gain,
            marker="o",
            linewidth=2.0,
            color=COLORS[method],
            label=LABELS[method],
        )
        for row in trajectory.loc[
            trajectory.budget_fraction.isin([0.2, 1.0])
        ].itertuples(index=False):
            axes[2].annotate(
                f"{int(row.budget_fraction * 100)}",
                (row.material_negative, row.mean_gain),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color=COLORS[method],
            )
    axes[0].axhline(0.0, color="#303840", linewidth=0.9)
    axes[0].set_ylabel("Mean held-out Brier gain")
    axes[0].set_title("A  20% audit-pool budget")
    axes[1].set_ylabel("Material negative-transfer frequency")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("B  20% audit-pool budget")
    axes[2].axvline(0.0, color="#303840", linewidth=0.9)
    axes[2].axhline(0.0, color="#303840", linewidth=0.9)
    axes[2].set_xlabel("Equal-domain mean material-negative frequency")
    axes[2].set_ylabel("Equal-domain mean held-out Brier gain")
    axes[2].set_title("C  Budget-dependent risk--gain frontier")
    for axis in axes[:2]:
        axis.set_xticks(x, datasets, rotation=25, ha="right")
    for axis in axes:
        axis.grid(axis="y", color="#dce2e8", linewidth=0.8)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=4, frameon=False)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_core_table(summary: pd.DataFrame, output: Path) -> pd.DataFrame:
    selected = summary.loc[
        summary.budget_fraction.isin([0.2, 1.0])
        & summary.method.isin(["dcs_selective", "target_platt"])
    ].copy()
    selected["budget"] = selected.budget_fraction.map({0.2: "Minimum", 1.0: "Full"})
    selected["gain_text"] = selected.mean_brier_gain.map(lambda value: f"{value:.4f}")
    selected["release_text"] = selected.release_rate.map(lambda value: f"{100 * value:.0f}")
    selected["negative_text"] = selected.material_negative_transfer_frequency.map(
        lambda value: f"{100 * value:.0f}"
    )
    rows = []
    for dataset in summary.dataset.drop_duplicates():
        for budget in ("Minimum", "Full"):
            part = selected.loc[selected.dataset.eq(dataset) & selected.budget.eq(budget)].set_index("method")
            rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "audit_configurations": int(part.audit_configurations.iloc[0]),
                    "dcs_release_percent": part.loc["dcs_selective", "release_text"],
                    "dcs_gain": part.loc["dcs_selective", "gain_text"],
                    "dcs_material_negative_percent": part.loc["dcs_selective", "negative_text"],
                    "platt_gain": part.loc["target_platt", "gain_text"],
                    "platt_material_negative_percent": part.loc["target_platt", "negative_text"],
                }
            )
    table = pd.DataFrame(rows)
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Budget & $n_L$ & \multicolumn{3}{c}{DCS-OPCT} & \multicolumn{2}{c}{Target Platt} \\",
        r"\hline",
        r" & & & Release (\%) & Gain & MNT (\%) & Gain & MNT (\%) \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"{row.dataset} & {row.budget} & {row.audit_configurations} & "
            f"{row.dcs_release_percent} & {row.dcs_gain} & "
            f"{row.dcs_material_negative_percent} & "
            f"{row.platt_gain} & {row.platt_material_negative_percent} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output.write_text("\n".join(lines), encoding="ascii")
    return table


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    summary = pd.read_csv(args.input)
    required = {
        "dataset",
        "budget_fraction",
        "audit_configurations",
        "method",
        "release_rate",
        "mean_brier_gain",
        "material_negative_transfer_frequency",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Summary lacks required columns: {missing}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    plot_tradeoff(summary, args.output_root / "fig_budget_risk_gain_tradeoff")
    table = write_core_table(summary, args.output_root / "table_budget_core.tex")
    table.to_csv(args.output_root / "table_budget_core.csv", index=False)
    manifest = {
        "status": "budget_submission_artifacts_complete",
        "date": "2026-09-08",
        "input_sha256": sha256(args.input),
        "script_sha256": sha256(Path(__file__)),
        "claim_boundary": (
            "Retrospective fixed-heldout sensitivity evidence. Equal-domain pooling in "
            "panel C is descriptive and does not create prospective external evidence."
        ),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
