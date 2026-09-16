from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = ROOT / "docs/elsarticle/figures"
TABLE_PATH = ROOT / "docs/elsarticle/tables/table_v11_independence_ladder.tex"


def load() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    transductive = pd.read_csv(
        ROOT / "outputs/distribution_covered_stratified_opct_v11_development/primary_heldout_results.csv"
    ).set_index("dataset")
    inductive = pd.read_csv(
        ROOT / "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/inductive_action_outcomes.csv"
    ).set_index("dataset")
    raw_group = pd.read_csv(
        ROOT / "outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914/raw_group_independent_action_outcomes.csv"
    ).set_index("dataset")
    raw_first = pd.read_csv(
        ROOT / "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/evaluation_outcome.csv"
    ).set_index("dataset")
    datasets = ["EPPVR", "CASE", "CEAP", "SEED-IV", "DREAMER"]
    protocols = ["Transductive", "Covariate-unseen", "Raw test-group", "Raw partition-first"]
    status = np.full((len(datasets), len(protocols)), np.nan)
    gain = np.full_like(status, np.nan)
    frames = [
        (transductive, "selected_method", "brier_gain"),
        (inductive, "selected_method", "heldout_brier_gain"),
        (raw_group, "selected_method", "heldout_brier_gain"),
        (raw_first, "selected_method", "evaluation_brier_gain"),
    ]
    for column, (frame, action_col, gain_col) in enumerate(frames):
        for row, dataset in enumerate(datasets):
            if dataset not in frame.index:
                continue
            action = str(frame.loc[dataset, action_col])
            status[row, column] = float(action != "identity")
            gain[row, column] = float(frame.loc[dataset, gain_col])
    return datasets, protocols, status, gain


def build_figure() -> None:
    datasets, protocols, status, gain = load()
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1.12, 1.0]})
    shown = np.ma.masked_invalid(status)
    cmap = ListedColormap(["#d8dde2", "#2a9d8f"])
    cmap.set_bad("white")
    axes[0].imshow(shown, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(protocols)), ["Transductive", "Covariate\nunseen", "Raw test\ngroups", "Raw partition\nfirst"])
    axes[0].set_yticks(range(len(datasets)), datasets)
    axes[0].set_title("A  Released action under stricter independence")
    for row in range(len(datasets)):
        for col in range(len(protocols)):
            label = "not run" if np.isnan(status[row, col]) else ("release" if status[row, col] else "original")
            color = "white" if status[row, col] == 1 else "#39424e"
            axes[0].text(col, row, label, ha="center", va="center", color=color, fontsize=8)
    axes[0].set_xticks(np.arange(-.5, len(protocols), 1), minor=True)
    axes[0].set_yticks(np.arange(-.5, len(datasets), 1), minor=True)
    axes[0].grid(which="minor", color="white", linewidth=1.5)
    axes[0].tick_params(which="minor", bottom=False, left=False)

    colors = {"EPPVR": "#2a9d8f", "CASE": "#457b9d", "CEAP": "#e76f51"}
    x = np.arange(len(protocols))
    for dataset in colors:
        row = datasets.index(dataset)
        axes[1].plot(x, gain[row], marker="o", linewidth=2, markersize=5, label=dataset, color=colors[dataset])
    axes[1].axhline(0, color="#5f6872", linewidth=1)
    axes[1].set_xticks(x, ["Transductive", "Covariate\nunseen", "Raw test\ngroups", "Raw partition\nfirst"])
    axes[1].set_ylabel("Evaluation Brier gain")
    axes[1].set_title("B  Observed gain of the released-or-original action")
    axes[1].grid(axis="y", color="#e5e8eb", linewidth=.8)
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].annotate("applicability stop", (3, 0), xytext=(2.45, 0.012), arrowprops={"arrowstyle": "->", "color": "#5f6872"}, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / "fig_v11_independence_ladder.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_ROOT / "fig_v11_independence_ladder.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_table() -> None:
    text = r"""\begin{table*}[t]
\centering
\caption{Independence ladder for the DCS-OPCT development evidence. Return original denotes non-intervention, not successful calibration.}
\label{tab:independence-ladder}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllllr}
\toprule
Protocol & Evaluation covariates & Endpoint raw groups & Emotion-model training & Released domains & EPPVR gain \\
\midrule
Transductive outcome holdout & Complete target batch used & Shared underlying observations & Shared & EPPVR, CASE, CEAP & 0.0263 \\
Covariate-unseen & Evaluation side unseen & Shared underlying observations & Shared & EPPVR, CASE & 0.0226 \\
Raw test-group separation & Evaluation side unseen & Disjoint test participants & Shared & EPPVR & 0.0135 \\
Raw partition first & Evaluation population unseen & Disjoint participants and rows & Disjoint & None; applicability stop & 0.0000 \\
\bottomrule
\end{tabular}%
}
\end{table*}
"""
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    build_figure()
    build_table()
    print(FIGURE_ROOT / "fig_v11_independence_ladder.pdf")
    print(TABLE_PATH)
