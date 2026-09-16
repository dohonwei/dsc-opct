from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

FREEZE = Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json")
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
SOURCE = Path("outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/component_same_assignment_leave_one_component.csv")
BUDGETS = Path("outputs/distribution_covered_stratified_opct_v11_development/primary_heldout_results.csv")
DATASETS = ["EPPVR", "SEED-IV", "DREAMER", "CASE", "CEAP"]
VARIANTS = ["risk_ranking_only", "certificate_only", "single_unlabeled_gate", "full_dcs_opct"]
LABELS = {
    "risk_ranking_only": "Risk ranking only",
    "certificate_only": "Certificate only",
    "single_unlabeled_gate": "Single unlabeled gate",
    "full_dcs_opct": "Full DCS-OPCT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build same-ceiling end-to-end simplified comparators for frozen DCS-OPCT v11.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dcs_opct_v11_end_to_end_simplified_comparators_20260910_v2"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_long_table(source: pd.DataFrame, budgets: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "minus_unlabeled_applicability": "certificate_only",
        "minus_quantile_witness": "single_unlabeled_gate",
        "full_v11": "full_dcs_opct",
    }
    audit_budget = budgets.set_index("dataset").n_audit_configurations.astype(int).to_dict()
    rows = []
    for dataset in tqdm(DATASETS, desc="Simplified comparators", unit="domain", dynamic_ncols=True):
        budget = audit_budget[dataset]
        rows.append({
            "dataset": dataset,
            "variant": "risk_ranking_only",
            "variant_label": LABELS["risk_ranking_only"],
            "candidate_eligible": False,
            "certificate_attempted": False,
            "audit_budget_ceiling": budget,
            "action_labels_required": budget,
            "selected_action": "identity",
            "heldout_brier_gain": 0.0,
            "material_negative_transfer": False,
            "interpretation": "Frozen risk score ranks audits but does not recalibrate probabilities.",
        })
        group = source.loc[source.dataset.eq(dataset)].set_index("variant")
        for source_variant, variant in mapping.items():
            item = group.loc[source_variant]
            eligible = bool(item.candidate_eligible)
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "variant_label": LABELS[variant],
                "candidate_eligible": eligible,
                "certificate_attempted": eligible,
                "audit_budget_ceiling": budget,
                "action_labels_required": budget if eligible else 0,
                "selected_action": item.selected_action,
                "heldout_brier_gain": float(item.heldout_brier_gain),
                "material_negative_transfer": bool(item.material_negative_transfer),
                "interpretation": {
                    "certificate_only": "CORAL candidate reaches the labeled certificate without an unlabeled screen.",
                    "single_unlabeled_gate": "CORAL invariant screen precedes the labeled certificate; quantile witness is omitted.",
                    "full_dcs_opct": "CORAL and quantile component screens, witness agreement, and labeled certificate are retained.",
                }[variant],
            })
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    maximum_budget = int(frame.loc[frame.variant.eq("risk_ranking_only"), "audit_budget_ceiling"].sum())
    rows = []
    for variant in VARIANTS:
        group = frame.loc[frame.variant.eq(variant)]
        labels = int(group.action_labels_required.sum())
        rows.append({
            "variant": variant,
            "variant_label": LABELS[variant],
            "same_audit_budget_ceiling": maximum_budget,
            "action_labels_required": labels,
            "action_label_fraction_of_ceiling": labels / maximum_budget,
            "certificate_attempts": int(group.certificate_attempted.sum()),
            "released_domains": int(group.selected_action.ne("identity").sum()),
            "released_domain_names": ";".join(group.loc[group.selected_action.ne("identity"), "dataset"]),
            "mean_heldout_brier_gain_all_domains": float(group.heldout_brier_gain.mean()),
            "sum_heldout_brier_gain_all_domains": float(group.heldout_brier_gain.sum()),
            "material_negative_transfer_count": int(group.material_negative_transfer.sum()),
        })
    return pd.DataFrame(rows)


def make_figure(frame: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    colors = ["#6B7280", "#D9822B", "#3C78A8", "#16826C"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    ordered = summary.set_index("variant").loc[VARIANTS]
    axes[0].bar(x, ordered.action_labels_required, color=colors)
    axes[0].axhline(ordered.same_audit_budget_ceiling.iloc[0], color="#9CA3AF", linestyle="--", linewidth=0.9)
    axes[0].set(ylabel="Action-stage audit labels", title="A  Label demand (ceiling = 720)")

    axes[1].bar(x, ordered.mean_heldout_brier_gain_all_domains, color=colors)
    axes[1].axhline(0, color="#374151", linewidth=0.8)
    axes[1].set(ylabel="Mean held-out Brier gain", title="B  End-to-end outcome")

    action = frame.assign(released=frame.selected_action.ne("identity").astype(int)).pivot(index="dataset", columns="variant", values="released").reindex(index=DATASETS, columns=VARIANTS)
    axes[2].imshow(action.to_numpy(float), cmap=plt.matplotlib.colors.ListedColormap(["#F1F5F9", "#16826C"]), vmin=0, vmax=1, aspect="auto")
    axes[2].set_yticks(range(len(DATASETS)), DATASETS)
    axes[2].set(title="C  Released actions")
    for row in range(len(DATASETS)):
        for column in range(len(VARIANTS)):
            axes[2].text(column, row, "release" if action.iloc[row, column] else "identity", ha="center", va="center", fontsize=5.8, color="white" if action.iloc[row, column] else "#374151")

    short = ["Risk", "Cert.", "1 gate", "Full"]
    for axis in axes[:2]:
        axis.set_xticks(x, short, rotation=25, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        axis.set_axisbelow(True)
    axes[2].set_xticks(range(4), short, rotation=25, ha="right")
    for spine in axes[2].spines.values():
        spine.set_visible(False)
    fig.suptitle("Same-ceiling simplified comparators for frozen DCS-OPCT v11", fontsize=10.5, fontweight="bold")
    fig.savefig(output / "fig_end_to_end_simplified_comparators.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_end_to_end_simplified_comparators.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for path in (FREEZE, SOURCE, BUDGETS):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(FREEZE) != EXPECTED_FREEZE:
        raise RuntimeError("Frozen v11 hash mismatch")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_root.resolve()}")
    frame = build_long_table(pd.read_csv(SOURCE), pd.read_csv(BUDGETS))
    summary = summarize(frame)
    args.output_root.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_root / "end_to_end_comparator_domains.csv", index=False)
    summary.to_csv(args.output_root / "end_to_end_comparator_summary.csv", index=False)
    make_figure(frame, summary, args.output_root)
    conclusion = {
        "same_budget_ceiling": int(summary.same_audit_budget_ceiling.iloc[0]),
        "full_action_labels_required": int(summary.loc[summary.variant.eq("full_dcs_opct"), "action_labels_required"].iloc[0]),
        "full_label_reduction_vs_certificate_only": float(1 - summary.loc[summary.variant.eq("full_dcs_opct"), "action_labels_required"].iloc[0] / summary.loc[summary.variant.eq("certificate_only"), "action_labels_required"].iloc[0]),
        "release_set_preserved_across_calibration_variants": summary.loc[summary.variant.ne("risk_ranking_only"), "released_domain_names"].nunique() == 1,
        "claim_boundary": "The unlabeled gates reduced retrospective action-stage label demand while preserving the observed release set. This is a same-assignment component diagnostic, not proof of prospective necessity, universal safety, or external effectiveness.",
    }
    (args.output_root / "conclusion.json").write_text(json.dumps(conclusion, indent=2), encoding="utf-8")
    names = ["end_to_end_comparator_domains.csv", "end_to_end_comparator_summary.csv", "fig_end_to_end_simplified_comparators.pdf", "fig_end_to_end_simplified_comparators.png", "conclusion.json"]
    manifest = {
        "status": "completed_retrospective_same_ceiling_simplified_comparators",
        "analysis_date": "2026-09-10",
        "frozen_v11_unchanged": True,
        "analysis_role": "Post-hoc same-assignment, same-budget-ceiling component diagnostic using frozen GPU outputs.",
        "freeze_sha256": sha256(FREEZE),
        "inputs": {str(path): sha256(path) for path in (FREEZE, SOURCE, BUDGETS)},
        "script_sha256": sha256(Path(__file__)),
        "outputs": {name: sha256(args.output_root / name) for name in names},
        "claim_boundary": conclusion["claim_boundary"],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
