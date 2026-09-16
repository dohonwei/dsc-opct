from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FREEZE = Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json")
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
STABILITY = Path("outputs/dcs_opct_v11_frozen_action_stability_20260910_v2")
BUDGETS = Path("outputs/distribution_covered_stratified_opct_v11_development/primary_heldout_results.csv")
OUTCOMES = Path("outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/six_dataset_action_outcome.csv")
EXTERNAL_AUDITS = {
    "AVDOS-VR": Path("outputs/avdos_v11_exploratory_ppg/primary_audit_labeled.csv"),
    "FACED": Path("outputs/faced_v11_post_access_common_montage_dcs_opct/primary_audit_labeled.csv"),
    "EEGEmotions-27": Path("outputs/eegemotions27_v11_external_robustness/outcome/primary_audit_labeled.csv"),
}
DATASETS = ["EPPVR", "CASE", "CEAP", "DREAMER", "SEED-IV", "AVDOS-VR", "FACED", "EEGEmotions-27"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a worked audit example for frozen DCS-OPCT v11.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dcs_opct_v11_worked_audit_example_20260910_v4"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_trace() -> pd.DataFrame:
    actions = pd.read_csv(STABILITY / "one_at_a_time_action_stability.csv")
    frozen = actions.loc[actions.grid_index.eq(2)].drop_duplicates("dataset").set_index("dataset")
    margins = pd.read_csv(STABILITY / "domain_gate_margins.csv")
    values = margins.pivot(index="dataset", columns="parameter", values="observed_value")
    budgets = pd.read_csv(BUDGETS).set_index("dataset")
    outcomes = pd.read_csv(OUTCOMES).set_index("dataset")
    external_roles = {
        "FACED": "post-access exploratory",
        "EEGEmotions-27": "separate pre-signal external robustness test",
    }
    external_gain = {"FACED": np.nan, "EEGEmotions-27": np.nan}
    rows = []
    for dataset in DATASETS:
        row = frozen.loc[dataset]
        applicable = bool(row.applicable)
        certified = bool(row.certified)
        if dataset in budgets.index:
            ceiling = int(budgets.loc[dataset, "n_audit_configurations"])
        else:
            ceiling = len(pd.read_csv(EXTERNAL_AUDITS[dataset]))
        role = external_roles.get(dataset, outcomes.loc[dataset, "evidence_role"] if dataset in outcomes.index else "retrospective development")
        gain = external_gain.get(dataset, float(outcomes.loc[dataset, "heldout_brier_gain"]) if dataset in outcomes.index else np.nan)
        rows.append({
            "dataset": dataset,
            "evidence_role": role,
            "component_gate": "pass" if bool(row.component_pass) else "stop",
            "max_absolute_component_shift": float(values.loc[dataset, "maximum_probability_mean_shift"]),
            "witness_gate": "not reached" if not bool(row.component_pass) else ("pass" if bool(row.witness_pass) else "stop"),
            "directional_agreement": float(values.loc[dataset, "minimum_directional_agreement"]),
            "displacement_cosine": float(values.loc[dataset, "minimum_displacement_cosine"]),
            "normalized_disagreement": float(values.loc[dataset, "maximum_normalized_disagreement"]),
            "audit_budget_ceiling": ceiling,
            "action_labels_accessed": ceiling if applicable else 0,
            "certificate_gate": "pass" if certified else ("not reached" if not applicable else "stop"),
            "candidate_brier_gain_lcb": float(values.loc[dataset, "minimum_brier_gain_lcb"]),
            "selected_action": row.selected_action,
            "heldout_brier_gain": gain,
            "claim_status": "retrospective calibrated release" if certified else "non-intervention; not effectiveness or safety success",
            "stop_reason": row.failure_stage,
        })
    return pd.DataFrame(rows)


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    release = frame.loc[frame.dataset.eq("EPPVR")].iloc[0]
    boundary = frame.loc[frame.dataset.eq("EEGEmotions-27")].iloc[0]
    text = f"""# Worked DCS-OPCT v11 audit example

This example uses frozen probabilities, thresholds, assignments, and certificates. It is a usage trace, not a new experiment or threshold-selection exercise.

## Released development pathway: EPPVR

1. **Unlabeled component screen.** The largest absolute component mean shift was {release.max_absolute_component_shift:.3f}, below the frozen 0.10 maximum; rank and inversion constraints also passed.
2. **Independent witness.** Directional agreement was {release.directional_agreement:.3f}, displacement cosine was {release.displacement_cosine:.3f}, and normalized disagreement was {release.normalized_disagreement:.3f}; all three frozen witness conditions passed.
3. **Distribution-covered audit.** Only after the unlabeled gates passed were {int(release.action_labels_accessed)} configuration labels used for the action certificate.
4. **Certificate and action.** The one-sided Brier-gain lower bound was {release.candidate_brier_gain_lcb:.4f}, above 0.001, so CORAL + OPCT was released. The held-out Brier gain was {release.heldout_brier_gain:.4f}.

## Pre-signal external boundary: EEGEmotions-27

1. **Unlabeled component screen.** The largest absolute component mean shift was {boundary.max_absolute_component_shift:.3f}, exceeding 0.10.
2. **Fail-closed stop.** The witness and labeled certificate were not allowed to rescue the candidate; action-stage labels accessed = 0.
3. **Action and interpretation.** The selected action was identity. This is non-intervention and cannot be counted as effectiveness, non-harm, or successful external calibration.

## Interpretation rule

The framework has three distinct outputs: released correction, pre-label abstention, or post-audit certificate rejection. Only a released correction with an independently evaluated outcome can contribute evidence about calibration effectiveness. Identity fallback records that the proposed correction was not applied; it does not validate the original probabilities.
"""
    path.write_text(text, encoding="utf-8")


def make_figure(frame: pd.DataFrame, output: Path) -> None:
    stages = ["Component", "Witness", "Certificate", "Release"]
    matrix = np.zeros((len(frame), len(stages)))
    for index, row in frame.iterrows():
        matrix[index, 0] = 1 if row.component_gate == "pass" else -1
        matrix[index, 1] = 1 if row.witness_gate == "pass" else (0 if row.witness_gate == "not reached" else -1)
        matrix[index, 2] = 1 if row.certificate_gate == "pass" else (0 if row.certificate_gate == "not reached" else -1)
        matrix[index, 3] = 1 if row.selected_action != "identity" else 0
    cmap = plt.matplotlib.colors.ListedColormap(["#C94C4C", "#EEF2F6", "#16826C"])
    norm = plt.matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(4), stages)
    ax.set_yticks(range(len(frame)), frame.dataset)
    for row, column in np.ndindex(matrix.shape):
        label = "PASS" if matrix[row, column] == 1 else "STOP" if matrix[row, column] == -1 else "N/A"
        ax.text(column, row, label, ha="center", va="center", fontsize=7, color="white" if matrix[row, column] else "#4B5563", fontweight="bold")
    for index, row in frame.iterrows():
        ax.text(4.25, index, f"labels: {int(row.action_labels_accessed)}", va="center", fontsize=7.2, color="#374151")
        ax.text(5.9, index, row.claim_status, va="center", fontsize=6.7, color="#16826C" if row.selected_action != "identity" else "#6B7280")
    ax.set_xlim(-0.5, 10.0)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Worked frozen audit trace: label access follows unlabeled eligibility", loc="left", pad=24, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6.6, pad=7)
    ax.text(4.25, -0.72, "Labels used", fontsize=6.8, fontweight="bold")
    ax.text(5.9, -0.72, "Interpretation", fontsize=6.8, fontweight="bold")
    fig.savefig(output / "fig_worked_audit_trace.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_worked_audit_trace.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    inputs = [FREEZE, STABILITY / "one_at_a_time_action_stability.csv", STABILITY / "domain_gate_margins.csv", BUDGETS, OUTCOMES, *EXTERNAL_AUDITS.values()]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(FREEZE) != EXPECTED_FREEZE:
        raise RuntimeError("Frozen v11 hash mismatch")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_root.resolve()}")
    frame = build_trace()
    args.output_root.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_root / "worked_audit_trace.csv", index=False)
    write_markdown(frame, args.output_root / "worked_audit_example.md")
    make_figure(frame, args.output_root)
    names = ["worked_audit_trace.csv", "worked_audit_example.md", "fig_worked_audit_trace.pdf", "fig_worked_audit_trace.png"]
    manifest = {
        "status": "completed_frozen_worked_audit_example",
        "analysis_date": "2026-09-10",
        "frozen_v11_unchanged": True,
        "analysis_role": "Worked usage trace from frozen artifacts; no new model fitting, threshold selection, or manuscript integration.",
        "freeze_sha256": sha256(FREEZE),
        "inputs": {str(path): sha256(path) for path in inputs},
        "script_sha256": sha256(Path(__file__)),
        "outputs": {name: sha256(args.output_root / name) for name in names},
        "claim_boundary": "Identity is non-intervention. EEGEmotions-27 remains a separate pre-signal external robustness test, not external effectiveness evidence.",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(frame[["dataset", "stop_reason", "action_labels_accessed", "selected_action"]].to_string(index=False))


if __name__ == "__main__":
    main()
