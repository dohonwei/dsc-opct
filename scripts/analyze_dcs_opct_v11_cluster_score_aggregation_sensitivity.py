from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs/material_event_threshold_robustness_final/leave_dataset_out_predictions.csv"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
OUTPUT = ROOT / "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
CLUSTER_COLUMNS = ["dataset", "task", "representation", "model", "split_seed"]
AGGREGATIONS = ("maximum", "mean", "median")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_pvalue(clusters: pd.DataFrame, selected: pd.DataFrame) -> float:
    distribution = np.array([1.0])
    for _, part in clusters.groupby("dataset", sort=True):
        draws = 12
        population = len(part)
        events = int(part.any_material_event.sum())
        support = np.arange(max(0, draws - (population - events)), min(draws, events) + 1)
        probability = np.zeros(support.max() + 1)
        probability[support] = hypergeom.pmf(support, population, events, draws)
        distribution = np.convolve(distribution, probability)
    return float(distribution[int(selected.any_material_event.sum()):].sum())


def main() -> None:
    if sha256(FREEZE) != EXPECTED_FREEZE:
        raise ValueError("Frozen v11 hash mismatch")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    frame = pd.read_csv(PREDICTIONS)
    frame = frame.loc[
        frame.held_out.eq(frame.dataset)
        & np.isclose(frame.material_effect_threshold, 0.02)
    ].copy()
    frame["material_excess"] = np.maximum(
        frame.dose_induced_amplification.to_numpy(float) - 0.02, 0.0
    )
    clusters = (
        frame.groupby(CLUSTER_COLUMNS, sort=True)
        .agg(
            maximum=("predicted_probability", "max"),
            mean=("predicted_probability", "mean"),
            median=("predicted_probability", "median"),
            event_configurations=("material_optimism_event", "sum"),
            any_material_event=("material_optimism_event", "max"),
            material_excess=("material_excess", "sum"),
        )
        .reset_index()
    )
    if len(clusters) != 120:
        raise ValueError("Expected 120 complete source-domain clusters")

    summary_rows = []
    selection_rows = []
    selections: dict[str, set[str]] = {}
    for aggregation in tqdm(
        AGGREGATIONS, desc="Cluster-score sensitivity", unit="aggregation"
    ):
        parts = []
        for _, part in clusters.groupby("dataset", sort=True):
            parts.append(
                part.sort_values(
                    [aggregation, "task", "representation", "model", "split_seed"],
                    ascending=[False, True, True, True, True],
                ).head(12)
            )
        selected = pd.concat(parts, ignore_index=True)
        selected["aggregation"] = aggregation
        selected["cluster_id"] = selected[CLUSTER_COLUMNS].astype(str).agg("|".join, axis=1)
        selections[aggregation] = set(selected.cluster_id)
        selection_rows.append(selected[["aggregation", "cluster_id", *CLUSTER_COLUMNS]])
        summary_rows.append(
            {
                "aggregation": aggregation,
                "selected_clusters": len(selected),
                "labeled_configurations": 4 * len(selected),
                "event_clusters_captured": int(selected.any_material_event.sum()),
                "cluster_event_sensitivity": float(selected.any_material_event.sum() / clusters.any_material_event.sum()),
                "event_configurations_captured": int(selected.event_configurations.sum()),
                "row_event_sensitivity": float(selected.event_configurations.sum() / clusters.event_configurations.sum()),
                "material_excess_captured": float(selected.material_excess.sum() / clusters.material_excess.sum()),
                "exact_stratified_p_one_sided": exact_pvalue(clusters, selected),
            }
        )

    overlap_rows = []
    for left, right in itertools.combinations(AGGREGATIONS, 2):
        intersection = selections[left] & selections[right]
        union = selections[left] | selections[right]
        overlap_rows.append(
            {
                "left": left,
                "right": right,
                "intersection_clusters": len(intersection),
                "jaccard": len(intersection) / len(union),
            }
        )

    OUTPUT.mkdir(parents=True)
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "aggregation_summary.csv", index=False)
    pd.concat(selection_rows, ignore_index=True).to_csv(
        OUTPUT / "selected_clusters.csv", index=False
    )
    pd.DataFrame(overlap_rows).to_csv(OUTPUT / "selection_overlap.csv", index=False)
    names = ["aggregation_summary.csv", "selected_clusters.csv", "selection_overlap.csv"]
    manifest = {
        "status": "completed_post_hoc_cluster_score_aggregation_sensitivity",
        "analysis_date": "2026-09-10",
        "frozen_v11_unchanged": True,
        "analysis_role": "post-hoc aggregation sensitivity, not aggregation selection",
        "primary_threshold": 0.02,
        "audit_budget_fraction": 0.20,
        "aggregations": list(AGGREGATIONS),
        "claim_boundary": (
            "Agreement across maximum, mean, and median cluster scores supports robustness "
            "of the retrospective source-domain audit-yield result only."
        ),
        "inputs": {str(PREDICTIONS.relative_to(ROOT)): sha256(PREDICTIONS), str(FREEZE.relative_to(ROOT)): sha256(FREEZE)},
        "script": sha256(Path(__file__).resolve()),
        "outputs": {name: sha256(OUTPUT / name) for name in names},
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nSelection overlap")
    print(pd.DataFrame(overlap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
