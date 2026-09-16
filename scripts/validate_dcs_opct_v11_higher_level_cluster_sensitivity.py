from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910"
SOURCE = ROOT / "outputs/material_event_threshold_robustness_final/leave_dataset_out_predictions.csv"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
MANUSCRIPT = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex"
SUPPLEMENT = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex"
EXPECTED = {
    FREEZE: "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee",
    MANUSCRIPT: "dcab80f4727be5bc67586989353c36164b66e2aedf4e364ffa0da32f22c1ed79",
    SUPPLEMENT: "eea5ea34e2a767cfde70fc62e831b2a12429b9aa178a44a3e73f6dccb5820b3f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    blocks = pd.read_csv(OUT / "dataset_task_seed_blocks.csv")
    summary = pd.read_csv(OUT / "higher_level_audit_yield.csv")
    null = pd.read_csv(OUT / "exact_random_block_allocations.csv")
    intervals = pd.read_csv(OUT / "higher_level_bootstrap_intervals.csv")
    conclusion = json.loads((OUT / "bounded_conclusion.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record("locked_hashes", all(sha256(path) == expected for path, expected in EXPECTED.items()),
           {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED})
    record("source_hash", sha256(SOURCE) == manifest["input_sha256"], sha256(SOURCE))
    record("block_dimensions", len(blocks) == 20 and set(blocks.rows) == {24}
           and not blocks.duplicated(["dataset", "task", "split_seed"]).any(),
           [len(blocks), sorted(blocks.rows.unique().tolist())])
    record("source_domain_count", blocks.dataset.nunique() == 2,
           sorted(blocks.dataset.unique().tolist()))
    expected_allocations = int(np.prod([
        len(list(itertools.combinations(range(len(group)), 2)))
        for _, group in blocks.groupby("dataset")
    ]))
    record("exact_null_complete", len(null) == expected_allocations == 2025,
           [len(null), expected_allocations])
    record("common_label_budget", set(summary.selected_blocks) == {4}
           and set(summary.labeled_rows) == {96},
           summary[["score_aggregation", "selected_blocks", "labeled_rows"]].to_dict("records"))
    record("aggregation_invariance",
           summary[["event_row_sensitivity", "material_excess_captured", "event_enrichment"]].nunique().eq(1).all(),
           summary.to_dict("records"))
    max_row = summary.loc[summary.score_aggregation.eq("max_risk")].iloc[0]
    recomputed_p_event = float(np.mean(null.event_row_sensitivity >= max_row.event_row_sensitivity - 1e-15))
    recomputed_p_excess = float(np.mean(null.material_excess_captured >= max_row.material_excess_captured - 1e-15))
    record("exact_pvalues_recompute",
           np.isclose(max_row.exact_random_p_event_row_sensitivity, recomputed_p_event)
           and np.isclose(max_row.exact_random_p_material_excess, recomputed_p_excess),
           [recomputed_p_event, recomputed_p_excess])
    record("higher_level_result_is_not_significant",
           max_row.exact_random_p_event_row_sensitivity > 0.05
           and max_row.exact_random_p_material_excess > 0.05,
           [float(max_row.exact_random_p_event_row_sensitivity), float(max_row.exact_random_p_material_excess)])
    record("bootstrap_intervals", len(intervals) == 2
           and set(intervals.independent_units) == {20}
           and (intervals.block_bootstrap_width > 0).all(), intervals.to_dict("records"))
    record("bounded_claim",
           all(phrase in conclusion["claim_boundary"] for phrase in [
               "conditional post-hoc sensitivity", "do not create independent future-domain replication",
               "prospective effectiveness", "universal non-harm",
           ]), conclusion["claim_boundary"])
    image = mpimg.imread(OUT / "fig_higher_level_cluster_sensitivity.png")
    record("figure_dimensions", image.shape[1] >= 3000 and image.shape[0] >= 1200, list(image.shape))
    record("manifest_outputs",
           all(sha256(OUT / name) == value for name, value in manifest["outputs"].items()),
           len(manifest["outputs"]))

    report = {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "claim_boundary": conclusion["claim_boundary"],
    }
    (OUT / "independent_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError([check["name"] for check in checks if not check["passed"]])
    print(f"Higher-level cluster sensitivity validation passed: {report['checks_passed']}/{report['checks_total']}")


if __name__ == "__main__":
    main()
