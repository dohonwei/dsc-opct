from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary = pd.read_csv(OUTPUT / "aggregation_summary.csv")
    selected = pd.read_csv(OUTPUT / "selected_clusters.csv")
    overlap = pd.read_csv(OUTPUT / "selection_overlap.csv")
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    checks = {
        "three_aggregations": set(summary.aggregation) == {"maximum", "mean", "median"},
        "fixed_budget": (summary.selected_clusters.eq(24) & summary.labeled_configurations.eq(96)).all(),
        "identical_event_yield": summary.event_clusters_captured.eq(19).all() and summary.event_configurations_captured.eq(43).all(),
        "identical_exact_p": np.allclose(summary.exact_stratified_p_one_sided, 0.0003446716618680),
        "strong_selection_overlap": len(overlap) == 3 and (overlap.jaccard >= 0.92).all(),
        "selection_rows": len(selected) == 72 and not selected.duplicated(["aggregation", "cluster_id"]).any(),
        "bounded_role": manifest["analysis_role"] == "post-hoc aggregation sensitivity, not aggregation selection",
        "output_hashes": all(sha256(OUTPUT / name) == value for name, value in manifest["outputs"].items()),
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "analysis_date": "2026-09-10",
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "checks": checks,
    }
    (OUTPUT / "independent_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"{report['checks_passed']}/{report['checks_total']} checks passed")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
