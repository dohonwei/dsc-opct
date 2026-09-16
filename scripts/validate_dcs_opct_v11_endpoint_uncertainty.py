from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ENDPOINT = Path("outputs/dcs_opct_v11_crossed_cluster_endpoint_uncertainty_20260914_v2")
RISK = Path("outputs/dcs_opct_v11_probabilistic_event_risk_20260914")

def main() -> None:
    endpoint = pd.read_csv(ENDPOINT / "crossed_cluster_event_uncertainty.csv")
    summary = pd.read_csv(ENDPOINT / "crossed_cluster_event_summary.csv").set_index("dataset")
    metrics = pd.read_csv(RISK / "probabilistic_event_lodo_metrics.csv").set_index("held_out")
    coefficients = pd.read_csv(RISK / "probabilistic_event_coefficients.csv")
    endpoint_report = json.loads((ENDPOINT / "analysis_report.json").read_text(encoding="utf-8"))
    risk_report = json.loads((RISK / "analysis_report.json").read_text(encoding="utf-8"))
    signs = coefficients.assign(sign=np.sign(coefficients.coefficient)).pivot(index="feature", columns="held_out", values="sign")
    checks = {
        "endpoint_rows_480": len(endpoint) == 480,
        "two_source_domains": set(summary.index) == {"DEAP", "MAHNOB-HCI"},
        "point_estimate_reconstructs": float(endpoint.point_reconstruction_error.max()) < 1e-12,
        "all_bootstraps_valid": bool(endpoint.valid_bootstraps.eq(5000).all()),
        "no_lcb_events": int(summary.lcb_events.sum()) == 0,
        "threshold_uncertainty_visible": int(summary.ci_crossing.sum()) >= 470,
        "soft_brier_improves_both": bool(metrics.probabilistic_model_soft_brier.lt(metrics.frozen_hard_model_soft_brier).all()),
        "hard_auroc_not_inflated": bool(metrics.probabilistic_model_hard_auroc.lt(metrics.frozen_hard_model_hard_auroc).all()),
        "coefficient_instability_visible": bool((signs.nunique(axis=1) > 1).any()),
        "cuda_recorded": "4070 Ti SUPER" in endpoint_report["device"],
        "endpoint_boundary_explicit": "shared training data" in endpoint_report["claim_boundary"],
        "risk_boundary_explicit": "does not replace" in risk_report["claim_boundary"],
    }
    report = {"status": "passed" if all(checks.values()) else "failed", "checks_passed": int(sum(checks.values())), "checks_total": len(checks), "checks": checks}
    (RISK / "independent_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
