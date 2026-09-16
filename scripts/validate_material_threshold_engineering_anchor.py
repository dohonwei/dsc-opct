from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "material_threshold_engineering_anchor"
FREEZE = ROOT / "docs" / "distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    margins = pd.read_csv(OUTPUT / "margin_vulnerability_summary.csv")
    audit = pd.read_csv(OUTPUT / "audit_budget_utility.csv")
    curve = pd.read_csv(OUTPUT / "configuration_decision_curve.csv")
    units = pd.read_csv(OUTPUT / "ranking_unit_margins.csv")
    pairs = pd.read_csv(OUTPUT / "candidate_pair_margins.csv")
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record("freeze_hash", sha256(FREEZE) == EXPECTED_FREEZE, sha256(FREEZE))
    record("frozen_boundary", manifest["frozen_v11_unchanged"] is True, manifest["analysis_role"])
    record("ranking_dimensions", len(units) == 20 and len(pairs) == 120, [len(units), len(pairs)])
    record("summary_dimensions", len(margins) == 5 and len(audit) == 25 and len(curve) == 63, [len(margins), len(audit), len(curve)])
    primary_margin = margins.loc[np.isclose(margins.distortion_threshold, 0.02)].iloc[0]
    record("primary_top2_fraction", np.isclose(primary_margin.top2_margin_at_or_below, 0.85), float(primary_margin.top2_margin_at_or_below))
    record("primary_pair_fraction", np.isclose(primary_margin.pair_margin_at_or_below, 65 / 120), float(primary_margin.pair_margin_at_or_below))
    primary_audit = audit.loc[np.isclose(audit.material_effect_threshold, 0.02) & np.isclose(audit.audit_budget_fraction, 0.20)].iloc[0]
    record("primary_budget_enrichment", primary_audit.event_enrichment > 1.0 and primary_audit.event_enrichment_ci_low > 1.0, [float(primary_audit.event_enrichment), float(primary_audit.event_enrichment_ci_low)])
    record("primary_event_capture_above_random", primary_audit.event_sensitivity_ci_low > 0.20, [float(primary_audit.event_sensitivity), float(primary_audit.event_sensitivity_ci_low)])
    record("primary_amplification_capture_above_random", primary_audit.material_amplification_captured_ci_low > 0.20, [float(primary_audit.material_amplification_captured), float(primary_audit.material_amplification_captured_ci_low)])
    risk = curve.loc[curve.policy.eq("risk_screen")]
    positive_costs = risk.loc[risk.net_benefit_vs_best_default > 0, "false_audit_cost_ratio"].tolist()
    record("bounded_decision_utility", positive_costs == [0.25, 0.30, 0.35, 0.40, 0.45], positive_costs)
    image = mpimg.imread(OUTPUT / "fig_material_threshold_engineering_anchor.png")
    record("figure_dimensions", image.shape[0] >= 1600 and image.shape[1] >= 2000, list(image.shape))
    record("output_hashes", all(sha256(OUTPUT / name) == value for name, value in manifest["outputs"].items()), len(manifest["outputs"]))

    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "analysis_date": "2026-09-08",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    (OUTPUT / "independent_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['checks_passed']}/{report['checks_total']} checks passed")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
