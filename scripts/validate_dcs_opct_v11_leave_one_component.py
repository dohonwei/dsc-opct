from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_leave_one_component"
MANIFEST = OUT / "manifest.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
DEV = ROOT / "outputs/distribution_covered_stratified_opct_v11_development"
SCRIPT = ROOT / "scripts/analyze_dcs_opct_v11_leave_one_component.py"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(actual, expected, rtol=0, atol=tolerance))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    same = pd.read_csv(OUT / "same_assignment_leave_one_component.csv")
    random_results = pd.read_csv(OUT / "random_half_allocation_repetitions.csv")
    random_summary = pd.read_csv(OUT / "distribution_coverage_summary.csv").set_index("dataset")
    frozen_certificates = pd.read_csv(DEV / "primary_certificates.csv").set_index("dataset")
    frozen_heldout = pd.read_csv(DEV / "primary_heldout_results.csv").set_index("dataset")
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record(
        "manifest_status_and_boundary",
        manifest.get("status") == "retrospective_frozen_leave_one_component_diagnostic_complete"
        and manifest.get("claim_status") == "diagnostic_only_no_gate_or_threshold_tuning",
        {"status": manifest.get("status"), "claim_status": manifest.get("claim_status")},
    )
    record(
        "freeze_hash_unchanged",
        sha256(FREEZE) == manifest.get("freeze_sha256") == EXPECTED_FREEZE_HASH,
        sha256(FREEZE),
    )
    record("analysis_script_hash", sha256(SCRIPT) == manifest.get("script_sha256"), sha256(SCRIPT))
    input_paths = {
        "v11_assignments": DEV / "distribution_covered_assignments.csv",
        "v11_components": DEV / "component_projection_diagnostics.csv",
        "v11_action_locks": DEV / "unlabeled_action_locks.csv",
        "v11_certificates": DEV / "primary_certificates.csv",
        "v11_heldout": DEV / "primary_heldout_results.csv",
        "v9_random_assignments": ROOT
        / "outputs/witness_gated_covariance_opct_v9_development/audit_cluster_assignments.csv",
    }
    input_mismatches = {
        name: {"recorded": manifest["input_sha256"].get(name), "actual": sha256(path)}
        for name, path in input_paths.items()
        if manifest["input_sha256"].get(name) != sha256(path)
    }
    record("input_hashes_current", not input_mismatches, input_mismatches)
    output_mismatches = {
        relative: {"recorded": expected, "actual": sha256(ROOT / relative)}
        for relative, expected in manifest.get("output_sha256", {}).items()
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected
    }
    record("output_hashes_current", not output_mismatches, output_mismatches)

    expected_variants = {
        "full_v11",
        "minus_quantile_witness",
        "minus_unlabeled_applicability",
    }
    record(
        "complete_same_assignment_grid",
        len(same) == 15
        and set(same.variant) == expected_variants
        and same.groupby("variant").dataset.nunique().eq(5).all(),
        same.groupby("variant").dataset.nunique().to_dict(),
    )
    full = same.loc[same.variant.eq("full_v11")].set_index("dataset")
    reproduction = {}
    reproduction_pass = True
    for dataset in full.index:
        expected_certified = bool(frozen_certificates.loc[dataset, "certified"])
        expected_gain = float(frozen_heldout.loc[dataset, "brier_gain"])
        actual_certified = bool(full.loc[dataset, "audit_certified"])
        actual_gain = float(full.loc[dataset, "heldout_brier_gain"])
        passed = actual_certified == expected_certified and close(actual_gain, expected_gain)
        reproduction_pass &= passed
        reproduction[dataset] = {
            "certified": actual_certified,
            "heldout_gain": actual_gain,
            "passed": passed,
        }
    record("full_v11_exactly_reproduced", reproduction_pass, reproduction)
    record(
        "full_v11_three_releases",
        int(full.audit_certified.sum()) == 3
        and set(full.index[full.audit_certified]) == {"EPPVR", "CASE", "CEAP"},
        full.audit_certified.to_dict(),
    )

    no_witness = same.loc[same.variant.eq("minus_quantile_witness")].set_index("dataset")
    record(
        "witness_removal_sends_dreamer_to_audit",
        bool(no_witness.loc["DREAMER", "candidate_eligible"])
        and not bool(no_witness.loc["DREAMER", "audit_certified"])
        and no_witness.loc["DREAMER", "audit_brier_gain_lcb"] < 0,
        no_witness.loc["DREAMER"].to_dict(),
    )
    no_screen = same.loc[same.variant.eq("minus_unlabeled_applicability")].set_index("dataset")
    record(
        "screen_removal_sends_seediv_to_negative_audit",
        bool(no_screen.loc["SEED-IV", "candidate_eligible"])
        and not bool(no_screen.loc["SEED-IV", "audit_certified"])
        and no_screen.loc["SEED-IV", "audit_brier_gain"] < -0.02,
        no_screen.loc["SEED-IV"].to_dict(),
    )
    record(
        "certificate_blocks_all_extra_candidates",
        int(no_witness.audit_certified.sum()) == 3
        and int(no_screen.audit_certified.sum()) == 3,
        {
            "without_witness": int(no_witness.audit_certified.sum()),
            "without_screen": int(no_screen.audit_certified.sum()),
        },
    )
    record(
        "no_released_material_negative_event",
        not same.material_negative_transfer.any() and not random_results.material_negative_transfer.any(),
        {
            "same_assignment_events": int(same.material_negative_transfer.sum()),
            "random_assignment_events": int(random_results.material_negative_transfer.sum()),
        },
    )
    record(
        "random_allocation_has_100_repetitions_per_domain",
        len(random_results) == 500
        and random_results.groupby("dataset").audit_repetition.nunique().eq(100).all(),
        random_results.groupby("dataset").audit_repetition.nunique().to_dict(),
    )
    expected_release = {"EPPVR": 0.99, "SEED-IV": 0.0, "DREAMER": 0.0, "CASE": 1.0, "CEAP": 0.60}
    actual_release = random_summary.random_release_rate.to_dict()
    record(
        "random_release_rates_reproduced",
        all(close(actual_release[key], value) for key, value in expected_release.items()),
        actual_release,
    )
    record(
        "distribution_coverage_resolves_ceap_instability",
        bool(random_summary.loc["CEAP", "covered_release"])
        and close(random_summary.loc["CEAP", "random_release_rate"], 0.60)
        and random_summary.loc["CEAP", "covered_heldout_brier_gain"]
        > random_summary.loc["CEAP", "random_mean_heldout_brier_gain"],
        random_summary.loc["CEAP"].to_dict(),
    )
    record(
        "order_preservation_unchanged",
        same.probability_rank.ge(0.999999).all() and same.order_inversions.eq(0).all(),
        {"minimum_rank": float(same.probability_rank.min()), "inversions": int(same.order_inversions.sum())},
    )
    table = (OUT / "table_leave_one_component.tex").read_text(encoding="utf-8")
    record(
        "latex_table_encodes_release_and_rejection",
        "R (+0.0263)" in table and "C-reject" in table and " & I & I & " in table,
        "R, C-reject, and I states present",
    )

    passed = sum(check["passed"] for check in checks)
    path = OUT / "independent_validation_report.json"
    report_content = {
        "status": "passed" if passed == len(checks) else "failed",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_content = {
            key: value for key, value in previous.items() if key != "generated_at_utc"
        }
        if previous_content == report_content:
            generated_at = previous.get("generated_at_utc", generated_at)
    report = {
        "status": report_content["status"],
        "generated_at_utc": generated_at,
        "checks_passed": report_content["checks_passed"],
        "checks_total": report_content["checks_total"],
        "checks": report_content["checks"],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"Leave-one-component validation failed: {failed}")
    print(f"Leave-one-component validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
