from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DATASETS = {"CASE", "CEAP", "DREAMER", "EPPVR", "SEED-IV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpcs = pd.read_csv(args.output_root / "cpcs_style_certified_outcomes.csv")
    comparison = pd.read_csv(args.output_root / "dcs_vs_cpcs_style.csv")
    atc = pd.read_csv(args.output_root / "atc_style_accuracy_estimation.csv")
    parameters = pd.read_csv(args.output_root / "cpcs_style_parameters.csv")
    report = json.loads((args.output_root / "analysis_report.json").read_text(encoding="utf-8"))
    checks = {
        "analysis_completed": report.get("status") == "completed",
        "cuda_device_recorded": "NVIDIA" in report.get("device", ""),
        "all_cpcs_datasets_present": set(cpcs.dataset) == DATASETS,
        "all_atc_datasets_present": set(atc.dataset) == DATASETS,
        "candidate_fit_uses_no_target_labels": cpcs.target_labels_for_candidate_fit.eq(0).all(),
        "certificate_budget_matches_inductive_dcs": (
            comparison.audit_labels_for_certificate == comparison.audit_configurations
        ).all(),
        "cpcs_positive_slope": parameters.scale.gt(0).all(),
        "importance_weight_ratio_bounded": (
            (parameters.weight_max / parameters.weight_min).le(400.0 + 1e-6)
        ).all(),
        "cpcs_release_result_explicit": not cpcs.released.any(),
        "cpcs_return_original_has_zero_gain": cpcs.heldout_brier_gain.eq(0).all(),
        "atc_is_noninterventional": (~atc.intervention_released).all(),
        "atc_errors_finite": atc.absolute_estimation_error.notna().all(),
        "atc_error_range_valid": atc.absolute_estimation_error.between(0, 1).all(),
        "transcal_noncomparability_boundary_recorded": "TransCal is not instantiated" in report.get("claim_boundary", ""),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    validation = {
        "status": "passed" if not failed else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "checks": {name: bool(value) for name, value in checks.items()},
        "failed_checks": failed,
    }
    (args.output_root / "independent_validation_report.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
