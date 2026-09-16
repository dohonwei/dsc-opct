from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dcs_opct_v11_inductive_target_sensitivity_20260914"),
    )
    return parser.parse_args()


def main() -> None:
    root = parse_args().output_root
    outcomes = pd.read_csv(root / "inductive_action_outcomes.csv").set_index("dataset")
    assignments = pd.read_csv(root / "covariate_blind_assignments.csv")
    predictions = pd.read_csv(root / "inductive_heldout_predictions.csv")
    comparison = pd.read_csv(root / "inductive_vs_transductive.csv").set_index("dataset")
    report = json.loads((root / "analysis_report.json").read_text(encoding="utf-8"))
    checks = {
        "five_domains": set(outcomes.index) == {"EPPVR", "SEED-IV", "DREAMER", "CASE", "CEAP"},
        "balanced_partitions": bool(
            assignments.groupby("dataset").partition.value_counts().unstack(fill_value=0).nunique(axis=1).eq(1).all()
        ),
        "assignment_is_covariate_blind": set(assignments.assignment_basis) == {"stratum_order_and_seed_only"},
        "prediction_partition_complete": bool(predictions.groupby("dataset").partition.nunique().eq(2).all()),
        "identity_reconstructs": bool(outcomes.identity_reconstruction_max_abs_error.lt(1e-6).all()),
        "eppvr_case_release": bool(outcomes.loc[["EPPVR", "CASE"], "selected_method"].eq("wg_opct").all()),
        "eppvr_case_positive_gain": bool(outcomes.loc[["EPPVR", "CASE"], "heldout_brier_gain"].gt(0).all()),
        "ceap_inductive_abstention": outcomes.loc["CEAP", "selected_method"] == "identity",
        "unsupported_domains_abstain": bool(outcomes.loc[["SEED-IV", "DREAMER"], "selected_method"].eq("identity").all()),
        "zero_material_negative_release": int(outcomes.material_negative_transfer.sum()) == 0,
        "transductive_reference_present": bool(comparison.transductive_heldout_brier_gain.notna().all()),
        "boundary_explicit": "participant or stimulus" in report["claim_boundary"],
    }
    validation = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "checks": checks,
    }
    (root / "independent_validation_report.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    if validation["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
