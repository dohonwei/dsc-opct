from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {"CASE": 320, "CEAP": 320, "DREAMER": 240, "EPPVR": 320, "SEED-IV": 240}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcomes = pd.read_csv(args.output_root / "raw_group_independent_action_outcomes.csv")
    endpoints = pd.read_csv(args.output_root / "raw_group_independent_endpoints.csv")
    metadata = pd.read_csv(args.output_root / "raw_partition_metadata.csv")
    report = json.loads((args.output_root / "analysis_report.json").read_text(encoding="utf-8"))
    indexed = outcomes.set_index("dataset")
    structure = metadata.set_index("dataset").raw_independence_structure.to_dict()
    checks = {
        "analysis_completed": report.get("status") == "completed",
        "cuda_device_recorded": "NVIDIA" in report.get("device", ""),
        "bootstrap_repetitions_at_least_5000": report.get("bootstrap_repetitions", 0) >= 5000,
        "all_datasets_present": set(outcomes.dataset) == set(EXPECTED),
        "configuration_counts_exact": endpoints.groupby("dataset").size().to_dict() == EXPECTED,
        "configuration_partitions_complete": set(endpoints.partition.unique()) == {"audit", "heldout"},
        "event_labels_binary": set(np.unique(endpoints[["audit_event", "evaluation_event"]].to_numpy())) <= {0, 1},
        "test_rows_nonempty": (endpoints[["audit_test_rows", "evaluation_test_rows"]] > 0).all().all(),
        "eppvr_participant_disjoint": structure.get("EPPVR") == "participant_disjoint_test_rows",
        "other_domains_participant_stimulus_disjoint": all(
            structure.get(dataset) == "participant_and_physical_stimulus_disjoint_test_rows"
            for dataset in set(EXPECTED) - {"EPPVR"}
        ),
        "only_eppvr_released": set(outcomes.loc[outcomes.selected_method.ne("identity"), "dataset"]) == {"EPPVR"},
        "released_gain_positive_and_no_material_harm": bool(
            indexed.loc["EPPVR", "heldout_brier_gain"] > 0
            and not indexed.loc["EPPVR", "material_negative_transfer"]
        ),
        "case_failed_certificate": bool(
            indexed.loc["CASE", "audit_brier_gain_lcb"] < 0.001
            and indexed.loc["CASE", "selected_method"] == "identity"
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    validation = {
        "status": "passed" if not failed else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "checks": {name: bool(value) for name, value in checks.items()},
        "failed_checks": failed,
        "claim_boundary": (
            "Raw test participants and, where identifiable, physical stimuli are disjoint. "
            "Shared model-training rows and repeated split-seed fits remain unresolved."
        ),
    }
    (args.output_root / "independent_validation_report.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
