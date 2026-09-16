from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CONFIGURATIONS = {
    "CASE": 320,
    "CEAP": 320,
    "DREAMER": 240,
    "EPPVR": 320,
    "SEED-IV": 240,
}
EXPECTED_HELDOUT = {dataset: count // 2 for dataset, count in EXPECTED_CONFIGURATIONS.items()}
RELEASE_DATASETS = {"CASE", "EPPVR"}
ABSTAIN_DATASETS = {"CEAP", "DREAMER", "SEED-IV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configurations = pd.read_csv(args.output_root / "target_configuration_soft_events.csv")
    summary = pd.read_csv(args.output_root / "inductive_action_soft_endpoint_summary.csv")
    report = json.loads((args.output_root / "analysis_report.json").read_text(encoding="utf-8"))

    heldout = configurations.loc[configurations.partition.eq("heldout")]
    checks = {
        "analysis_completed": report.get("status") == "completed",
        "cuda_device_recorded": "NVIDIA" in report.get("device", ""),
        "bootstrap_repetitions_at_least_5000": report.get("bootstrap_repetitions", 0) >= 5000,
        "all_datasets_present": set(configurations.dataset.unique()) == set(EXPECTED_CONFIGURATIONS),
        "configuration_counts_exact": configurations.groupby("dataset").size().to_dict() == EXPECTED_CONFIGURATIONS,
        "heldout_counts_exact": heldout.groupby("dataset").size().to_dict() == EXPECTED_HELDOUT,
        "partitions_complete": set(configurations.partition.unique()) == {"audit", "heldout"},
        "soft_probabilities_valid": configurations.probability_material_event.between(0, 1).all(),
        "point_reconstruction_exact": configurations.point_reconstruction_error.max() < 1e-12,
        "bootstrap_structure_explicit": (
            configurations.groupby("dataset").bootstrap_structure.unique().map(list).to_dict()
            == {
                "CASE": ["participant_x_stimulus"],
                "CEAP": ["participant_x_stimulus"],
                "DREAMER": ["participant_x_stimulus"],
                "EPPVR": ["participant"],
                "SEED-IV": ["participant_x_stimulus"],
            }
        ),
        "released_actions_positive_under_soft_endpoint": (
            summary.set_index("dataset").loc[list(RELEASE_DATASETS), "soft_brier_gain"] > 0
        ).all(),
        "abstained_actions_unchanged": np.allclose(
            summary.set_index("dataset").loc[list(ABSTAIN_DATASETS), ["hard_brier_gain", "soft_brier_gain"]],
            0.0,
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    validation = {
        "status": "passed" if not failed else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": int(sum(bool(value) for value in checks.values())),
        "checks_total": len(checks),
        "checks": {name: bool(value) for name, value in checks.items()},
        "failed_checks": failed,
        "maximum_point_reconstruction_error": float(configurations.point_reconstruction_error.max()),
        "claim_boundary": (
            "This validates a post-hoc probabilistic-endpoint sensitivity, not the frozen primary endpoint. "
            "EPPVR is participant-clustered because a shared physical-stimulus identifier is unavailable."
        ),
    }
    output = args.output_root / "independent_validation_report.json"
    output.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
