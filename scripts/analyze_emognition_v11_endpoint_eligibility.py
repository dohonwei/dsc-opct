from __future__ import annotations

import argparse
import json
from pathlib import Path

from emognition_v11_contract import DEFAULT_DATA_ROOT, ROOT, verify_preaccess_contract
from external_endpoint_eligibility_core import (
    CONFIGURATIONS_PER_CLUSTER,
    INTRACLUSTER_CORRELATIONS,
    PREVALENCES,
    make_figure,
    maximum_boundary_error,
    simulate_grid,
)


DEFAULT_OUTPUT = ROOT / "outputs/emognition_v11_preaccess"
CLUSTERS_PER_HALF = 30
DEFAULT_REPETITIONS = 50_000
DEFAULT_SEED = 20260909


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Emognition endpoint-eligibility operating characteristics "
            "without accessing participant data or changing the frozen gate."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reservation, _, freeze = verify_preaccess_contract()
    if DEFAULT_DATA_ROOT.exists():
        raise RuntimeError(
            "This planning analysis must be completed before the Emognition archive is present."
        )
    if args.repetitions < 10_000:
        raise ValueError("At least 10,000 repetitions are required")
    counterfactual = reservation["counterfactual_contract"]
    expected_clusters = (
        len(reservation["task_contract"]["tasks"])
        * len(counterfactual["models"])
        * len(reservation["feature_contract"]["representations"])
        * len(counterfactual["split_seeds"])
    )
    if expected_clusters != 2 * CLUSTERS_PER_HALF:
        raise RuntimeError("The registered Emognition cluster count is not 60")

    results = simulate_grid(
        dataset="Emognition",
        clusters_per_half=CLUSTERS_PER_HALF,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "endpoint_eligibility_operating_characteristics.csv"
    results.to_csv(csv_path, index=False)
    make_figure(
        results,
        args.output_root,
        "emognition_v11_endpoint_eligibility",
        "60 four-dose clusters; 30 audit and 30 held out. The five-event panel is not an acceptance gate.",
    )
    report = {
        "status": "preaccess_planning_analysis_complete",
        "date": "2026-09-09",
        "dataset": "Emognition Wearable Dataset 2020",
        "participant_values_accessed": False,
        "archive_present_during_execution": False,
        "registered_design": {
            "configuration_clusters": expected_clusters,
            "configurations_per_cluster": CONFIGURATIONS_PER_CLUSTER,
            "audit_clusters": CLUSTERS_PER_HALF,
            "heldout_clusters": CLUSTERS_PER_HALF,
            "material_event_threshold": counterfactual["material_event_threshold"],
            "endpoint_eligibility": reservation["acceptance_gate"]["endpoint_eligibility"],
        },
        "simulation": {
            "model": "exchangeable beta-binomial four-dose clusters",
            "event_prevalences": list(PREVALENCES),
            "intracluster_correlations": list(INTRACLUSTER_CORRELATIONS),
            "repetitions_per_cell": args.repetitions,
            "seed": args.seed,
            "maximum_exact_boundary_absolute_error": maximum_boundary_error(results),
        },
        "interpretation_boundary": (
            "This analysis quantifies only the probability that both partitions contain "
            "event and non-event configurations under hypothetical prevalence and "
            "within-cluster dependence. It does not estimate Emognition prevalence, "
            "calibration power, applicability, effectiveness, non-harm, safety, or replication."
        ),
        "non_gate_diagnostic": (
            "The probability that both halves contain at least five events is reported "
            "for planning context only and is not a registered acceptance criterion."
        ),
        "freeze_id": freeze["freeze_id"],
    }
    (args.output_root / "endpoint_eligibility_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "Emognition pre-access endpoint-eligibility analysis complete; "
        f"maximum exact-boundary error={report['simulation']['maximum_exact_boundary_absolute_error']:.4f}"
    )


if __name__ == "__main__":
    main()
