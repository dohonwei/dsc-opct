from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from emognition_v11_contract import MATERIAL_THRESHOLD, ROOT


OUTPUT = ROOT / "outputs/emognition_v11_preaccess"
CSV = OUTPUT / "endpoint_eligibility_operating_characteristics.csv"
REPORT = OUTPUT / "endpoint_eligibility_report.json"
FIGURES = (
    OUTPUT / "emognition_v11_endpoint_eligibility.png",
    OUTPUT / "emognition_v11_endpoint_eligibility.pdf",
)


def main() -> None:
    frame = pd.read_csv(CSV)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected_prevalence = {0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20}
    expected_icc = {0.0, 0.25, 0.50, 0.75, 1.0}
    checks = {
        "complete_grid": len(frame) == 35
        and set(frame.event_prevalence) == expected_prevalence
        and set(frame.intracluster_correlation) == expected_icc,
        "registered_cluster_design": bool(
            (frame.clusters_per_half == 30).all()
            and (frame.configurations_per_cluster == 4).all()
        ),
        "adequate_monte_carlo_repetitions": bool((frame.repetitions >= 50_000).all()),
        "probabilities_bounded": bool(
            frame.both_halves_mixed_probability.between(0, 1).all()
            and frame.both_halves_at_least_five_events_probability.between(0, 1).all()
        ),
        "eligibility_monotone_in_prevalence": all(
            np.all(
                np.diff(
                    group.sort_values("event_prevalence").both_halves_mixed_probability
                )
                >= -0.005
            )
            for _, group in frame.groupby("intracluster_correlation")
        ),
        "exact_boundaries_reproduced": bool(
            report["simulation"]["maximum_exact_boundary_absolute_error"] < 0.006
        ),
        "material_threshold_unchanged": bool(
            report["registered_design"]["material_event_threshold"] == MATERIAL_THRESHOLD
        ),
        "participant_values_not_accessed": report["participant_values_accessed"] is False
        and report["archive_present_during_execution"] is False,
        "diagnostic_not_promoted_to_gate": "not a registered acceptance criterion"
        in report["non_gate_diagnostic"],
        "replication_not_inferred": "replication"
        in report["interpretation_boundary"],
        "figures_present": all(
            path.is_file() and path.stat().st_size > 10_000 for path in FIGURES
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Emognition endpoint-eligibility checks failed: {failed}")
    print(f"Emognition endpoint-eligibility test passed: {len(checks)}/{len(checks)}")


if __name__ == "__main__":
    main()
