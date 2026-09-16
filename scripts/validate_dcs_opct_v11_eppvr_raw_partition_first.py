from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PARTITIONS = {"audit_population", "evaluation_population"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914"),
    )
    return parser.parse_args()


def subjects(value: object) -> set[str]:
    if pd.isna(value) or str(value) == "":
        return set()
    return set(str(value).split(";"))


def main() -> None:
    args = parse_args()
    required = [
        "participant_population_assignment.csv",
        "population_predictions.csv",
        "population_risk_tables.csv",
        "fit_provenance.csv",
        "audit_population_certificate_assignment.csv",
        "component_fit_diagnostics.csv",
        "certificate.csv",
        "evaluation_outcome.csv",
        "evaluation_population_action_predictions.csv",
        "analysis_manifest.json",
    ]
    missing = [name for name in required if not (args.output_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw-partition-first artifacts: {missing}")

    assignment = pd.read_csv(
        args.output_root / "participant_population_assignment.csv", dtype={"subject_id": str}
    )
    predictions = pd.read_csv(
        args.output_root / "population_predictions.csv", dtype={"subject_id": str}
    )
    risk = pd.read_csv(args.output_root / "population_risk_tables.csv")
    provenance = pd.read_csv(args.output_root / "fit_provenance.csv")
    audit_assignment = pd.read_csv(args.output_root / "audit_population_certificate_assignment.csv")
    components = pd.read_csv(args.output_root / "component_fit_diagnostics.csv")
    certificate = pd.read_csv(args.output_root / "certificate.csv")
    outcome = pd.read_csv(args.output_root / "evaluation_outcome.csv")
    action = pd.read_csv(args.output_root / "evaluation_population_action_predictions.csv")
    manifest = json.loads(
        (args.output_root / "analysis_manifest.json").read_text(encoding="utf-8")
    )

    allowed = {
        population: set(assignment.loc[assignment.population.eq(population), "subject_id"])
        for population in PARTITIONS
    }
    fit_subject_contract = True
    for row in provenance.itertuples(index=False):
        fit_subject_contract &= subjects(row.train_subjects) <= allowed[row.population]
        fit_subject_contract &= subjects(row.test_subjects) <= allowed[row.population]

    expected_risk_rows = (
        len(manifest["tasks"]) * len(manifest["representations"])
        * len(manifest["models"]) * len(manifest["seeds"])
        * (len(manifest["doses"]) - 1)
    )
    expected_probe_fits = (
        len(PARTITIONS) * len(manifest["representations"]) * len(manifest["seeds"])
        * manifest["record_folds"] * 2
    )
    full_run = manifest["fold_cells_per_seed"] == manifest["subject_folds"] * manifest["record_folds"]
    checks = {
        "analysis_completed": manifest.get("status") == "completed",
        "cuda_device_recorded": "NVIDIA" in manifest.get("device", ""),
        "participant_assignment_is_15_15": assignment.population.value_counts().to_dict() == {
            "audit_population": 15, "evaluation_population": 15,
        },
        "participant_sets_are_disjoint": not (
            allowed["audit_population"] & allowed["evaluation_population"]
        ),
        "all_30_participants_are_assigned_once": assignment.subject_id.nunique() == len(assignment) == 30,
        "assignment_is_outcome_blind": assignment.assignment_basis.eq(
            "lexicographic_subject_id_parity_only"
        ).all(),
        "prediction_subjects_stay_in_population": all(
            set(group.subject_id) <= allowed[population]
            for population, group in predictions.groupby("population")
        ),
        "fit_subjects_stay_in_population": fit_subject_contract,
        "all_fits_have_zero_train_test_row_overlap": provenance.row_overlap.eq(0).all(),
        "emotion_fit_count_exact": int(
            provenance.fit_kind.str.startswith("emotion_").sum()
        ) == manifest["expected_emotion_model_fits"],
        "identity_probe_fit_count_exact": int(
            provenance.fit_kind.eq("identity_probe").sum()
        ) == expected_probe_fits,
        "risk_configuration_counts_exact": risk.groupby("population").size().to_dict() == {
            "audit_population": expected_risk_rows,
            "evaluation_population": expected_risk_rows,
        },
        "risk_events_are_binary": set(risk.material_optimism_event.unique()) <= {0, 1},
        "risk_probabilities_are_finite_unit_interval": bool(
            np.isfinite(risk.probability_identity).all()
            and risk.probability_identity.between(0, 1).all()
        ),
        "certificate_budget_is_160_in_full_run": (
            not full_run
            or audit_assignment.partition.value_counts().to_dict() == {"audit": 40, "heldout": 40}
        ),
        "components_fit_on_audit_and_apply_to_evaluation": bool(
            components.fit_population.eq("audit_population").all()
            and components.application_population.eq("evaluation_population").all()
        ),
        "single_certificate_and_outcome": len(certificate) == len(outcome) == 1,
        "evaluation_action_covers_all_evaluation_configs": len(action) == expected_risk_rows,
        "evaluation_action_probabilities_finite": bool(
            np.isfinite(action[["probability_identity", "probability_wg_opct"]]).all().all()
        ),
        "selection_and_certificate_are_consistent": bool(
            (outcome.iloc[0].selected_method == "wg_opct") == bool(certificate.iloc[0].certified)
        ),
        "claim_boundary_is_explicit": "not a new-dataset" in manifest.get("claim_boundary", ""),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    report = {
        "status": "passed" if not failed else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "checks": {name: bool(value) for name, value in checks.items()},
        "failed_checks": failed,
        "claim_boundary": manifest.get("claim_boundary"),
    }
    (args.output_root / "independent_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
