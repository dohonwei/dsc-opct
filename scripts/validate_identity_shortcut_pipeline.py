from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Q1 identity-shortcut experiment chain.")
    parser.add_argument(
        "--counterfactual-root",
        type=Path,
        default=Path("outputs/counterfactual_identity_dose_crossed_valid"),
    )
    parser.add_argument(
        "--ranking-root",
        type=Path,
        default=Path("outputs/model_ranking_consequences_crossed_valid"),
    )
    parser.add_argument(
        "--risk-root",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier"),
    )
    parser.add_argument(
        "--eppvr-root",
        type=Path,
        default=Path("outputs/eppvr_subject_shortcut_external_validation"),
    )
    parser.add_argument(
        "--external-risk-root",
        type=Path,
        default=Path("outputs/eppvr_frozen_risk_validation"),
    )
    parser.add_argument(
        "--transport-root",
        type=Path,
        default=Path("outputs/eppvr_risk_transport_diagnostics"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/identity_shortcut_validation_report.json"),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_counterfactual(root: Path) -> list[dict]:
    audit = pd.read_csv(root / "split_audit.csv")
    predictions = pd.read_csv(root / "predictions.csv", low_memory=False)
    summary = pd.read_csv(root / "summary.csv")
    require((audit.row_overlap_with_test == 0).all(), "Counterfactual train-test overlap detected")
    require(np.allclose(audit.identity_coverage, 1.0), "Counterfactual identity coverage failed")
    require((audit.n_train > 0).all() and (audit.n_test > 0).all(), "Empty counterfactual split")
    monotonic_keys = ["dataset", "task", "axis", "split_seed", "fold"]
    for _, group in audit.groupby(monotonic_keys):
        ordered = group.sort_values("nominal_dose")
        require(
            np.all(np.diff(ordered.normalized_mutual_information) >= -1e-12),
            "Counterfactual opportunity is not monotonic in nominal dose",
        )
    prediction_keys = [
        "dataset",
        "task",
        "axis",
        "representation",
        "model",
        "split_seed",
        "fold",
        "condition",
        "nominal_dose",
        "row_id",
    ]
    require(not predictions.duplicated(prediction_keys).any(), "Duplicate counterfactual predictions")
    require(np.isfinite(summary.exposure_effect).all(), "Non-finite counterfactual effect")
    return [
        {"check": "counterfactual_split_contract", "rows": len(audit), "status": "passed"},
        {"check": "counterfactual_prediction_contract", "rows": len(predictions), "status": "passed"},
        {"check": "counterfactual_summary", "rows": len(summary), "status": "passed"},
    ]


def check_ranking(root: Path) -> list[dict]:
    per_seed = pd.read_csv(root / "per_seed_ranking_consequences.csv")
    summary = pd.read_csv(root / "ranking_consequences_summary.csv")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    threshold = float(manifest["minimum_practical_difference"])
    expected = per_seed.winner_changed.astype(bool) & (per_seed.deployment_regret >= threshold)
    require(
        np.array_equal(expected, per_seed.material_winner_reversal.astype(bool)),
        "Material model-reversal rule is inconsistent",
    )
    require((per_seed.deployment_regret >= -1e-12).all(), "Negative deployment regret detected")
    return [
        {"check": "model_ranking_consequences", "rows": len(per_seed), "status": "passed"},
        {"check": "model_ranking_bootstrap", "rows": len(summary), "status": "passed"},
    ]


def check_risk(root: Path) -> list[dict]:
    freeze = json.loads((root / "freeze_manifest.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "admissibility_report.json").read_text(encoding="utf-8"))
    model = root / "frozen_risk_model.joblib"
    require(freeze.get("admissible") is True, "Risk classifier did not pass its freeze gate")
    require(freeze.get("status") == "admissible_freeze", "Risk freeze status is invalid")
    require(gate.get("admissible") is True, "Risk admissibility report failed")
    require(all(item["passed"] for item in gate["criteria"]), "A risk gate criterion failed")
    require(model.is_file(), "Frozen risk model is missing")
    require(sha256(model) == freeze["model_sha256"], "Frozen risk-model hash mismatch")
    features = freeze["numeric_features"] + freeze["categorical_features"]
    require("dataset" not in features, "Dataset identity leaked into the shortcut-risk predictors")
    forbidden_structure = {
        "log_n_rows",
        "n_rows",
        "n_subjects",
        "n_repeated_units",
        "repeated_measure_completeness",
        "class_balance",
    }
    require(
        not forbidden_structure.intersection(features),
        "Dataset-structure proxies leaked into the shortcut-risk predictors",
    )
    require(
        set(freeze["training_datasets"]) == {"DEAP", "MAHNOB-HCI"},
        "Risk model must be frozen on DEAP and MAHNOB-HCI only",
    )
    require(
        freeze.get("training_source") == "counterfactual_only",
        "Primary risk model must be frozen on counterfactual dose rows only",
    )
    table = pd.read_csv(root / "risk_learning_table.csv")
    require(
        set(table.source) == {"counterfactual_dose"},
        "Observational rows leaked into the primary frozen risk model",
    )
    require(set(table.axis) == {"subject"}, "Primary risk table must use subject identity only")
    require((table.nominal_dose > 0).all(), "Dose-zero rows must be anchors, not risk cases")
    metrics = pd.read_csv(root / "grouped_cv_metrics.csv")
    require(
        np.isfinite(table.dose_induced_amplification).all(),
        "Anchored risk target contains non-finite values",
    )
    expected_event = (
        table.dose_induced_amplification >= float(freeze["material_effect_threshold"])
    ).astype(int)
    require(
        np.array_equal(expected_event, table.material_optimism_event.astype(int)),
        "Material-risk event definition is inconsistent",
    )
    require(set(metrics.scheme) == {
        "leave_dataset_out",
        "leave_task_out",
        "leave_representation_out",
        "leave_model_out",
    }, "Grouped validation schemes are incomplete")
    return [
        {"check": "frozen_risk_model", "rows": len(table), "status": "passed"},
        {"check": "grouped_risk_validation", "rows": len(metrics), "status": "passed"},
    ]


def check_eppvr(root: Path) -> list[dict]:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(root / "split_audit.csv")
    predictions = pd.read_csv(root / "predictions.csv", low_memory=False)
    probes = pd.read_csv(root / "identity_probe_summary.csv")
    require(manifest["validation_axis"] == "subject identity only", "EPPVR axis contract failed")
    forbidden = set(manifest["forbidden_claims"])
    require("stimulus exposure" in forbidden, "EPPVR stimulus prohibition is missing")
    require((audit.row_overlap_with_test == 0).all(), "EPPVR train-test overlap detected")
    require(np.allclose(audit.subject_coverage, 1.0), "EPPVR subject exposure is incomplete")
    require(set(probes.axis) == {"subject"}, "EPPVR identity probes include a forbidden axis")
    require(set(predictions.condition) == {"unseen", "dose"}, "EPPVR conditions are incomplete")
    require(
        bool(manifest.get("risk_model_sha256_frozen_before_external_run")),
        "EPPVR run did not record a pre-existing public-data risk freeze",
    )
    return [
        {"check": "eppvr_subject_axis_contract", "rows": len(audit), "status": "passed"},
        {"check": "eppvr_predictions", "rows": len(predictions), "status": "passed"},
        {"check": "eppvr_identity_encoding", "rows": len(probes), "status": "passed"},
    ]


def check_external_risk(root: Path) -> list[dict]:
    predictions = pd.read_csv(root / "eppvr_risk_predictions.csv")
    metrics = json.loads((root / "external_validation_metrics.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "external_validation_gate.json").read_text(encoding="utf-8"))
    require(set(predictions.axis) == {"subject"}, "External risk predictions include a forbidden axis")
    require(
        np.isfinite(predictions.predicted_material_risk).all(),
        "Non-finite external risk probabilities",
    )
    require(
        predictions.predicted_material_risk.between(0, 1).all(),
        "External risk probabilities fall outside [0, 1]",
    )
    require(metrics["n_rows"] == len(predictions), "External risk row count mismatch")
    expected_event = (
        predictions.dose_induced_amplification
        >= float(metrics["material_effect_threshold"])
    ).astype(int)
    require(
        np.array_equal(expected_event, predictions.material_optimism_event.astype(int)),
        "External material-risk target is inconsistent",
    )
    expected_prediction = (
        predictions.predicted_material_risk >= float(metrics["decision_threshold"])
    ).astype(int)
    require(
        np.array_equal(
            expected_prediction,
            predictions.predicted_material_optimism_event.astype(int),
        ),
        "External frozen-threshold predictions are inconsistent",
    )
    require(
        gate["model_sha256"] == metrics["risk_model_sha256"],
        "External gate and metric files refer to different risk models",
    )
    return [{
        "check": "eppvr_frozen_risk_application",
        "rows": len(predictions),
        "status": "passed",
        "external_claim_admissible": bool(gate["admissible"]),
    }]


def check_transport_diagnostics(
    root: Path,
    risk_root: Path,
    external_root: Path,
) -> list[dict]:
    required = [
        "predictor_shift.csv",
        "calibration_bins.csv",
        "calibration_summary.json",
        "risk_quintile_enrichment.csv",
        "subgroup_performance.csv",
        "transport_diagnostics.json",
    ]
    require(root.is_dir(), f"Missing transport diagnostic root: {root}")
    require(
        all((root / name).is_file() for name in required),
        "A formal EPPVR transport diagnostic output is missing",
    )
    shift = pd.read_csv(root / "predictor_shift.csv")
    calibration_bins = pd.read_csv(root / "calibration_bins.csv")
    quintiles = pd.read_csv(root / "risk_quintile_enrichment.csv")
    subgroups = pd.read_csv(root / "subgroup_performance.csv")
    calibration = json.loads((root / "calibration_summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((root / "transport_diagnostics.json").read_text(encoding="utf-8"))
    freeze = json.loads((risk_root / "freeze_manifest.json").read_text(encoding="utf-8"))
    external_metrics = json.loads(
        (external_root / "external_validation_metrics.json").read_text(encoding="utf-8")
    )
    external_gate = json.loads(
        (external_root / "external_validation_gate.json").read_text(encoding="utf-8")
    )
    require(
        diagnostics["analysis_role"] == "post_external_explanatory_diagnostic",
        "Transport analysis role is not explicitly post-external",
    )
    require(
        diagnostics["frozen_model_or_threshold_modified"] is False,
        "Post-external diagnostics altered a frozen artifact",
    )
    require(
        diagnostics["frozen_model_sha256"] == freeze["model_sha256"],
        "Transport diagnostics refer to the wrong frozen model",
    )
    require(
        diagnostics["external_gate_admissible"] == external_gate["admissible"],
        "Transport diagnostics changed the locked external decision",
    )
    require(
        diagnostics["bootstrap"]["repetitions"] >= 5000,
        "Formal transport diagnostics require at least 5000 bootstrap repetitions",
    )
    require(
        set(shift.feature) == set(freeze["numeric_features"]),
        "Predictor-shift table does not cover every frozen numeric feature",
    )
    expected_bins = set(range(1, len(calibration_bins) + 1))
    require(
        set(calibration_bins.risk_bin) == expected_bins,
        "Calibration risk bins are incomplete",
    )
    require(
        calibration_bins.n_rows.sum() == external_metrics["n_rows"],
        "Calibration bins do not partition the external prediction table",
    )
    require(
        np.allclose(
            calibration_bins.observed_event_rate,
            quintiles.observed_event_rate,
        ),
        "Calibration and enrichment tables use different risk bins",
    )
    require(
        np.isclose(
            calibration["mean_predicted_probability"],
            external_metrics["mean_predicted_probability"],
        ),
        "Transport calibration changed the locked predictions",
    )
    require(
        set(subgroups.subgroup_family)
        == {"task", "representation", "model", "nominal_dose"},
        "External subgroup sensitivity families are incomplete",
    )
    return [{
        "check": "post_external_transport_diagnostics",
        "rows": int(calibration_bins.n_rows.sum()),
        "status": "passed",
        "interpretation": diagnostics["status"],
    }]


def main() -> None:
    args = parse_args()
    stages = [
        ("counterfactual", args.counterfactual_root, check_counterfactual),
        ("ranking", args.ranking_root, check_ranking),
        ("risk", args.risk_root, check_risk),
        ("eppvr", args.eppvr_root, check_eppvr),
        ("external_risk", args.external_risk_root, check_external_risk),
    ]
    report = {"status": "running", "checks": []}
    for name, root, function in tqdm(stages, desc="Identity-shortcut validation", unit="stage"):
        require(root.is_dir(), f"Missing {name} result root: {root}")
        report["checks"].extend(function(root))
    report["checks"].extend(
        check_transport_diagnostics(
            args.transport_root,
            args.risk_root,
            args.external_risk_root,
        )
    )
    report["status"] = "passed"
    external = [
        item for item in report["checks"] if item["check"] == "eppvr_frozen_risk_application"
    ]
    report["scientific_claim_status"] = (
        "external_risk_gate_passed"
        if external and external[0]["external_claim_admissible"]
        else "external_risk_gate_failed"
    )
    transport = [
        item
        for item in report["checks"]
        if item["check"] == "post_external_transport_diagnostics"
    ]
    report["transport_interpretation"] = (
        transport[0]["interpretation"] if transport else "not_available"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
