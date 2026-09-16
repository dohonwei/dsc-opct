from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "outputs" / "dcs_opct_v11_submission_artifacts_crossfit_v11"
DEV = ROOT / "outputs" / "distribution_covered_stratified_opct_v11_development"
AVDOS = ROOT / "outputs" / "avdos_v11_exploratory_ppg"
STRONG = ROOT / "outputs" / "dcs_opct_v11_strong_calibration_baselines"
CROSSFIT = ROOT / "outputs" / "dcs_opct_v11_crossfit_certified_baselines"
PAIRED = ROOT / "outputs" / "dcs_opct_v11_paired_inference_crossfit"
ENGINEERING = ROOT / "outputs" / "material_threshold_engineering_anchor"
FACED = ROOT / "outputs" / "faced_v11_post_access_common_montage_artifacts_v2"
COMPONENT = ROOT / "outputs" / "dcs_opct_v11_leave_one_component"
FREEZE = ROOT / "docs" / "distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_DATASETS = ["EPPVR", "CASE", "CEAP", "SEED-IV", "DREAMER", "AVDOS-VR"]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(actual, expected, rtol=0, atol=tolerance))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    matrix = pd.read_csv(ARTIFACTS / "six_dataset_action_outcome.csv")
    diagnostics = pd.read_csv(ARTIFACTS / "support_gate_diagnostics.csv")
    source_certificates = pd.read_csv(DEV / "primary_certificates.csv").set_index("dataset")
    source_heldout = pd.read_csv(DEV / "primary_heldout_results.csv").set_index("dataset")
    avdos_gate = read_json(AVDOS / "external_confirmation_gate.json")
    faced_evidence = read_json(FACED / "faced_post_access_evidence_report.json")
    faced_validation = read_json(FACED / "independent_validation_report.json")
    component_validation = read_json(COMPONENT / "independent_validation_report.json")
    indexed = matrix.set_index("dataset")

    checks: list[dict] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record("dataset_order", matrix.dataset.tolist() == EXPECTED_DATASETS, matrix.dataset.tolist())
    for dataset in EXPECTED_DATASETS[:-1]:
        row = indexed.loc[dataset]
        source_h = source_heldout.loc[dataset]
        record(
            f"{dataset}_heldout_gain",
            close(row.heldout_brier_gain, source_h.brier_gain),
            float(row.heldout_brier_gain),
        )
        source_c = source_certificates.loc[dataset]
        if np.isfinite(source_c.brier_gain):
            record(
                f"{dataset}_audit_gain",
                close(row.audit_brier_gain, source_c.brier_gain),
                float(row.audit_brier_gain),
            )
            record(
                f"{dataset}_audit_lcb",
                close(row.audit_brier_gain_lcb, source_c.brier_gain_lcb),
                float(row.audit_brier_gain_lcb),
            )

    avdos_row = indexed.loc["AVDOS-VR"]
    record(
        "avdos_role",
        avdos_row.evidence_role == "post-access exploratory",
        avdos_row.evidence_role,
    )
    record(
        "avdos_nonintervention_not_effective",
        bool(avdos_row.safety_pass) and not bool(avdos_row.effectiveness_pass),
        {
            "artifact_safety": bool(avdos_row.safety_pass),
            "artifact_effectiveness": bool(avdos_row.effectiveness_pass),
            "source_selected_method": avdos_gate.get("selected_method", "identity"),
        },
    )
    record(
        "all_order_inversions_zero",
        bool((diagnostics.order_inversions == 0).all()),
        int(diagnostics.order_inversions.sum()),
    )
    avdos_components = diagnostics.loc[diagnostics.dataset == "AVDOS-VR"].set_index("method")
    record(
        "avdos_coral_exceeds_shift_budget",
        abs(avdos_components.loc["coral", "probability_mean_shift"]) > 0.10,
        float(avdos_components.loc["coral", "probability_mean_shift"]),
    )
    record(
        "avdos_quantile_exceeds_shift_budget",
        abs(avdos_components.loc["quantile_mapping", "probability_mean_shift"]) > 0.10,
        float(avdos_components.loc["quantile_mapping", "probability_mean_shift"]),
    )
    record(
        "faced_role_exploratory_only",
        faced_evidence["evidence_role"] == "post_access_exploratory_external_stress_test_only"
        and faced_evidence["confirmatory_claim_permitted"] is False,
        faced_evidence["evidence_role"],
    )
    record(
        "faced_identity_without_material_events",
        faced_evidence["identity_probe_balanced_accuracy_range"][0] >= 0.885
        and faced_evidence["chance_identity_accuracy"] < 0.009
        and faced_evidence["material_events"] == 0
        and faced_evidence["maximum_dose_induced_amplification"] < 0.02,
        {
            "identity_range": faced_evidence["identity_probe_balanced_accuracy_range"],
            "chance": faced_evidence["chance_identity_accuracy"],
            "events": faced_evidence["material_events"],
            "maximum_amplification": faced_evidence["maximum_dose_induced_amplification"],
        },
    )
    record(
        "faced_identity_fallback_not_effectiveness",
        faced_evidence["dcs_candidate_method"] == "identity"
        and faced_evidence["dcs_selected_method"] == "identity"
        and faced_evidence["dcs_claim_supported"] is False,
        {
            "candidate": faced_evidence["dcs_candidate_method"],
            "selected": faced_evidence["dcs_selected_method"],
            "claim_supported": faced_evidence["dcs_claim_supported"],
        },
    )
    record(
        "faced_independent_validation",
        faced_validation["status"] == "passed"
        and faced_validation["passed_checks"] == 18
        and faced_validation["total_checks"] == 18,
        {"passed": faced_validation["passed_checks"], "total": faced_validation["total_checks"]},
    )
    record(
        "leave_one_component_independent_validation",
        component_validation["status"] == "passed"
        and component_validation["checks_passed"] == 17
        and component_validation["checks_total"] == 17,
        {
            "passed": component_validation["checks_passed"],
            "total": component_validation["checks_total"],
        },
    )
    component_same = pd.read_csv(COMPONENT / "same_assignment_leave_one_component.csv")
    component_random = pd.read_csv(COMPONENT / "distribution_coverage_summary.csv").set_index("dataset")
    record(
        "leave_one_component_role_separation",
        bool(
            component_same.loc[
                component_same.dataset.eq("DREAMER")
                & component_same.variant.eq("minus_quantile_witness"),
                "candidate_eligible",
            ].iloc[0]
        )
        and not bool(
            component_same.loc[
                component_same.dataset.eq("DREAMER")
                & component_same.variant.eq("minus_quantile_witness"),
                "audit_certified",
            ].iloc[0]
        )
        and component_same.loc[
            component_same.dataset.eq("SEED-IV")
            & component_same.variant.eq("minus_unlabeled_applicability"),
            "audit_brier_gain",
        ].iloc[0]
        < -0.02,
        "DREAMER and SEED-IV reach but fail the unchanged audit after gate removal",
    )
    record(
        "distribution_coverage_ceap_diagnostic",
        bool(component_random.loc["CEAP", "covered_release"])
        and close(component_random.loc["CEAP", "random_release_rate"], 0.60),
        component_random.loc["CEAP"].to_dict(),
    )

    figure_names = [
        "fig_v11_six_dataset_action_matrix.png",
        "fig_v11_audit_and_heldout_gain.png",
        "fig_v11_support_gate_geometry.png",
        "fig_avdos_exploratory_identity_and_dose.png",
        "fig_v11_failure_and_protocol_timeline.png",
        "fig_v11_crossfit_certification_stress_test.png",
        "engineering_anchor_fig_material_threshold_engineering_anchor.png",
        "faced_fig_faced_post_access_external_boundary.png",
    ]
    for name in figure_names:
        path = ARTIFACTS / name
        image = mpimg.imread(path)
        record(
            f"figure_{name}",
            path.exists() and image.shape[0] >= 1000 and image.shape[1] >= 2000,
            list(image.shape),
        )

    action_table = (ARTIFACTS / "table_v11_action_outcome.tex").read_text(encoding="utf-8")
    record(
        "latex_table_preserves_avdos_boundary",
        "AVDOS-VR" in action_table and "Exploratory" in action_table,
        "AVDOS-VR marked Exploratory",
    )

    strong_manifest = read_json(STRONG / "manifest.json")
    legacy = read_json(STRONG / "legacy_reproduction.json")
    record(
        "strong_baseline_freeze_hash",
        strong_manifest["frozen_v11_sha256"] == sha256(FREEZE),
        strong_manifest["frozen_v11_sha256"],
    )
    record(
        "strong_baseline_legacy_reproduction",
        legacy["status"] == "passed"
        and legacy["maximum_absolute_discrepancy"] <= legacy["tolerance"],
        legacy,
    )
    strong = pd.read_csv(STRONG / "decision_curve_summary.csv")
    minimum = strong.loc[strong.budget_fraction.eq(0.2)].set_index(["dataset", "method"])
    dcs_material = minimum.xs("dcs_selective", level="method")[
        "material_negative_transfer_frequency"
    ]
    record(
        "minimum_budget_dcs_material_negative_zero",
        bool(dcs_material.eq(0).all()),
        dcs_material.to_dict(),
    )
    beta_expected = {
        "EPPVR": 0.06,
        "SEED-IV": 0.69,
        "DREAMER": 0.53,
        "CASE": 0.06,
        "CEAP": 0.28,
    }
    beta_actual = {
        dataset: float(minimum.loc[(dataset, "target_beta"), "material_negative_transfer_frequency"])
        for dataset in beta_expected
    }
    record(
        "minimum_budget_beta_material_negative_reproduced",
        all(close(beta_actual[key], value) for key, value in beta_expected.items()),
        beta_actual,
    )
    crossfit = pd.read_csv(CROSSFIT / "decision_curve_summary.csv")
    crossfit_minimum = crossfit.loc[crossfit.budget_fraction.eq(0.2)].set_index(
        ["dataset", "method"]
    )
    crossfit_expected = {
        ("SEED-IV", "crossfit_certified_platt"): 0.21,
        ("SEED-IV", "crossfit_certified_beta"): 0.20,
        ("DREAMER", "crossfit_certified_platt"): 0.17,
        ("DREAMER", "crossfit_certified_beta"): 0.16,
        ("CEAP", "crossfit_certified_platt"): 0.13,
        ("CEAP", "crossfit_certified_beta"): 0.12,
    }
    crossfit_actual = {
        f"{dataset}:{method}": float(
            crossfit_minimum.loc[(dataset, method), "material_negative_transfer_frequency"]
        )
        for dataset, method in crossfit_expected
    }
    record(
        "minimum_budget_crossfit_harm_reproduced",
        all(
            close(
                crossfit_minimum.loc[key, "material_negative_transfer_frequency"],
                expected,
            )
            for key, expected in crossfit_expected.items()
        ),
        crossfit_actual,
    )
    paired_gain = pd.read_csv(PAIRED / "paired_gain_inference.csv")
    paired_safety = pd.read_csv(PAIRED / "paired_safety_inference.csv")
    record(
        "paired_inference_has_150_cells",
        len(paired_gain) == 150 and len(paired_safety) == 150,
        {"gain_rows": len(paired_gain), "safety_rows": len(paired_safety)},
    )
    dcs_bounds = pd.read_csv(PAIRED / "domain_budget_safety_upper_bounds.csv")
    dcs_bounds = dcs_bounds.loc[
        dcs_bounds.method.eq("dcs_selective") & dcs_bounds.release_count.gt(0)
    ]
    record(
        "dcs_zero_event_upper_bounds_preserved",
        close(dcs_bounds.released_action_95pct_upper_bound.min(), 0.0295130496070399)
        and close(dcs_bounds.released_action_95pct_upper_bound.max(), 0.0464384514839382),
        {
            "minimum": float(dcs_bounds.released_action_95pct_upper_bound.min()),
            "maximum": float(dcs_bounds.released_action_95pct_upper_bound.max()),
        },
    )
    engineering_validation = read_json(ENGINEERING / "independent_validation_report.json")
    record(
        "engineering_anchor_validation",
        engineering_validation["status"] == "passed"
        and engineering_validation["checks_passed"] == 12,
        engineering_validation["checks_passed"],
    )
    copied_pairs = {
        "fig_strong_direct_calibration.pdf": STRONG / "fig_strong_direct_calibration.pdf",
        "fig_split_certified_calibration.pdf": STRONG / "fig_split_certified_calibration.pdf",
        "strong_calibration_decision_curve_summary.csv": STRONG / "decision_curve_summary.csv",
        "strong_calibration_low_label_summary.csv": STRONG / "low_label_summary.csv",
        "strong_calibration_paired_dcs_comparisons.csv": STRONG / "paired_dcs_comparisons.csv",
        "strong_calibration_legacy_reproduction.json": STRONG / "legacy_reproduction.json",
        "strong_calibration_manifest.json": STRONG / "manifest.json",
        "crossfit_certified_decision_curve_summary.csv": CROSSFIT / "decision_curve_summary.csv",
        "crossfit_certified_diagnostics.csv": CROSSFIT / "certification_diagnostics.csv",
        "crossfit_certified_manifest.json": CROSSFIT / "manifest.json",
        "paired_gain_inference.csv": PAIRED / "paired_gain_inference.csv",
        "paired_safety_inference.csv": PAIRED / "paired_safety_inference.csv",
        "domain_budget_safety_upper_bounds.csv": PAIRED / "domain_budget_safety_upper_bounds.csv",
        "engineering_anchor_margin_vulnerability_summary.csv": ENGINEERING / "margin_vulnerability_summary.csv",
        "engineering_anchor_audit_budget_utility.csv": ENGINEERING / "audit_budget_utility.csv",
        "engineering_anchor_configuration_decision_curve.csv": ENGINEERING / "configuration_decision_curve.csv",
        "engineering_anchor_fig_material_threshold_engineering_anchor.pdf": ENGINEERING / "fig_material_threshold_engineering_anchor.pdf",
        "engineering_anchor_manifest.json": ENGINEERING / "manifest.json",
        "engineering_anchor_independent_validation_report.json": ENGINEERING / "independent_validation_report.json",
        "faced_configuration_outcomes.csv": FACED / "faced_configuration_outcomes.csv",
        "faced_dcs_applicability.csv": FACED / "faced_dcs_applicability.csv",
        "faced_dose_amplification_summary.csv": FACED / "faced_dose_amplification_summary.csv",
        "faced_identity_encoding_summary.csv": FACED / "faced_identity_encoding_summary.csv",
        "faced_post_access_evidence_report.json": FACED / "faced_post_access_evidence_report.json",
        "faced_fig_faced_post_access_external_boundary.pdf": FACED / "fig_faced_post_access_external_boundary.pdf",
        "faced_fig_faced_post_access_external_boundary.png": FACED / "fig_faced_post_access_external_boundary.png",
        "faced_independent_validation_report.json": FACED / "independent_validation_report.json",
        "faced_table_faced_post_access_dose.tex": FACED / "table_faced_post_access_dose.tex",
        "component_same_assignment_leave_one_component.csv": COMPONENT / "same_assignment_leave_one_component.csv",
        "component_random_half_allocation_repetitions.csv": COMPONENT / "random_half_allocation_repetitions.csv",
        "component_distribution_coverage_summary.csv": COMPONENT / "distribution_coverage_summary.csv",
        "component_table_leave_one_component.tex": COMPONENT / "table_leave_one_component.tex",
        "component_table_distribution_coverage_diagnostic.tex": COMPONENT / "table_distribution_coverage_diagnostic.tex",
        "component_manifest.json": COMPONENT / "manifest.json",
        "component_independent_validation_report.json": COMPONENT / "independent_validation_report.json",
    }
    for copied_name, source_path in copied_pairs.items():
        copied_path = ARTIFACTS / copied_name
        record(
            f"copied_{copied_name}",
            copied_path.exists() and sha256(copied_path) == sha256(source_path),
            copied_name,
        )
    strong_table = (ARTIFACTS / "table_v11_strong_calibration_low_label.tex").read_text(
        encoding="utf-8"
    )
    record(
        "strong_table_preserves_claim_boundary",
        all(
            token in strong_table
            for token in [
                "DCS selective", "Target beta", "Split-certified beta",
                "Cross-fit-certified Platt", "Cross-fit-certified beta", "0.000",
            ]
        ),
        "DCS and strong comparators present",
    )

    artifact_manifest = read_json(ARTIFACTS / "submission_artifact_manifest.json")
    hash_mismatches = {
        relative: digest
        for relative, digest in artifact_manifest["output_sha256"].items()
        if sha256(ROOT / relative) != digest
    }
    record("submission_manifest_output_hashes", not hash_mismatches, hash_mismatches)

    passed = sum(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "date": "2026-09-09",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }
    (ARTIFACTS / "independent_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"Independent validation failed: {failed}")
    print(f"Independent validation passed: {passed}/{len(checks)} checks")


if __name__ == "__main__":
    main()
