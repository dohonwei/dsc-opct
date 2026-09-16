from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    partition_frame,
    stratified_certificate,
)
from develop_order_preserving_transport_v6 import RULE  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402


RESERVATION = ROOT / "docs/avdos_v7_external_confirmation_reservation.json"
AMENDMENT = ROOT / "docs/avdos_v11_external_confirmation_amendment.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
IMPLEMENTATION_LOCK = ROOT / "docs/avdos_v11_external_confirmation_implementation_lock.json"
FEATURE_ROOT = ROOT / "outputs/avdos_trial_features"
DOSE_ROOT = ROOT / "outputs/avdos_counterfactual_identity_dose"
CONFIRMATION_ROOT = ROOT / "outputs/avdos_v11_external_confirmation"
REPORT = CONFIRMATION_ROOT / "independent_validation_report.json"
SOURCE = Path(
    "E:/AA发表论文的数据/dataset/AVDOS-VR-main/notebooks/temp/2_affect/"
    "Dataset_AVDOS_ManualFeaturesWithAnnotations.csv"
)
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, passed: bool, detail: object) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    reservation = json.loads(RESERVATION.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    implementation = json.loads(IMPLEMENTATION_LOCK.read_text(encoding="utf-8"))
    feature_manifest = json.loads(
        (FEATURE_ROOT / "build_manifest.json").read_text(encoding="utf-8")
    )
    dose_manifest = json.loads(
        (DOSE_ROOT / "run_manifest.json").read_text(encoding="utf-8")
    )
    confirmation_manifest = json.loads(
        (CONFIRMATION_ROOT / "external_confirmation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    action_lock = json.loads(
        (CONFIRMATION_ROOT / "primary_action_and_audit_lock.json").read_text(
            encoding="utf-8"
        )
    )
    gate = json.loads(
        (CONFIRMATION_ROOT / "external_confirmation_gate.json").read_text(
            encoding="utf-8"
        )
    )
    saved_certificate = json.loads(
        (CONFIRMATION_ROOT / "primary_candidate_certificate.json").read_text(
            encoding="utf-8"
        )
    )
    checks = []

    checks.append(
        check(
            "source_hash",
            sha256(SOURCE) == reservation["source"]["sha256"],
            sha256(SOURCE),
        )
    )
    code_hashes = {
        hash_group: {
            relative: sha256(ROOT / relative)
            for relative in implementation[hash_group]
        }
        for hash_group in ("analysis_code_sha256", "dependency_code_sha256")
    }
    expected_code_hashes = {
        hash_group: implementation[hash_group]
        for hash_group in ("analysis_code_sha256", "dependency_code_sha256")
    }
    checks.append(
        check(
            "implementation_hashes",
            code_hashes == expected_code_hashes,
            code_hashes,
        )
    )
    checks.append(
        check(
            "freeze_and_amendment_chain",
            amendment["reservation_sha256"] == sha256(RESERVATION)
            and implementation["reservation_amendment_sha256"] == sha256(AMENDMENT)
            and implementation["freeze_sha256"] == sha256(FREEZE),
            {
                "reservation": sha256(RESERVATION),
                "amendment": sha256(AMENDMENT),
                "freeze": sha256(FREEZE),
            },
        )
    )
    frozen_hashes = {
        relative: sha256(ROOT / relative) for relative in freeze["locked_artifacts"]
    }
    checks.append(
        check(
            "frozen_v11_artifacts",
            frozen_hashes == freeze["locked_artifacts"],
            frozen_hashes,
        )
    )

    trials = pd.read_csv(FEATURE_ROOT / "avdos_trial_features.csv")
    trial_contract = bool(
        len(trials) == 333
        and trials.subject_id.nunique() == 37
        and trials.groupby("subject_id").size().eq(9).all()
        and set(trials.segment) == {"Positive", "Neutral", "Negative"}
    )
    checks.append(check("trial_contract", trial_contract, feature_manifest))
    settings = dose_manifest["settings"]
    gpu_contract = bool(
        dose_manifest["gpu_name"]
        and dose_manifest["gpu_epochs"] == 40
        and settings["subject_folds"] == 5
        and settings["stimulus_folds"] == 3
        and settings["seeds"] == [20260813, 20260829, 20260911, 20260923, 20261007]
        and settings["doses"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    checks.append(check("registered_gpu_and_split_settings", gpu_contract, settings))

    summary = pd.read_csv(DOSE_ROOT / "summary.csv")
    identity = pd.read_csv(DOSE_ROOT / "identity_encoding_margin.csv")
    risk_rows = summary.loc[summary.axis.eq("subject") & summary.nominal_dose.gt(0)]
    dose_contract = bool(
        len(risk_rows) == 320
        and set(risk_rows.task) == {"arousal", "valence"}
        and len(identity) == 80
    )
    checks.append(
        check(
            "counterfactual_risk_contract",
            dose_contract,
            {"risk_rows": len(risk_rows), "identity_rows": len(identity)},
        )
    )

    mechanism = pd.read_csv(CONFIRMATION_ROOT / "mechanism_probabilities_blinded.csv")
    assignments = pd.read_csv(CONFIRMATION_ROOT / "distribution_covered_assignments.csv")
    components = pd.read_csv(CONFIRMATION_ROOT / "component_projection_diagnostics.csv")
    assignment_contract = bool(
        assignments.partition.value_counts().to_dict() == {"audit": 40, "heldout": 40}
        and "material_optimism_event" not in assignments.columns
        and "material_optimism_event" not in mechanism.columns
        and action_lock["assignment_sha256"]
        == sha256(CONFIRMATION_ROOT / "distribution_covered_assignments.csv")
        and action_lock["mechanism_sha256"]
        == sha256(CONFIRMATION_ROOT / "mechanism_probabilities_blinded.csv")
    )
    checks.append(check("pre_outcome_action_and_audit_lock", assignment_contract, action_lock))

    audit = pd.read_csv(CONFIRMATION_ROOT / "primary_audit_labeled.csv")
    heldout = pd.read_csv(CONFIRMATION_ROOT / "primary_heldout_results.csv")
    config = RiskControlledConfig(
        configuration_budgets=(freeze["certification"]["audit_configuration_budget"],),
        bootstrap_repetitions=freeze["certification"]["bootstrap_repetitions"],
        auc_noninferiority_margin=freeze["certification"]["auc_noninferiority_margin"],
        minimum_brier_gain=freeze["certification"]["minimum_brier_gain_lcb"],
        minimum_valid_auc_bootstraps=freeze["certification"]["minimum_valid_auc_bootstraps"],
    )
    coral = components.loc[components.method.eq("coral")].iloc[0]
    candidate_applicable = action_lock["candidate_method"] == "wg_opct"
    recomputed_certificate = stratified_certificate(
        audit,
        config,
        int(freeze["certification"]["primary_audit_seed"]) + RULE.configuration_budget * 1009,
        candidate_applicable,
        float(coral.probability_rank),
        int(coral.order_inversions),
    )
    certificate_match = all(
        (
            bool(recomputed_certificate[key]) == bool(saved_certificate[key])
            if isinstance(recomputed_certificate[key], (bool, np.bool_))
            else (
                np.isnan(recomputed_certificate[key]) and np.isnan(saved_certificate[key])
                if isinstance(recomputed_certificate[key], float)
                and np.isnan(recomputed_certificate[key])
                else abs(float(recomputed_certificate[key]) - float(saved_certificate[key]))
                <= TOLERANCE
            )
        )
        for key in ("brier_gain", "brier_gain_lcb", "auc_delta", "auc_delta_lcb", "certified")
    )
    checks.append(
        check(
            "certificate_reproduces",
            certificate_match,
            {"saved": saved_certificate, "recomputed": recomputed_certificate},
        )
    )
    selected = gate["selected_method"]
    heldout_metrics = evaluate_action(heldout, selected)
    heldout_match = all(
        abs(float(heldout_metrics[key]) - float(gate["heldout_metrics"][key])) <= TOLERANCE
        for key in ("brier_gain", "auc_delta")
    ) and all(
        bool(heldout_metrics[key]) == bool(gate["heldout_metrics"][key])
        for key in (
            "adapted",
            "negative_transfer",
            "material_negative_transfer",
            "auc_noninferior",
        )
    )
    checks.append(check("heldout_metrics_reproduce", heldout_match, heldout_metrics))
    checks.append(
        check(
            "manifest_and_gate_consistency",
            bool(confirmation_manifest["claim_supported"]) == bool(gate["claim_supported"])
            and confirmation_manifest["action_lock_sha256"]
            == sha256(CONFIRMATION_ROOT / "primary_action_and_audit_lock.json")
            and feature_manifest["output_sha256"]
            == sha256(FEATURE_ROOT / "avdos_trial_features.csv")
            and dose_manifest["feature_sha256"]
            == sha256(FEATURE_ROOT / "avdos_trial_features.csv"),
            gate,
        )
    )

    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "date": "2026-09-08",
        "checks_passed": int(sum(item["passed"] for item in checks)),
        "checks_total": len(checks),
        "validator_sha256": sha256(Path(__file__)),
        "external_gate": gate,
        "checks": checks,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
