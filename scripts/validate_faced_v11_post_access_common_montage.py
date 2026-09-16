from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "outputs/faced_v11_post_access_common_montage_features"
DOSE_ROOT = ROOT / "outputs/faced_v11_post_access_common_montage_dose"
DCS_ROOT = ROOT / "outputs/faced_v11_post_access_common_montage_dcs_opct"
ARTIFACT_ROOT = ROOT / "outputs/faced_v11_post_access_common_montage_artifacts_v2"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_SHA256 = (
    "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, condition: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": bool(condition), "detail": detail}


def main() -> None:
    feature_manifest = json.loads(
        (FEATURE_ROOT / "build_manifest.json").read_text(encoding="utf-8")
    )
    dose_manifest_path = DOSE_ROOT / "run_manifest.json"
    dose_manifest = json.loads(dose_manifest_path.read_text(encoding="utf-8"))
    gate = json.loads(
        (DCS_ROOT / "external_confirmation_gate.json").read_text(encoding="utf-8")
    )
    dcs_manifest = json.loads(
        (DCS_ROOT / "external_confirmation_manifest.json").read_text(encoding="utf-8")
    )
    action_lock_path = DCS_ROOT / "primary_action_and_audit_lock.json"
    summary = pd.read_csv(DOSE_ROOT / "summary.csv")
    audit = pd.read_csv(DOSE_ROOT / "split_audit.csv")
    identity = pd.read_csv(DOSE_ROOT / "identity_encoding_margin.csv")
    components = pd.read_csv(DCS_ROOT / "component_projection_diagnostics.csv")
    audit_labeled = pd.read_csv(DCS_ROOT / "primary_audit_labeled.csv")
    heldout = pd.read_csv(DCS_ROOT / "primary_heldout_results.csv")

    keys = ["representation", "model", "split_seed"]
    anchors = summary.loc[
        summary.nominal_dose.eq(0), keys + ["exposure_effect"]
    ].rename(columns={"exposure_effect": "zero"})
    nonzero = summary.loc[summary.nominal_dose.gt(0)].merge(
        anchors, on=keys, validate="many_to_one"
    )
    amplification = nonzero.exposure_effect - nonzero.zero
    checks = [
        check(
            "frozen_v11_unchanged",
            sha256(FREEZE) == EXPECTED_FREEZE_SHA256,
            sha256(FREEZE),
        ),
        check(
            "feature_build_complete",
            feature_manifest.get("status")
            == "faced_post_access_common_montage_trial_features_built",
            str(feature_manifest.get("status")),
        ),
        check(
            "feature_cohort_complete",
            feature_manifest.get("usable_participants") == 123
            and feature_manifest.get("failed_participants") == 0
            and feature_manifest.get("rows") == 2952,
            "123 participants, 0 failures, 2952 participant-video rows",
        ),
        check(
            "feature_role_exploratory",
            feature_manifest.get("confirmatory_claim_permitted") is False,
            "confirmatory_claim_permitted=false",
        ),
        check(
            "dose_run_complete",
            dose_manifest.get("status")
            == "faced_post_access_common_montage_counterfactual_complete",
            str(dose_manifest.get("status")),
        ),
        check(
            "dose_role_exploratory",
            dose_manifest.get("confirmatory_claim_permitted") is False,
            "confirmatory_claim_permitted=false",
        ),
        check(
            "dose_summary_shape",
            len(summary) == 150
            and not summary.duplicated(
                ["representation", "model", "split_seed", "nominal_dose"]
            ).any(),
            f"rows={len(summary)}",
        ),
        check(
            "split_audit_shape",
            len(audit) == 625,
            f"rows={len(audit)}",
        ),
        check(
            "split_contract",
            audit.row_overlap_with_test.eq(0).all()
            and np.isclose(audit.identity_coverage, 1.0).all(),
            "zero row overlap and complete identity coverage",
        ),
        check(
            "identity_probe_shape",
            len(identity) == 30 and identity.chance.nunique() == 1,
            f"rows={len(identity)}, chance={identity.chance.mean():.6f}",
        ),
        check(
            "no_material_events",
            int((amplification >= 0.02).sum()) == 0,
            f"max amplification={amplification.max():.6f}",
        ),
        check(
            "dcs_components_inapplicable",
            len(components) == 2 and components.applicable.eq(False).all(),
            components[["method", "probability_mean_shift", "applicable"]]
            .to_dict(orient="records")
            .__repr__(),
        ),
        check(
            "dcs_identity_fallback",
            gate.get("candidate_method") == "identity"
            and gate.get("selected_method") == "identity",
            f"candidate={gate.get('candidate_method')}, selected={gate.get('selected_method')}",
        ),
        check(
            "dcs_claim_rejected",
            gate.get("claim_supported") is False
            and gate.get("confirmatory_claim_supported") is False
            and gate.get("confirmatory_claim_permitted") is False,
            "exploratory and confirmatory claims both rejected",
        ),
        check(
            "outcome_partition_complete",
            len(audit_labeled) == 60
            and len(heldout) == 60
            and audit_labeled.material_optimism_event.eq(0).all()
            and heldout.material_optimism_event.eq(0).all(),
            f"audit={len(audit_labeled)}, heldout={len(heldout)}, all events=0",
        ),
        check(
            "action_lock_stable",
            dcs_manifest.get("action_lock_sha256") == sha256(action_lock_path),
            sha256(action_lock_path),
        ),
        check(
            "dcs_manifest_role_exploratory",
            dcs_manifest.get("confirmatory_claim_permitted") is False,
            str(dcs_manifest.get("evidence_role")),
        ),
        check(
            "dose_manifest_linked",
            dcs_manifest.get("dose_manifest_sha256") == sha256(dose_manifest_path),
            sha256(dose_manifest_path),
        ),
    ]
    passed = sum(item["passed"] for item in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "passed_checks": passed,
        "total_checks": len(checks),
        "evidence_role": "post_access_exploratory_external_stress_test_only",
        "confirmatory_claim_permitted": False,
        "checks": checks,
        "validator_sha256": sha256(Path(__file__)),
    }
    output_path = ARTIFACT_ROOT / "independent_validation_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
