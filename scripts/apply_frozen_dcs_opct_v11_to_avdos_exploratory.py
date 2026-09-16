from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_identity_shortcut_risk import MODEL_CAPACITY  # noqa: E402
from apply_frozen_transport_policy_to_seediv import probability  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    cluster_geometry,
    partition_frame,
    select_distribution_covered_audit,
    stratified_certificate,
)
from develop_order_preserving_transport_v6 import RULE, fit_opct  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import witness_statistics  # noqa: E402
from identity_shortcut.domain_transport import coral, quantile_map  # noqa: E402
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402


KEYS = ["dataset", "task", "axis", "representation", "model", "split_seed"]
RESERVATION = ROOT / "docs/avdos_v7_external_confirmation_reservation.json"
AMENDMENT = ROOT / "docs/avdos_v11_external_confirmation_amendment.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
IMPLEMENTATION_LOCK = ROOT / "docs/avdos_v11_external_confirmation_implementation_lock.json"
EXPLORATORY_LOCK = ROOT / "docs/avdos_v11_post_access_exploratory_implementation_lock_amendment_003.json"
DOSE_ROOT = ROOT / "outputs/avdos_exploratory_ppg_dose"
RISK_ROOT = ROOT / "outputs/identity_shortcut_risk_classifier"
OUTPUT_ROOT = ROOT / "outputs/avdos_v11_exploratory_ppg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply frozen DCS-OPCT v11 to exploratory AVDOS PPG data.")
    parser.add_argument("--dose-root", type=Path, default=DOSE_ROOT)
    parser.add_argument("--risk-root", type=Path, default=RISK_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_native(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def verify_locks() -> tuple[dict[str, object], dict[str, object]]:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    lock = json.loads(IMPLEMENTATION_LOCK.read_text(encoding="utf-8"))
    exploratory_lock = json.loads(EXPLORATORY_LOCK.read_text(encoding="utf-8"))
    if amendment["reservation_sha256"] != sha256(RESERVATION):
        raise RuntimeError("AVDOS amendment does not match the reservation")
    if lock["reservation_amendment_sha256"] != sha256(AMENDMENT):
        raise RuntimeError("AVDOS implementation lock does not match the amendment")
    if lock["freeze_sha256"] != sha256(FREEZE):
        raise RuntimeError("AVDOS implementation lock does not match the freeze")
    for relative, expected in freeze["locked_artifacts"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen v11 artifact changed: {relative}")
    for hash_group in ("analysis_code_sha256", "dependency_code_sha256"):
        for relative, expected in lock[hash_group].items():
            if sha256(ROOT / relative) != expected:
                raise RuntimeError(f"Registered AVDOS code changed: {relative}")
    for relative, expected in exploratory_lock["analysis_code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Exploratory AVDOS code changed: {relative}")
    return freeze, lock


def load_unlabeled_target(root: Path) -> pd.DataFrame:
    columns = KEYS + ["nominal_dose", "achieved_opportunity", "metadata_prior_opportunity"]
    table = pd.read_csv(root / "summary.csv", usecols=columns)
    table = table.loc[table.axis.eq("subject")].rename(
        columns={"achieved_opportunity": "label_opportunity"}
    )
    encoding = pd.read_csv(root / "identity_encoding_margin.csv")
    encoding = (
        encoding.groupby(["dataset", "representation", "split_seed"], as_index=False)
        .agg(n_features=("n_features", "first"), encoding_margin=("encoding_margin", "mean"))
    )
    table = table.merge(
        encoding,
        on=["dataset", "representation", "split_seed"],
        validate="many_to_one",
    )
    anchors = table.loc[
        table.nominal_dose.eq(0),
        KEYS + ["label_opportunity", "metadata_prior_opportunity"],
    ].rename(
        columns={
            "label_opportunity": "dose_zero_label_opportunity",
            "metadata_prior_opportunity": "dose_zero_metadata_prior_opportunity",
        }
    )
    if anchors.duplicated(KEYS).any():
        raise ValueError("AVDOS dose-zero anchors are not unique")
    table = table.merge(anchors, on=KEYS, validate="many_to_one")
    table = table.loc[table.nominal_dose.gt(0)].copy()
    table["opportunity_delta"] = table.label_opportunity - table.dose_zero_label_opportunity
    table["metadata_prior_delta"] = (
        table.metadata_prior_opportunity - table.dose_zero_metadata_prior_opportunity
    )
    table["dose_encoding_interaction"] = table.opportunity_delta * table.encoding_margin
    table["model_capacity"] = table.model.map(MODEL_CAPACITY)
    if table.model_capacity.isna().any() or len(table) != 160:
        raise ValueError(f"AVDOS unlabeled risk table must contain 160 rows, observed {len(table)}")
    return table.sort_values(KEYS + ["nominal_dose"]).reset_index(drop=True)


def attach_outcomes(table: pd.DataFrame, summary_path: Path, threshold: float) -> pd.DataFrame:
    columns = KEYS + ["nominal_dose", "exposure_effect"]
    summary = pd.read_csv(summary_path, usecols=columns)
    summary = summary.loc[summary.axis.eq("subject")]
    anchors = summary.loc[
        summary.nominal_dose.eq(0), KEYS + ["exposure_effect"]
    ].rename(columns={"exposure_effect": "dose_zero_exposure_effect"})
    outcomes = summary.merge(anchors, on=KEYS, validate="many_to_one")
    outcomes = outcomes.loc[outcomes.nominal_dose.gt(0)].copy()
    outcomes["dose_induced_amplification"] = (
        outcomes.exposure_effect - outcomes.dose_zero_exposure_effect
    )
    outcomes["material_optimism_event"] = (
        outcomes.dose_induced_amplification >= threshold
    ).astype(int)
    return table.merge(outcomes, on=KEYS + ["nominal_dose"], validate="one_to_one")


def config_from_freeze(freeze: dict[str, object]) -> RiskControlledConfig:
    certificate = freeze["certification"]
    return RiskControlledConfig(
        configuration_budgets=(certificate["audit_configuration_budget"],),
        bootstrap_repetitions=certificate["bootstrap_repetitions"],
        auc_noninferiority_margin=certificate["auc_noninferiority_margin"],
        minimum_brier_gain=certificate["minimum_brier_gain_lcb"],
        minimum_valid_auc_bootstraps=certificate["minimum_valid_auc_bootstraps"],
    )


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite AVDOS confirmation: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen AVDOS DCS-OPCT application")
    freeze, implementation_lock = verify_locks()
    paths = {
        "risk_manifest": args.risk_root / "freeze_manifest.json",
        "risk_model": args.risk_root / "frozen_risk_model.joblib",
        "risk_table": args.risk_root / "risk_learning_table.csv",
        "summary": args.dose_root / "summary.csv",
        "identity": args.dose_root / "identity_encoding_margin.csv",
        "dose_manifest": args.dose_root / "run_manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    risk_manifest = json.loads(paths["risk_manifest"].read_text(encoding="utf-8"))
    risk_model = joblib.load(paths["risk_model"])
    features = risk_manifest["numeric_features"]
    public = pd.read_csv(paths["risk_table"], usecols=features)
    target_table = load_unlabeled_target(args.dose_root)
    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    source = scaler.transform(public[features]).astype(np.float32)
    target = scaler.transform(target_table[features]).astype(np.float32)
    transforms = {
        "identity": target.copy(),
        "coral": coral(source, target),
        "quantile_mapping": quantile_map(source, target),
    }
    mechanism = target_table.copy()
    for method, transformed in transforms.items():
        mechanism[f"probability_{method}"] = probability(logistic, transformed)

    config = config_from_freeze(freeze)
    identity = mechanism.probability_identity.to_numpy(float)
    component_rows = []
    progress = tqdm(total=2 * RULE.epochs, desc="AVDOS CUDA OPCT components", unit="epoch")
    projections = {}
    for index, method in enumerate(("coral", "quantile_mapping")):
        projected, diagnostics = fit_opct(
            identity,
            mechanism[f"probability_{method}"].to_numpy(float),
            RULE,
            int(freeze["certification"]["primary_audit_seed"]) + index * 1009,
            "cuda",
            progress,
        )
        projections[method] = projected
        component_rows.append({"method": method, **diagnostics})
    progress.close()
    component_table = pd.DataFrame(component_rows)
    both_pass = bool(component_table.applicable.eq(True).all())
    witness = (
        witness_statistics(identity, projections["coral"], projections["quantile_mapping"])
        if both_pass
        else {
            "directional_agreement": np.nan,
            "displacement_cosine": np.nan,
            "normalized_disagreement": np.nan,
            "witness_pass": False,
        }
    )
    candidate_applicable = bool(both_pass and witness["witness_pass"])
    mechanism["probability_wg_opct"] = (
        projections["coral"] if candidate_applicable else identity
    )
    geometry = cluster_geometry(mechanism)
    assignment = select_distribution_covered_audit(geometry)

    args.output_root.mkdir(parents=True, exist_ok=False)
    component_table.to_csv(args.output_root / "component_projection_diagnostics.csv", index=False)
    geometry.to_csv(args.output_root / "unlabeled_cluster_geometry.csv", index=False)
    assignment.to_csv(args.output_root / "distribution_covered_assignments.csv", index=False)
    mechanism.to_csv(args.output_root / "mechanism_probabilities_blinded.csv", index=False)
    action_lock = {
        "status": "avdos_v11_exploratory_action_and_audit_locked_before_outcomes",
        "evidence_role": "post_access_exploratory_only",
        "candidate_method": "wg_opct" if candidate_applicable else "identity",
        "component_diagnostics": component_rows,
        "witness": witness,
        "assignment_sha256": sha256(args.output_root / "distribution_covered_assignments.csv"),
        "mechanism_sha256": sha256(args.output_root / "mechanism_probabilities_blinded.csv"),
        "freeze_sha256": sha256(FREEZE),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "summary_sha256": sha256(paths["summary"]),
    }
    action_lock_path = args.output_root / "primary_action_and_audit_lock.json"
    action_lock_path.write_text(
        json.dumps(json_native(action_lock), indent=2), encoding="utf-8"
    )
    action_lock_hash = sha256(action_lock_path)

    evaluation = attach_outcomes(
        mechanism,
        paths["summary"],
        float(risk_manifest["material_effect_threshold"]),
    )
    if sha256(action_lock_path) != action_lock_hash:
        raise RuntimeError("AVDOS action lock changed after outcomes were loaded")
    audit, heldout = partition_frame(evaluation, assignment)
    coral_diagnostics = component_table.loc[component_table.method.eq("coral")].iloc[0]
    certificate = stratified_certificate(
        audit,
        config,
        int(freeze["certification"]["primary_audit_seed"]) + RULE.configuration_budget * 1009,
        candidate_applicable,
        float(coral_diagnostics.probability_rank),
        int(coral_diagnostics.order_inversions),
    )
    selected = "wg_opct" if bool(certificate["certified"]) else "identity"
    heldout_metrics = evaluate_action(heldout, selected)
    effectiveness_pass = bool(selected == "wg_opct" and heldout_metrics["brier_gain"] > 0.0)
    safety_pass = bool(
        certificate["certified"]
        and not heldout_metrics["negative_transfer"]
        and heldout_metrics["auc_noninferior"]
        and float(coral_diagnostics.probability_rank) >= RULE.minimum_rank
        and int(coral_diagnostics.order_inversions) == 0
    ) if selected == "wg_opct" else bool(
        not heldout_metrics["negative_transfer"] and heldout_metrics["auc_noninferior"]
    )
    claim_supported = bool(effectiveness_pass and safety_pass)
    gate = {
        "status": "passed" if claim_supported else "failed",
        "claim_supported": claim_supported,
        "confirmatory_claim_supported": False,
        "evidence_role": "post_access_exploratory_robustness",
        "candidate_method": "wg_opct" if candidate_applicable else "identity",
        "selected_method": selected,
        "certificate_pass": bool(certificate["certified"]),
        "effectiveness_pass": effectiveness_pass,
        "safety_pass": safety_pass,
        "heldout_metrics": heldout_metrics,
        "interpretation": (
            "DCS-OPCT v11 passed the post-access exploratory AVDOS PPG gates"
            if claim_supported
            else (
                "safe abstention without evidence of external effectiveness"
                if selected == "identity" and safety_pass
                else "post-access exploratory AVDOS PPG analysis did not support the target claim"
            )
        ),
    }
    audit.to_csv(args.output_root / "primary_audit_labeled.csv", index=False)
    heldout.to_csv(args.output_root / "primary_heldout_results.csv", index=False)
    (args.output_root / "primary_candidate_certificate.json").write_text(
        json.dumps(json_native(certificate), indent=2), encoding="utf-8"
    )
    (args.output_root / "external_confirmation_gate.json").write_text(
        json.dumps(json_native(gate), indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "avdos_v11_post_access_exploratory_analysis_complete",
        "evidence_role": "exploratory_only_not_untouched_confirmation",
        "confirmatory_claim_supported": False,
        "date": "2026-09-08",
        "claim_supported": claim_supported,
        "gpu_name": torch.cuda.get_device_name(0),
        "action_lock_sha256": action_lock_hash,
        "reservation_sha256": sha256(RESERVATION),
        "reservation_amendment_sha256": sha256(AMENDMENT),
        "freeze_sha256": sha256(FREEZE),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "exploratory_lock_sha256": sha256(EXPLORATORY_LOCK),
        "dose_manifest_sha256": sha256(paths["dose_manifest"]),
        "script_sha256": sha256(Path(__file__)),
        "implementation_lock_status": implementation_lock["status"],
    }
    (args.output_root / "external_confirmation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(json_native(gate), indent=2))


if __name__ == "__main__":
    main()
