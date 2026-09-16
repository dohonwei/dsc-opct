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
from develop_minimal_intervention_transport_v5 import (  # noqa: E402
    RULE,
    rank_minimal_interventions,
)
from develop_risk_controlled_transport_v5 import enrich_signatures, evaluate_action  # noqa: E402
from identity_shortcut.domain_transport import (  # noqa: E402
    SCMTConfig,
    coral,
    fit_scmt,
    mean_shift,
    quantile_map,
    support_clip,
)
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    balanced_cluster_order,
    certify_checkpoint,
    probability_methods,
    rows_for_clusters,
)
from identity_shortcut.transport_policy import unlabeled_signature  # noqa: E402


KEYS = ["dataset", "task", "axis", "representation", "model", "split_seed"]
SCMT_SEED = 20260905


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply frozen minimal-intervention v5 to CASE.")
    parser.add_argument("--registration", type=Path, default=Path("docs/case_v5_external_confirmation_registration.json"))
    parser.add_argument("--registration-amendment", type=Path, default=Path("docs/case_v5_external_confirmation_registration_amendment_002.json"))
    parser.add_argument("--freeze", type=Path, default=Path("docs/risk_controlled_transport_v5_freeze.json"))
    parser.add_argument("--case-root", type=Path, default=Path("outputs/case_counterfactual_identity_dose"))
    parser.add_argument("--risk-root", type=Path, default=Path("outputs/identity_shortcut_risk_classifier"))
    parser.add_argument("--scmt-root", type=Path, default=Path("outputs/scmt_public_development"))
    parser.add_argument("--policy-root", type=Path, default=Path("outputs/transport_policy_public_development"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/case_v5_external_confirmation"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_empty(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite one-shot CASE output: {path}")


def load_unlabeled_target(case_root: Path) -> pd.DataFrame:
    columns = KEYS + [
        "nominal_dose",
        "achieved_opportunity",
        "metadata_prior_opportunity",
    ]
    table = pd.read_csv(case_root / "summary.csv", usecols=columns)
    table = table.loc[table.axis.eq("subject")].rename(
        columns={"achieved_opportunity": "label_opportunity"}
    )
    encoding = pd.read_csv(case_root / "identity_encoding_margin.csv")
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
        raise ValueError("CASE dose-zero anchors are not unique")
    table = table.merge(anchors, on=KEYS, validate="many_to_one")
    table = table.loc[table.nominal_dose.gt(0)].copy()
    table["opportunity_delta"] = table.label_opportunity - table.dose_zero_label_opportunity
    table["metadata_prior_delta"] = (
        table.metadata_prior_opportunity - table.dose_zero_metadata_prior_opportunity
    )
    table["dose_encoding_interaction"] = table.opportunity_delta * table.encoding_margin
    table["model_capacity"] = table.model.map(MODEL_CAPACITY)
    if table.model_capacity.isna().any() or len(table) != 320:
        raise ValueError(f"CASE unlabeled risk table violates the registered 320-row contract: {len(table)}")
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


def main() -> None:
    args = parse_args()
    assert_empty(args.output_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered CASE application")
    registration = json.loads(args.registration.read_text("utf-8"))
    amendment = json.loads(args.registration_amendment.read_text("utf-8"))
    freeze = json.loads(args.freeze.read_text("utf-8"))
    if sha256(args.freeze) != registration["v5_freeze"]["sha256"]:
        raise ValueError("CASE registration does not match the frozen v5 artifact")
    if amendment["registration_sha256"] != sha256(args.registration):
        raise ValueError("CASE registration amendment hash mismatch")
    for relative, expected in amendment["analysis_code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Registered CASE analysis code changed: {relative}")

    paths = {
        "risk_manifest": args.risk_root / "freeze_manifest.json",
        "risk_model": args.risk_root / "frozen_risk_model.joblib",
        "risk_table": args.risk_root / "risk_learning_table.csv",
        "scmt_manifest": args.scmt_root / "scmt_freeze_manifest.json",
        "policy_artifact": args.policy_root / "frozen_transport_policy.joblib",
        "public_candidates": args.policy_root / "synthetic_candidate_results.csv",
        "case_summary": args.case_root / "summary.csv",
        "case_identity": args.case_root / "identity_encoding_margin.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    risk_manifest = json.loads(paths["risk_manifest"].read_text("utf-8"))
    scmt_manifest = json.loads(paths["scmt_manifest"].read_text("utf-8"))
    risk_model = joblib.load(paths["risk_model"])
    policy = joblib.load(paths["policy_artifact"])
    public_candidates = pd.read_csv(paths["public_candidates"])
    features = risk_manifest["numeric_features"]
    public = pd.read_csv(paths["risk_table"], usecols=features + ["material_optimism_event"])
    target_table = load_unlabeled_target(args.case_root)
    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    source = scaler.transform(public[features]).astype(np.float32)
    source_labels = public.material_optimism_event.to_numpy(int)
    target = scaler.transform(target_table[features]).astype(np.float32)

    scmt_config = SCMTConfig(**scmt_manifest["selected_config"])
    progress = tqdm(total=1, desc="CASE frozen SCMT candidate", unit="fit", dynamic_ncols=True)
    transformed_scmt, scmt_diagnostics = fit_scmt(
        source,
        target,
        lambda values: probability(logistic, values),
        scmt_config,
        SCMT_SEED,
        "cuda",
        lambda epoch, loss: progress.set_postfix(loss=f"{loss['total']:.3f}", refresh=False)
        if epoch % 40 == 0
        else None,
    )
    progress.update(1)
    progress.close()
    transforms = {
        "identity": target.copy(),
        "mean_shift": mean_shift(source, target),
        "coral": coral(source, target),
        "quantile_mapping": quantile_map(source, target),
        "support_clipping": support_clip(source, target),
        "scmt": transformed_scmt,
    }
    decision_threshold = float(risk_manifest["decision_threshold"])
    raw_signatures = pd.DataFrame(
        [
            {
                "scenario_id": "CASE",
                "method": method,
                **unlabeled_signature(
                    source,
                    source_labels,
                    target,
                    transformed,
                    lambda values: probability(logistic, values),
                    decision_threshold,
                    registration["one_shot_v5_contract"]["primary_audit_seed"],
                ),
            }
            for method, transformed in transforms.items()
        ]
    )
    signatures = enrich_signatures(raw_signatures, public_candidates, policy)
    config = RiskControlledConfig(**freeze["config"])
    rankings, candidate, action_reason = rank_minimal_interventions(
        signatures, list(transforms), config, RULE
    )
    mechanism = target_table.copy()
    for method, transformed in transforms.items():
        mechanism[f"probability_{method}"] = probability(logistic, transformed)

    args.output_root.mkdir(parents=True, exist_ok=False)
    signatures.to_csv(args.output_root / "candidate_signatures.csv", index=False)
    rankings.to_csv(args.output_root / "candidate_distortion_rankings.csv", index=False)
    mechanism.to_csv(args.output_root / "mechanism_probabilities_blinded.csv", index=False)
    lock = {
        "status": "case_action_locked_before_outcomes_loaded",
        "candidate_method": candidate,
        "action_lock_reason": action_reason,
        "registration_sha256": sha256(args.registration),
        "registration_amendment_sha256": sha256(args.registration_amendment),
        "v5_freeze_sha256": sha256(args.freeze),
        "case_summary_sha256": sha256(paths["case_summary"]),
        "candidate_signatures_sha256": sha256(args.output_root / "candidate_signatures.csv"),
        "mechanism_probabilities_sha256": sha256(args.output_root / "mechanism_probabilities_blinded.csv"),
    }
    lock_path = args.output_root / "primary_action_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    lock_hash = sha256(lock_path)

    evaluation = attach_outcomes(
        mechanism,
        paths["case_summary"],
        float(risk_manifest["material_effect_threshold"]),
    )
    seed = int(registration["one_shot_v5_contract"]["primary_audit_seed"])
    order = balanced_cluster_order(evaluation, config, seed)
    audit_clusters = order[: config.cluster_budgets[0]]
    audit_mask = rows_for_clusters(evaluation, config.cluster_columns, audit_clusters)
    audit = evaluation.loc[audit_mask].reset_index(drop=True)
    heldout = evaluation.loc[~audit_mask].reset_index(drop=True)
    if candidate == "identity":
        certificate = None
        selected = "identity"
        selection_reason = "abstain_unlabeled_actionability_gate_failed"
    else:
        certificate = certify_checkpoint(
            audit,
            [candidate],
            config,
            seed + RULE.configuration_budget * 1009,
            total_registered_candidates=1,
        ).iloc[0]
        selected = candidate if bool(certificate.certified) else "identity"
        selection_reason = (
            "selected_frozen_v5_certificate"
            if selected != "identity"
            else f"abstain_confirmation_failed:{certificate.failure_reason}"
        )
    heldout_metrics = evaluate_action(heldout, selected)
    certificate_pass = bool(certificate is not None and certificate.certified)
    effectiveness_pass = bool(
        selected != "identity" and heldout_metrics["brier_gain"] > 0.0
    )
    safety_pass = bool(
        (selected == "identity" or certificate_pass)
        and not heldout_metrics["negative_transfer"]
        and heldout_metrics["auc_noninferior"]
    )
    claim_supported = bool(certificate_pass and effectiveness_pass and safety_pass)
    assignments = pd.DataFrame(
        [
            {
                "cluster_order_position": position,
                "partition": "audit" if position < config.cluster_budgets[0] else "heldout",
                **dict(zip(config.cluster_columns, cluster, strict=True)),
            }
            for position, cluster in enumerate(order)
        ]
    )
    certificate_payload = {
        "candidate_method": candidate,
        "selected_method": selected,
        "selection_reason": selection_reason,
        "n_audit_configurations": len(audit),
        "n_heldout_configurations": len(heldout),
        "certificate": None if certificate is None else certificate.to_dict(),
    }
    gate = {
        "status": "passed" if claim_supported else "failed",
        "claim_supported": claim_supported,
        "certificate_pass": certificate_pass,
        "effectiveness_pass": effectiveness_pass,
        "safety_pass": safety_pass,
        "candidate_method": candidate,
        "selected_method": selected,
        "heldout_metrics": heldout_metrics,
        "interpretation": (
            "frozen v5 passed the untouched CASE effectiveness and safety gate"
            if claim_supported
            else (
                "safe abstention without evidence of effectiveness"
                if selected == "identity" and safety_pass
                else "untouched CASE confirmation did not support the target claim"
            )
        ),
    }
    assignments.to_csv(args.output_root / "primary_audit_cluster_order.csv", index=False)
    audit.to_csv(args.output_root / "primary_audit_labeled.csv", index=False)
    heldout.to_csv(args.output_root / "primary_heldout_results.csv", index=False)
    (args.output_root / "primary_candidate_certificate.json").write_text(
        json.dumps(certificate_payload, indent=2), encoding="utf-8"
    )
    (args.output_root / "external_confirmation_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    diagnostics = {
        key: value for key, value in scmt_diagnostics.items() if not isinstance(value, dict)
    } | {f"check_{key}": value for key, value in scmt_diagnostics["checks"].items()}
    pd.DataFrame([diagnostics]).to_csv(
        args.output_root / "scmt_candidate_diagnostics.csv", index=False
    )
    manifest = {
        "status": "case_v5_one_shot_external_confirmation_complete",
        "dataset": "CASE",
        "claim_supported": claim_supported,
        "action_lock_sha256": lock_hash,
        "registration_sha256": sha256(args.registration),
        "registration_amendment_sha256": sha256(args.registration_amendment),
        "freeze_sha256": sha256(args.freeze),
        "script_sha256": sha256(Path(__file__)),
        "gpu_name": torch.cuda.get_device_name(0),
        "statistical_backend": "numpy_cpu_cluster_bootstrap",
        "outcome_access_boundary": (
            "primary_action_lock.json was written and hashed before attach_outcomes read "
            "the exposure_effect column from the CASE summary"
        ),
    }
    (args.output_root / "external_confirmation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
