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
    RULE as V5_RULE,
    rank_minimal_interventions,
)
from develop_order_preserving_transport_v6 import RULE, fit_opct  # noqa: E402
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
    rows_for_clusters,
)
from identity_shortcut.transport_policy import unlabeled_signature  # noqa: E402


KEYS = ["dataset", "task", "axis", "representation", "model", "split_seed"]
SCMT_SEED = 20260905


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply frozen OPCT v6 to CEAP once.")
    parser.add_argument(
        "--reservation",
        type=Path,
        default=Path("docs/ceap_v6_external_confirmation_reservation.json"),
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "docs/ceap_v6_external_confirmation_implementation_lock_amendment_001.json"
        ),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("docs/order_preserving_transport_v6_final_freeze.json"),
    )
    parser.add_argument(
        "--ceap-root",
        type=Path,
        default=Path("outputs/ceap_counterfactual_identity_dose"),
    )
    parser.add_argument(
        "--risk-root",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier"),
    )
    parser.add_argument("--scmt-root", type=Path, default=Path("outputs/scmt_public_development"))
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=Path("outputs/transport_policy_public_development"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ceap_v6_external_confirmation"),
    )
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
        raise ValueError("CEAP dose-zero anchors are not unique")
    table = table.merge(anchors, on=KEYS, validate="many_to_one")
    table = table.loc[table.nominal_dose.gt(0)].copy()
    table["opportunity_delta"] = table.label_opportunity - table.dose_zero_label_opportunity
    table["metadata_prior_delta"] = (
        table.metadata_prior_opportunity - table.dose_zero_metadata_prior_opportunity
    )
    table["dose_encoding_interaction"] = table.opportunity_delta * table.encoding_margin
    table["model_capacity"] = table.model.map(MODEL_CAPACITY)
    if table.model_capacity.isna().any() or len(table) != 320:
        raise ValueError(f"CEAP unlabeled risk table violates the registered 320 rows: {len(table)}")
    return table.sort_values(KEYS + ["nominal_dose"]).reset_index(drop=True)


def attach_outcomes(table: pd.DataFrame, summary_path: Path, threshold: float) -> pd.DataFrame:
    columns = KEYS + ["nominal_dose", "exposure_effect"]
    summary = pd.read_csv(summary_path, usecols=columns)
    summary = summary.loc[summary.axis.eq("subject")]
    anchors = summary.loc[summary.nominal_dose.eq(0), KEYS + ["exposure_effect"]].rename(
        columns={"exposure_effect": "dose_zero_exposure_effect"}
    )
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
        raise FileExistsError(f"Refusing to overwrite one-shot CEAP output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for registered CEAP OPCT application")
    reservation = json.loads(args.reservation.read_text(encoding="utf-8"))
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if amendment["reservation_sha256"] != sha256(args.reservation):
        raise ValueError("CEAP implementation lock does not match the reservation")
    if amendment["freeze_sha256"] != sha256(args.freeze):
        raise ValueError("CEAP implementation lock does not match the OPCT v6 freeze")
    for relative, expected in amendment["analysis_code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Registered CEAP analysis code changed: {relative}")

    paths = {
        "risk_manifest": args.risk_root / "freeze_manifest.json",
        "risk_model": args.risk_root / "frozen_risk_model.joblib",
        "risk_table": args.risk_root / "risk_learning_table.csv",
        "scmt_manifest": args.scmt_root / "scmt_freeze_manifest.json",
        "policy": args.policy_root / "frozen_transport_policy.joblib",
        "public_candidates": args.policy_root / "synthetic_candidate_results.csv",
        "summary": args.ceap_root / "summary.csv",
        "identity": args.ceap_root / "identity_encoding_margin.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    risk_manifest = json.loads(paths["risk_manifest"].read_text(encoding="utf-8"))
    scmt_manifest = json.loads(paths["scmt_manifest"].read_text(encoding="utf-8"))
    risk_model = joblib.load(paths["risk_model"])
    policy = joblib.load(paths["policy"])
    public_candidates = pd.read_csv(paths["public_candidates"])
    features = risk_manifest["numeric_features"]
    public = pd.read_csv(paths["risk_table"], usecols=features + ["material_optimism_event"])
    target_table = load_unlabeled_target(args.ceap_root)
    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    source = scaler.transform(public[features]).astype(np.float32)
    source_labels = public.material_optimism_event.to_numpy(int)
    target = scaler.transform(target_table[features]).astype(np.float32)

    scmt_config = SCMTConfig(**scmt_manifest["selected_config"])
    progress = tqdm(total=1, desc="CEAP frozen SCMT base candidate", unit="fit", dynamic_ncols=True)
    transformed_scmt, _ = fit_scmt(
        source,
        target,
        lambda values: probability(logistic, values),
        scmt_config,
        SCMT_SEED,
        "cuda",
        lambda epoch, loss: progress.set_postfix(loss=f"{loss['total']:.3f}", refresh=False)
        if epoch % 40 == 0 else None,
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
                "scenario_id": "CEAP-360VR",
                "method": method,
                **unlabeled_signature(
                    source, source_labels, target, transformed,
                    lambda values: probability(logistic, values),
                    decision_threshold,
                    reservation["confirmation_contract"]["primary_audit_seed"],
                ),
            }
            for method, transformed in transforms.items()
        ]
    )
    signatures = enrich_signatures(raw_signatures, public_candidates, policy)
    config = config_from_freeze(freeze)
    rankings, base_method, base_reason = rank_minimal_interventions(
        signatures, list(transforms), config, V5_RULE
    )
    mechanism = target_table.copy()
    for method, transformed in transforms.items():
        mechanism[f"probability_{method}"] = probability(logistic, transformed)
    if base_method == "identity":
        mechanism["probability_opct"] = mechanism.probability_identity
        diagnostics = {
            "scale": 1.0, "shift": 0.0, "final_loss": 0.0,
            "probability_mean_shift": 0.0, "probability_rank": 1.0,
            "decision_flip_rate": 0.0, "order_inversions": 0, "applicable": False,
        }
        candidate = "identity"
        projection_reason = "abstain_v5_unlabeled_actionability_gate_failed"
    else:
        projected, diagnostics = fit_opct(
            mechanism.probability_identity.to_numpy(float),
            mechanism[f"probability_{base_method}"].to_numpy(float),
            RULE,
            reservation["confirmation_contract"]["primary_audit_seed"],
            "cuda",
        )
        mechanism["probability_opct"] = projected
        candidate = "opct" if bool(diagnostics["applicable"]) else "identity"
        projection_reason = (
            "opct_locked_before_target_outcomes" if candidate == "opct"
            else "abstain_opct_unlabeled_invariant_failed"
        )

    args.output_root.mkdir(parents=True, exist_ok=False)
    signatures.to_csv(args.output_root / "candidate_signatures.csv", index=False)
    rankings.to_csv(args.output_root / "candidate_distortion_rankings.csv", index=False)
    mechanism.to_csv(args.output_root / "mechanism_probabilities_blinded.csv", index=False)
    action_lock = {
        "status": "ceap_opct_action_locked_before_outcomes_loaded",
        "base_method": base_method,
        "base_action_reason": base_reason,
        "candidate_method": candidate,
        "projection_reason": projection_reason,
        "opct_diagnostics": diagnostics,
        "reservation_sha256": sha256(args.reservation),
        "implementation_lock_sha256": sha256(args.amendment),
        "v6_freeze_sha256": sha256(args.freeze),
        "ceap_summary_sha256": sha256(paths["summary"]),
        "candidate_signatures_sha256": sha256(args.output_root / "candidate_signatures.csv"),
        "mechanism_probabilities_sha256": sha256(
            args.output_root / "mechanism_probabilities_blinded.csv"
        ),
    }
    lock_path = args.output_root / "primary_action_lock.json"
    lock_path.write_text(json.dumps(json_native(action_lock), indent=2), encoding="utf-8")
    lock_hash = sha256(lock_path)

    evaluation = attach_outcomes(
        mechanism, paths["summary"], float(risk_manifest["material_effect_threshold"])
    )
    if sha256(lock_path) != lock_hash:
        raise RuntimeError("CEAP action lock changed after outcomes were loaded")
    seed = int(reservation["confirmation_contract"]["primary_audit_seed"])
    order = balanced_cluster_order(evaluation, config, seed)
    audit_clusters = order[: config.cluster_budgets[0]]
    audit_mask = rows_for_clusters(evaluation, config.cluster_columns, audit_clusters)
    audit = evaluation.loc[audit_mask].reset_index(drop=True)
    heldout = evaluation.loc[~audit_mask].reset_index(drop=True)
    if candidate == "identity":
        certificate = None
        selected = "identity"
        selection_reason = "abstain_no_applicable_opct_candidate"
    else:
        certificate = certify_checkpoint(
            audit,
            ["opct"],
            config,
            seed + RULE.configuration_budget * 1009,
            total_registered_candidates=1,
        ).iloc[0]
        selected = "opct" if bool(certificate.certified) else "identity"
        selection_reason = (
            "selected_frozen_opct_v6_certificate" if selected == "opct"
            else f"abstain_confirmation_failed:{certificate.failure_reason}"
        )
    heldout_metrics = evaluate_action(heldout, selected)
    certificate_pass = bool(certificate is not None and certificate.certified)
    invariant_pass = bool(
        diagnostics["probability_rank"] >= RULE.minimum_rank
        and diagnostics["order_inversions"] == 0
    )
    effectiveness_pass = bool(selected == "opct" and heldout_metrics["brier_gain"] > 0)
    safety_pass = bool(
        (selected == "identity" or certificate_pass)
        and not heldout_metrics["negative_transfer"]
        and heldout_metrics["auc_noninferior"]
        and invariant_pass
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
        "order_preservation_pass": invariant_pass,
        "base_method": base_method,
        "candidate_method": candidate,
        "selected_method": selected,
        "heldout_metrics": heldout_metrics,
        "interpretation": (
            "frozen OPCT v6 passed untouched CEAP effectiveness and safety gates"
            if claim_supported else (
                "safe abstention without evidence of external effectiveness"
                if selected == "identity" and safety_pass
                else "untouched CEAP confirmation did not support the target claim"
            )
        ),
    }
    assignments.to_csv(args.output_root / "primary_audit_cluster_order.csv", index=False)
    audit.to_csv(args.output_root / "primary_audit_labeled.csv", index=False)
    heldout.to_csv(args.output_root / "primary_heldout_results.csv", index=False)
    (args.output_root / "primary_candidate_certificate.json").write_text(
        json.dumps(json_native(certificate_payload), indent=2), encoding="utf-8"
    )
    (args.output_root / "external_confirmation_gate.json").write_text(
        json.dumps(json_native(gate), indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "ceap_v6_one_shot_external_confirmation_complete",
        "claim_supported": claim_supported,
        "gpu_name": torch.cuda.get_device_name(0),
        "action_lock_sha256": lock_hash,
        "reservation_sha256": sha256(args.reservation),
        "implementation_lock_sha256": sha256(args.amendment),
        "v6_freeze_sha256": sha256(args.freeze),
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "external_confirmation_manifest.json").write_text(
        json.dumps(json_native(manifest), indent=2), encoding="utf-8"
    )
    print(json.dumps(json_native(gate), indent=2))


if __name__ == "__main__":
    main()
