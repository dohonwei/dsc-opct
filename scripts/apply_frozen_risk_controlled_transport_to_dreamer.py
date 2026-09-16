from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_identity_shortcut_risk import MODEL_CAPACITY  # noqa: E402
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
    applicability_table,
    attach_conformal_ood,
    balanced_cluster_order,
    certify_checkpoint,
    probability_methods,
    rows_for_clusters,
    run_registered_checkpoints,
    select_certified_action,
    validate_prediction_contract,
)
from identity_shortcut.transport_policy import (  # noqa: E402
    committee_distribution,
    unlabeled_signature,
)


KEYS = ["dataset", "task", "axis", "representation", "model", "split_seed"]
CLUSTER_COLUMNS = ["task", "representation", "model", "split_seed"]
SCMT_SEED = 20260905


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot external application of frozen v4 to DREAMER."
    )
    parser.add_argument(
        "--execution-freeze",
        type=Path,
        default=Path("docs/dreamer_counterfactual_execution_freeze.json"),
    )
    parser.add_argument(
        "--v4-freeze", type=Path, default=Path("docs/risk_controlled_transport_v4_freeze.json")
    )
    parser.add_argument(
        "--application-freeze",
        type=Path,
        default=Path("docs/dreamer_v4_application_freeze.json"),
    )
    parser.add_argument(
        "--counterfactual-root",
        type=Path,
        default=Path("outputs/dreamer_counterfactual_external"),
    )
    parser.add_argument(
        "--identity-probe",
        type=Path,
        default=Path("outputs/dreamer_identity_probes/per_seed_summary.csv"),
    )
    parser.add_argument(
        "--risk-root", type=Path, default=Path("outputs/identity_shortcut_risk_classifier")
    )
    parser.add_argument(
        "--scmt-root", type=Path, default=Path("outputs/scmt_public_development")
    )
    parser.add_argument(
        "--policy-root", type=Path, default=Path("outputs/transport_policy_public_development")
    )
    parser.add_argument(
        "--v4-development-root",
        type=Path,
        default=Path("outputs/risk_controlled_transport_v4_development"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dreamer_v4_external_confirmation"),
    )
    parser.add_argument("--primary-audit-seed", type=int, default=20260907)
    parser.add_argument("--repeated-audits", type=int, default=100)
    parser.add_argument("--audit-seed-step", type=int, default=7919)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probability(logistic, values: np.ndarray) -> np.ndarray:
    return logistic.predict_proba(np.asarray(values))[:, 1]


def safe_auc(labels: np.ndarray, probability_values: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, probability_values))


def assert_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"One-shot DREAMER output already exists at {path}; refusing to overwrite it"
        )


def validate_frozen_inputs(args: argparse.Namespace) -> tuple[dict, dict, dict[str, Path]]:
    execution = json.loads(args.execution_freeze.read_text(encoding="utf-8"))
    v4_freeze = json.loads(args.v4_freeze.read_text(encoding="utf-8"))
    application = json.loads(args.application_freeze.read_text(encoding="utf-8"))
    if sha256(Path(__file__)) != application["application_script_sha256"]:
        raise ValueError("DREAMER application script changed after freeze")
    if sha256(args.execution_freeze) != application["execution_freeze_sha256"]:
        raise ValueError("DREAMER counterfactual execution freeze changed")
    registered = application["audit_contract"]
    observed = {
        "primary_audit_seed": args.primary_audit_seed,
        "repeated_audits": args.repeated_audits,
        "audit_seed_step": args.audit_seed_step,
        "bootstrap_repetitions": args.bootstrap_repetitions,
    }
    if observed != registered:
        raise ValueError(f"Audit contract differs from application freeze: {observed}")
    frozen_paths = {
        key: ROOT / key for key in execution["hashes"]
    }
    for key, path in frozen_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        expected = execution["hashes"][key]
        if observed != expected:
            raise ValueError(f"Frozen input hash mismatch for {key}: {observed} != {expected}")
    if sha256(args.v4_freeze) != execution["upstream_freezes"]["v4_policy_freeze_sha256"]:
        raise ValueError("v4 freeze hash mismatch")
    v4_code = ROOT / "src" / "identity_shortcut" / "risk_controlled_transport.py"
    if sha256(v4_code) != v4_freeze["hashes"]["risk_controlled_transport_code_sha256"]:
        raise ValueError("Frozen v4 implementation changed")
    paths = {
        "risk_manifest": args.risk_root / "freeze_manifest.json",
        "risk_model": args.risk_root / "frozen_risk_model.joblib",
        "risk_table": args.risk_root / "risk_learning_table.csv",
        "scmt_manifest": args.scmt_root / "scmt_freeze_manifest.json",
        "policy_manifest": args.policy_root / "transport_policy_freeze_manifest.json",
        "policy_artifact": args.policy_root / "frozen_transport_policy.joblib",
        "public_candidates": args.policy_root / "synthetic_candidate_results.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return execution, v4_freeze, paths


def build_target_mechanism_table(summary_path: Path, identity_probe_path: Path) -> pd.DataFrame:
    columns = KEYS + [
        "nominal_dose",
        "achieved_opportunity",
        "metadata_prior_opportunity",
    ]
    table = pd.read_csv(summary_path, usecols=columns)
    table = table.loc[(table.dataset == "DREAMER") & (table.axis == "subject")].copy()
    if set(table.task) != {"arousal", "valence"}:
        raise ValueError("DREAMER task contract mismatch")
    anchors = table.loc[
        table.nominal_dose.eq(0.0),
        KEYS + ["achieved_opportunity", "metadata_prior_opportunity"],
    ].rename(
        columns={
            "achieved_opportunity": "dose_zero_label_opportunity",
            "metadata_prior_opportunity": "dose_zero_metadata_prior_opportunity",
        }
    )
    if anchors.duplicated(KEYS).any():
        raise ValueError("DREAMER dose-zero anchors are not unique")
    table = table.merge(anchors, on=KEYS, validate="many_to_one")
    table = table.loc[table.nominal_dose.gt(0.0)].copy()
    table["opportunity_delta"] = (
        table.achieved_opportunity - table.dose_zero_label_opportunity
    )
    table["metadata_prior_delta"] = (
        table.metadata_prior_opportunity - table.dose_zero_metadata_prior_opportunity
    )

    probe = pd.read_csv(identity_probe_path)
    probe = probe.loc[
        (probe.dataset == "DREAMER") & (probe.axis == "subject_across_stimuli")
    ].copy()
    probe["encoding_margin"] = probe.balanced_accuracy - probe.chance
    probe = probe.groupby("split_seed", as_index=False).encoding_margin.mean()
    table = table.merge(probe, on="split_seed", validate="many_to_one")
    table["dose_encoding_interaction"] = table.opportunity_delta * table.encoding_margin
    table["model_capacity"] = table.model.map(MODEL_CAPACITY)
    if table.model_capacity.isna().any():
        raise ValueError("DREAMER contains a model absent from the frozen capacity mapping")
    if len(table) != 240:
        raise ValueError(f"Expected 240 DREAMER configurations, observed {len(table)}")
    if table.duplicated(CLUSTER_COLUMNS + ["nominal_dose"]).any():
        raise ValueError("Duplicate DREAMER cluster-dose rows")
    return table.sort_values(KEYS + ["nominal_dose"]).reset_index(drop=True)


def enrich_signatures(
    raw_signatures: pd.DataFrame,
    public_candidates: pd.DataFrame,
    policy_artifact: dict[str, object],
) -> pd.DataFrame:
    signatures = attach_conformal_ood(public_candidates, raw_signatures)
    signatures["predicted_brier_gain"] = 0.0
    signatures["predicted_auc_delta"] = 0.0
    candidate_mask = signatures.method.ne("identity")
    candidates = signatures.loc[candidate_mask]
    if len(candidates):
        gain = committee_distribution(policy_artifact["gain_committee"], candidates)
        auc = committee_distribution(policy_artifact["auc_committee"], candidates)
        signatures.loc[candidate_mask, "predicted_brier_gain"] = gain.mean(axis=1)
        signatures.loc[candidate_mask, "predicted_auc_delta"] = auc.mean(axis=1)
    return signatures


def cluster_key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        row["task"],
        row["representation"],
        row["model"],
        int(float(row["split_seed"])),
    )


def load_counterfactual_outcomes(
    summary_path: Path,
    threshold: float,
    allowed_clusters: set[tuple[object, ...]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = set(KEYS + ["nominal_dose", "exposure_effect"])
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Counterfactual summary lacks frozen outcome columns")
        for raw in reader:
            if raw["dataset"] != "DREAMER" or raw["axis"] != "subject":
                continue
            key = cluster_key(raw)
            if allowed_clusters is not None and key not in allowed_clusters:
                continue
            rows.append(
                {
                    "dataset": raw["dataset"],
                    "task": raw["task"],
                    "axis": raw["axis"],
                    "representation": raw["representation"],
                    "model": raw["model"],
                    "split_seed": int(float(raw["split_seed"])),
                    "nominal_dose": float(raw["nominal_dose"]),
                    "exposure_effect": float(raw["exposure_effect"]),
                }
            )
    outcomes = pd.DataFrame(rows)
    anchors = outcomes.loc[outcomes.nominal_dose.eq(0.0), KEYS + ["exposure_effect"]].rename(
        columns={"exposure_effect": "dose_zero_exposure_effect"}
    )
    if anchors.duplicated(KEYS).any():
        raise ValueError("Outcome dose-zero anchors are not unique")
    outcomes = outcomes.merge(anchors, on=KEYS, validate="many_to_one")
    outcomes = outcomes.loc[outcomes.nominal_dose.gt(0.0)].copy()
    outcomes["dose_induced_amplification"] = (
        outcomes.exposure_effect - outcomes.dose_zero_exposure_effect
    )
    outcomes["material_optimism_event"] = (
        outcomes.dose_induced_amplification >= threshold
    ).astype(int)
    return outcomes


def attach_outcomes(
    mechanism: pd.DataFrame,
    summary_path: Path,
    threshold: float,
    allowed_clusters: list[tuple[object, ...]] | None = None,
) -> pd.DataFrame:
    allowed = set(allowed_clusters) if allowed_clusters is not None else None
    outcomes = load_counterfactual_outcomes(summary_path, threshold, allowed)
    base = mechanism
    if allowed_clusters is not None:
        mask = rows_for_clusters(mechanism, CLUSTER_COLUMNS, allowed_clusters)
        base = mechanism.loc[mask]
    output = base.merge(outcomes, on=KEYS + ["nominal_dose"], validate="one_to_one")
    if len(output) != len(base):
        raise ValueError("Incomplete DREAMER outcome merge")
    return output.reset_index(drop=True)


def select_primary_action(
    mechanism: pd.DataFrame,
    signatures: pd.DataFrame,
    summary_path: Path,
    threshold: float,
    config: RiskControlledConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[object, ...]]]:
    methods = probability_methods(mechanism)
    applicability = applicability_table(signatures, methods, config)
    applicable = applicability.loc[
        applicability.applicable & applicability.method.ne("identity"), "method"
    ].tolist()
    order = balanced_cluster_order(mechanism, config, seed)
    registered_candidates = max(1, len([method for method in methods if method != "identity"]))
    locked_method: str | None = None
    locked_budget: int | None = None
    decision_rows: list[dict[str, object]] = []
    certificate_rows: list[pd.DataFrame] = []
    for configuration_budget, cluster_budget in zip(
        config.configuration_budgets, config.cluster_budgets, strict=True
    ):
        if locked_method is None:
            audit_clusters = order[:cluster_budget]
            audit = attach_outcomes(
                mechanism, summary_path, threshold, audit_clusters
            )
            certification = certify_checkpoint(
                audit,
                applicable,
                config,
                seed + configuration_budget * 1009,
                total_registered_candidates=registered_candidates,
            )
            selected, reason = select_certified_action(certification)
            if selected != "identity":
                locked_method = selected
                locked_budget = cluster_budget
            if not certification.empty:
                certification.insert(0, "configuration_budget", configuration_budget)
                certification.insert(1, "cluster_budget", cluster_budget)
                certificate_rows.append(certification)
        else:
            selected = locked_method
            reason = "carried_forward_locked_certificate"
            audit_clusters = order[: int(locked_budget)]
            audit = attach_outcomes(mechanism, summary_path, threshold, audit_clusters)
        decision_rows.append(
            {
                "configuration_budget": configuration_budget,
                "cluster_budget": cluster_budget,
                "actual_configuration_budget": len(audit),
                "actual_cluster_budget": len(audit_clusters),
                "n_audit_events": int(audit.material_optimism_event.sum()),
                "n_applicable_candidates": len(applicable),
                "selected_method": selected,
                "selection_reason": reason,
            }
        )
    certificates = pd.concat(certificate_rows, ignore_index=True) if certificate_rows else pd.DataFrame()
    return pd.DataFrame(decision_rows), certificates, order


def evaluate_action(frame: pd.DataFrame, method: str) -> dict[str, float | bool]:
    labels = frame.material_optimism_event.to_numpy(int)
    identity = frame.probability_identity.to_numpy(float)
    selected = frame[f"probability_{method}"].to_numpy(float)
    identity_brier = float(np.mean((labels - identity) ** 2))
    selected_brier = float(np.mean((labels - selected) ** 2))
    identity_auc = safe_auc(labels, identity)
    selected_auc = safe_auc(labels, selected)
    gain = identity_brier - selected_brier
    auc_delta = selected_auc - identity_auc if np.isfinite(identity_auc) and np.isfinite(selected_auc) else np.nan
    return {
        "brier_gain": gain,
        "auc_delta": auc_delta,
        "negative_transfer": bool(method != "identity" and gain < 0.0),
        "adapted": method != "identity",
    }


def paired_cluster_interval(
    frame: pd.DataFrame,
    method: str,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    point = evaluate_action(frame, method)
    groups = [
        group.index.to_numpy(int)
        for _, group in frame.reset_index(drop=True).groupby(CLUSTER_COLUMNS, sort=False)
    ]
    labels = frame.material_optimism_event.to_numpy(int)
    identity = frame.probability_identity.to_numpy(float)
    selected = frame[f"probability_{method}"].to_numpy(float)
    identity_loss = (labels - identity) ** 2
    selected_loss = (labels - selected) ** 2
    rng = np.random.default_rng(seed)
    gains = np.empty(repetitions, dtype=float)
    auc_deltas = np.full(repetitions, np.nan, dtype=float)
    progress = tqdm(total=repetitions, desc="DREAMER held-out cluster bootstrap", unit="draw", dynamic_ncols=True)
    for repetition in range(repetitions):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        gains[repetition] = float(np.mean(identity_loss[indices] - selected_loss[indices]))
        identity_auc = safe_auc(labels[indices], identity[indices])
        selected_auc = safe_auc(labels[indices], selected[indices])
        if np.isfinite(identity_auc) and np.isfinite(selected_auc):
            auc_deltas[repetition] = selected_auc - identity_auc
        progress.update(1)
    progress.close()
    valid_auc = auc_deltas[np.isfinite(auc_deltas)]
    return {
        **point,
        "brier_gain_ci95": [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))],
        "auc_delta_ci95": (
            [float(np.quantile(valid_auc, 0.025)), float(np.quantile(valid_auc, 0.975))]
            if len(valid_auc) >= 200 else [None, None]
        ),
        "valid_auc_bootstraps": int(len(valid_auc)),
    }


def main() -> None:
    args = parse_args()
    assert_empty_output(args.output_root)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for debugging.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    execution, v4_freeze, paths = validate_frozen_inputs(args)
    summary_path = args.counterfactual_root / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    mechanism = build_target_mechanism_table(summary_path, args.identity_probe)

    risk_manifest = json.loads(paths["risk_manifest"].read_text(encoding="utf-8"))
    scmt_manifest = json.loads(paths["scmt_manifest"].read_text(encoding="utf-8"))
    policy_manifest = json.loads(paths["policy_manifest"].read_text(encoding="utf-8"))
    if not policy_manifest.get("admissible"):
        raise ValueError("Frozen public transport policy is not admissible")
    risk_model = joblib.load(paths["risk_model"])
    policy_artifact = joblib.load(paths["policy_artifact"])
    features = policy_manifest["features"]
    public = pd.read_csv(paths["risk_table"], usecols=features + ["material_optimism_event"])
    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    source = scaler.transform(public[features]).astype(np.float32)
    source_labels = public.material_optimism_event.to_numpy(int)
    target = scaler.transform(mechanism[features]).astype(np.float32)

    scmt_config = SCMTConfig(**scmt_manifest["selected_config"])
    progress = tqdm(total=1, desc="DREAMER frozen SCMT candidate", unit="fit", dynamic_ncols=True)
    transformed_scmt, scmt_diagnostics = fit_scmt(
        source,
        target,
        lambda values: probability(logistic, values),
        scmt_config,
        SCMT_SEED,
        device,
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
                "scenario_id": "DREAMER",
                "method": method,
                **unlabeled_signature(
                    source,
                    source_labels,
                    target,
                    transformed,
                    lambda values: probability(logistic, values),
                    decision_threshold,
                    args.primary_audit_seed,
                ),
            }
            for method, transformed in transforms.items()
        ]
    )
    public_candidates = pd.read_csv(paths["public_candidates"])
    signatures = enrich_signatures(raw_signatures, public_candidates, policy_artifact)
    probabilities = {method: probability(logistic, transformed) for method, transformed in transforms.items()}
    for method, values in probabilities.items():
        mechanism[f"probability_{method}"] = values

    config = RiskControlledConfig(
        bootstrap_repetitions=args.bootstrap_repetitions,
        minimum_valid_auc_bootstraps=max(200, args.bootstrap_repetitions // 10),
    )
    validate_prediction_contract(
        mechanism.assign(material_optimism_event=0), config
    )
    applicability = applicability_table(signatures, probability_methods(mechanism), config)
    threshold = float(risk_manifest["material_effect_threshold"])
    decisions, certificates, order = select_primary_action(
        mechanism,
        signatures,
        summary_path,
        threshold,
        config,
        args.primary_audit_seed,
    )
    final_decision = decisions.iloc[-1]
    selected_method = str(final_decision.selected_method)
    actual_cluster_budget = int(final_decision.actual_cluster_budget)

    args.output_root.mkdir(parents=True, exist_ok=False)
    mechanism.to_csv(args.output_root / "mechanism_probabilities_blinded.csv", index=False)
    signatures.to_csv(args.output_root / "candidate_signatures.csv", index=False)
    applicability.to_csv(args.output_root / "applicability_gates.csv", index=False)
    decisions.to_csv(args.output_root / "primary_checkpoint_decisions.csv", index=False)
    certificates.to_csv(args.output_root / "primary_candidate_certificates.csv", index=False)
    pd.DataFrame(order, columns=CLUSTER_COLUMNS).to_csv(
        args.output_root / "primary_audit_cluster_order.csv", index=False
    )
    lock = {
        "status": "primary_action_locked_before_heldout_outcomes_loaded",
        "selected_method": selected_method,
        "selection_reason": str(final_decision.selection_reason),
        "primary_audit_seed": args.primary_audit_seed,
        "actual_configuration_budget": int(final_decision.actual_configuration_budget),
        "actual_cluster_budget": actual_cluster_budget,
        "execution_freeze_sha256": sha256(args.execution_freeze),
        "v4_freeze_sha256": sha256(args.v4_freeze),
        "counterfactual_summary_sha256": sha256(summary_path),
    }
    lock_path = args.output_root / "primary_action_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    lock_hash = sha256(lock_path)

    evaluation = attach_outcomes(mechanism, summary_path, threshold)
    audit_clusters = order[:actual_cluster_budget]
    audit_mask = rows_for_clusters(evaluation, CLUSTER_COLUMNS, audit_clusters)
    held_out = evaluation.loc[~audit_mask].reset_index(drop=True)
    primary_interval = paired_cluster_interval(
        held_out,
        selected_method,
        args.bootstrap_repetitions,
        args.primary_audit_seed + 104729,
    )
    held_out.to_csv(args.output_root / "primary_heldout_predictions.csv", index=False)

    repeated_rows: list[dict[str, object]] = []
    repeated_decisions: list[pd.DataFrame] = []
    repeated_seeds = [args.primary_audit_seed + index * args.audit_seed_step for index in range(args.repeated_audits)]
    progress = tqdm(total=len(repeated_seeds), desc="DREAMER repeated audit robustness", unit="audit", dynamic_ncols=True)
    for repetition, seed in enumerate(repeated_seeds):
        repeat_decision, _, repeat_order = run_registered_checkpoints(
            evaluation, signatures, config, seed
        )
        repeat_decision.insert(0, "audit_repetition", repetition)
        repeat_decision.insert(1, "audit_seed", seed)
        repeated_decisions.append(repeat_decision)
        selected = repeat_decision.iloc[-1]
        clusters = repeat_order[: int(selected.actual_cluster_budget)]
        mask = rows_for_clusters(evaluation, CLUSTER_COLUMNS, clusters)
        test = evaluation.loc[~mask].reset_index(drop=True)
        metrics = evaluate_action(test, str(selected.selected_method))
        repeated_rows.append(
            {
                "audit_repetition": repetition,
                "audit_seed": seed,
                "selected_method": str(selected.selected_method),
                "actual_configuration_budget": int(selected.actual_configuration_budget),
                "actual_cluster_budget": int(selected.actual_cluster_budget),
                "n_heldout_configurations": len(test),
                **metrics,
            }
        )
        progress.update(1)
        progress.set_postfix(method=str(selected.selected_method), refresh=False)
    progress.close()
    robustness = pd.DataFrame(repeated_rows)
    pd.concat(repeated_decisions, ignore_index=True).to_csv(
        args.output_root / "repeated_checkpoint_decisions.csv", index=False
    )
    robustness.to_csv(args.output_root / "repeated_audit_results.csv", index=False)
    adapted = robustness.loc[robustness.adapted]
    negative_transfer_frequency = float(robustness.negative_transfer.mean())
    coverage = float(robustness.adapted.mean())
    negative_when_adapted = float(adapted.negative_transfer.mean()) if len(adapted) else 0.0

    gain_ci = primary_interval["brier_gain_ci95"]
    auc_ci = primary_interval["auc_delta_ci95"]
    criteria = {
        "nonidentity_certified": selected_method != "identity",
        "positive_heldout_brier_gain_lower_bound": gain_ci[0] > 0.0,
        "maximum_negative_transfer_frequency": negative_transfer_frequency <= 0.05,
        "maximum_brier_worsening_upper_bound": -gain_ci[0] <= 0.001,
        "minimum_auc_delta_lower_bound": auc_ci[0] is not None and auc_ci[0] >= -0.02,
    }
    gate = {
        "status": "external_confirmation_complete",
        "claim": "effective_and_safe_cross_domain_transport",
        "claim_supported": bool(all(criteria.values())),
        "primary_selected_method": selected_method,
        "primary": primary_interval,
        "repeated_audit_coverage": coverage,
        "repeated_audit_negative_transfer_frequency": negative_transfer_frequency,
        "repeated_audit_negative_transfer_when_adapted": negative_when_adapted,
        "criteria": criteria,
        "interpretation": (
            "The frozen external gate passed."
            if all(criteria.values())
            else "The frozen external gate did not pass; the effectiveness-and-safety claim must be narrowed."
        ),
    }
    (args.output_root / "external_confirmation_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {key: value for key, value in scmt_diagnostics.items() if not isinstance(value, dict)}
            | {f"check_{key}": value for key, value in scmt_diagnostics["checks"].items()}
        ]
    ).to_csv(args.output_root / "scmt_candidate_diagnostics.csv", index=False)
    manifest = {
        "status": "dreamer_v4_one_shot_external_confirmation_complete",
        "dataset": "DREAMER",
        "n_configurations": len(evaluation),
        "primary_action_lock_sha256": lock_hash,
        "primary_audit_seed": args.primary_audit_seed,
        "repeated_audit_seed_scheme": f"{args.primary_audit_seed} + repetition * {args.audit_seed_step}",
        "repeated_audits": args.repeated_audits,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_unit": "task x representation x model x split_seed cluster",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "execution_freeze_sha256": sha256(args.execution_freeze),
        "v4_freeze_sha256": sha256(args.v4_freeze),
        "application_script_sha256": sha256(Path(__file__)),
        "label_blinding_contract": (
            "The primary audit order was built without outcome values. Only registered audit-cluster "
            "outcomes were interpreted for certification; the action lock was persisted before "
            "held-out outcomes were loaded for evaluation."
        ),
    }
    (args.output_root / "external_confirmation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
