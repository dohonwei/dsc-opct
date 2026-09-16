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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from scipy.special import logit
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_identity_shortcut_risk import MODEL_CAPACITY  # noqa: E402
from apply_frozen_scmt_to_eppvr import paired_bootstrap, performance  # noqa: E402
from identity_shortcut.domain_transport import (  # noqa: E402
    SCMTConfig,
    coral,
    fit_scmt,
    mean_shift,
    quantile_map,
    support_clip,
)
from identity_shortcut.transport_policy import (  # noqa: E402
    PolicyConfig,
    select_actions,
    unlabeled_signature,
)


SCMT_SEED = 20260905
KEYS = ["dataset", "task", "axis", "representation", "model", "split_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot application of the frozen unlabeled transport policy to SEED-IV."
    )
    parser.add_argument(
        "--registration", type=Path, default=Path("docs/seediv_external_freeze.json")
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
        "--valence-root", type=Path, default=Path("outputs/seediv_valence_counterfactual_external")
    )
    parser.add_argument(
        "--arousal-root", type=Path, default=Path("outputs/seediv_arousal_counterfactual_external")
    )
    parser.add_argument(
        "--valence-audit", type=Path, default=Path("outputs/seediv_valence_identity_audit")
    )
    parser.add_argument(
        "--arousal-audit", type=Path, default=Path("outputs/seediv_arousal_identity_audit")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/seediv_transport_policy_external")
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260907)
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


def assert_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"One-shot external output already exists at {path}; refusing to overwrite it"
        )


def load_unlabeled_target(args: argparse.Namespace) -> pd.DataFrame:
    summary_columns = [
        "dataset",
        "task",
        "axis",
        "representation",
        "model",
        "split_seed",
        "nominal_dose",
        "achieved_opportunity",
        "metadata_prior_opportunity",
    ]
    summaries = []
    audits = []
    for task, summary_root, audit_root in (
        ("valence", args.valence_root, args.valence_audit),
        ("arousal", args.arousal_root, args.arousal_audit),
    ):
        summary = pd.read_csv(summary_root / "summary.csv", usecols=summary_columns)
        if set(summary.task) != {task}:
            raise ValueError(f"{task} counterfactual summary contains an unexpected task")
        summaries.append(summary)
        audit = pd.read_csv(audit_root / "per_seed_summary.csv")
        audit = audit.loc[audit.axis == "subject_across_stimuli"].copy()
        audit["encoding_margin"] = audit.balanced_accuracy - audit.chance
        audits.append(
            audit.groupby(["dataset", "representation", "n_features", "split_seed"])
            .encoding_margin.mean()
            .reset_index()
            .assign(task=task)
        )
    table = pd.concat(summaries, ignore_index=True).rename(
        columns={"achieved_opportunity": "label_opportunity"}
    )
    probes = pd.concat(audits, ignore_index=True)
    table = table.merge(
        probes,
        on=["dataset", "task", "representation", "split_seed"],
        validate="many_to_one",
    )
    anchors = table.loc[
        table.nominal_dose == 0,
        KEYS + ["label_opportunity", "metadata_prior_opportunity"],
    ].rename(
        columns={
            "label_opportunity": "dose_zero_label_opportunity",
            "metadata_prior_opportunity": "dose_zero_metadata_prior_opportunity",
        }
    )
    if anchors.duplicated(KEYS).any():
        raise ValueError("SEED-IV dose-zero anchors are not unique")
    table = table.merge(anchors, on=KEYS, validate="many_to_one")
    table = table.loc[table.nominal_dose > 0].copy()
    table["opportunity_delta"] = (
        table.label_opportunity - table.dose_zero_label_opportunity
    )
    table["metadata_prior_delta"] = (
        table.metadata_prior_opportunity
        - table.dose_zero_metadata_prior_opportunity
    )
    table["dose_encoding_interaction"] = table.opportunity_delta * table.encoding_margin
    table["model_capacity"] = table.model.map(MODEL_CAPACITY)
    if table.model_capacity.isna().any():
        raise ValueError("SEED-IV contains a model absent from the frozen capacity mapping")
    if len(table) != 240:
        raise ValueError(f"Expected 240 frozen SEED-IV configurations, observed {len(table)}")
    return table.sort_values(KEYS + ["nominal_dose"]).reset_index(drop=True)


def attach_outcomes(table: pd.DataFrame, args: argparse.Namespace, threshold: float) -> pd.DataFrame:
    columns = KEYS + ["nominal_dose", "exposure_effect"]
    summaries = pd.concat(
        [
            pd.read_csv(args.valence_root / "summary.csv", usecols=columns),
            pd.read_csv(args.arousal_root / "summary.csv", usecols=columns),
        ],
        ignore_index=True,
    )
    anchors = summaries.loc[
        summaries.nominal_dose == 0, KEYS + ["exposure_effect"]
    ].rename(columns={"exposure_effect": "dose_zero_exposure_effect"})
    outcomes = summaries.merge(anchors, on=KEYS, validate="many_to_one")
    outcomes = outcomes.loc[outcomes.nominal_dose > 0].copy()
    outcomes["dose_induced_amplification"] = (
        outcomes.exposure_effect - outcomes.dose_zero_exposure_effect
    )
    outcomes["material_optimism_event"] = (
        outcomes.dose_induced_amplification >= threshold
    ).astype(int)
    return table.merge(outcomes, on=KEYS + ["nominal_dose"], validate="one_to_one")


def supervised_recalibration_oracle(
    table: pd.DataFrame, raw_probability: np.ndarray
) -> np.ndarray:
    predictor = logit(np.clip(raw_probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    target = table.material_optimism_event.to_numpy(int)
    groups = table[["task", "representation", "model", "split_seed"]].astype(str).agg("|".join, axis=1)
    estimate = np.full(len(table), np.nan)
    for train, test in GroupKFold(5).split(predictor, target, groups):
        model = LogisticRegression(C=1.0, max_iter=3000)
        model.fit(predictor[train], target[train])
        estimate[test] = model.predict_proba(predictor[test])[:, 1]
    if not np.isfinite(estimate).all():
        raise ValueError("Incomplete supervised recalibration oracle predictions")
    return estimate


def decision_utility(table: pd.DataFrame, probability_values: np.ndarray) -> pd.DataFrame:
    target = table.material_optimism_event.to_numpy(int)
    ordering = np.argsort(-probability_values)
    total_events = int(target.sum())
    rows = []
    for fraction in (0.05, 0.10, 0.20, 0.30, 0.50):
        count = max(1, int(np.ceil(len(table) * fraction)))
        selected = target[ordering[:count]]
        found = int(selected.sum())
        rows.append(
            {
                "audit_budget_fraction": fraction,
                "n_audited": count,
                "events_found": found,
                "event_recall": found / total_events if total_events else np.nan,
                "event_precision": found / count,
                "events_per_10_audits": 10.0 * found / count,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    assert_empty_output(args.output_root)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for debugging.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    paths = {
        "risk_manifest": args.risk_root / "freeze_manifest.json",
        "risk_model": args.risk_root / "frozen_risk_model.joblib",
        "risk_table": args.risk_root / "risk_learning_table.csv",
        "scmt_manifest": args.scmt_root / "scmt_freeze_manifest.json",
        "policy_manifest": args.policy_root / "transport_policy_freeze_manifest.json",
        "policy_artifact": args.policy_root / "frozen_transport_policy.joblib",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(paths["policy_manifest"]) != registration["policy_manifest_sha256"]:
        raise ValueError("Registered policy-manifest hash mismatch")
    if sha256(paths["policy_artifact"]) != registration["policy_artifact_sha256"]:
        raise ValueError("Registered policy-artifact hash mismatch")
    if sha256(paths["risk_model"]) != registration["risk_model_sha256"]:
        raise ValueError("Registered risk-model hash mismatch")
    for task, item in registration["feature_files"].items():
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Registered SEED-IV {task} feature hash mismatch")

    risk_manifest = json.loads(paths["risk_manifest"].read_text(encoding="utf-8"))
    scmt_manifest = json.loads(paths["scmt_manifest"].read_text(encoding="utf-8"))
    policy_manifest = json.loads(paths["policy_manifest"].read_text(encoding="utf-8"))
    if not policy_manifest.get("admissible"):
        raise ValueError("The public transport policy did not pass its frozen gate")
    if sha256(ROOT / "src" / "identity_shortcut" / "transport_policy.py") != policy_manifest["policy_code_sha256"]:
        raise ValueError("Transport-policy implementation changed after freeze")

    features = policy_manifest["features"]
    public = pd.read_csv(paths["risk_table"], usecols=features + ["material_optimism_event"])
    target_table = load_unlabeled_target(args)
    risk_model = joblib.load(paths["risk_model"])
    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    source = scaler.transform(public[features]).astype(np.float32)
    source_labels = public.material_optimism_event.to_numpy(int)
    target = scaler.transform(target_table[features]).astype(np.float32)

    scmt_config = SCMTConfig(**scmt_manifest["selected_config"])
    progress = tqdm(total=1, desc="SEED-IV frozen-policy SCMT candidate", unit="fit", dynamic_ncols=True)
    transformed_scmt, scmt_diagnostics = fit_scmt(
        source,
        target,
        lambda values: probability(logistic, values),
        scmt_config,
        SCMT_SEED,
        device,
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
    signatures = pd.DataFrame(
        [
            {
                "scenario_id": "SEED-IV",
                "method": method,
                **unlabeled_signature(
                    source,
                    source_labels,
                    target,
                    transformed,
                    lambda values: probability(logistic, values),
                    decision_threshold,
                    args.bootstrap_seed,
                ),
            }
            for method, transformed in transforms.items()
        ]
    )
    artifact = joblib.load(paths["policy_artifact"])
    policy_config = PolicyConfig(**artifact["policy_config"])
    decision = select_actions(
        signatures, artifact["gain_committee"], artifact["auc_committee"], policy_config
    )
    selected_method = str(decision.iloc[0].selected_method)
    selection_reason = str(decision.iloc[0].selection_reason)

    # Persist the action before any target outcome column is read.
    args.output_root.mkdir(parents=True, exist_ok=False)
    signatures.to_csv(args.output_root / "unlabeled_candidate_signatures.csv", index=False)
    decision.to_csv(args.output_root / "frozen_policy_decision.csv", index=False)
    decision_lock = {
        "status": "action_locked_before_target_outcomes_loaded",
        "selected_method": selected_method,
        "selection_reason": selection_reason,
        "policy_manifest_sha256": sha256(paths["policy_manifest"]),
        "policy_artifact_sha256": sha256(paths["policy_artifact"]),
        "risk_model_sha256": sha256(paths["risk_model"]),
        "registration_sha256": sha256(args.registration),
    }
    lock_path = args.output_root / "decision_lock.json"
    lock_path.write_text(json.dumps(decision_lock, indent=2), encoding="utf-8")
    lock_hash = sha256(lock_path)

    evaluation = attach_outcomes(
        target_table, args, float(risk_manifest["material_effect_threshold"])
    )
    labels = evaluation.material_optimism_event.to_numpy(int)
    null_probability = float(risk_manifest["public_training_event_prevalence"])
    probabilities = {
        method: probability(logistic, transformed) for method, transformed in transforms.items()
    }
    probabilities["supervised_recalibration_oracle"] = supervised_recalibration_oracle(
        evaluation, probabilities["identity"]
    )
    metrics = pd.DataFrame(
        [
            {
                "method": method,
                "selected_by_policy": method == selected_method,
                "oracle_only": method == "supervised_recalibration_oracle",
                **performance(labels, values, decision_threshold, null_probability),
            }
            for method, values in probabilities.items()
        ]
    )
    for method, values in probabilities.items():
        evaluation[f"probability_{method}"] = values
    intervals = paired_bootstrap(
        evaluation,
        probabilities,
        decision_threshold,
        null_probability,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    selected_metrics = metrics.loc[metrics.method == selected_method].iloc[0]
    identity_metrics = metrics.loc[metrics.method == "identity"].iloc[0]
    if selected_method == "identity":
        brier_delta_ci = (0.0, 0.0)
        auc_delta_ci = (0.0, 0.0)
    else:
        chosen_intervals = intervals.loc[intervals.method == selected_method]
        brier_row = chosen_intervals.loc[chosen_intervals.metric == "brier_score"].iloc[0]
        auc_row = chosen_intervals.loc[chosen_intervals.metric == "roc_auc"].iloc[0]
        brier_delta_ci = (float(brier_row.delta_ci_low), float(brier_row.delta_ci_high))
        auc_delta_ci = (float(auc_row.delta_ci_low), float(auc_row.delta_ci_high))
    brier_delta = float(selected_metrics.brier_score - identity_metrics.brier_score)
    auc_delta = float(selected_metrics.roc_auc - identity_metrics.roc_auc)
    safety_pass = (
        brier_delta_ci[1] <= registration["safety_endpoints"]["maximum_brier_worsening"]
        and auc_delta_ci[0] >= registration["safety_endpoints"]["auc_noninferiority_margin"]
    )
    benefit_pass = selected_method != "identity" and brier_delta_ci[1] < 0.0
    gate = {
        "safety_pass": bool(safety_pass),
        "benefit_pass": bool(benefit_pass),
        "selected_method": selected_method,
        "selection_reason": selection_reason,
        "brier_gain": -brier_delta,
        "brier_gain_ci95": [-brier_delta_ci[1], -brier_delta_ci[0]],
        "auc_delta": auc_delta,
        "auc_delta_ci95": list(auc_delta_ci),
        "interpretation": (
            "external safety gate failed; adaptation benefit is not demonstrated"
            if not safety_pass
            else (
                "adaptation benefit demonstrated"
                if benefit_pass
                else "safety passed, but adaptation benefit is not demonstrated"
            )
        ),
    }
    utility = decision_utility(evaluation, probabilities[selected_method])
    metrics.to_csv(args.output_root / "method_performance.csv", index=False)
    intervals.to_csv(args.output_root / "paired_bootstrap_differences.csv", index=False)
    evaluation.to_csv(args.output_root / "seediv_policy_predictions.csv", index=False)
    utility.to_csv(args.output_root / "audit_budget_utility.csv", index=False)
    pd.DataFrame(
        [
            {key: value for key, value in scmt_diagnostics.items() if not isinstance(value, dict)}
            | {f"check_{key}": value for key, value in scmt_diagnostics["checks"].items()}
        ]
    ).to_csv(args.output_root / "scmt_candidate_diagnostics.csv", index=False)
    (args.output_root / "external_validation_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "untouched_external_validation_complete",
        "dataset": "SEED-IV",
        "n_configurations": len(evaluation),
        "selected_method": selected_method,
        "selection_reason": selection_reason,
        "decision_lock_sha256": lock_hash,
        "registration_sha256": sha256(args.registration),
        "label_blinding_contract": (
            "The domain action and its immutable lock were written using only public labels, "
            "public features, and unlabeled SEED-IV mechanism features before SEED-IV "
            "exposure-effect outcomes were loaded."
        ),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_unit": "task x representation x model x split_seed configuration",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scmt_seed": SCMT_SEED,
    }
    (args.output_root / "external_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(decision[["selected_method", "selection_reason", "estimated_label_shift"]].to_string(index=False))
    print(metrics.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
