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
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identity_shortcut.domain_transport import (  # noqa: E402
    SCMTConfig,
    coral,
    fit_scmt,
    mean_shift,
    quantile_map,
    support_clip,
)
from identity_shortcut.transport_policy import (  # noqa: E402
    POLICY_FEATURES,
    PolicyConfig,
    choose_alpha,
    fit_family_committee,
    select_actions,
    unlabeled_signature,
)


SEEDS = [17, 29, 43, 71, 101]
SHIFT_FAMILIES = [
    "mean",
    "variance",
    "covariance",
    "tails",
    "mixed",
    "label_shift",
    "conditional_shift",
    "support_failure",
]
SHIFT_DOSES = [0.5, 1.0, 1.5]
CANDIDATE_METHODS = [
    "identity",
    "mean_shift",
    "coral",
    "quantile_mapping",
    "support_clipping",
    "scmt",
]
CLUSTER_COLUMNS = ["dataset", "task", "representation", "model", "split_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop an unlabeled adapter-selection and abstention policy on public data."
    )
    parser.add_argument(
        "--risk-root", type=Path, default=Path("outputs/identity_shortcut_risk_classifier")
    )
    parser.add_argument(
        "--scmt-root", type=Path, default=Path("outputs/scmt_public_development")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/transport_policy_public_development")
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def random_rotation(dimensions: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimensions, dimensions))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))[None, :]
    return q


def synthetic_target(
    values: np.ndarray,
    labels: np.ndarray,
    family: str,
    dose: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dimensions = values.shape[1]
    direction = rng.normal(size=dimensions)
    direction /= np.linalg.norm(direction) + 1e-8
    selected = np.arange(len(values))
    shifted = values.copy()
    if family == "label_shift":
        sign = 1 if seed % 2 else -1
        logits = (labels * 2 - 1) * sign * dose
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        selected = rng.choice(len(values), size=len(values), replace=True, p=weights)
        shifted = values[selected].copy()
    elif family == "mean":
        shifted = values + dose * direction
    elif family == "variance":
        scales = np.exp(dose * np.linspace(-0.35, 0.35, dimensions))
        shifted = values * scales
    elif family == "covariance":
        rotation = random_rotation(dimensions, rng)
        matrix = (1.0 - 0.22 * dose) * np.eye(dimensions) + 0.22 * dose * rotation
        shifted = values @ matrix
    elif family == "tails":
        projection = values @ direction
        mask = projection >= np.quantile(projection, 0.80)
        shifted[mask] += dose * (0.5 + np.abs(projection[mask, None])) * direction
    elif family == "mixed":
        scales = np.exp(dose * np.linspace(-0.35, 0.35, dimensions))
        rotation = random_rotation(dimensions, rng)
        matrix = (1.0 - 0.22 * dose) * np.eye(dimensions) + 0.22 * dose * rotation
        shifted = values @ matrix * scales + 0.65 * dose * direction
        projection = shifted @ direction
        mask = projection >= np.quantile(projection, 0.85)
        shifted[mask] += 0.35 * dose * direction
    elif family == "conditional_shift":
        sign = np.where(labels[:, None] == 1, 1.0, -1.0)
        shifted = values + sign * 0.55 * dose * direction
    elif family == "support_failure":
        projection = values @ direction
        mask = projection >= np.quantile(projection, max(0.55, 0.85 - 0.10 * dose))
        shifted[mask] += (1.5 + dose) * direction
    else:
        raise ValueError(f"Unknown shift family: {family}")
    return shifted.astype(np.float32), labels[selected], selected


def load_contract(args: argparse.Namespace):
    risk_manifest_path = args.risk_root / "freeze_manifest.json"
    risk_model_path = args.risk_root / "frozen_risk_model.joblib"
    risk_table_path = args.risk_root / "risk_learning_table.csv"
    scmt_manifest_path = args.scmt_root / "scmt_freeze_manifest.json"
    for path in (risk_manifest_path, risk_model_path, risk_table_path, scmt_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    risk_manifest = json.loads(risk_manifest_path.read_text(encoding="utf-8"))
    scmt_manifest = json.loads(scmt_manifest_path.read_text(encoding="utf-8"))
    if not risk_manifest.get("admissible") or not scmt_manifest.get("admissible"):
        raise ValueError("Both frozen public risk and SCMT manifests must be admissible")
    if sha256(risk_model_path) != risk_manifest["model_sha256"]:
        raise ValueError("Frozen risk model hash mismatch")
    if scmt_manifest["source_risk_model_sha256"] != risk_manifest["model_sha256"]:
        raise ValueError("Frozen SCMT and risk model disagree")
    transport_code = ROOT / "src" / "identity_shortcut" / "domain_transport.py"
    if sha256(transport_code) != scmt_manifest["domain_transport_code_sha256"]:
        raise ValueError("Frozen SCMT implementation has changed")
    table = pd.read_csv(risk_table_path)
    features = risk_manifest["numeric_features"]
    risk_model = joblib.load(risk_model_path)
    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    standardized = scaler.transform(table[features]).astype(np.float32)
    config_values = dict(scmt_manifest["selected_config"])
    if args.epochs is not None:
        config_values["epochs"] = args.epochs
    if args.smoke:
        config_values["epochs"] = min(int(config_values["epochs"]), 12)
    return (
        table,
        standardized,
        features,
        logistic,
        risk_manifest,
        risk_model_path,
        scmt_manifest,
        scmt_manifest_path,
        SCMTConfig(**config_values),
    )


def split_clusters(table: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    clusters = table[CLUSTER_COLUMNS].astype(str).agg("|".join, axis=1)
    unique = clusters.unique()
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(unique, size=max(1, len(unique) // 5), replace=False))
    target = clusters.isin(held_out).to_numpy()
    return ~target, target


def probability(logistic, values: np.ndarray) -> np.ndarray:
    return logistic.predict_proba(np.asarray(values))[:, 1]


def candidate_transforms(
    source: np.ndarray,
    target: np.ndarray,
    logistic,
    scmt_config: SCMTConfig,
    seed: int,
    device: str,
    progress: tqdm,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    transformed_scmt, diagnostics = fit_scmt(
        source,
        target,
        lambda x: probability(logistic, x),
        scmt_config,
        seed,
        device,
        lambda epoch, loss: progress.set_postfix(
            loss=f"{loss['total']:.3f}", refresh=False
        ) if epoch % 40 == 0 else None,
    )
    return {
        "identity": target.copy(),
        "mean_shift": mean_shift(source, target),
        "coral": coral(source, target),
        "quantile_mapping": quantile_map(source, target),
        "support_clipping": support_clip(source, target),
        "scmt": transformed_scmt,
    }, diagnostics


def evaluate_scenarios(
    table: pd.DataFrame,
    standardized: np.ndarray,
    logistic,
    threshold: float,
    scmt_config: SCMTConfig,
    device: str,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = SEEDS[:2] if smoke else SEEDS
    families = ["mean", "label_shift", "support_failure"] if smoke else SHIFT_FAMILIES
    doses = SHIFT_DOSES[:1] if smoke else SHIFT_DOSES
    rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    total = len(seeds) * len(families) * len(doses)
    progress = tqdm(total=total, desc="Public transport-policy scenarios", unit="scenario", dynamic_ncols=True)
    for seed in seeds:
        source_mask, target_mask = split_clusters(table, seed)
        source = standardized[source_mask]
        source_labels = table.loc[source_mask, "material_optimism_event"].to_numpy(int)
        original = standardized[target_mask]
        original_labels = table.loc[target_mask, "material_optimism_event"].to_numpy(int)
        for family_index, family in enumerate(families):
            for dose in doses:
                scenario_seed = seed + 1009 * (family_index + 1) + int(100 * dose)
                shifted, labels, selected = synthetic_target(
                    original, original_labels, family, dose, scenario_seed
                )
                reference_probability = probability(logistic, original[selected])
                transforms, scmt_diagnostics = candidate_transforms(
                    source,
                    shifted,
                    logistic,
                    scmt_config,
                    scenario_seed + 7919,
                    device,
                    progress,
                )
                scenario_id = f"s{seed}_{family}_d{dose:.1f}"
                identity_probability = probability(logistic, shifted)
                identity_brier = float(brier_score_loss(labels, identity_probability))
                identity_auc = float(roc_auc_score(labels, identity_probability))
                identity_ba = float(
                    balanced_accuracy_score(labels, identity_probability >= threshold)
                )
                for method, transformed in transforms.items():
                    method_probability = probability(logistic, transformed)
                    brier = float(brier_score_loss(labels, method_probability))
                    auc = float(roc_auc_score(labels, method_probability))
                    balanced_accuracy = float(
                        balanced_accuracy_score(labels, method_probability >= threshold)
                    )
                    rows.append({
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "shift_family": family,
                        "shift_dose": dose,
                        "method": method,
                        "n_target": len(labels),
                        "target_prevalence": float(np.mean(labels)),
                        "brier_score": brier,
                        "roc_auc": auc,
                        "balanced_accuracy": balanced_accuracy,
                        "prediction_recovery_mae": float(
                            np.mean(np.abs(method_probability - reference_probability))
                        ),
                        "brier_gain": identity_brier - brier,
                        "auc_delta": auc - identity_auc,
                        "balanced_accuracy_delta": balanced_accuracy - identity_ba,
                        **unlabeled_signature(
                            source,
                            source_labels,
                            shifted,
                            transformed,
                            lambda x: probability(logistic, x),
                            threshold,
                            scenario_seed,
                        ),
                    })
                diagnostics_rows.append({
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "shift_family": family,
                    "shift_dose": dose,
                    **{
                        key: value
                        for key, value in scmt_diagnostics.items()
                        if not isinstance(value, dict)
                    },
                    **{
                        f"check_{key}": value
                        for key, value in scmt_diagnostics["checks"].items()
                    },
                })
                progress.update(1)
    progress.close()
    return pd.DataFrame(rows), pd.DataFrame(diagnostics_rows)


def best_fixed_method(training: pd.DataFrame, margin: float) -> str:
    summary = training.groupby("method").agg(
        mean_brier_gain=("brier_gain", "mean"),
        mean_auc_delta=("auc_delta", "mean"),
    )
    safe = summary.loc[summary.mean_auc_delta >= -margin]
    return str(safe.mean_brier_gain.idxmax()) if not safe.empty else "identity"


def evaluate_losfo(
    results: pd.DataFrame,
    config: PolicyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = results.loc[results.method != "identity"].copy()
    decisions = []
    tuning_rows = []
    for held_family in sorted(results.shift_family.unique()):
        training = candidates.loc[candidates.shift_family != held_family]
        held_out = results.loc[results.shift_family == held_family]
        gain_alpha, gain_scores = choose_alpha(training, "brier_gain", config)
        auc_alpha, auc_scores = choose_alpha(training, "auc_delta", config)
        gain_scores["target"] = "brier_gain"
        auc_scores["target"] = "auc_delta"
        gain_scores["outer_held_family"] = held_family
        auc_scores["outer_held_family"] = held_family
        tuning_rows.extend([gain_scores, auc_scores])
        selected = select_actions(
            held_out,
            fit_family_committee(training, "brier_gain", gain_alpha),
            fit_family_committee(training, "auc_delta", auc_alpha),
            config,
        )
        fixed = best_fixed_method(training, config.auc_noninferiority_margin)
        for _, row in selected.iterrows():
            scenario = results.loc[results.scenario_id == row.scenario_id]
            identity = scenario.loc[scenario.method == "identity"].iloc[0]
            oracle_pool = scenario.loc[
                scenario.auc_delta >= -config.auc_noninferiority_margin
            ]
            oracle = oracle_pool.loc[oracle_pool.brier_score.idxmin()]
            fixed_row = scenario.loc[scenario.method == fixed].iloc[0]
            item = row.to_dict()
            item.update({
                "outer_held_family": held_family,
                "fixed_method": fixed,
                "fixed_brier_gain": float(fixed_row.brier_gain),
                "fixed_auc_delta": float(fixed_row.auc_delta),
                "oracle_method": str(oracle.method),
                "oracle_brier_gain": float(oracle.brier_gain),
                "oracle_auc_delta": float(oracle.auc_delta),
                "policy_regret": float(oracle.brier_score - row.brier_score) * -1.0,
                "identity_brier": float(identity.brier_score),
            })
            decisions.append(item)
    decision_table = pd.DataFrame(decisions)
    family_summary = decision_table.groupby("outer_held_family").agg(
        n_scenarios=("scenario_id", "size"),
        adaptation_rate=("selected_method", lambda x: float(np.mean(x != "identity"))),
        mean_brier_gain=("brier_gain", "mean"),
        median_brier_gain=("brier_gain", "median"),
        negative_transfer_rate=("brier_gain", lambda x: float(np.mean(x < 0))),
        auc_noninferiority_rate=(
            "auc_delta", lambda x: float(np.mean(x >= -config.auc_noninferiority_margin))
        ),
        mean_fixed_brier_gain=("fixed_brier_gain", "mean"),
        mean_oracle_brier_gain=("oracle_brier_gain", "mean"),
        mean_policy_regret=("policy_regret", "mean"),
    ).reset_index()
    overall = pd.DataFrame([{
        "outer_held_family": "OVERALL",
        "n_scenarios": len(decision_table),
        "adaptation_rate": float(np.mean(decision_table.selected_method != "identity")),
        "mean_brier_gain": float(decision_table.brier_gain.mean()),
        "median_brier_gain": float(decision_table.brier_gain.median()),
        "negative_transfer_rate": float(np.mean(decision_table.brier_gain < 0)),
        "auc_noninferiority_rate": float(
            np.mean(decision_table.auc_delta >= -config.auc_noninferiority_margin)
        ),
        "mean_fixed_brier_gain": float(decision_table.fixed_brier_gain.mean()),
        "mean_oracle_brier_gain": float(decision_table.oracle_brier_gain.mean()),
        "mean_policy_regret": float(decision_table.policy_regret.mean()),
    }])
    return decision_table, pd.concat([family_summary, overall], ignore_index=True), pd.concat(tuning_rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for a software smoke test.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (
        table,
        standardized,
        features,
        logistic,
        risk_manifest,
        risk_model_path,
        scmt_manifest,
        scmt_manifest_path,
        scmt_config,
    ) = load_contract(args)
    results, scmt_diagnostics = evaluate_scenarios(
        table,
        standardized,
        logistic,
        float(risk_manifest["decision_threshold"]),
        scmt_config,
        device,
        args.smoke,
    )
    policy_config = PolicyConfig()
    decisions, summary, tuning = evaluate_losfo(results, policy_config)
    candidates = results.loc[results.method != "identity"].copy()
    gain_alpha, gain_scores = choose_alpha(candidates, "brier_gain", policy_config)
    auc_alpha, auc_scores = choose_alpha(candidates, "auc_delta", policy_config)
    policy_artifact = {
        "gain_committee": fit_family_committee(candidates, "brier_gain", gain_alpha),
        "auc_committee": fit_family_committee(candidates, "auc_delta", auc_alpha),
        "gain_alpha": gain_alpha,
        "auc_alpha": auc_alpha,
        "policy_features": list(POLICY_FEATURES),
        "candidate_methods": CANDIDATE_METHODS,
        "policy_config": policy_config.to_dict(),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_root / "synthetic_candidate_results.csv", index=False)
    scmt_diagnostics.to_csv(args.output_root / "synthetic_scmt_diagnostics.csv", index=False)
    decisions.to_csv(args.output_root / "losfo_policy_decisions.csv", index=False)
    summary.to_csv(args.output_root / "losfo_policy_summary.csv", index=False)
    tuning.to_csv(args.output_root / "nested_tuning_scores.csv", index=False)
    pd.concat([
        gain_scores.assign(target="brier_gain"),
        auc_scores.assign(target="auc_delta"),
    ]).to_csv(args.output_root / "final_tuning_scores.csv", index=False)
    artifact_name = "smoke_policy.joblib" if args.smoke else "frozen_transport_policy.joblib"
    artifact_path = args.output_root / artifact_name
    joblib.dump(policy_artifact, artifact_path)
    overall = summary.loc[summary.outer_held_family == "OVERALL"].iloc[0]
    family_only = summary.loc[summary.outer_held_family != "OVERALL"]
    criteria = [
        {
            "criterion": "losfo_positive_brier_gain",
            "passed": bool(overall.mean_brier_gain > 0),
            "observed": float(overall.mean_brier_gain),
            "required": "> 0",
        },
        {
            "criterion": "losfo_auc_noninferiority",
            "passed": bool(overall.auc_noninferiority_rate >= 0.90),
            "observed": float(overall.auc_noninferiority_rate),
            "required": ">= 0.90 at margin -0.02",
        },
        {
            "criterion": "losfo_beats_fixed_adapter",
            "passed": bool(overall.mean_brier_gain > overall.mean_fixed_brier_gain),
            "observed": float(overall.mean_brier_gain - overall.mean_fixed_brier_gain),
            "required": "> 0 mean Brier-gain advantage",
        },
        {
            "criterion": "bounded_negative_transfer",
            "passed": bool(overall.negative_transfer_rate <= 0.20),
            "observed": float(overall.negative_transfer_rate),
            "required": "<= 0.20",
        },
        {
            "criterion": "worst_family_brier_tolerance",
            "passed": bool(family_only.mean_brier_gain.min() >= -0.0001),
            "observed": float(family_only.mean_brier_gain.min()),
            "required": ">= -0.0001 mean Brier gain in every held-out shift family",
        },
        {
            "criterion": "worst_family_auc_noninferiority",
            "passed": bool(family_only.auc_noninferiority_rate.min() >= 0.90),
            "observed": float(family_only.auc_noninferiority_rate.min()),
            "required": ">= 0.90 in every held-out shift family",
        },
    ]
    admitted = all(item["passed"] for item in criteria) and not args.smoke
    manifest = {
        "status": "admissible_policy_freeze" if admitted else "development_not_admissible",
        "admissible": admitted,
        "scope": "public synthetic shift development on DEAP and MAHNOB-HCI only",
        "scientific_role": (
            "Select one unlabeled target-domain adapter or abstain before target outcomes are read."
        ),
        "claim_boundary": (
            "This is synthetic-shift development evidence. EPPVR is retrospective because its "
            "outcomes informed earlier SCMT development; an untouched fourth dataset is required."
        ),
        "features": features,
        "candidate_methods": CANDIDATE_METHODS,
        "shift_families": sorted(results.shift_family.unique()),
        "shift_doses": sorted(float(value) for value in results.shift_dose.unique()),
        "seeds": sorted(int(value) for value in results.seed.unique()),
        "validation_design": "nested leave-shift-family-out",
        "primary_endpoint": "Brier gain versus identity input",
        "safety_endpoint": "AUROC non-inferiority margin -0.02",
        "abstention_definition": "select identity when no adapter has a positive safe lower bound",
        "label_shift_guard": (
            "Abstain when the soft-confusion prior estimate differs from source prevalence by "
            f"more than {policy_config.max_estimated_label_shift:.2f}."
        ),
        "local_support_ambiguity_guard": (
            "Abstain when raw support violation exceeds 0.50 while mean distance is below 0.80 "
            "and covariance distance is below 1.80."
        ),
        "development_revision": (
            "Policy v2 added the label-shift ambiguity guard after the first public-only LOSFO run "
            "failed on held-out label shift. Policy v3 added the local-support ambiguity guard "
            "after the second public-only run failed on held-out conditional shift. No EPPVR "
            "results were used for either revision."
        ),
        "policy_config": policy_config.to_dict(),
        "gain_alpha": gain_alpha,
        "auc_alpha": auc_alpha,
        "criteria": criteria,
        "source_risk_model_sha256": sha256(risk_model_path),
        "source_scmt_manifest_sha256": sha256(scmt_manifest_path),
        "policy_artifact": artifact_name,
        "policy_artifact_sha256": sha256(artifact_path),
        "development_script_sha256": sha256(Path(__file__)),
        "policy_code_sha256": sha256(ROOT / "src" / "identity_shortcut" / "transport_policy.py"),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "scmt_epochs": scmt_config.epochs,
    }
    manifest_name = "smoke_manifest.json" if args.smoke else "transport_policy_freeze_manifest.json"
    (args.output_root / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
