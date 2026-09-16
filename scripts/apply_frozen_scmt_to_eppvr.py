from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy import stats
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from identity_shortcut.domain_transport import (  # noqa: E402
    SCMTConfig,
    coral,
    mean_shift,
    quantile_map,
    sliced_wasserstein,
    support_clip,
    support_violation,
)
from identity_shortcut.domain_transport import fit_scmt  # noqa: E402


CLUSTER_COLUMNS = ["task", "representation", "model", "split_seed"]
TARGET_FIT_SEEDS = [17, 29, 43, 71, 101]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrospectively apply the public-frozen SCMT algorithm to EPPVR."
    )
    parser.add_argument(
        "--risk-root", type=Path, default=Path("outputs/identity_shortcut_risk_classifier")
    )
    parser.add_argument(
        "--scmt-root", type=Path, default=Path("outputs/scmt_public_development")
    )
    parser.add_argument(
        "--external-root", type=Path, default=Path("outputs/eppvr_frozen_risk_validation")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/eppvr_scmt_transport")
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260905)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    predictor = logit(np.clip(probability, 1e-6, 1 - 1e-6))
    design = sm.add_constant(predictor)
    try:
        fitted = sm.GLM(target, design, family=sm.families.Binomial()).fit()
        return float(fitted.params[0]), float(fitted.params[1])
    except Exception:
        return float("nan"), float("nan")


def performance(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    null_probability: float,
) -> dict[str, float]:
    brier = float(brier_score_loss(target, probability))
    null_brier = float(brier_score_loss(target, np.full(len(target), null_probability)))
    intercept, slope = calibration(target, probability)
    prevalence = float(np.mean(target))
    ap = float(average_precision_score(target, probability))
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": ap,
        "average_precision_lift": ap / prevalence,
        "brier_score": brier,
        "brier_skill_public_null": 1.0 - brier / null_brier,
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, probability >= threshold)
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_probability": float(np.mean(probability)),
    }


def domain_classifier_auc(source: np.ndarray, target: np.ndarray, seed: int) -> float:
    values = np.vstack([source, target])
    labels = np.concatenate([np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)])
    model = LogisticRegression(C=0.1, max_iter=3000)
    probability = cross_val_predict(
        model,
        values,
        labels,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    auc = float(roc_auc_score(labels, probability))
    return max(auc, 1.0 - auc)


def rbf_mmd(source: np.ndarray, target: np.ndarray) -> float:
    pooled = np.vstack([source, target])
    squared = np.sum((pooled[:, None] - pooled[None, :]) ** 2, axis=2)
    positive = squared[squared > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / max(2.0 * bandwidth, 1e-8)
    kxx = np.exp(-gamma * np.sum((source[:, None] - source[None, :]) ** 2, axis=2))
    kyy = np.exp(-gamma * np.sum((target[:, None] - target[None, :]) ** 2, axis=2))
    kxy = np.exp(-gamma * np.sum((source[:, None] - target[None, :]) ** 2, axis=2))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def domain_metrics(source: np.ndarray, target: np.ndarray, seed: int) -> dict[str, float]:
    feature_wasserstein = [
        stats.wasserstein_distance(source[:, index], target[:, index])
        for index in range(source.shape[1])
    ]
    return {
        "domain_classifier_auc": domain_classifier_auc(source, target, seed),
        "rbf_mmd": rbf_mmd(source, target),
        "mean_feature_wasserstein": float(np.mean(feature_wasserstein)),
        "sliced_wasserstein": sliced_wasserstein(source, target, seed=seed),
        "support_violation": support_violation(source, target),
    }


def cluster_indices(table: pd.DataFrame) -> list[np.ndarray]:
    labels = table[CLUSTER_COLUMNS].astype(str).agg("|".join, axis=1).to_numpy()
    return [np.flatnonzero(labels == value) for value in np.unique(labels)]


def paired_bootstrap(
    table: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    threshold: float,
    null_probability: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    clusters = cluster_indices(table)
    target = table.material_optimism_event.to_numpy(int)
    rng = np.random.default_rng(seed)
    metric_names = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "brier_skill_public_null",
        "log_loss",
        "balanced_accuracy",
        "calibration_intercept",
        "calibration_slope",
    ]
    methods = [method for method in probabilities if method != "identity"]
    draws = {(method, metric): [] for method in methods for metric in metric_names}
    for _ in tqdm(
        range(repetitions),
        desc="EPPVR paired cluster bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        chosen = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([clusters[index] for index in chosen])
        y = target[indices]
        if len(np.unique(y)) < 2:
            continue
        baseline = performance(y, probabilities["identity"][indices], threshold, null_probability)
        for method in methods:
            adapted = performance(y, probabilities[method][indices], threshold, null_probability)
            for metric in metric_names:
                draws[(method, metric)].append(adapted[metric] - baseline[metric])
    rows = []
    for (method, metric), values in draws.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        rows.append({
            "method": method,
            "metric": metric,
            "delta_definition": "adapted_minus_identity",
            "bootstrap_valid_repetitions": len(finite),
            "delta_median": float(np.median(finite)) if len(finite) else float("nan"),
            "delta_ci_low": float(np.quantile(finite, 0.025)) if len(finite) else float("nan"),
            "delta_ci_high": float(np.quantile(finite, 0.975)) if len(finite) else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for debugging.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    risk_manifest_path = args.risk_root / "freeze_manifest.json"
    risk_model_path = args.risk_root / "frozen_risk_model.joblib"
    scmt_manifest_path = args.scmt_root / "scmt_freeze_manifest.json"
    predictions_path = args.external_root / "eppvr_risk_predictions.csv"
    locked_gate_path = args.external_root / "external_validation_gate.json"
    for path in (
        risk_manifest_path,
        risk_model_path,
        scmt_manifest_path,
        predictions_path,
        locked_gate_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    risk_manifest = json.loads(risk_manifest_path.read_text(encoding="utf-8"))
    scmt_manifest = json.loads(scmt_manifest_path.read_text(encoding="utf-8"))
    if not scmt_manifest.get("admissible"):
        raise ValueError("SCMT public-development algorithm was not admitted")
    if sha256(risk_model_path) != risk_manifest["model_sha256"]:
        raise ValueError("Frozen risk-model hash mismatch")
    if scmt_manifest["source_risk_model_sha256"] != risk_manifest["model_sha256"]:
        raise ValueError("SCMT and frozen risk model manifests disagree")
    transport_code = ROOT / "src" / "identity_shortcut" / "domain_transport.py"
    if sha256(transport_code) != scmt_manifest["domain_transport_code_sha256"]:
        raise ValueError("SCMT implementation changed after the public-only freeze")
    features = scmt_manifest["features"]
    public = pd.read_csv(args.risk_root / "risk_learning_table.csv", usecols=features)
    # The first target read is deliberately feature-only: labels are unavailable to fitting.
    target_features = pd.read_csv(predictions_path, usecols=features)
    risk_model = joblib.load(risk_model_path)
    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    source = scaler.transform(public[features]).astype(np.float32)
    target = scaler.transform(target_features[features]).astype(np.float32)
    risk_probability = lambda values: logistic.predict_proba(np.asarray(values))[:, 1]
    config = SCMTConfig(**scmt_manifest["selected_config"])
    transformed_seeds = []
    scmt_diagnostics = []
    progress = tqdm(total=len(TARGET_FIT_SEEDS), desc="EPPVR SCMT fits", unit="seed", dynamic_ncols=True)
    for seed in TARGET_FIT_SEEDS:
        transformed, diagnostics = fit_scmt(
            source,
            target,
            risk_probability,
            config,
            seed,
            device,
            lambda epoch, loss: progress.set_postfix(
                seed=seed, loss=f"{loss['total']:.3f}", refresh=False
            ) if epoch % 40 == 0 else None,
        )
        transformed_seeds.append(transformed)
        scmt_diagnostics.append({
            "seed": seed,
            **{key: value for key, value in diagnostics.items() if not isinstance(value, dict)},
            **{f"check_{key}": value for key, value in diagnostics["checks"].items()},
        })
        progress.update(1)
    progress.close()
    transforms = {
        "identity": target,
        "mean_shift": mean_shift(source, target),
        "coral": coral(source, target),
        "quantile_mapping": quantile_map(source, target),
        "support_clipping": support_clip(source, target),
        "scmt_ensemble": np.mean(transformed_seeds, axis=0),
    }
    probabilities = {method: risk_probability(values) for method, values in transforms.items()}
    # Labels and configuration metadata enter only after all unsupervised transformations are fixed.
    evaluation = pd.read_csv(predictions_path)
    target_label = evaluation.material_optimism_event.to_numpy(int)
    threshold = float(risk_manifest["decision_threshold"])
    null_probability = float(risk_manifest["public_training_event_prevalence"])
    metric_rows = []
    domain_rows = []
    for method, probability in probabilities.items():
        metric_rows.append({"method": method, **performance(
            target_label, probability, threshold, null_probability
        )})
        domain_rows.append({"method": method, **domain_metrics(
            source, transforms[method], args.bootstrap_seed
        )})
        evaluation[f"probability_{method}"] = probability
    metrics = pd.DataFrame(metric_rows)
    domains = pd.DataFrame(domain_rows)
    intervals = paired_bootstrap(
        evaluation,
        probabilities,
        threshold,
        null_probability,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_root / "method_performance.csv", index=False)
    domains.to_csv(args.output_root / "domain_discrepancy.csv", index=False)
    intervals.to_csv(args.output_root / "paired_bootstrap_differences.csv", index=False)
    evaluation.to_csv(args.output_root / "eppvr_transport_predictions.csv", index=False)
    pd.DataFrame(scmt_diagnostics).to_csv(
        args.output_root / "scmt_seed_diagnostics.csv", index=False
    )
    locked_gate_hash_before = sha256(locked_gate_path)
    report = {
        "status": "post_external_retrospective_transport_evaluation",
        "claim_boundary": (
            "EPPVR outcomes were inspected before SCMT development. These results cannot replace "
            "the locked zero-shot external failure and are not independent external validation."
        ),
        "label_blinding_contract": (
            "SCMT and baseline transformations were fitted from public features and EPPVR feature "
            "columns only; outcome labels were loaded after transformed features were fixed."
        ),
        "source_risk_model_sha256": sha256(risk_model_path),
        "scmt_manifest_sha256": sha256(scmt_manifest_path),
        "locked_external_gate_sha256_before": locked_gate_hash_before,
        "locked_external_gate_sha256_after": sha256(locked_gate_path),
        "locked_external_gate_unchanged": locked_gate_hash_before == sha256(locked_gate_path),
        "target_fit_seeds": TARGET_FIT_SEEDS,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_unit": "task x representation x model x split_seed configuration",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "scmt_gate_application_rate": float(
            np.mean([row["applied"] for row in scmt_diagnostics])
        ),
    }
    (args.output_root / "retrospective_evaluation_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(domains.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

