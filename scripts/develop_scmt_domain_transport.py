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
    discrepancy,
    fit_scmt,
    mean_shift,
    quantile_map,
    support_clip,
    support_violation,
)


CLUSTER_COLUMNS = ["dataset", "task", "representation", "model", "split_seed"]
SEEDS = [17, 29, 43, 71, 101]
SHIFT_FAMILIES = ["mean", "variance", "covariance", "tails", "mixed"]
SHIFT_DOSES = [0.5, 1.0, 1.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop and freeze SCMT using public datasets only."
    )
    parser.add_argument(
        "--risk-root",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/scmt_public_development"),
    )
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(args: argparse.Namespace):
    table_path = args.risk_root / "risk_learning_table.csv"
    model_path = args.risk_root / "frozen_risk_model.joblib"
    manifest_path = args.risk_root / "freeze_manifest.json"
    for path in (table_path, model_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "admissible_freeze" or not manifest.get("admissible"):
        raise ValueError("SCMT development requires an admissible frozen public risk model")
    if sha256(model_path) != manifest["model_sha256"]:
        raise ValueError("Frozen risk-model hash mismatch")
    if set(manifest["training_datasets"]) != {"DEAP", "MAHNOB-HCI"}:
        raise ValueError("SCMT public development is locked to DEAP and MAHNOB-HCI")
    table = pd.read_csv(table_path)
    if set(table.dataset) != {"DEAP", "MAHNOB-HCI"}:
        raise ValueError("Risk table contains an inadmissible dataset")
    features = manifest["numeric_features"]
    if manifest.get("categorical_features"):
        raise ValueError("SCMT currently supports the frozen numeric mechanism vector only")
    model = joblib.load(model_path)
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = model.named_steps["model"]
    standardized = scaler.transform(table[features]).astype(np.float32)
    return table, standardized, features, model, scaler, logistic, manifest, model_path


def random_rotation(dimensions: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimensions, dimensions))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))[None, :]
    return q


def synthetic_shift(
    values: np.ndarray,
    family: str,
    dose: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dimensions = values.shape[1]
    direction = rng.normal(size=dimensions)
    direction /= np.linalg.norm(direction)
    scales = np.exp(dose * np.linspace(-0.35, 0.35, dimensions))
    rotation = random_rotation(dimensions, rng)
    rotated = values @ ((1.0 - 0.22 * dose) * np.eye(dimensions) + 0.22 * dose * rotation)
    shifted = values.copy()
    if family == "mean":
        shifted = values + dose * direction
    elif family == "variance":
        shifted = values * scales
    elif family == "covariance":
        shifted = rotated
    elif family == "tails":
        projection = values @ direction
        mask = projection >= np.quantile(projection, 0.80)
        shifted[mask] += dose * (0.5 + np.abs(projection[mask, None])) * direction
    elif family == "mixed":
        shifted = rotated * scales + 0.65 * dose * direction
        projection = shifted @ direction
        mask = projection >= np.quantile(projection, 0.85)
        shifted[mask] += 0.35 * dose * direction
    else:
        raise ValueError(f"Unknown shift family: {family}")
    return shifted.astype(np.float32)


def split_configuration_clusters(table: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cluster = table[CLUSTER_COLUMNS].astype(str).agg("|".join, axis=1)
    unique = cluster.unique()
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(unique, size=max(1, len(unique) // 5), replace=False))
    target = cluster.isin(held_out).to_numpy()
    return ~target, target


def risk_probability(logistic, standardized: np.ndarray) -> np.ndarray:
    return logistic.predict_proba(np.asarray(standardized))[:, 1]


def evaluate(
    target: np.ndarray,
    probability: np.ndarray,
    original_probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    output = {
        "brier_score": float(brier_score_loss(target, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(target, probability >= threshold)),
        "prediction_recovery_mae": float(np.mean(np.abs(probability - original_probability))),
        "mean_probability": float(np.mean(probability)),
    }
    output["roc_auc"] = (
        float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else float("nan")
    )
    return output


def profile_configs(epochs: int) -> dict[str, SCMTConfig]:
    return {
        "conservative": SCMTConfig(
            epochs=epochs,
            lambda_geometry=0.35,
            lambda_identity=0.55,
            lambda_rank=0.45,
            lambda_displacement=0.15,
            min_relative_alignment_gain=0.06,
            min_prediction_rank=0.97,
            max_mean_displacement=0.90,
        ),
        "balanced": SCMTConfig(epochs=epochs),
        "support_strict": SCMTConfig(
            epochs=epochs,
            lambda_support=0.35,
            lambda_identity=0.45,
            gate_floor=0.05,
            min_relative_alignment_gain=0.10,
            min_prediction_rank=0.95,
        ),
    }


def baseline_transforms(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": target.copy(),
        "mean_shift": mean_shift(source, target),
        "coral": coral(source, target),
        "quantile_mapping": quantile_map(source, target),
        "support_clipping": support_clip(source, target),
    }


def synthetic_development(
    table: pd.DataFrame,
    standardized: np.ndarray,
    logistic,
    threshold: float,
    profiles: dict[str, SCMTConfig],
    device: str,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = SEEDS[:1] if smoke else SEEDS
    families = SHIFT_FAMILIES[:2] if smoke else SHIFT_FAMILIES
    doses = SHIFT_DOSES[:1] if smoke else SHIFT_DOSES
    rows = []
    diagnostic_rows = []
    total_fits = len(seeds) * len(families) * len(doses) * len(profiles)
    progress = tqdm(total=total_fits, desc="SCMT public synthetic fits", unit="fit", dynamic_ncols=True)
    for seed in seeds:
        source_mask, target_mask = split_configuration_clusters(table, seed)
        source = standardized[source_mask]
        original = standardized[target_mask]
        labels = table.loc[target_mask, "material_optimism_event"].to_numpy(int)
        original_probability = risk_probability(logistic, original)
        for family in families:
            for dose in doses:
                shifted = synthetic_shift(original, family, dose, seed + 1009 * (families.index(family) + 1))
                transforms = baseline_transforms(source, shifted)
                for method, transformed in transforms.items():
                    probability = risk_probability(logistic, transformed)
                    values = evaluate(labels, probability, original_probability, threshold)
                    before = discrepancy(source, shifted, seed)
                    after = discrepancy(source, transformed, seed)
                    rows.append({
                        "seed": seed,
                        "shift_family": family,
                        "shift_dose": dose,
                        "method": method,
                        "applied": True,
                        **values,
                        "swd_before": before["sliced_wasserstein"],
                        "swd_after": after["sliced_wasserstein"],
                        "support_violation": support_violation(source, transformed),
                    })
                for profile_index, (profile_name, config) in enumerate(profiles.items()):
                    callback = lambda epoch, loss: progress.set_postfix(
                        profile=profile_name, loss=f"{loss['total']:.3f}", refresh=False
                    ) if epoch % 40 == 0 else None
                    transformed, diagnostics = fit_scmt(
                        source,
                        shifted,
                        lambda x: risk_probability(logistic, x),
                        config,
                        seed + 7919 * profile_index,
                        device,
                        callback,
                    )
                    probability = risk_probability(logistic, transformed)
                    values = evaluate(labels, probability, original_probability, threshold)
                    rows.append({
                        "seed": seed,
                        "shift_family": family,
                        "shift_dose": dose,
                        "method": f"scmt_{profile_name}",
                        "applied": diagnostics["applied"],
                        **values,
                        "swd_before": diagnostics["raw_discrepancy"]["sliced_wasserstein"],
                        "swd_after": discrepancy(source, transformed, seed)["sliced_wasserstein"],
                        "support_violation": support_violation(source, transformed),
                    })
                    diagnostic_rows.append({
                        "seed": seed,
                        "shift_family": family,
                        "shift_dose": dose,
                        "profile": profile_name,
                        **{key: value for key, value in diagnostics.items() if not isinstance(value, dict)},
                        **{f"check_{key}": value for key, value in diagnostics["checks"].items()},
                    })
                    progress.update(1)
    progress.close()
    return pd.DataFrame(rows), pd.DataFrame(diagnostic_rows)


def summarize_synthetic(results: pd.DataFrame) -> pd.DataFrame:
    identity = results.loc[results.method == "identity", [
        "seed", "shift_family", "shift_dose", "brier_score", "prediction_recovery_mae"
    ]].rename(columns={
        "brier_score": "identity_brier",
        "prediction_recovery_mae": "identity_recovery_mae",
    })
    paired = results.merge(identity, on=["seed", "shift_family", "shift_dose"], validate="many_to_one")
    paired["brier_improvement_vs_shifted"] = paired.identity_brier - paired.brier_score
    paired["recovery_improvement_vs_shifted"] = (
        paired.identity_recovery_mae - paired.prediction_recovery_mae
    )
    return (
        paired.groupby("method")
        .agg(
            n_scenarios=("brier_score", "size"),
            application_rate=("applied", "mean"),
            median_brier=("brier_score", "median"),
            median_brier_improvement=("brier_improvement_vs_shifted", "median"),
            worst_brier_improvement=("brier_improvement_vs_shifted", "min"),
            median_recovery_mae=("prediction_recovery_mae", "median"),
            median_recovery_improvement=("recovery_improvement_vs_shifted", "median"),
            median_roc_auc=("roc_auc", "median"),
            median_balanced_accuracy=("balanced_accuracy", "median"),
            median_swd_after=("swd_after", "median"),
            median_support_violation=("support_violation", "median"),
        )
        .reset_index()
        .sort_values(["median_brier_improvement", "median_recovery_improvement"], ascending=False)
    )


def select_profile(summary: pd.DataFrame, strict: bool = True) -> str:
    candidates = summary.loc[summary.method.str.startswith("scmt_")].copy()
    candidates = candidates.loc[
        (candidates.application_rate >= 0.50)
        & (candidates.median_brier_improvement > 0)
        & (candidates.median_recovery_improvement > 0)
    ]
    if candidates.empty:
        if strict:
            raise RuntimeError("No SCMT profile passed the synthetic-development admission rule")
        candidates = summary.loc[summary.method.str.startswith("scmt_")].copy()
    score = (
        candidates.median_brier_improvement
        + candidates.median_recovery_improvement
        - 0.25 * candidates.median_support_violation
    )
    return str(candidates.loc[score.idxmax(), "method"]).removeprefix("scmt_")


def real_public_negative_control(
    table: pd.DataFrame,
    standardized: np.ndarray,
    logistic,
    threshold: float,
    profile_name: str,
    config: SCMTConfig,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diagnostic_rows = []
    progress = tqdm(total=2, desc="SCMT public negative controls", unit="direction", dynamic_ncols=True)
    for source_name, target_name in [("DEAP", "MAHNOB-HCI"), ("MAHNOB-HCI", "DEAP")]:
        source = standardized[table.dataset.to_numpy() == source_name]
        target_mask = table.dataset.to_numpy() == target_name
        target = standardized[target_mask]
        labels = table.loc[target_mask, "material_optimism_event"].to_numpy(int)
        original_probability = risk_probability(logistic, target)
        raw = evaluate(labels, original_probability, original_probability, threshold)
        rows.append({"source": source_name, "target": target_name, "method": "identity", **raw})
        transformed, diagnostics = fit_scmt(
            source,
            target,
            lambda x: risk_probability(logistic, x),
            config,
            3001 + len(rows),
            device,
        )
        probability = risk_probability(logistic, transformed)
        adapted = evaluate(labels, probability, original_probability, threshold)
        rows.append({
            "source": source_name,
            "target": target_name,
            "method": f"scmt_{profile_name}",
            **adapted,
        })
        diagnostic_rows.append({
            "source": source_name,
            "target": target_name,
            "profile": profile_name,
            **{key: value for key, value in diagnostics.items() if not isinstance(value, dict)},
            **{f"check_{key}": value for key, value in diagnostics["checks"].items()},
        })
        progress.update(1)
    progress.close()
    results = pd.DataFrame(rows)
    baseline = results.loc[results.method == "identity", ["target", "brier_score"]].rename(
        columns={"brier_score": "identity_brier"}
    )
    results = results.merge(baseline, on="target", validate="many_to_one")
    results["brier_change_vs_identity"] = results.brier_score - results.identity_brier
    return results, pd.DataFrame(diagnostic_rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for a software smoke test.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table, standardized, features, model, scaler, logistic, freeze, model_path = load_contract(args)
    profiles = profile_configs(20 if args.smoke else args.epochs)
    if args.smoke:
        profiles = {"balanced": profiles["balanced"]}
    synthetic, diagnostics = synthetic_development(
        table,
        standardized,
        logistic,
        float(freeze["decision_threshold"]),
        profiles,
        device,
        args.smoke,
    )
    summary = summarize_synthetic(synthetic)
    selected = select_profile(summary, strict=not args.smoke)
    negative, negative_diagnostics = real_public_negative_control(
        table,
        standardized,
        logistic,
        float(freeze["decision_threshold"]),
        selected,
        profiles[selected],
        device,
    )
    selected_summary = summary.loc[summary.method == f"scmt_{selected}"].iloc[0]
    adapted_negative = negative.loc[negative.method == f"scmt_{selected}"]
    criteria = [
        {
            "criterion": "synthetic_probability_recovery",
            "passed": bool(selected_summary.median_recovery_improvement > 0),
            "observed": float(selected_summary.median_recovery_improvement),
            "required": "> 0 median improvement",
        },
        {
            "criterion": "synthetic_brier_recovery",
            "passed": bool(selected_summary.median_brier_improvement > 0),
            "observed": float(selected_summary.median_brier_improvement),
            "required": "> 0 median improvement",
        },
        {
            "criterion": "synthetic_gate_coverage",
            "passed": bool(selected_summary.application_rate >= 0.50),
            "observed": float(selected_summary.application_rate),
            "required": ">= 0.50",
        },
        {
            "criterion": "real_public_negative_transfer",
            "passed": bool((adapted_negative.brier_change_vs_identity <= 0.005).all()),
            "observed": float(adapted_negative.brier_change_vs_identity.max()),
            "required": "Brier worsening <= 0.005 in each direction",
        },
    ]
    admitted = all(item["passed"] for item in criteria) and not args.smoke
    args.output_root.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(args.output_root / "synthetic_shift_results.csv", index=False)
    diagnostics.to_csv(args.output_root / "synthetic_scmt_diagnostics.csv", index=False)
    summary.to_csv(args.output_root / "synthetic_method_summary.csv", index=False)
    negative.to_csv(args.output_root / "public_negative_control_results.csv", index=False)
    negative_diagnostics.to_csv(
        args.output_root / "public_negative_control_diagnostics.csv", index=False
    )
    run_manifest = {
        "status": "admissible_algorithm_freeze" if admitted else "development_not_admissible",
        "admissible": admitted,
        "scope": "public-only SCMT development; EPPVR was not read",
        "development_version": "SCMT-v2 trust-region residual projection",
        "revision_disclosure": (
            "The trust-region projection was introduced after inspecting the first retrospective "
            "EPPVR SCMT application. Independent validation therefore requires an untouched dataset."
        ),
        "selected_profile": selected,
        "selected_config": profiles[selected].to_dict(),
        "features": features,
        "source_risk_model_sha256": sha256(model_path),
        "domain_transport_code_sha256": sha256(ROOT / "src" / "identity_shortcut" / "domain_transport.py"),
        "development_script_sha256": sha256(Path(__file__)),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seeds": SEEDS[:1] if args.smoke else SEEDS,
        "shift_families": SHIFT_FAMILIES[:2] if args.smoke else SHIFT_FAMILIES,
        "shift_doses": SHIFT_DOSES[:1] if args.smoke else SHIFT_DOSES,
        "selection_rule": (
            "Highest public synthetic recovery score among profiles with >=50% unsupervised "
            "gate coverage and positive median Brier and prediction-recovery improvements."
        ),
        "negative_control_rule": "Brier worsening must be <=0.005 in both real public directions.",
        "criteria": criteria,
        "integrity_note": (
            "This manifest freezes an algorithm configuration, not target-specific weights. "
            "No EPPVR features or outcomes are loaded by this script."
        ),
    }
    manifest_name = "smoke_manifest.json" if args.smoke else "scmt_freeze_manifest.json"
    (args.output_root / manifest_name).write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()
