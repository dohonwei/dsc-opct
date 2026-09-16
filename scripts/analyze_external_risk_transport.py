from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
import statsmodels.api as sm
from tqdm.auto import tqdm


CLUSTER_COLUMNS = ["task", "representation", "model", "split_seed"]
SUBGROUP_COLUMNS = ["task", "representation", "model", "nominal_dose"]
PRIMARY_METRICS = [
    "roc_auc",
    "average_precision",
    "average_precision_lift",
    "brier_score",
    "brier_skill_public_null",
    "balanced_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose ranking, calibration, and predictor-support transport of the "
            "locked public-data shortcut-risk model on EPPVR."
        )
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("outputs/eppvr_frozen_risk_validation"),
    )
    parser.add_argument(
        "--risk-root",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/eppvr_risk_transport_diagnostics"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    parser.add_argument("--calibration-bins", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_quantile(values: list[float], probability: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan")
    return float(np.quantile(finite, probability))


def interval(values: list[float]) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    return (
        finite_quantile(values, 0.025),
        finite_quantile(values, 0.975),
        int(np.isfinite(finite).sum()),
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def assign_equal_frequency_bins(probability: pd.Series, n_bins: int) -> pd.Series:
    if n_bins < 2:
        raise ValueError("At least two calibration bins are required")
    ranks = probability.rank(method="first")
    return pd.qcut(ranks, q=n_bins, labels=np.arange(1, n_bins + 1)).astype(int)


def performance_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    public_null_probability: float,
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = probability >= threshold
    prevalence = float(np.mean(target))
    brier = float(brier_score_loss(target, probability))
    null_brier = float(
        brier_score_loss(
            target,
            np.full(len(target), public_null_probability, dtype=float),
        )
    )
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(target)) == 2
    average_precision = (
        float(average_precision_score(target, probability))
        if has_both_classes
        else float("nan")
    )
    return {
        "n_rows": int(len(target)),
        "n_events": int(np.sum(target)),
        "prevalence": prevalence,
        "roc_auc": (
            float(roc_auc_score(target, probability))
            if has_both_classes
            else float("nan")
        ),
        "average_precision": average_precision,
        "average_precision_lift": safe_divide(average_precision, prevalence),
        "brier_score": brier,
        "brier_skill_public_null": 1.0 - brier / null_brier,
        "balanced_accuracy": (
            float(balanced_accuracy_score(target, prediction))
            if has_both_classes
            else float("nan")
        ),
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "mean_predicted_probability": float(np.mean(probability)),
        "predicted_positive_fraction": float(np.mean(prediction)),
    }


def fit_logistic_calibration(
    target: np.ndarray,
    probability: np.ndarray,
    max_iterations: int = 100,
) -> tuple[float, float]:
    target = np.asarray(target, dtype=float)
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    design = np.column_stack([np.ones(len(clipped)), logit(clipped)])
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(max_iterations):
        fitted = expit(design @ beta)
        weights = np.clip(fitted * (1.0 - fitted), 1e-8, None)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (target - fitted)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta += step
        if not np.isfinite(beta).all():
            return float("nan"), float("nan")
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def calibration_in_the_large(target: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    linear_predictor = logit(clipped)

    def score(intercept: float) -> float:
        return float(np.sum(target - expit(intercept + linear_predictor)))

    low, high = -30.0, 30.0
    if score(low) * score(high) > 0:
        return float("nan")
    return float(optimize.brentq(score, low, high))


def cluster_robust_calibration(
    target: np.ndarray,
    probability: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    predictor = logit(clipped)
    design = sm.add_constant(predictor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joint = sm.GLM(target, design, family=sm.families.Binomial()).fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_ids},
        )
        citl = sm.GLM(
            target,
            np.ones((len(target), 1)),
            family=sm.families.Binomial(),
            offset=predictor,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": cluster_ids},
        )
    joint_ci = np.asarray(joint.conf_int(alpha=0.05), dtype=float)
    citl_ci = np.asarray(citl.conf_int(alpha=0.05), dtype=float)
    return {
        "weak_calibration_intercept": float(joint.params[0]),
        "weak_calibration_intercept_cluster_ci_low": float(joint_ci[0, 0]),
        "weak_calibration_intercept_cluster_ci_high": float(joint_ci[0, 1]),
        "weak_calibration_slope": float(joint.params[1]),
        "weak_calibration_slope_cluster_ci_low": float(joint_ci[1, 0]),
        "weak_calibration_slope_cluster_ci_high": float(joint_ci[1, 1]),
        "calibration_in_the_large": float(citl.params[0]),
        "calibration_in_the_large_cluster_ci_low": float(citl_ci[0, 0]),
        "calibration_in_the_large_cluster_ci_high": float(citl_ci[0, 1]),
    }


def subgroup_definitions(table: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    definitions = []
    for column in SUBGROUP_COLUMNS:
        for value in sorted(table[column].unique()):
            definitions.append((column, str(value), table[column].to_numpy() == value))
    return definitions


def bootstrap_indices(table: pd.DataFrame) -> list[np.ndarray]:
    cluster_ids = table[CLUSTER_COLUMNS].astype(str).agg("|".join, axis=1).to_numpy()
    return [np.flatnonzero(cluster_ids == value) for value in np.unique(cluster_ids)]


def main() -> None:
    args = parse_args()
    predictions_path = args.external_root / "eppvr_risk_predictions.csv"
    metrics_path = args.external_root / "external_validation_metrics.json"
    gate_path = args.external_root / "external_validation_gate.json"
    public_path = args.risk_root / "risk_learning_table.csv"
    freeze_path = args.risk_root / "freeze_manifest.json"
    model_path = args.risk_root / "frozen_risk_model.joblib"
    for path in [
        predictions_path,
        metrics_path,
        gate_path,
        public_path,
        freeze_path,
        model_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    table = pd.read_csv(predictions_path)
    public = pd.read_csv(public_path)
    external_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    external_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    model_hash = sha256(model_path)
    if model_hash != freeze["model_sha256"]:
        raise ValueError("Frozen model hash does not match the freeze manifest")
    if model_hash != external_metrics["risk_model_sha256"]:
        raise ValueError("External predictions were not generated by the frozen model")
    if not isinstance(external_gate.get("admissible"), bool):
        raise ValueError("External validation gate is missing its locked decision")
    if set(table.axis) != {"subject"}:
        raise ValueError("EPPVR transport diagnostics permit the subject axis only")
    if table.duplicated(CLUSTER_COLUMNS + ["nominal_dose"]).any():
        raise ValueError("EPPVR external rows are not unique by configuration and dose")
    dose_counts = table.groupby(CLUSTER_COLUMNS).nominal_dose.nunique()
    if not (dose_counts == table.nominal_dose.nunique()).all():
        raise ValueError("Configuration clusters do not contain the complete dose grid")

    feature_names = freeze["numeric_features"] + freeze["categorical_features"]
    if freeze["categorical_features"]:
        raise ValueError("Predictor-support diagnostics currently require numeric frozen features")
    missing_external = set(feature_names) - set(table.columns)
    missing_public = set(feature_names) - set(public.columns)
    if missing_external or missing_public:
        raise ValueError(
            "Missing frozen transport predictors: "
            f"external={sorted(missing_external)}, public={sorted(missing_public)}"
        )

    table = table.copy()
    table["risk_bin"] = assign_equal_frequency_bins(
        table.predicted_material_risk,
        args.calibration_bins,
    )
    target = table.material_optimism_event.to_numpy(int)
    probability = table.predicted_material_risk.to_numpy(float)
    amplification = table.dose_induced_amplification.to_numpy(float)
    threshold = float(freeze["decision_threshold"])
    public_prevalence = float(freeze["public_training_event_prevalence"])
    cluster_labels = table[CLUSTER_COLUMNS].astype(str).agg("|".join, axis=1).to_numpy()
    clusters = bootstrap_indices(table)
    subgroup_masks = subgroup_definitions(table)

    support_indicators: dict[str, np.ndarray] = {}
    central_support_indicators: dict[str, np.ndarray] = {}
    shift_rows = []
    for feature in feature_names:
        source = public[feature].to_numpy(float)
        external = table[feature].to_numpy(float)
        lower, upper = float(np.min(source)), float(np.max(source))
        central_lower, central_upper = np.quantile(source, [0.025, 0.975])
        support_indicators[feature] = (external < lower) | (external > upper)
        central_support_indicators[feature] = (
            (external < central_lower) | (external > central_upper)
        )
        source_sd = float(np.std(source, ddof=1))
        shift_rows.append({
            "feature": feature,
            "public_n": len(source),
            "public_min": lower,
            "public_q025": float(np.quantile(source, 0.025)),
            "public_median": float(np.median(source)),
            "public_q975": float(np.quantile(source, 0.975)),
            "public_max": upper,
            "public_mean": float(np.mean(source)),
            "public_sd": source_sd,
            "eppvr_n": len(external),
            "eppvr_min": float(np.min(external)),
            "eppvr_q025": float(np.quantile(external, 0.025)),
            "eppvr_median": float(np.median(external)),
            "eppvr_q975": float(np.quantile(external, 0.975)),
            "eppvr_max": float(np.max(external)),
            "eppvr_mean": float(np.mean(external)),
            "eppvr_sd": float(np.std(external, ddof=1)),
            "standardized_mean_shift": safe_divide(
                float(np.mean(external) - np.mean(source)),
                source_sd,
            ),
            "below_public_min_n": int(np.sum(external < lower)),
            "below_public_min_rate": float(np.mean(external < lower)),
            "above_public_max_n": int(np.sum(external > upper)),
            "above_public_max_rate": float(np.mean(external > upper)),
            "outside_public_range_n": int(np.sum(support_indicators[feature])),
            "outside_public_range_rate": float(np.mean(support_indicators[feature])),
            "outside_public_central_95_n": int(
                np.sum(central_support_indicators[feature])
            ),
            "outside_public_central_95_rate": float(
                np.mean(central_support_indicators[feature])
            ),
        })

    quintile_rows = []
    calibration_rows = []
    for risk_bin in range(1, args.calibration_bins + 1):
        mask = table.risk_bin.to_numpy() == risk_bin
        observed_rate = float(np.mean(target[mask]))
        predicted_mean = float(np.mean(probability[mask]))
        row = {
            "risk_bin": risk_bin,
            "n_rows": int(np.sum(mask)),
            "n_events": int(np.sum(target[mask])),
            "score_min": float(np.min(probability[mask])),
            "score_max": float(np.max(probability[mask])),
            "mean_predicted_probability": predicted_mean,
            "observed_event_rate": observed_rate,
            "absolute_calibration_gap": abs(predicted_mean - observed_rate),
            "risk_enrichment_vs_overall": safe_divide(observed_rate, float(np.mean(target))),
        }
        quintile_rows.append(row.copy())
        calibration_rows.append(row.copy())

    overall = performance_metrics(target, probability, threshold, public_prevalence)
    spearman = stats.spearmanr(probability, amplification)
    bottom_rate = quintile_rows[0]["observed_event_rate"]
    top_rate = quintile_rows[-1]["observed_event_rate"]
    top_bottom_risk_ratio = safe_divide(top_rate, bottom_rate)
    top_bottom_risk_difference = float(top_rate - bottom_rate)
    robust_calibration = cluster_robust_calibration(target, probability, cluster_labels)

    subgroup_rows = []
    for family, subgroup, mask in subgroup_masks:
        values = performance_metrics(
            target[mask],
            probability[mask],
            threshold,
            public_prevalence,
        )
        subgroup_rows.append({"subgroup_family": family, "subgroup": subgroup, **values})

    bootstrap = {
        "top_bottom_risk_ratio": [],
        "top_bottom_risk_difference": [],
        "spearman_rho": [],
        "weak_calibration_intercept": [],
        "weak_calibration_slope": [],
        "calibration_in_the_large": [],
    }
    quintile_bootstrap = {
        risk_bin: {"observed": [], "predicted": []}
        for risk_bin in range(1, args.calibration_bins + 1)
    }
    support_bootstrap = {
        feature: {"outside_range": [], "outside_central_95": []}
        for feature in feature_names
    }
    subgroup_bootstrap = {
        (family, subgroup): {metric: [] for metric in PRIMARY_METRICS}
        for family, subgroup, _ in subgroup_masks
    }
    rng = np.random.default_rng(args.bootstrap_seed)
    for _ in tqdm(
        range(args.bootstrap_repetitions),
        desc="EPPVR transport bootstrap",
        unit="rep",
        dynamic_ncols=True,
    ):
        chosen = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([clusters[index] for index in chosen])
        sampled_target = target[indices]
        sampled_probability = probability[indices]
        sampled_amplification = amplification[indices]
        sampled_bins = table.risk_bin.to_numpy()[indices]

        bottom_mask = sampled_bins == 1
        top_mask = sampled_bins == args.calibration_bins
        if bottom_mask.any() and top_mask.any():
            sampled_bottom = float(np.mean(sampled_target[bottom_mask]))
            sampled_top = float(np.mean(sampled_target[top_mask]))
            bootstrap["top_bottom_risk_ratio"].append(
                safe_divide(sampled_top, sampled_bottom)
            )
            bootstrap["top_bottom_risk_difference"].append(
                sampled_top - sampled_bottom
            )
        correlation = stats.spearmanr(
            sampled_probability,
            sampled_amplification,
        ).statistic
        bootstrap["spearman_rho"].append(float(correlation))
        if len(np.unique(sampled_target)) == 2:
            intercept, slope = fit_logistic_calibration(
                sampled_target,
                sampled_probability,
            )
            bootstrap["weak_calibration_intercept"].append(intercept)
            bootstrap["weak_calibration_slope"].append(slope)
            bootstrap["calibration_in_the_large"].append(
                calibration_in_the_large(sampled_target, sampled_probability)
            )

        for risk_bin in range(1, args.calibration_bins + 1):
            mask = sampled_bins == risk_bin
            if mask.any():
                quintile_bootstrap[risk_bin]["observed"].append(
                    float(np.mean(sampled_target[mask]))
                )
                quintile_bootstrap[risk_bin]["predicted"].append(
                    float(np.mean(sampled_probability[mask]))
                )

        for feature in feature_names:
            support_bootstrap[feature]["outside_range"].append(
                float(np.mean(support_indicators[feature][indices]))
            )
            support_bootstrap[feature]["outside_central_95"].append(
                float(np.mean(central_support_indicators[feature][indices]))
            )

        sampled_frame = table.iloc[indices]
        for family, subgroup, _ in subgroup_masks:
            mask = sampled_frame[family].astype(str).to_numpy() == subgroup
            if not mask.any():
                continue
            values = performance_metrics(
                sampled_target[mask],
                sampled_probability[mask],
                threshold,
                public_prevalence,
            )
            for metric in PRIMARY_METRICS:
                subgroup_bootstrap[(family, subgroup)][metric].append(values[metric])

    for row in shift_rows:
        values = support_bootstrap[row["feature"]]
        low, high, valid = interval(values["outside_range"])
        row["outside_public_range_ci_low"] = low
        row["outside_public_range_ci_high"] = high
        row["outside_public_range_bootstrap_valid"] = valid
        low, high, valid = interval(values["outside_central_95"])
        row["outside_public_central_95_ci_low"] = low
        row["outside_public_central_95_ci_high"] = high
        row["outside_public_central_95_bootstrap_valid"] = valid

    for rows in [quintile_rows, calibration_rows]:
        for row in rows:
            values = quintile_bootstrap[row["risk_bin"]]
            low, high, valid = interval(values["observed"])
            row["observed_event_rate_ci_low"] = low
            row["observed_event_rate_ci_high"] = high
            row["observed_event_rate_bootstrap_valid"] = valid
            low, high, valid = interval(values["predicted"])
            row["mean_predicted_probability_ci_low"] = low
            row["mean_predicted_probability_ci_high"] = high
            row["mean_predicted_probability_bootstrap_valid"] = valid

    for row in subgroup_rows:
        values = subgroup_bootstrap[(row["subgroup_family"], row["subgroup"])]
        for metric in PRIMARY_METRICS:
            low, high, valid = interval(values[metric])
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_bootstrap_valid"] = valid

    calibration = {
        "n_rows": len(table),
        "n_configuration_clusters": len(clusters),
        "observed_event_prevalence": float(np.mean(target)),
        "mean_predicted_probability": float(np.mean(probability)),
        "observed_to_expected_ratio": safe_divide(
            float(np.mean(target)),
            float(np.mean(probability)),
        ),
        "expected_calibration_error_equal_frequency": float(
            sum(
                row["n_rows"] / len(table) * row["absolute_calibration_gap"]
                for row in calibration_rows
            )
        ),
        **robust_calibration,
    }
    for name in [
        "weak_calibration_intercept",
        "weak_calibration_slope",
        "calibration_in_the_large",
    ]:
        low, high, valid = interval(bootstrap[name])
        calibration[f"{name}_bootstrap_ci_low"] = low
        calibration[f"{name}_bootstrap_ci_high"] = high
        calibration[f"{name}_bootstrap_valid"] = valid

    top_bottom_low, top_bottom_high, top_bottom_valid = interval(
        bootstrap["top_bottom_risk_ratio"]
    )
    risk_difference_low, risk_difference_high, risk_difference_valid = interval(
        bootstrap["top_bottom_risk_difference"]
    )
    spearman_low, spearman_high, spearman_valid = interval(bootstrap["spearman_rho"])
    diagnostic_status = (
        "locked_gate_passed_with_transport_diagnostics"
        if external_gate["admissible"]
        else "ranking_transport_with_calibration_failure"
    )
    diagnostics = {
        "status": diagnostic_status,
        "analysis_role": "post_external_explanatory_diagnostic",
        "frozen_model_sha256": model_hash,
        "frozen_model_or_threshold_modified": False,
        "external_gate_admissible": bool(external_gate["admissible"]),
        "bootstrap": {
            "unit": "configuration",
            "cluster_columns": CLUSTER_COLUMNS,
            "n_clusters": len(clusters),
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
            "interval": "percentile 95%",
        },
        "locked_external_performance": overall,
        "ranking_enrichment": {
            "bottom_bin_event_rate": bottom_rate,
            "top_bin_event_rate": top_rate,
            "top_vs_bottom_risk_ratio": top_bottom_risk_ratio,
            "top_vs_bottom_risk_ratio_ci_low": top_bottom_low,
            "top_vs_bottom_risk_ratio_ci_high": top_bottom_high,
            "top_vs_bottom_risk_ratio_bootstrap_valid": top_bottom_valid,
            "top_minus_bottom_risk_difference": top_bottom_risk_difference,
            "top_minus_bottom_risk_difference_ci_low": risk_difference_low,
            "top_minus_bottom_risk_difference_ci_high": risk_difference_high,
            "top_minus_bottom_risk_difference_bootstrap_valid": risk_difference_valid,
        },
        "continuous_association": {
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value_descriptive": float(spearman.pvalue),
            "spearman_rho_ci_low": spearman_low,
            "spearman_rho_ci_high": spearman_high,
            "spearman_bootstrap_valid": spearman_valid,
        },
        "calibration": calibration,
        "support_shift": {
            row["feature"]: {
                "standardized_mean_shift": row["standardized_mean_shift"],
                "outside_public_range_rate": row["outside_public_range_rate"],
                "outside_public_central_95_rate": row[
                    "outside_public_central_95_rate"
                ],
            }
            for row in shift_rows
        },
        "claim": (
            "The frozen mechanism-derived score transports as an audit-priority "
            "ranking under substantial predictor shift, but its absolute probability "
            "calibration and frozen operating threshold do not transport to EPPVR."
        ),
        "claim_boundary": (
            "Post-external diagnostics are explanatory and do not alter the frozen model, "
            "material-effect threshold, decision threshold, or failed external gate. "
            "EPPVR supports subject-axis inference only."
        ),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(shift_rows).to_csv(
        args.output_root / "predictor_shift.csv",
        index=False,
    )
    pd.DataFrame(calibration_rows).to_csv(
        args.output_root / "calibration_bins.csv",
        index=False,
    )
    pd.DataFrame(quintile_rows).to_csv(
        args.output_root / "risk_quintile_enrichment.csv",
        index=False,
    )
    pd.DataFrame(subgroup_rows).to_csv(
        args.output_root / "subgroup_performance.csv",
        index=False,
    )
    (args.output_root / "calibration_summary.json").write_text(
        json.dumps(calibration, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "transport_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
