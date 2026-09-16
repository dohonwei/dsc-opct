from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon


LEVELS = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
UNCERTAINTY_METHOD_PATTERN = r"(regional|regional_no_geo|regional_no_gate|regional_no_uncertainty|mc_dropout)"


def _fold_rows(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    match = re.fullmatch(rf"subject_(\d+)_(M\d+)_{UNCERTAINTY_METHOD_PATTERN}", path.stem)
    if match is None:
        raise ValueError(f"Unexpected prediction filename: {path.name}")
    with np.load(path) as payload:
        if (
            "true_standardized" not in payload
            or "predicted_standardized" not in payload
            or "standardized_log_variance" not in payload
        ):
            raise ValueError(f"{path.name} predates the scale-consistent BSPC export contract.")
        true = payload["true_standardized"].astype(np.float64)
        predicted = payload["predicted_standardized"].astype(np.float64)
        log_variance = payload["standardized_log_variance"].astype(np.float64)
    residual = true - predicted
    sigma = np.exp(0.5 * log_variance)
    standardized_error = np.abs(residual) / np.maximum(sigma, 1e-8)
    metadata = {
        "held_out_subject": int(match.group(1)),
        "budget": match.group(2),
        "method": match.group(3),
    }
    rows = []
    calibration_errors = []
    for level in LEVELS:
        empirical = float(np.mean(standardized_error <= norm.ppf((1.0 + level) / 2.0)))
        calibration_errors.append(abs(empirical - level))
        rows.append({**metadata, "nominal_coverage": level, "empirical_coverage": empirical})
    return rows, {**metadata, "coverage_ece": float(np.mean(calibration_errors))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate uncertainty interval calibration across coverage levels.")
    parser.add_argument("--dataset", choices=["deap", "hci"], required=True)
    parser.add_argument("--run-name", default="loso_bspc")
    parser.add_argument("--output-root", default="outputs/virtual_reconstruction")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / args.output_root
    prediction_root = root / "predictions" / f"{args.dataset}_{args.run_name}"
    coverage_rows, score_rows = [], []
    for path in sorted(prediction_root.glob("*.npz")):
        try:
            rows, score = _fold_rows(path)
        except ValueError:
            continue
        coverage_rows.extend(rows)
        score_rows.append(score)
    coverage = pd.DataFrame(coverage_rows)
    scores = pd.DataFrame(score_rows)
    if coverage.empty:
        raise FileNotFoundError(f"No uncertainty predictions found under {prediction_root}")
    coverage_summary = coverage.groupby(["budget", "method", "nominal_coverage"]).agg(
        mean=("empirical_coverage", "mean"),
        std=("empirical_coverage", "std"),
        count=("empirical_coverage", "count"),
    ).reset_index()
    score_summary = scores.groupby(["budget", "method"]).agg(
        mean=("coverage_ece", "mean"),
        std=("coverage_ece", "std"),
        count=("coverage_ece", "count"),
    ).reset_index()
    comparisons = []
    for budget, group in scores.groupby("budget"):
        paired = group.pivot(index="held_out_subject", columns="method", values="coverage_ece").dropna()
        if not {"regional", "mc_dropout"}.issubset(paired.columns):
            continue
        improvement = paired["mc_dropout"] - paired["regional"]
        p_value = 1.0 if np.allclose(improvement, 0) else float(wilcoxon(improvement, alternative="greater").pvalue)
        comparisons.append({
            "budget": budget,
            "contrast": "regional_vs_mc_dropout",
            "n_subjects": len(improvement),
            "mean_ece_improvement": float(improvement.mean()),
            "p_value": p_value,
        })
    comparison = pd.DataFrame(comparisons)
    if not comparison.empty:
        comparison["p_holm"] = np.nan
        ordered = comparison["p_value"].sort_values().index
        running = 0.0
        for rank, index in enumerate(ordered):
            running = max(running, (len(ordered) - rank) * comparison.loc[index, "p_value"])
            comparison.loc[index, "p_holm"] = min(running, 1.0)
    coverage.to_csv(root / f"{args.dataset}_{args.run_name}_calibration_per_subject.csv", index=False)
    coverage_summary.to_csv(root / f"{args.dataset}_{args.run_name}_calibration_summary.csv", index=False)
    scores.to_csv(root / f"{args.dataset}_{args.run_name}_calibration_error_per_subject.csv", index=False)
    score_summary.to_csv(root / f"{args.dataset}_{args.run_name}_calibration_error_summary.csv", index=False)
    comparison.to_csv(root / f"{args.dataset}_{args.run_name}_calibration_method_comparison.csv", index=False)
    print(coverage_summary.round(4).to_string(index=False))
    print(score_summary.round(4).to_string(index=False))
    if not comparison.empty:
        print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
