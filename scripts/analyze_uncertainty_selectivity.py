from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _risk_at_coverage(errors: np.ndarray, uncertainty: np.ndarray, coverage: float) -> float:
    count = max(1, int(np.ceil(coverage * len(errors))))
    selected = np.argpartition(uncertainty, count - 1)[:count]
    return float(errors[selected].mean())


UNCERTAINTY_METHOD_PATTERN = r"(regional|regional_no_geo|regional_no_gate|regional_no_uncertainty|mc_dropout)"


def _fold_metrics(path: Path) -> dict[str, object]:
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
    errors = np.abs(true - predicted).ravel()
    uncertainty = np.exp(0.5 * log_variance).ravel()
    full_risk = float(errors.mean())
    risk_80 = _risk_at_coverage(errors, uncertainty, 0.8)
    risk_50 = _risk_at_coverage(errors, uncertainty, 0.5)
    coverages = np.linspace(0.1, 1.0, 19)
    risks = np.asarray([_risk_at_coverage(errors, uncertainty, value) for value in coverages])
    aurc = float(np.trapz(risks, coverages) / (coverages[-1] - coverages[0]))
    return {
        "held_out_subject": int(match.group(1)),
        "budget": match.group(2),
        "method": match.group(3),
        "full_standardized_mae": full_risk,
        "risk_at_80": risk_80,
        "risk_at_50": risk_50,
        "relative_gain_at_80": (full_risk - risk_80) / full_risk,
        "relative_gain_at_50": (full_risk - risk_50) / full_risk,
        "aurc": aurc,
        "aurc_gain_vs_random": (full_risk - aurc) / full_risk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate feature-wise uncertainty selective reconstruction.")
    parser.add_argument("--dataset", choices=["deap", "hci"], required=True)
    parser.add_argument("--run-name", default="loso_v2")
    parser.add_argument("--output-root", default="outputs/virtual_reconstruction")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / args.output_root
    prediction_root = root / "predictions" / f"{args.dataset}_{args.run_name}"
    rows = []
    for path in sorted(prediction_root.glob("*.npz")):
        try:
            rows.append(_fold_metrics(path))
        except ValueError:
            continue
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No uncertainty predictions found under {prediction_root}")
    frame.insert(0, "dataset", args.dataset)
    summaries = []
    for (budget, method), group in frame.groupby(["budget", "method"]):
        gain = group["aurc_gain_vs_random"].to_numpy()
        p_value = 1.0 if np.allclose(gain, 0) else float(wilcoxon(gain, alternative="greater").pvalue)
        summaries.append(
            {
                "dataset": args.dataset,
                "budget": budget,
                "method": method,
                "n_subjects": len(group),
                **group[["full_standardized_mae", "risk_at_80", "risk_at_50", "relative_gain_at_80",
                         "relative_gain_at_50", "aurc", "aurc_gain_vs_random"]].mean().to_dict(),
                "aurc_gain_p_value": p_value,
            }
        )
    summary = pd.DataFrame(summaries)
    summary["aurc_gain_p_holm"] = np.nan
    for _, indices in summary.groupby("method").groups.items():
        ordered = summary.loc[indices, "aurc_gain_p_value"].sort_values().index
        running = 0.0
        for rank, index in enumerate(ordered):
            running = max(running, (len(ordered) - rank) * summary.loc[index, "aurc_gain_p_value"])
            summary.loc[index, "aurc_gain_p_holm"] = min(running, 1.0)

    comparisons = []
    for budget, group in frame.groupby("budget"):
        paired = group.pivot(index="held_out_subject", columns="method", values="aurc").dropna()
        if not {"regional", "mc_dropout"}.issubset(paired.columns):
            continue
        improvement = paired["mc_dropout"] - paired["regional"]
        p_value = 1.0 if np.allclose(improvement, 0) else float(wilcoxon(improvement, alternative="greater").pvalue)
        comparisons.append({
            "dataset": args.dataset,
            "budget": budget,
            "contrast": "regional_vs_mc_dropout",
            "n_subjects": len(improvement),
            "mean_aurc_improvement": float(improvement.mean()),
            "p_value": p_value,
        })
    comparison = pd.DataFrame(comparisons)
    if not comparison.empty:
        ordered = comparison["p_value"].sort_values().index
        running = 0.0
        comparison["p_holm"] = np.nan
        for rank, index in enumerate(ordered):
            running = max(running, (len(ordered) - rank) * comparison.loc[index, "p_value"])
            comparison.loc[index, "p_holm"] = min(running, 1.0)
    per_subject_path = root / f"{args.dataset}_{args.run_name}_uncertainty_selectivity_per_subject.csv"
    summary_path = root / f"{args.dataset}_{args.run_name}_uncertainty_selectivity_summary.csv"
    comparison_path = root / f"{args.dataset}_{args.run_name}_uncertainty_method_comparison.csv"
    frame.to_csv(per_subject_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    print(summary.round(4).to_string(index=False))
    if not comparison.empty:
        print(comparison.round(4).to_string(index=False))
    print(f"Selectivity: {per_subject_path} | {summary_path} | {comparison_path}")


if __name__ == "__main__":
    main()
