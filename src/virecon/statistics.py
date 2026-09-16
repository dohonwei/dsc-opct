from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import warnings


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def _paired_summary(
    first: pd.DataFrame,
    second: pd.DataFrame,
    metric: str,
    improvement_sign: float,
    seed: int,
) -> dict[str, float | int]:
    paired = first[["held_out_subject", metric]].merge(
        second[["held_out_subject", metric]], on="held_out_subject", suffixes=("_first", "_second")
    ).dropna()
    difference = improvement_sign * (paired[f"{metric}_first"] - paired[f"{metric}_second"]).to_numpy(float)
    if len(difference) == 0:
        return {"n_subjects": 0, "mean_improvement": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "cohens_dz": np.nan, "p_value": np.nan}
    rng = np.random.default_rng(seed)
    samples = rng.choice(difference, size=(10000, len(difference)), replace=True).mean(axis=1)
    std = difference.std(ddof=1) if len(difference) > 1 else np.nan
    if np.allclose(difference, 0):
        p_value = 1.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p_value = float(wilcoxon(difference, alternative="two-sided").pvalue)
    return {
        "n_subjects": len(difference),
        "mean_improvement": float(difference.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "cohens_dz": float(difference.mean() / std) if std and np.isfinite(std) else np.nan,
        "p_value": p_value,
    }


def reconstruction_comparisons(frame: pd.DataFrame, seed: int = 10) -> pd.DataFrame:
    rows = []
    for (dataset, budget), group in frame.groupby(["dataset", "budget"]):
        proposed = group[group["method"] == "regional"]
        for baseline in ("interpolation", "ridge", "mlp", "mc_dropout"):
            reference = group[group["method"] == baseline]
            if proposed.empty or reference.empty:
                continue
            metric_specs = (
                ("log_mae", reference, proposed),
                ("log_rmse", reference, proposed),
                ("log_feature_correlation", proposed, reference),
            )
            for metric, first, second in metric_specs:
                rows.append({
                    "family": "reconstruction",
                    "dataset": dataset,
                    "budget": budget,
                    "target": "fp12",
                    "contrast": f"regional_vs_{baseline}",
                    "metric": metric,
                    **_paired_summary(first, second, metric, 1.0, seed),
                })
    return _adjust(pd.DataFrame(rows))


def ablation_comparisons(frame: pd.DataFrame, seed: int = 10) -> pd.DataFrame:
    rows = []
    ablations = ("regional_no_geo", "regional_no_gate", "regional_no_uncertainty")
    for (dataset, budget), group in frame.groupby(["dataset", "budget"]):
        proposed = group[group["method"] == "regional"]
        for ablation in ablations:
            reference = group[group["method"] == ablation]
            if proposed.empty or reference.empty:
                continue
            metric_specs = (
                ("log_mae", reference, proposed),
                ("log_rmse", reference, proposed),
                ("log_feature_correlation", proposed, reference),
                ("gaussian_nll", reference, proposed),
                ("error_uncertainty_correlation", proposed, reference),
            )
            for metric, first, second in metric_specs:
                if metric not in group.columns:
                    continue
                rows.append({
                    "family": "ablation",
                    "dataset": dataset,
                    "budget": budget,
                    "target": "fp12",
                    "contrast": f"regional_vs_{ablation}",
                    "metric": metric,
                    **_paired_summary(first, second, metric, 1.0, seed),
                })
    return _adjust(pd.DataFrame(rows))


def feature_group_comparisons(frame: pd.DataFrame, seed: int = 10) -> pd.DataFrame:
    rows = []
    for (dataset, budget, feature_group), group in frame.groupby(["dataset", "budget", "feature_group"]):
        proposed = group[group["method"] == "regional"]
        for baseline in ("interpolation", "ridge", "mlp", "mc_dropout"):
            reference = group[group["method"] == baseline]
            if proposed.empty or reference.empty:
                continue
            for metric, first, second in (
                ("log_mae", reference, proposed),
                ("log_feature_correlation", proposed, reference),
            ):
                rows.append({
                    "family": "feature_group",
                    "dataset": dataset,
                    "budget": budget,
                    "target": feature_group,
                    "contrast": f"regional_vs_{baseline}",
                    "metric": metric,
                    **_paired_summary(first, second, metric, 1.0, seed),
                })
    return _adjust(pd.DataFrame(rows))


def downstream_comparisons(frame: pd.DataFrame, seed: int = 10) -> pd.DataFrame:
    rows = []
    proposed = frame[frame["method"] == "regional"]
    for (dataset, budget, target), group in proposed.groupby(["dataset", "budget", "target"]):
        contrasts = (
            ("augmentation_vs_source", "source_plus_reconstructed", "source_only"),
            ("reconstructed_vs_oracle", "reconstructed_fp12", "oracle_real_fp12"),
        )
        for contrast, first_name, second_name in contrasts:
            first = group[group["representation"] == first_name]
            second = group[group["representation"] == second_name]
            rows.append({
                "family": "downstream",
                "dataset": dataset,
                "budget": budget,
                "target": target,
                "contrast": contrast,
                "metric": "balanced_accuracy",
                **_paired_summary(first, second, "balanced_accuracy", 1.0, seed),
            })
    return _adjust(pd.DataFrame(rows))


def _adjust(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["p_holm"] = np.nan
    for _, indices in frame.groupby(["family", "dataset", "metric"]).groups.items():
        frame.loc[indices, "p_holm"] = _holm_adjust(frame.loc[indices, "p_value"].to_numpy(float))
    return frame
