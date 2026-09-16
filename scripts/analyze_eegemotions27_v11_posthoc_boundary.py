from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from tqdm.auto import tqdm


CONFIG_COLUMNS = ["task", "representation", "model", "split_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc boundary analysis of EEGEmotions-27 DCS-OPCT v11."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_external_robustness"),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_posthoc_boundary_analysis"),
    )
    return parser.parse_args()


def safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(probability, method="average")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def counterfactual_metrics(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    labels = frame.material_optimism_event.to_numpy(int)
    identity = frame.probability_identity.to_numpy(float)
    candidate = frame[f"probability_{method}"].to_numpy(float)
    identity_auc = safe_auc(labels, identity)
    candidate_auc = safe_auc(labels, candidate)
    return {
        "n": int(len(frame)),
        "events": int(labels.sum()),
        "event_rate": float(labels.mean()),
        "brier_gain": float(
            np.mean((labels - identity) ** 2 - (labels - candidate) ** 2)
        ),
        "identity_auc": identity_auc,
        "candidate_auc": candidate_auc,
        "auc_delta": float(candidate_auc - identity_auc),
        "probability_mean_shift": float(np.mean(candidate - identity)),
        "probability_rank_correlation": float(
            pd.Series(identity).corr(pd.Series(candidate), method="spearman")
        ),
        "decision_flip_rate": float(
            np.mean((identity >= 0.5) != (candidate >= 0.5))
        ),
    }


def descriptive_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    value: str,
) -> pd.DataFrame:
    return (
        frame.groupby(group_columns, dropna=False)[value]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
    )


def ranking_reversals(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = summary.loc[summary.nominal_dose.eq(0.0)].copy()
    model_rows = []
    for (representation, seed), group in baseline.groupby(
        ["representation", "split_seed"], sort=True
    ):
        baseline_winner = str(
            group.sort_values(
                ["dose_balanced_accuracy", "model"], ascending=[False, True]
            ).iloc[0].model
        )
        doses = summary.loc[
            summary.representation.eq(representation)
            & summary.split_seed.eq(seed)
            & summary.nominal_dose.gt(0.0)
        ]
        for dose, dose_group in doses.groupby("nominal_dose", sort=True):
            winner = str(
                dose_group.sort_values(
                    ["dose_balanced_accuracy", "model"], ascending=[False, True]
                ).iloc[0].model
            )
            model_rows.append(
                {
                    "representation": representation,
                    "split_seed": int(seed),
                    "nominal_dose": float(dose),
                    "baseline_winner": baseline_winner,
                    "dose_winner": winner,
                    "ranking_reversal": winner != baseline_winner,
                }
            )

    representation_rows = []
    for (model, seed), group in baseline.groupby(["model", "split_seed"], sort=True):
        baseline_winner = str(
            group.sort_values(
                ["dose_balanced_accuracy", "representation"], ascending=[False, True]
            ).iloc[0].representation
        )
        doses = summary.loc[
            summary.model.eq(model)
            & summary.split_seed.eq(seed)
            & summary.nominal_dose.gt(0.0)
        ]
        for dose, dose_group in doses.groupby("nominal_dose", sort=True):
            winner = str(
                dose_group.sort_values(
                    ["dose_balanced_accuracy", "representation"],
                    ascending=[False, True],
                ).iloc[0].representation
            )
            representation_rows.append(
                {
                    "model": model,
                    "split_seed": int(seed),
                    "nominal_dose": float(dose),
                    "baseline_winner": baseline_winner,
                    "dose_winner": winner,
                    "ranking_reversal": winner != baseline_winner,
                }
            )
    return pd.DataFrame(model_rows), pd.DataFrame(representation_rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite post-hoc analysis: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    gate = json.loads(
        (args.input_root / "outcome/external_confirmation_gate.json").read_text(
            encoding="utf-8"
        )
    )
    validation = json.loads(
        (args.input_root / "independent_validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    if validation.get("status") != "passed":
        raise RuntimeError("Primary result must pass independent validation first")

    summary = pd.read_csv(args.input_root / "model_outcomes/summary.csv")
    encoding = pd.read_csv(
        args.input_root / "model_outcomes/identity_encoding_margin.csv"
    )
    components = pd.read_csv(
        args.input_root / "action/component_projection_diagnostics.csv"
    )
    partitions = {
        "audit": pd.read_csv(args.input_root / "outcome/primary_audit_labeled.csv"),
        "heldout": pd.read_csv(args.input_root / "outcome/primary_heldout_results.csv"),
    }
    progress = tqdm(total=8, desc="EEGEmotions-27 post-hoc boundary analysis", unit="table")

    threshold = float(
        freeze["probability_action"]["maximum_probability_mean_shift"]
    )
    applicability = components.copy()
    applicability["absolute_probability_mean_shift"] = applicability[
        "probability_mean_shift"
    ].abs()
    applicability["maximum_allowed_shift"] = threshold
    applicability["threshold_excess"] = (
        applicability.absolute_probability_mean_shift - threshold
    )
    applicability["relative_threshold_excess"] = (
        applicability.threshold_excess / threshold
    )
    applicability["analysis_status"] = "post_hoc_boundary_description"
    applicability.to_csv(args.output_root / "applicability_thresholds.csv", index=False)
    progress.update()

    baseline = summary.loc[summary.nominal_dose.eq(0.0)].copy()
    baseline_table = descriptive_summary(
        baseline,
        ["representation", "model"],
        "unseen_balanced_accuracy",
    ).rename(
        columns={
            "mean": "balanced_accuracy_mean",
            "std": "balanced_accuracy_sd",
            "min": "balanced_accuracy_min",
            "median": "balanced_accuracy_median",
            "max": "balanced_accuracy_max",
        }
    )
    baseline_table["chance_reference"] = 0.5
    baseline_table["mean_minus_chance"] = (
        baseline_table.balanced_accuracy_mean - 0.5
    )
    baseline_table["analysis_status"] = "descriptive_correlated_seed_summary"
    baseline_table.to_csv(
        args.output_root / "baseline_dual_generalization.csv", index=False
    )
    progress.update()

    encoding_table = descriptive_summary(
        encoding,
        ["representation", "model"],
        "encoding_margin",
    ).rename(
        columns={
            "model": "identity_probe",
            "mean": "encoding_margin_mean",
            "std": "encoding_margin_sd",
            "min": "encoding_margin_min",
            "median": "encoding_margin_median",
            "max": "encoding_margin_max",
        }
    )
    encoding_table["analysis_status"] = "descriptive_correlated_seed_summary"
    encoding_table.to_csv(
        args.output_root / "identity_encoding_summary.csv", index=False
    )
    progress.update()

    trajectory = descriptive_summary(
        summary,
        ["representation", "model", "nominal_dose"],
        "exposure_effect",
    ).rename(
        columns={
            "mean": "exposure_effect_mean",
            "std": "exposure_effect_sd",
            "min": "exposure_effect_min",
            "median": "exposure_effect_median",
            "max": "exposure_effect_max",
        }
    )
    trajectory["analysis_status"] = "post_hoc_descriptive_dose_trajectory"
    trajectory.to_csv(args.output_root / "dose_trajectory_summary.csv", index=False)
    progress.update()

    dose_correlations = []
    for key, group in summary.groupby(CONFIG_COLUMNS, sort=True, dropna=False):
        correlation = spearmanr(group.nominal_dose, group.exposure_effect).statistic
        dose_correlations.append(
            {
                **dict(zip(CONFIG_COLUMNS, key, strict=True)),
                "spearman_dose_effect": float(correlation),
                "monotonic_positive": bool(correlation >= 0.9),
                "monotonic_negative": bool(correlation <= -0.9),
                "analysis_status": "post_hoc_descriptive_dose_pattern",
            }
        )
    dose_correlations = pd.DataFrame(dose_correlations)
    dose_correlations.to_csv(
        args.output_root / "dose_monotonicity_by_configuration.csv", index=False
    )
    progress.update()

    event_rows = []
    for partition, frame in partitions.items():
        for grouping, columns in {
            "overall": [],
            "representation": ["representation"],
            "model": ["model"],
            "dose": ["nominal_dose"],
        }.items():
            grouped = [((), frame)] if not columns else frame.groupby(columns, sort=True)
            for level, group in grouped:
                levels = level if isinstance(level, tuple) else (level,)
                row = {
                    "partition": partition,
                    "grouping": grouping,
                    "level": "all" if not columns else "|".join(map(str, levels)),
                    "n": int(len(group)),
                    "positive_material_events": int(
                        (group.dose_induced_amplification >= 0.02).sum()
                    ),
                    "negative_material_events": int(
                        (group.dose_induced_amplification <= -0.02).sum()
                    ),
                    "nonmaterial_events": int(
                        (group.dose_induced_amplification.abs() < 0.02).sum()
                    ),
                    "amplification_mean": float(group.dose_induced_amplification.mean()),
                    "amplification_sd": float(group.dose_induced_amplification.std(ddof=1)),
                    "analysis_status": "post_hoc_boundary_description",
                }
                event_rows.append(row)
    event_table = pd.DataFrame(event_rows)
    event_table.to_csv(args.output_root / "material_event_summary.csv", index=False)
    progress.update()

    model_reversals, representation_reversals = ranking_reversals(summary)
    model_reversals["analysis_status"] = "post_hoc_ranking_stability_analysis"
    representation_reversals["analysis_status"] = (
        "post_hoc_ranking_stability_analysis"
    )
    model_reversals.to_csv(args.output_root / "model_ranking_reversals.csv", index=False)
    representation_reversals.to_csv(
        args.output_root / "representation_ranking_reversals.csv", index=False
    )
    progress.update()

    counterfactual_rows = []
    for partition, frame in partitions.items():
        for method in ("coral", "quantile_mapping"):
            counterfactual_rows.append(
                {
                    "partition": partition,
                    "method": method,
                    **counterfactual_metrics(frame, method),
                    "released": False,
                    "analysis_status": "post_hoc_nonreleased_counterfactual",
                }
            )
    counterfactual = pd.DataFrame(counterfactual_rows)
    counterfactual.to_csv(
        args.output_root / "nonreleased_counterfactual_diagnostics.csv", index=False
    )
    progress.update()
    progress.close()

    overall_events = event_table.loc[event_table.grouping.eq("overall")].set_index(
        "partition"
    )
    summary_payload = {
        "status": "post_hoc_boundary_analysis_complete",
        "date": "2026-09-09",
        "dataset": "EEGEmotions-27",
        "primary_gate_changed": False,
        "primary_claim_supported": bool(gate["claim_supported"]),
        "selected_method": gate["selected_method"],
        "failure_stage": "pre-outcome_unlabeled_action_applicability",
        "maximum_probability_mean_shift": threshold,
        "component_threshold_excess": {
            row.method: float(row.threshold_excess)
            for row in applicability.itertuples(index=False)
        },
        "all_baseline_dual_generalization_means_below_chance": bool(
            baseline_table.balanced_accuracy_mean.lt(0.5).all()
        ),
        "identity_encoding_margin_range": [
            float(encoding.encoding_margin.min()),
            float(encoding.encoding_margin.max()),
        ],
        "monotonic_positive_configurations": int(
            dose_correlations.monotonic_positive.sum()
        ),
        "monotonic_negative_configurations": int(
            dose_correlations.monotonic_negative.sum()
        ),
        "total_configurations": int(len(dose_correlations)),
        "audit_material_positive_events": int(
            overall_events.loc["audit", "positive_material_events"]
        ),
        "audit_material_negative_events": int(
            overall_events.loc["audit", "negative_material_events"]
        ),
        "heldout_material_positive_events": int(
            overall_events.loc["heldout", "positive_material_events"]
        ),
        "heldout_material_negative_events": int(
            overall_events.loc["heldout", "negative_material_events"]
        ),
        "model_ranking_reversals": int(model_reversals.ranking_reversal.sum()),
        "model_ranking_comparisons": int(len(model_reversals)),
        "representation_ranking_reversals": int(
            representation_reversals.ranking_reversal.sum()
        ),
        "representation_ranking_comparisons": int(len(representation_reversals)),
        "interpretation_boundary": (
            "Every output is exploratory and post-hoc. These diagnostics explain the "
            "observed boundary but cannot rescue, repeat, redefine, or supersede the "
            "frozen one-shot gate. Category-derived polarity is not experienced valence."
        ),
    }
    (args.output_root / "posthoc_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
