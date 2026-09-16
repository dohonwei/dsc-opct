from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the frozen transport policy's untouched SEED-IV failure."
    )
    parser.add_argument(
        "--public-root", type=Path, default=Path("outputs/transport_policy_public_development")
    )
    parser.add_argument(
        "--external-root", type=Path, default=Path("outputs/seediv_transport_policy_external")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/seediv_transport_policy_diagnostics")
    )
    return parser.parse_args()


def safe_auc(target: pd.Series, probability: pd.Series) -> float:
    return (
        float(roc_auc_score(target.astype(int), probability.astype(float)))
        if target.nunique() == 2
        else float("nan")
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(
        (args.public_root / "transport_policy_freeze_manifest.json").read_text(encoding="utf-8")
    )
    public = pd.read_csv(args.public_root / "synthetic_candidate_results.csv")
    signatures = pd.read_csv(args.external_root / "unlabeled_candidate_signatures.csv")
    decision = pd.read_csv(args.external_root / "frozen_policy_decision.csv").iloc[0]
    predictions = pd.read_csv(args.external_root / "seediv_policy_predictions.csv")
    metrics = pd.read_csv(args.external_root / "method_performance.csv")
    intervals = pd.read_csv(args.external_root / "paired_bootstrap_differences.csv")

    signature_columns = [
        column
        for column in signatures.columns
        if column not in {"scenario_id", "method"}
        and pd.api.types.is_numeric_dtype(signatures[column])
    ]
    support_rows = []
    for _, target_row in signatures.iterrows():
        method = str(target_row.method)
        reference = public.loc[public.method == method]
        outside = 0
        robust_distances = []
        for feature in signature_columns:
            low = float(reference[feature].min())
            high = float(reference[feature].max())
            value = float(target_row[feature])
            is_outside = value < low or value > high
            outside += int(is_outside)
            q1, q3 = reference[feature].quantile([0.25, 0.75])
            scale = max(float(q3 - q1), 1e-8)
            robust_distances.append(abs(value - float(reference[feature].median())) / scale)
            support_rows.append(
                {
                    "method": method,
                    "signature_feature": feature,
                    "seediv_value": value,
                    "public_min": low,
                    "public_max": high,
                    "outside_public_range": is_outside,
                    "robust_distance_from_public_median": robust_distances[-1],
                }
            )
        support_rows.append(
            {
                "method": method,
                "signature_feature": "__SUMMARY__",
                "seediv_value": np.nan,
                "public_min": np.nan,
                "public_max": np.nan,
                "outside_public_range": outside > 0,
                "n_features_outside_public_range": outside,
                "fraction_features_outside_public_range": outside / len(signature_columns),
                "maximum_robust_distance": max(robust_distances),
            }
        )
    support = pd.DataFrame(support_rows)

    selected = str(decision.selected_method)
    selected_metrics = metrics.loc[metrics.method == selected].iloc[0]
    identity_metrics = metrics.loc[metrics.method == "identity"].iloc[0]
    plausibility = pd.DataFrame(
        [
            {
                "quantity": "predicted_brier_gain",
                "prediction": float(decision.predicted_brier_gain),
                "theoretical_min": -1.0,
                "theoretical_max": 1.0,
                "within_theoretical_range": -1.0 <= float(decision.predicted_brier_gain) <= 1.0,
                "public_observed_min": float(public.brier_gain.min()),
                "public_observed_max": float(public.brier_gain.max()),
            },
            {
                "quantity": "predicted_auc_delta",
                "prediction": float(decision.predicted_auc_delta),
                "theoretical_min": -1.0,
                "theoretical_max": 1.0,
                "within_theoretical_range": -1.0 <= float(decision.predicted_auc_delta) <= 1.0,
                "public_observed_min": float(public.auc_delta.min()),
                "public_observed_max": float(public.auc_delta.max()),
            },
        ]
    )

    group_columns = ["task", "representation", "model", "nominal_dose"]
    subgroup_rows = []
    for keys, group in predictions.groupby(group_columns, sort=True):
        target = group.material_optimism_event.astype(int)
        identity = group.probability_identity.astype(float)
        adapted = group[f"probability_{selected}"].astype(float)
        subgroup_rows.append(
            dict(
                zip(group_columns, keys, strict=True),
                n=len(group),
                n_events=int(target.sum()),
                event_prevalence=float(target.mean()),
                identity_brier=float(np.mean((identity - target) ** 2)),
                selected_brier=float(np.mean((adapted - target) ** 2)),
                brier_gain=float(np.mean((identity - target) ** 2) - np.mean((adapted - target) ** 2)),
                identity_auc=safe_auc(target, identity),
                selected_auc=safe_auc(target, adapted),
            )
        )
    subgroups = pd.DataFrame(subgroup_rows)
    subgroups["auc_delta"] = subgroups.selected_auc - subgroups.identity_auc

    brier_interval = intervals.loc[
        (intervals.method == selected) & (intervals.metric == "brier_score")
    ].iloc[0]
    auc_interval = intervals.loc[
        (intervals.method == selected) & (intervals.metric == "roc_auc")
    ].iloc[0]
    summary = {
        "status": "frozen_external_failure_diagnosed_without_policy_revision",
        "selected_method": selected,
        "selection_reason": str(decision.selection_reason),
        "n_configurations": len(predictions),
        "n_material_events": int(predictions.material_optimism_event.sum()),
        "event_prevalence": float(predictions.material_optimism_event.mean()),
        "identity_brier": float(identity_metrics.brier_score),
        "selected_brier": float(selected_metrics.brier_score),
        "brier_gain": float(identity_metrics.brier_score - selected_metrics.brier_score),
        "brier_gain_ci95": [
            -float(brier_interval.delta_ci_high),
            -float(brier_interval.delta_ci_low),
        ],
        "identity_auc": float(identity_metrics.roc_auc),
        "selected_auc": float(selected_metrics.roc_auc),
        "auc_delta": float(selected_metrics.roc_auc - identity_metrics.roc_auc),
        "auc_delta_ci95": [float(auc_interval.delta_ci_low), float(auc_interval.delta_ci_high)],
        "raw_support_violation": float(decision.raw_support_violation),
        "predicted_brier_gain": float(decision.predicted_brier_gain),
        "predicted_auc_delta": float(decision.predicted_auc_delta),
        "primary_failure_mechanism": (
            "Unbounded ridge-committee extrapolation outside the public synthetic-signature "
            "support produced impossible predicted metric changes, while the frozen ambiguity "
            "guards did not require abstention for large global displacement."
        ),
        "interpretation": (
            "The untouched external safety and benefit gates failed. SEED-IV cannot be reused "
            "to revise and validate the same policy. Any successor policy requires development "
            "without SEED-IV outcomes and a new untouched dataset for confirmation."
        ),
        "nomenclature_correction": (
            "The output row named supervised_recalibration_oracle is an exploratory grouped-CV "
            "target-label recalibration reference, not a guaranteed performance upper bound."
        ),
        "policy_features": manifest["features"],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    support.to_csv(args.output_root / "signature_support_audit.csv", index=False)
    plausibility.to_csv(args.output_root / "prediction_plausibility_audit.csv", index=False)
    subgroups.to_csv(args.output_root / "subgroup_failure_analysis.csv", index=False)
    (args.output_root / "failure_diagnosis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
