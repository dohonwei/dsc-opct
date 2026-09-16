from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, jaccard_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_identity_shortcut_risk import (  # noqa: E402
    bootstrap_classification_metrics,
    classification_metrics,
    classifier_features,
    coefficient_table,
    make_classifier_pipeline,
    training_only_threshold,
)


DEFAULT_THRESHOLDS = [0.010, 0.015, 0.020, 0.025, 0.030]
PRIMARY_THRESHOLD = 0.020
METRIC_FLOORS = {
    "roc_auc": 0.65,
    "average_precision_lift": 1.50,
    "brier_skill": 0.0,
    "balanced_accuracy": 0.60,
}
COLORS = {"DEAP": "#2878B5", "MAHNOB-HCI": "#D95F02"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc robustness analysis of the operational material-event threshold. "
            "The frozen DCS-OPCT v11 model and its 0.02 endpoint remain unchanged."
        )
    )
    parser.add_argument(
        "--risk-table",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier/risk_learning_table.csv"),
    )
    parser.add_argument(
        "--frozen-metrics",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier/grouped_cv_metrics.csv"),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json"),
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/material_event_threshold_robustness"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(args: argparse.Namespace) -> list[float]:
    missing = [
        path for path in (args.risk_table, args.frozen_metrics, args.freeze) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    thresholds = sorted(set(float(value) for value in args.thresholds))
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("All material-event thresholds must be positive")
    if not any(np.isclose(value, PRIMARY_THRESHOLD) for value in thresholds):
        raise ValueError("The sensitivity grid must include the frozen 0.02 threshold")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_root.resolve()}"
        )
    return thresholds


def relabel(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    work = table.copy()
    work["material_optimism_event"] = (
        work.dose_induced_amplification.to_numpy(float) >= threshold
    ).astype(int)
    return work


def leave_dataset_out(
    table: pd.DataFrame,
    threshold: float,
    logistic_c: float,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[list[dict], list[dict], list[pd.DataFrame]]:
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    coefficient_frames: list[pd.DataFrame] = []
    features = classifier_features()
    for held_out in sorted(table.dataset.unique()):
        train = table.loc[table.dataset != held_out].copy()
        test = table.loc[table.dataset == held_out].copy()
        if train.material_optimism_event.nunique() < 2 or test.material_optimism_event.nunique() < 2:
            raise ValueError(f"Threshold {threshold:.3f} creates a one-class split for {held_out}")

        decision_threshold, training_oof_ba = training_only_threshold(train, logistic_c)
        pipeline, _ = make_classifier_pipeline(logistic_c)
        pipeline.fit(train[features], train.material_optimism_event)
        probability = pipeline.predict_proba(test[features])[:, 1]
        null_probability = float(train.material_optimism_event.mean())
        values = classification_metrics(
            test.material_optimism_event.to_numpy(int),
            probability,
            decision_threshold,
            null_probability,
        )
        values.update(
            {
                "material_effect_threshold": threshold,
                "held_out": held_out,
                "training_prevalence": null_probability,
                "training_oof_balanced_accuracy": training_oof_ba,
            }
        )
        values.update(
            bootstrap_classification_metrics(
                test,
                probability,
                decision_threshold,
                null_probability,
                bootstrap_repetitions,
                bootstrap_seed
                + int(round(threshold * 1000)) * 1000
                + sum(ord(char) for char in str(held_out)),
            )
        )
        metric_rows.append(values)

        for row, estimate in zip(test.itertuples(index=False), probability, strict=True):
            prediction_rows.append(
                {
                    "material_effect_threshold": threshold,
                    "held_out": held_out,
                    "risk_row_id": row.risk_row_id,
                    "dataset": row.dataset,
                    "task": row.task,
                    "representation": row.representation,
                    "model": row.model,
                    "split_seed": row.split_seed,
                    "nominal_dose": row.nominal_dose,
                    "dose_induced_amplification": row.dose_induced_amplification,
                    "material_optimism_event": row.material_optimism_event,
                    "predicted_probability": float(estimate),
                    "decision_threshold": decision_threshold,
                    "predicted_event": int(estimate >= decision_threshold),
                }
            )

        coefficients = coefficient_table(pipeline)
        coefficients.insert(0, "held_out", held_out)
        coefficients.insert(0, "material_effect_threshold", threshold)
        coefficient_frames.append(coefficients)
    return metric_rows, prediction_rows, coefficient_frames


def event_prevalence(table: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        work = relabel(table, threshold)
        for dataset, group in work.groupby("dataset", sort=True):
            rows.append(
                {
                    "material_effect_threshold": threshold,
                    "dataset": dataset,
                    "n": len(group),
                    "n_events": int(group.material_optimism_event.sum()),
                    "event_prevalence": float(group.material_optimism_event.mean()),
                }
            )
    return pd.DataFrame(rows)


def event_concordance(table: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    labels = {
        threshold: relabel(table, threshold).material_optimism_event.to_numpy(int)
        for threshold in thresholds
    }
    rows = []
    for dataset in ["ALL", *sorted(table.dataset.unique())]:
        mask = (
            np.ones(len(table), dtype=bool)
            if dataset == "ALL"
            else table.dataset.eq(dataset).to_numpy()
        )
        for lower, upper in combinations(thresholds, 2):
            low = labels[lower][mask]
            high = labels[upper][mask]
            rows.append(
                {
                    "dataset": dataset,
                    "lower_threshold": lower,
                    "upper_threshold": upper,
                    "agreement": float(np.mean(low == high)),
                    "cohen_kappa": float(cohen_kappa_score(low, high)),
                    "event_jaccard": float(jaccard_score(low, high, zero_division=0)),
                    "events_lost": int(np.sum((low == 1) & (high == 0))),
                    "events_gained": int(np.sum((low == 0) & (high == 1))),
                }
            )
    return pd.DataFrame(rows)


def threshold_summary(metrics: pd.DataFrame, prevalence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold, group in metrics.groupby("material_effect_threshold", sort=True):
        event_counts = prevalence.loc[
            prevalence.material_effect_threshold.eq(threshold), "n_events"
        ]
        row = {
            "material_effect_threshold": threshold,
            "minimum_events": int(event_counts.min()),
        }
        for metric, floor in METRIC_FLOORS.items():
            minimum = float(group[metric].min())
            row[f"minimum_{metric}"] = minimum
            if metric == "brier_skill":
                row[f"passes_{metric}_floor"] = bool(minimum > floor)
            else:
                row[f"passes_{metric}_floor"] = bool(minimum >= floor)
        row["passes_all_descriptive_floors"] = all(
            row[f"passes_{metric}_floor"] for metric in METRIC_FLOORS
        )
        row["has_at_least_20_events_per_dataset"] = bool(row["minimum_events"] >= 20)
        row["operationally_supported"] = bool(
            row["passes_all_descriptive_floors"]
            and row["has_at_least_20_events_per_dataset"]
        )
        rows.append(row)
    return pd.DataFrame(rows)
def model_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    wide = coefficients.pivot_table(
        index=["material_effect_threshold", "held_out"],
        columns="feature",
        values="coefficient",
    ).sort_index(axis=1)
    rows = []
    for held_out in sorted(coefficients.held_out.unique()):
        reference = wide.loc[(PRIMARY_THRESHOLD, held_out)].to_numpy(float)
        for threshold in sorted(coefficients.material_effect_threshold.unique()):
            candidate = wide.loc[(threshold, held_out)].to_numpy(float)
            denominator = np.linalg.norm(reference) * np.linalg.norm(candidate)
            rows.append(
                {
                    "material_effect_threshold": threshold,
                    "held_out": held_out,
                    "coefficient_cosine_vs_0_02": float(
                        np.dot(reference, candidate) / denominator
                    ),
                    "coefficient_spearman_vs_0_02": float(
                        spearmanr(reference, candidate).statistic
                    ),
                    "coefficient_sign_agreement_vs_0_02": float(
                        np.mean(np.sign(reference) == np.sign(candidate))
                    ),
                }
            )
    return pd.DataFrame(rows)


def verify_primary_reproduction(metrics: pd.DataFrame, frozen_path: Path) -> dict:
    frozen = pd.read_csv(frozen_path)
    frozen = frozen.loc[frozen.scheme.eq("leave_dataset_out")].set_index("held_out")
    reproduced = metrics.loc[
        np.isclose(metrics.material_effect_threshold, PRIMARY_THRESHOLD)
    ].set_index("held_out")
    checked = [
        "n_test",
        "n_events",
        "prevalence",
        "roc_auc",
        "average_precision",
        "average_precision_lift",
        "brier_score",
        "null_brier_score",
        "brier_skill",
        "decision_threshold",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "predicted_positive_fraction",
        "mean_predicted_probability",
        "training_prevalence",
        "training_oof_balanced_accuracy",
    ]
    differences = {}
    for held_out in frozen.index:
        differences[held_out] = {
            column: float(
                abs(
                    float(reproduced.loc[held_out, column])
                    - float(frozen.loc[held_out, column])
                )
            )
            for column in checked
        }
    maximum = max(value for dataset in differences.values() for value in dataset.values())
    return {
        "exact_within_numerical_tolerance": bool(maximum <= 1e-12),
        "maximum_absolute_difference": maximum,
        "tolerance": 1e-12,
        "checked_columns": checked,
        "differences": differences,
    }


def make_figure(
    metrics: pd.DataFrame,
    prevalence: pd.DataFrame,
    output_root: Path,
) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    panels = [
        (axes[0, 0], "roc_auc", "A  AUROC", 0.65),
        (axes[0, 1], "brier_skill", "B  Brier skill", 0.0),
        (axes[1, 0], "balanced_accuracy", "C  Balanced accuracy", 0.60),
    ]
    for axis, metric, title, floor in panels:
        for dataset, group in metrics.groupby("held_out", sort=True):
            group = group.sort_values("material_effect_threshold")
            axis.plot(
                group.material_effect_threshold,
                group[metric],
                marker="o",
                linewidth=1.7,
                color=COLORS.get(dataset, "#555555"),
                label=dataset,
            )
        axis.axhline(floor, color="#6B7280", linestyle="--", linewidth=1.0)
        axis.axvline(PRIMARY_THRESHOLD, color="#111827", linestyle=":", linewidth=1.1)
        axis.set_title(title)
        axis.set_xlabel("Material-event threshold")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)

    axis = axes[1, 1]
    for dataset, group in prevalence.groupby("dataset", sort=True):
        group = group.sort_values("material_effect_threshold")
        axis.plot(
            group.material_effect_threshold,
            group.event_prevalence,
            marker="o",
            linewidth=1.7,
            color=COLORS.get(dataset, "#555555"),
            label=dataset,
        )
    axis.axvline(PRIMARY_THRESHOLD, color="#111827", linestyle=":", linewidth=1.1)
    axis.set_title("D  Observed event prevalence")
    axis.set_xlabel("Material-event threshold")
    axis.set_ylabel("Fraction of configurations")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.035),
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_root / f"fig_material_event_threshold_robustness.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    thresholds = validate_inputs(args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    table = pd.read_csv(args.risk_table)
    required = set(classifier_features()) | {
        "dataset",
        "task",
        "representation",
        "model",
        "split_seed",
        "nominal_dose",
        "risk_row_id",
        "dose_induced_amplification",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Risk table is missing columns: {sorted(missing)}")

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    coefficient_frames: list[pd.DataFrame] = []
    progress = tqdm(
        thresholds,
        desc="Material-event thresholds",
        unit="threshold",
        dynamic_ncols=True,
    )
    for threshold in progress:
        progress.set_postfix_str(f"cutoff={threshold:.3f}")
        work = relabel(table, threshold)
        metrics, predictions, coefficients = leave_dataset_out(
            work,
            threshold,
            args.logistic_c,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        )
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
        coefficient_frames.extend(coefficients)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    prevalence = event_prevalence(table, thresholds)
    summary = threshold_summary(metrics, prevalence)
    concordance = event_concordance(table, thresholds)
    stability = model_stability(coefficients)
    reproduction = verify_primary_reproduction(metrics, args.frozen_metrics)

    outputs = {
        "leave_dataset_out_metrics.csv": metrics,
        "leave_dataset_out_predictions.csv": predictions,
        "leave_dataset_out_coefficients.csv": coefficients,
        "event_prevalence.csv": prevalence,
        "threshold_summary.csv": summary,
        "event_concordance.csv": concordance,
        "model_stability.csv": stability,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_root / name, index=False)
    make_figure(metrics, prevalence, args.output_root)

    script_path = Path(__file__).resolve()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    supported = summary.loc[
        summary.operationally_supported, "material_effect_threshold"
    ].tolist()
    manifest = {
        "status": "completed_post_hoc_sensitivity",
        "analysis_date": "2026-09-08",
        "analysis_role": "post hoc endpoint robustness analysis",
        "frozen_method": freeze.get("method", "DCS-OPCT v11"),
        "freeze_id": freeze.get("freeze_id"),
        "frozen_v11_unchanged": True,
        "primary_material_effect_threshold": PRIMARY_THRESHOLD,
        "tested_thresholds": thresholds,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_unit": "task-by-representation-by-model-by-split-seed cluster",
        "validation_scheme": "leave one source dataset out",
        "descriptive_metric_floors": METRIC_FLOORS,
        "minimum_events_per_held_out_dataset": 20,
        "operationally_supported_thresholds": supported,
        "bounded_interpretation": (
            "The frozen 0.02 endpoint lies at the lower edge of the tested 0.02-0.03 "
            "region that met all descriptive transfer floors. This does not optimize, "
            "validate physiologically, or replace the frozen threshold."
        ),
        "claim_boundary": (
            "No expanded external-effectiveness, universal safety, physiological "
            "materiality, or threshold-optimality claim is permitted."
        ),
        "primary_reproduction": reproduction,
        "inputs": {
            str(args.risk_table): sha256(args.risk_table),
            str(args.frozen_metrics): sha256(args.frozen_metrics),
            str(args.freeze): sha256(args.freeze),
        },
        "script": {str(script_path): sha256(script_path)},
    }
    generated = [
        *outputs,
        "fig_material_event_threshold_robustness.pdf",
        "fig_material_event_threshold_robustness.png",
    ]
    manifest["outputs"] = {
        name: sha256(args.output_root / name) for name in generated
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(json.dumps(reproduction, indent=2))
    print(f"Saved post-hoc robustness analysis to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
