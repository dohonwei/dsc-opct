from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_identity_shortcut_risk import (  # noqa: E402
    classification_metrics,
    classifier_features,
    make_classifier_pipeline,
    training_only_threshold,
)


CONFIG = ["dataset", "task", "axis", "representation", "model", "split_seed"]
THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/counterfactual_identity_dose_crossed_valid/predictions.csv"),
    )
    parser.add_argument(
        "--risk-table",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier/risk_learning_table.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dcs_opct_v11_endpoint_measurement_uncertainty_20260914"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def fold_bacc(frame: pd.DataFrame) -> float:
    return float(balanced_accuracy_score(frame.target, frame.probability.ge(0.5)))


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    if args.bootstrap_repetitions < 5000:
        raise ValueError("At least 5,000 bootstrap repetitions are required")
    predictions = pd.read_csv(args.predictions)
    risk = pd.read_csv(args.risk_table)
    risk = risk.loc[risk.dataset.isin(["DEAP", "MAHNOB-HCI"])].copy()
    unseen = predictions.loc[predictions.condition.eq("unseen")]
    dose = predictions.loc[predictions.condition.eq("dose")]
    unseen_scores = (
        unseen.groupby(CONFIG + ["fold"], sort=False).apply(fold_bacc)
        .rename("unseen_fold_bacc").reset_index()
    )
    dose_scores = (
        dose.groupby(CONFIG + ["nominal_dose", "fold"], sort=False).apply(fold_bacc)
        .rename("dose_fold_bacc").reset_index()
    )
    paired = dose_scores.merge(unseen_scores, on=CONFIG + ["fold"], validate="many_to_one")
    paired["fold_amplification"] = paired.dose_fold_bacc - paired.unseen_fold_bacc

    rng = np.random.default_rng(args.seed)
    rows = []
    groups = paired.groupby(CONFIG + ["nominal_dose"], sort=True)
    for key, group in tqdm(groups, total=groups.ngroups, desc="Endpoint fold bootstrap", unit="config"):
        values = group.fold_amplification.to_numpy(float)
        draws = values[rng.integers(0, len(values), size=(args.bootstrap_repetitions, len(values)))].mean(axis=1)
        rows.append({
            **dict(zip(CONFIG + ["nominal_dose"], key, strict=True)),
            "n_folds": len(values),
            "fold_mean_amplification": float(values.mean()),
            "fold_se_amplification": float(values.std(ddof=1) / np.sqrt(len(values))),
            "amplification_ci_low": float(np.quantile(draws, 0.025)),
            "amplification_ci_high": float(np.quantile(draws, 0.975)),
            "probability_material_event": float(np.mean(draws >= THRESHOLD)),
        })
    uncertainty = pd.DataFrame(rows)
    merged = risk.merge(uncertainty, on=CONFIG + ["nominal_dose"], validate="one_to_one")
    merged["uncertain_at_0_02"] = (
        (merged.amplification_ci_low < THRESHOLD) & (merged.amplification_ci_high >= THRESHOLD)
    )
    merged["lcb_material_event"] = merged.amplification_ci_low.ge(THRESHOLD).astype(int)
    merged["ucb_non_event"] = merged.amplification_ci_high.lt(THRESHOLD).astype(int)

    features = classifier_features()
    metric_rows = []
    for held_out in sorted(merged.dataset.unique()):
        train = merged.loc[merged.dataset.ne(held_out) & ~merged.uncertain_at_0_02].copy()
        test = merged.loc[merged.dataset.eq(held_out) & ~merged.uncertain_at_0_02].copy()
        decision_threshold, _ = training_only_threshold(train, 0.1)
        model, _ = make_classifier_pipeline(0.1)
        model.fit(train[features], train.material_optimism_event)
        probability = model.predict_proba(test[features])[:, 1]
        metrics = classification_metrics(
            test.material_optimism_event.to_numpy(int), probability, decision_threshold,
            float(train.material_optimism_event.mean()),
        )
        metric_rows.append({"held_out": held_out, "analysis": "exclude_ci_crossing_0_02", "n_train": len(train), "n_test": len(test), **metrics})
    metrics = pd.DataFrame(metric_rows)
    summary = merged.groupby("dataset").agg(
        configurations=("risk_row_id", "size"),
        original_events=("material_optimism_event", "sum"),
        uncertain_events=("uncertain_at_0_02", "sum"),
        lcb_events=("lcb_material_event", "sum"),
        median_event_probability=("probability_material_event", "median"),
    ).reset_index()

    args.output_root.mkdir(parents=True, exist_ok=False)
    paired.to_csv(args.output_root / "fold_level_paired_bacc.csv", index=False)
    merged.to_csv(args.output_root / "configuration_event_uncertainty.csv", index=False)
    metrics.to_csv(args.output_root / "uncertainty_exclusion_lodo_metrics.csv", index=False)
    summary.to_csv(args.output_root / "event_uncertainty_summary.csv", index=False)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "resampling_unit": "paired joint-unseen fold",
        "total_configurations": int(len(merged)),
        "uncertain_configurations": int(merged.uncertain_at_0_02.sum()),
        "uncertain_fraction": float(merged.uncertain_at_0_02.mean()),
        "claim_boundary": "Fold bootstrap propagates split-level endpoint variability; it is not a participant-level sampling distribution and is reported as sensitivity analysis.",
    }
    (args.output_root / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
