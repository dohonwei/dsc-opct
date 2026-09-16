from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm


CONFIG = ["dataset", "task", "axis", "representation", "model", "split_seed", "nominal_dose"]
BASE_CONFIG = CONFIG[:-1]
THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("outputs/counterfactual_identity_dose_crossed_valid/predictions.csv"),
    )
    parser.add_argument(
        "--risk-table", type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier/risk_learning_table.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dcs_opct_v11_crossed_cluster_endpoint_uncertainty_20260914_v2"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def multinomial_counts(
    rng: np.random.Generator, repetitions: int, levels: int
) -> np.ndarray:
    probabilities = np.full(levels, 1.0 / levels)
    return rng.multinomial(levels, probabilities, size=repetitions).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.bootstrap_repetitions < 5000:
        raise ValueError("At least 5,000 bootstrap repetitions are required")

    risk = pd.read_csv(args.risk_table)
    risk = risk.loc[risk.dataset.isin(["DEAP", "MAHNOB-HCI"])].copy()
    predictions = pd.read_csv(args.predictions)
    predictions = predictions.loc[predictions.dataset.isin(risk.dataset.unique())]
    all_dose = predictions.loc[predictions.condition.eq("dose")].copy()
    dose = all_dose.merge(
        risk[CONFIG].drop_duplicates(), on=CONFIG, how="inner", validate="many_to_one"
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    progress = tqdm(
        total=len(risk), desc="GPU crossed participant-stimulus bootstrap",
        unit="config", dynamic_ncols=True,
    )

    for dataset, configs in risk.groupby("dataset", sort=True):
        dataset_dose = all_dose.loc[all_dose.dataset.eq(dataset)]
        subjects = sorted(dataset_dose.subject_id.unique())
        stimuli = sorted(dataset_dose.trial_id.unique())
        subject_lookup = {value: index for index, value in enumerate(subjects)}
        stimulus_lookup = {value: index for index, value in enumerate(stimuli)}
        subject_counts = multinomial_counts(rng, args.bootstrap_repetitions, len(subjects))
        stimulus_counts = multinomial_counts(rng, args.bootstrap_repetitions, len(stimuli))

        for config in configs.sort_values(CONFIG).itertuples(index=False):
            base_mask = dataset_dose.nominal_dose.eq(0.0).to_numpy()
            for column in BASE_CONFIG:
                base_mask &= dataset_dose[column].to_numpy() == getattr(config, column)
            base = dataset_dose.loc[base_mask].sort_values("row_id")
            treated_mask = np.ones(len(dose), dtype=bool)
            for column in CONFIG:
                treated_mask &= dose[column].to_numpy() == getattr(config, column)
            treated = dose.loc[treated_mask].sort_values("row_id")
            if len(base) != len(treated) or not np.array_equal(base.row_id, treated.row_id):
                raise ValueError(f"Unpaired dose-zero rows for risk_row_id={config.risk_row_id}")
            if not np.array_equal(base.target, treated.target):
                raise ValueError(f"Target mismatch for risk_row_id={config.risk_row_id}")

            subject_index = np.array([subject_lookup[x] for x in base.subject_id])
            stimulus_index = np.array([stimulus_lookup[x] for x in base.trial_id])
            weights = subject_counts[:, subject_index] * stimulus_counts[:, stimulus_index]
            target = base.target.to_numpy(int)
            base_class = base.probability.to_numpy(float) >= 0.5
            dose_class = treated.probability.to_numpy(float) >= 0.5
            positive = target == 1
            negative = ~positive
            positive_delta = (dose_class.astype(float) - base_class.astype(float)) * positive
            negative_delta = (base_class.astype(float) - dose_class.astype(float)) * negative

            device = torch.device("cuda")
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
            positive_tensor = torch.as_tensor(positive.astype(np.float32), device=device)
            negative_tensor = torch.as_tensor(negative.astype(np.float32), device=device)
            positive_delta_tensor = torch.as_tensor(positive_delta.astype(np.float32), device=device)
            negative_delta_tensor = torch.as_tensor(negative_delta.astype(np.float32), device=device)
            positive_denominator = weight_tensor @ positive_tensor
            negative_denominator = weight_tensor @ negative_tensor
            valid = (positive_denominator > 0) & (negative_denominator > 0)
            draws = 0.5 * (
                (weight_tensor @ positive_delta_tensor) / positive_denominator
                + (weight_tensor @ negative_delta_tensor) / negative_denominator
            )
            draws = draws[valid].cpu().numpy()
            point = balanced_accuracy_score(target, dose_class) - balanced_accuracy_score(
                target, base_class
            )
            rows.append(
                {
                    "risk_row_id": int(config.risk_row_id),
                    **{column: getattr(config, column) for column in CONFIG},
                    "n_observations": len(base),
                    "n_subjects": len(subjects),
                    "n_stimuli": len(stimuli),
                    "valid_bootstraps": len(draws),
                    "paired_point_amplification": float(point),
                    "amplification_ci_low": float(np.quantile(draws, 0.025)),
                    "amplification_ci_high": float(np.quantile(draws, 0.975)),
                    "probability_material_event": float(np.mean(draws >= THRESHOLD)),
                }
            )
            progress.update(1)
    progress.close()

    endpoint = pd.DataFrame(rows)
    merged = risk.merge(endpoint, on=["risk_row_id", *CONFIG], validate="one_to_one")
    merged["point_reconstruction_error"] = (
        merged.paired_point_amplification - merged.dose_induced_amplification
    ).abs()
    merged["ci_crosses_threshold"] = (
        (merged.amplification_ci_low < THRESHOLD)
        & (merged.amplification_ci_high >= THRESHOLD)
    )
    merged["lcb_event"] = merged.amplification_ci_low.ge(THRESHOLD)
    summary = (
        merged.groupby("dataset")
        .agg(
            configurations=("risk_row_id", "size"),
            original_events=("material_optimism_event", "sum"),
            ci_crossing=("ci_crosses_threshold", "sum"),
            lcb_events=("lcb_event", "sum"),
            median_event_probability=("probability_material_event", "median"),
            maximum_reconstruction_error=("point_reconstruction_error", "max"),
        )
        .reset_index()
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    merged.to_csv(args.output_root / "crossed_cluster_event_uncertainty.csv", index=False)
    summary.to_csv(args.output_root / "crossed_cluster_event_summary.csv", index=False)
    report = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "resampling_unit": "crossed participant and physical-stimulus multinomial weights paired across conditions",
        "claim_boundary": "Repeated model fits and shared training data across configurations remain outside this crossed test-sample bootstrap.",
    }
    (args.output_root / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
