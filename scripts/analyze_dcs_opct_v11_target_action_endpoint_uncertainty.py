from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from develop_witness_gated_covariance_opct_v9 import load_development  # noqa: E402


CONFIG = ["task", "axis", "representation", "model", "split_seed", "nominal_dose"]
BASE_CONFIG = CONFIG[:-1]
THRESHOLD = 0.02
PREDICTION_PATHS = {
    "EPPVR": [Path("outputs/eppvr_subject_shortcut_external_validation/predictions.csv")],
    "CASE": [Path("outputs/case_counterfactual_identity_dose/predictions.csv")],
    "CEAP": [Path("outputs/ceap_counterfactual_identity_dose/predictions.csv")],
    "SEED-IV": [
        Path("outputs/seediv_arousal_counterfactual_external/predictions.csv"),
        Path("outputs/seediv_valence_counterfactual_external/predictions.csv"),
    ],
    "DREAMER": [Path("outputs/dreamer_counterfactual_external/predictions.csv")],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inductive-predictions", type=Path,
        default=Path(
            "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/"
            "inductive_heldout_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def counts(rng: np.random.Generator, repetitions: int, levels: int) -> np.ndarray:
    return rng.multinomial(
        levels, np.full(levels, 1.0 / levels), size=repetitions
    ).astype(np.float32)


def normalize_predictions(dataset: str, paths: list[Path]) -> pd.DataFrame:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{dataset}: missing prediction files: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if "dataset" not in frame:
        frame["dataset"] = dataset
    if "axis" not in frame:
        frame["axis"] = "subject"
    if dataset == "EPPVR":
        frame = frame.rename(columns={"record_id": "trial_id"})
    if dataset == "CEAP":
        frame["dataset"] = "CEAP"
    duplicate_keys = [*CONFIG, "condition", "row_id"]
    if frame.duplicated(duplicate_keys).any():
        raise ValueError(f"{dataset}: duplicate prediction rows after file merge")
    return frame


def verify_configuration_coverage(
    dataset: str, configurations: pd.DataFrame, predictions: pd.DataFrame
) -> None:
    expected = configurations[CONFIG].drop_duplicates()
    available = predictions.loc[
        predictions.condition.eq("dose"), CONFIG
    ].drop_duplicates()
    coverage = expected.merge(available, on=CONFIG, how="left", indicator=True)
    missing = coverage.loc[coverage._merge.ne("both"), CONFIG]
    if not missing.empty:
        raise ValueError(
            f"{dataset}: {len(missing)} configurations lack dose predictions; "
            f"examples={missing.head(5).to_dict('records')}"
        )


def bootstrap_soft_events(
    dataset: str,
    configurations: pd.DataFrame,
    predictions: pd.DataFrame,
    repetitions: int,
    rng: np.random.Generator,
    progress: tqdm,
) -> pd.DataFrame:
    all_dose = predictions.loc[predictions.condition.eq("dose")].copy()
    subjects = sorted(all_dose.subject_id.unique())
    stimuli = sorted(all_dose.trial_id.unique())
    subject_lookup = {value: index for index, value in enumerate(subjects)}
    stimulus_lookup = {value: index for index, value in enumerate(stimuli)}
    subject_counts = counts(rng, repetitions, len(subjects))
    stimulus_counts = None if dataset == "EPPVR" else counts(rng, repetitions, len(stimuli))
    rows = []
    for config in configurations.sort_values(CONFIG).itertuples(index=False):
        baseline_mask = all_dose.nominal_dose.eq(0.0).to_numpy()
        treated_mask = np.ones(len(all_dose), dtype=bool)
        for column in BASE_CONFIG:
            value = getattr(config, column)
            baseline_mask &= all_dose[column].to_numpy() == value
            treated_mask &= all_dose[column].to_numpy() == value
        treated_mask &= all_dose.nominal_dose.to_numpy() == config.nominal_dose
        baseline = all_dose.loc[baseline_mask].sort_values("row_id")
        treated = all_dose.loc[treated_mask].sort_values("row_id")
        if len(baseline) != len(treated) or not np.array_equal(baseline.row_id, treated.row_id):
            raise ValueError(f"{dataset}: unpaired rows for configuration {config}")
        target = baseline.target.to_numpy(int)
        baseline_class = baseline.probability.to_numpy(float) >= 0.5
        treated_class = treated.probability.to_numpy(float) >= 0.5
        if baseline.empty:
            raise ValueError(f"{dataset}: empty matched configuration {config}")
        sidx = np.asarray(
            [subject_lookup[x] for x in baseline.subject_id], dtype=np.int64
        )
        weights = subject_counts[:, sidx]
        if stimulus_counts is not None:
            tidx = np.asarray(
                [stimulus_lookup[x] for x in baseline.trial_id], dtype=np.int64
            )
            weights = weights * stimulus_counts[:, tidx]
        positive = target == 1
        negative = ~positive
        positive_delta = (treated_class.astype(float) - baseline_class.astype(float)) * positive
        negative_delta = (baseline_class.astype(float) - treated_class.astype(float)) * negative
        device = torch.device("cuda")
        w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        pos = torch.as_tensor(positive.astype(np.float32), device=device)
        neg = torch.as_tensor(negative.astype(np.float32), device=device)
        pos_delta = torch.as_tensor(positive_delta.astype(np.float32), device=device)
        neg_delta = torch.as_tensor(negative_delta.astype(np.float32), device=device)
        pos_den = w @ pos
        neg_den = w @ neg
        valid = (pos_den > 0) & (neg_den > 0)
        draws = 0.5 * ((w @ pos_delta) / pos_den + (w @ neg_delta) / neg_den)
        draws = draws[valid].cpu().numpy()
        point = balanced_accuracy_score(target, treated_class) - balanced_accuracy_score(
            target, baseline_class
        )
        rows.append(
            {
                "dataset": dataset,
                **{column: getattr(config, column) for column in CONFIG},
                "bootstrap_structure": (
                    "participant" if dataset == "EPPVR" else "participant_x_stimulus"
                ),
                "paired_point_amplification": float(point),
                "amplification_ci_low": float(np.quantile(draws, 0.025)),
                "amplification_ci_high": float(np.quantile(draws, 0.975)),
                "probability_material_event": float(np.mean(draws >= THRESHOLD)),
            }
        )
        progress.update(1)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.bootstrap_repetitions < 5000:
        raise ValueError("At least 5,000 bootstrap repetitions are required")
    development = load_development()
    inductive = pd.read_csv(args.inductive_predictions)
    rng = np.random.default_rng(args.seed)
    prepared = {}
    for dataset, (raw, _) in sorted(development.items()):
        prediction = normalize_predictions(dataset, PREDICTION_PATHS[dataset])
        verify_configuration_coverage(dataset, raw, prediction)
        prepared[dataset] = prediction
    total = sum(len(value[0]) for value in development.values())
    progress = tqdm(
        total=total, desc="GPU target endpoint uncertainty", unit="config", dynamic_ncols=True
    )
    endpoint_frames = []
    target_frames = []
    for dataset, (raw, _) in sorted(development.items()):
        target = raw.copy().reset_index().rename(columns={"index": "row_index"})
        target["dataset"] = dataset
        prediction = prepared[dataset]
        endpoint_frames.append(
            bootstrap_soft_events(
                dataset, target[CONFIG], prediction, args.bootstrap_repetitions, rng, progress
            )
        )
        target_frames.append(target)
    progress.close()
    endpoints = pd.concat(endpoint_frames, ignore_index=True)
    targets = pd.concat(target_frames, ignore_index=True)
    target = targets.merge(endpoints, on=["dataset", *CONFIG], validate="one_to_one")
    target = target.merge(
        inductive[
            ["dataset", "row_index", "partition", "probability_identity", "probability_inductive_action"]
        ],
        on=["dataset", "row_index"],
        suffixes=("_archived", "_inductive"),
        validate="one_to_one",
    )
    target["point_reconstruction_error"] = (
        target.paired_point_amplification - target.dose_induced_amplification
    ).abs()
    soft = target.probability_material_event
    target["soft_brier_gain"] = (
        (soft - target.probability_identity_inductive) ** 2
        - (soft - target.probability_inductive_action) ** 2
    )
    target["hard_brier_gain"] = (
        (target.material_optimism_event - target.probability_identity_inductive) ** 2
        - (target.material_optimism_event - target.probability_inductive_action) ** 2
    )
    heldout = target.loc[target.partition.eq("heldout")]
    summary = (
        heldout.groupby("dataset")
        .agg(
            configurations=("row_index", "size"),
            hard_event_prevalence=("material_optimism_event", "mean"),
            soft_event_prevalence=("probability_material_event", "mean"),
            hard_brier_gain=("hard_brier_gain", "mean"),
            soft_brier_gain=("soft_brier_gain", "mean"),
            maximum_reconstruction_error=("point_reconstruction_error", "max"),
        )
        .reset_index()
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    target.to_csv(args.output_root / "target_configuration_soft_events.csv", index=False)
    summary.to_csv(args.output_root / "inductive_action_soft_endpoint_summary.csv", index=False)
    report = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "claim_boundary": "Soft-event gains are post-hoc measurement-error sensitivities. EPPVR uses participant clustering because no shared physical-stimulus identifier is verified.",
    }
    (args.output_root / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
