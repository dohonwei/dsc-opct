from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import run_matched_exposure_benchmark as matched
import run_paired_block_exposure_benchmark as paired
import run_protocol_model_benchmark as benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metadata-only priors for matched identity exposures.")
    parser.add_argument("--datasets", nargs="+", default=list(matched.DEFAULT_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(matched.DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--design",
        choices=["independent", "paired-block"],
        default="paired-block",
        help="Reconstruct the training sets used by the selected matched design.",
    )
    parser.add_argument(
        "--matched-root",
        type=Path,
        default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid"),
    )
    return parser.parse_args()


def mapped_prior(
    frame: pd.DataFrame,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    identity: str,
) -> tuple[np.ndarray, np.ndarray]:
    global_probability = float(labels[train].mean())
    prior = pd.Series(labels[train]).groupby(frame.iloc[train][identity].to_numpy()).mean()
    probability = frame.iloc[test][identity].map(prior).to_numpy(float)
    available = np.isfinite(probability)
    probability[~available] = global_probability
    return probability, available


def build_predictions(args: argparse.Namespace) -> pd.DataFrame:
    total = len(args.seeds) * len(args.datasets) * len(args.tasks) * args.folds**2 * len(matched.EXPOSURES)
    progress = tqdm(total=total, desc="Matched identity priors", unit="fold", dynamic_ncols=True)
    rows = []
    for split_seed in args.seeds:
        for specification in args.datasets:
            dataset, path = specification.split("=", 1)
            frame = pd.read_csv(path).reset_index(drop=True)
            frame["row_id"] = np.arange(len(frame))
            subject_fold, stimulus_fold = matched.fold_assignments(frame, args.folds, split_seed)
            for task in args.tasks:
                labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
                for subject_test in range(args.folds):
                    for stimulus_test in range(args.folds):
                        train_set_builder = (
                            matched.train_sets_for_fold if args.design == "independent" else paired.paired_train_sets
                        )
                        test, train_sets, _ = train_set_builder(
                            frame,
                            labels,
                            subject_fold,
                            stimulus_fold,
                            subject_test,
                            stimulus_test,
                            dataset,
                            task,
                            split_seed,
                        )
                        fold = f"s{subject_test}_t{stimulus_test}"
                        for exposure, train in train_sets.items():
                            global_probability = np.full(len(test), labels[train].mean(), dtype=float)
                            subject_probability, subject_available = mapped_prior(
                                frame, labels, train, test, "subject_id"
                            )
                            stimulus_probability, stimulus_available = mapped_prior(
                                frame, labels, train, test, "trial_id"
                            )
                            for model, probability, available in (
                                ("global_prevalence", global_probability, np.ones(len(test), dtype=bool)),
                                ("subject_label_prior", subject_probability, subject_available),
                                ("stimulus_label_prior", stimulus_probability, stimulus_available),
                            ):
                                for index, estimate, is_available in zip(test, probability, available, strict=True):
                                    rows.append(
                                        {
                                            "split_seed": split_seed,
                                            "dataset": dataset,
                                            "task": task,
                                            "exposure": exposure,
                                            "model": model,
                                            "fold": fold,
                                            "row_id": int(frame.iloc[index].row_id),
                                            "subject_id": str(frame.iloc[index].subject_id),
                                            "trial_id": int(frame.iloc[index].trial_id),
                                            "target": int(labels[index]),
                                            "probability": float(estimate),
                                            "identity_available": bool(is_available),
                                        }
                                    )
                            progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def validate_predictions(predictions: pd.DataFrame, args: argparse.Namespace) -> None:
    key = ["split_seed", "dataset", "task", "exposure", "model", "row_id"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate metadata-prior OOF predictions")
    if not np.isfinite(predictions.probability.to_numpy(float)).all():
        raise ValueError("Non-finite metadata-prior probability")
    expected_coverage = {
        ("unseen_both", "subject_label_prior"): 0.0,
        ("unseen_both", "stimulus_label_prior"): 0.0,
        ("seen_subject", "subject_label_prior"): 1.0,
        ("seen_subject", "stimulus_label_prior"): 0.0,
        ("seen_stimulus", "subject_label_prior"): 0.0,
        ("seen_stimulus", "stimulus_label_prior"): 1.0,
        ("seen_both", "subject_label_prior"): 1.0,
        ("seen_both", "stimulus_label_prior"): 1.0,
    }
    observed = predictions.groupby(["exposure", "model"]).identity_available.mean()
    for key_value, expected in expected_coverage.items():
        if not np.isclose(observed.loc[key_value], expected):
            raise ValueError(f"Unexpected prior coverage for {key_value}: {observed.loc[key_value]}")
    dataset_counts = {
        dataset: len(pd.read_csv(path, usecols=["subject_id"]))
        for dataset, path in (item.split("=", 1) for item in args.datasets)
    }
    groups = predictions.groupby(["split_seed", "dataset", "task", "exposure", "model"]).size()
    expected_groups = len(args.seeds) * len(args.datasets) * len(args.tasks) * len(matched.EXPOSURES) * 3
    if len(groups) != expected_groups:
        raise ValueError(f"Observed {len(groups)} groups, expected {expected_groups}")
    for (_, dataset, _, _, _), count in groups.items():
        if count != dataset_counts[dataset]:
            raise ValueError(f"Incomplete OOF prior predictions for {dataset}: {count}")


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["split_seed", "dataset", "task", "exposure", "model"]):
        rows.append(
            {
                "split_seed": keys[0],
                "dataset": keys[1],
                "task": keys[2],
                "exposure": keys[3],
                "model": keys[4],
                "balanced_accuracy": benchmark.balanced_accuracy(
                    group.target.to_numpy(int), group.probability.to_numpy(float)
                ),
                "identity_coverage": group.identity_available.mean(),
            }
        )
    return pd.DataFrame(rows)


def diagnostic_table(summary: pd.DataFrame, matched_summary: pd.DataFrame) -> pd.DataFrame:
    physiology_models = list(matched.DEFAULT_MODELS)
    physiology = (
        matched_summary.loc[matched_summary.model.isin(physiology_models)]
        .groupby(["split_seed", "dataset", "task", "exposure"])
        .pooled_balanced_accuracy.mean()
        .rename("physiology_panel_balanced_accuracy")
        .reset_index()
    )
    physiology_wide = physiology.pivot(
        index=["split_seed", "dataset", "task"],
        columns="exposure",
        values="physiology_panel_balanced_accuracy",
    )
    prior_wide = summary.pivot(
        index=["split_seed", "dataset", "task"],
        columns=["exposure", "model"],
        values="balanced_accuracy",
    )
    rows = []
    for index in physiology_wide.index:
        split_seed, dataset, task = index
        unseen = physiology_wide.loc[index, "unseen_both"]
        for exposure, prior_model in (
            ("seen_subject", "subject_label_prior"),
            ("seen_stimulus", "stimulus_label_prior"),
        ):
            global_score = prior_wide.loc[index, (exposure, "global_prevalence")]
            prior_score = prior_wide.loc[index, (exposure, prior_model)]
            physiology_score = physiology_wide.loc[index, exposure]
            rows.append(
                {
                    "split_seed": split_seed,
                    "dataset": dataset,
                    "task": task,
                    "exposure": exposure,
                    "identity_prior": prior_model,
                    "global_balanced_accuracy": global_score,
                    "identity_prior_balanced_accuracy": prior_score,
                    "identity_prior_opportunity": prior_score - global_score,
                    "physiology_panel_balanced_accuracy": physiology_score,
                    "physiology_exposure_delta": physiology_score - unseen,
                }
            )
    per_seed = pd.DataFrame(rows)
    aggregated = (
        per_seed.groupby(["dataset", "task", "exposure", "identity_prior"])
        .agg(
            n_split_seeds=("split_seed", "nunique"),
            identity_prior_balanced_accuracy=("identity_prior_balanced_accuracy", "mean"),
            identity_prior_opportunity=("identity_prior_opportunity", "mean"),
            physiology_exposure_delta=("physiology_exposure_delta", "mean"),
            physiology_delta_min=("physiology_exposure_delta", "min"),
            physiology_delta_max=("physiology_exposure_delta", "max"),
        )
        .reset_index()
    )
    return per_seed, aggregated


def main() -> None:
    args = parse_args()
    predictions = build_predictions(args)
    validate_predictions(predictions, args)
    predictions.to_csv(args.matched_root / "identity_prior_predictions.csv", index=False)
    summary = summarize(predictions)
    summary.to_csv(args.matched_root / "identity_prior_summary.csv", index=False)
    matched_summary = pd.read_csv(args.matched_root / "per_seed_summary.csv")
    per_seed, diagnostic = diagnostic_table(summary, matched_summary)
    per_seed.to_csv(args.matched_root / "identity_opportunity_per_seed.csv", index=False)
    diagnostic.to_csv(args.matched_root / "identity_opportunity_summary.csv", index=False)
    print(diagnostic.to_string(index=False))


if __name__ == "__main__":
    main()
