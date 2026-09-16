from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import run_protocol_model_benchmark as benchmark
from run_multiseed_protocol_sensitivity import add_one_two_sided_p, holm_adjust
from stimulus_identity_contract import CROSSED_DATASET_NAMES, stable_analysis_seed


MODELS = ("linear_logistic", "rbf_svm", "extra_trees", "hist_gradient_boosting")
CONTRASTS = {
    "subject_exposure_main_effect": {
        "unseen_both": -0.5,
        "seen_subject": 0.5,
        "seen_stimulus": -0.5,
        "seen_both": 0.5,
    },
    "stimulus_exposure_main_effect": {
        "unseen_both": -0.5,
        "seen_subject": -0.5,
        "seen_stimulus": 0.5,
        "seen_both": 0.5,
    },
    "subject_by_stimulus_interaction": {
        "unseen_both": 1.0,
        "seen_subject": -1.0,
        "seen_stimulus": -1.0,
        "seen_both": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factorial effects in the matched 2x2 identity design.")
    parser.add_argument(
        "--matched-root",
        type=Path,
        default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid"),
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def load_predictions(
    root: Path,
    models: list[str],
    datasets: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    manifest = pd.read_json(root / "run_manifest.json", typ="series")
    frames = []
    for split_seed in manifest.split_seeds:
        path = root / f"seed_{split_seed}" / "predictions.csv"
        frame = pd.read_csv(path, low_memory=False)
        frame = frame.loc[frame.model.isin(models)].copy()
        frame["subject_id"] = frame.subject_id.astype(str)
        frame["trial_id"] = frame.trial_id.astype(int)
        frame["split_seed"] = int(split_seed)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    admitted = set(CROSSED_DATASET_NAMES if datasets is None else datasets)
    invalid = sorted(admitted - CROSSED_DATASET_NAMES)
    if invalid:
        raise ValueError(f"Datasets are not admitted by the stimulus contract: {invalid}")
    predictions = predictions.loc[predictions.dataset.isin(admitted)].copy()
    if set(predictions.dataset.unique()) != admitted:
        raise ValueError(
            f"Requested factorial datasets {sorted(admitted)} do not match available admitted data "
            f"{sorted(predictions.dataset.unique())}"
        )
    key = ["split_seed", "dataset", "task", "exposure", "model", "row_id"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate matched OOF predictions")
    expected_exposures = set(next(iter(CONTRASTS.values())))
    observed_exposures = set(predictions.exposure)
    if observed_exposures != expected_exposures:
        raise ValueError(f"Expected exposures {expected_exposures}, observed {observed_exposures}")
    return predictions


def contrast_estimates(
    merged: pd.DataFrame,
    seeds: np.ndarray,
    models: list[str],
    weights: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    truth = merged.target.to_numpy(int)
    outputs = {contrast: np.empty(len(seeds), dtype=float) for contrast in CONTRASTS}
    for seed_index, split_seed in enumerate(seeds):
        model_scores = {exposure: [] for exposure in next(iter(CONTRASTS.values()))}
        for model in models:
            indices = np.flatnonzero(
                (merged.split_seed.to_numpy() == split_seed) & (merged.model.to_numpy() == model)
            )
            model_weights = None if weights is None else weights[indices]
            for exposure in model_scores:
                model_scores[exposure].append(
                    benchmark.balanced_accuracy(
                        truth[indices],
                        merged[f"probability_{exposure}"].to_numpy(float)[indices],
                        model_weights,
                    )
                )
        exposure_scores = {
            exposure: float(np.mean(scores)) for exposure, scores in model_scores.items()
        }
        for contrast, coefficients in CONTRASTS.items():
            outputs[contrast][seed_index] = sum(
                coefficients[exposure] * exposure_scores[exposure] for exposure in coefficients
            )
    return outputs


def analyze(
    predictions: pd.DataFrame,
    models: list[str],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    seed_rows = []
    groups = list(predictions.groupby(["dataset", "task"], sort=True))
    progress = tqdm(
        total=len(groups) * len(CONTRASTS),
        desc="Factorial triple-cluster bootstrap",
        unit="contrast",
        dynamic_ncols=True,
    )
    for dataset_task, group in groups:
        dataset, task = dataset_task
        merge_keys = ["split_seed", "model", "row_id", "subject_id", "trial_id", "target"]
        merged = None
        for exposure in next(iter(CONTRASTS.values())):
            exposure_frame = group.loc[group.exposure == exposure, merge_keys + ["probability"]].rename(
                columns={"probability": f"probability_{exposure}"}
            )
            merged = exposure_frame if merged is None else merged.merge(
                exposure_frame,
                on=merge_keys,
                validate="one_to_one",
            )
        seeds = np.sort(merged.split_seed.unique())
        observed = contrast_estimates(merged, seeds, models)
        for contrast, estimates in observed.items():
            for split_seed, estimate in zip(seeds, estimates, strict=True):
                seed_rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "contrast": contrast,
                        "split_seed": split_seed,
                        "balanced_accuracy_effect": estimate,
                    }
                )
        subject_codes, subjects = pd.factorize(merged.subject_id, sort=True)
        stimulus_codes, stimuli = pd.factorize(merged.trial_id, sort=True)
        rng = np.random.default_rng(stable_analysis_seed(seed, dataset, task, "factorial"))
        bootstrap = {contrast: np.empty(repetitions, dtype=float) for contrast in CONTRASTS}
        valid = 0
        for _ in range(repetitions):
            subject_frequency = np.bincount(
                rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)
            )
            stimulus_frequency = np.bincount(
                rng.integers(0, len(stimuli), len(stimuli)), minlength=len(stimuli)
            )
            weights = subject_frequency[subject_codes] * stimulus_frequency[stimulus_codes]
            estimates = contrast_estimates(merged, seeds, models, weights)
            if not all(np.isfinite(values).all() for values in estimates.values()):
                continue
            seed_frequency = np.bincount(
                rng.integers(0, len(seeds), len(seeds)), minlength=len(seeds)
            )
            for contrast, values in estimates.items():
                bootstrap[contrast][valid] = np.average(values, weights=seed_frequency)
            valid += 1
        if valid < repetitions * 0.95:
            raise RuntimeError(f"Only {valid}/{repetitions} valid replicates for {dataset}/{task}")
        for contrast, values in bootstrap.items():
            values = values[:valid]
            observed_values = observed[contrast]
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "contrast": contrast,
                    "n_models": len(models),
                    "n_split_seeds": len(seeds),
                    "balanced_accuracy_effect": np.mean(observed_values),
                    "split_seed_sd": np.std(observed_values, ddof=1),
                    "split_seed_min": np.min(observed_values),
                    "split_seed_max": np.max(observed_values),
                    "positive_seed_fraction": np.mean(observed_values > 0),
                    "ci_low": np.quantile(values, 0.025),
                    "ci_high": np.quantile(values, 0.975),
                    "p_bootstrap": add_one_two_sided_p(values),
                    "bootstrap_valid": valid,
                }
            )
            progress.update(1)
    progress.close()
    output = pd.DataFrame(rows)
    output["p_holm_within_contrast"] = np.nan
    for _, indices in output.groupby("contrast").groups.items():
        positions = np.asarray(list(indices), dtype=int)
        output.loc[positions, "p_holm_within_contrast"] = holm_adjust(
            output.loc[positions, "p_bootstrap"].to_numpy(float)
        )
    return pd.DataFrame(seed_rows), output


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.matched_root, args.models)
    per_seed, effects = analyze(
        predictions,
        args.models,
        args.bootstrap_repetitions,
        args.seed,
    )
    per_seed.to_csv(args.matched_root / "factorial_effects_per_seed_crossed_valid.csv", index=False)
    effects.to_csv(args.matched_root / "factorial_effects_crossed_valid.csv", index=False)
    manifest = {
        "input_root": str(args.matched_root),
        "admitted_datasets": sorted(predictions.dataset.unique()),
        "models": args.models,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "bootstrap_seed_scheme": "sha256(base_seed|dataset|task|factorial)",
        "bootstrap_clusters": ["subject_id", "trial_id", "split_seed"],
        "multiplicity_family": "four dataset-task comparisons within each factorial contrast",
    }
    (args.matched_root / "factorial_effects_crossed_valid_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(effects.to_string(index=False))


if __name__ == "__main__":
    main()
