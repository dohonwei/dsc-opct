from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm.auto import tqdm

from analyze_matched_factorial_effects import CONTRASTS, contrast_estimates, load_predictions
from run_multiseed_protocol_sensitivity import holm_adjust
from stimulus_identity_contract import stable_analysis_seed


DEFAULT_MODELS = ("linear_logistic", "rbf_svm", "extra_trees", "hist_gradient_boosting")
DEFAULT_DATASETS = ("DEAP", "MAHNOB-HCI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc SESOI sensitivity analysis for matched factorial effects."
    )
    parser.add_argument(
        "--matched-root",
        type=Path,
        default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid"),
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets admitted by the physical stimulus-identity contract.",
    )
    parser.add_argument("--sesoi", nargs="+", type=float, default=[0.02, 0.03, 0.05])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def merged_exposure_predictions(group: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["split_seed", "model", "row_id", "subject_id", "trial_id", "target"]
    merged = None
    for exposure in next(iter(CONTRASTS.values())):
        frame = group.loc[
            group.exposure == exposure, merge_keys + ["probability"]
        ].rename(columns={"probability": f"probability_{exposure}"})
        merged = frame if merged is None else merged.merge(
            frame,
            on=merge_keys,
            validate="one_to_one",
        )
    if merged is None:
        raise ValueError("No exposure predictions were available")
    return merged


def cluster_bootstrap_effects(
    merged: pd.DataFrame,
    models: list[str],
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    seeds = np.sort(merged.split_seed.unique())
    observed_by_seed = contrast_estimates(merged, seeds, models)
    observed = {name: float(values.mean()) for name, values in observed_by_seed.items()}

    subject_codes, subjects = pd.factorize(merged.subject_id, sort=True)
    stimulus_codes, stimuli = pd.factorize(merged.trial_id, sort=True)
    bootstrap = {name: np.empty(repetitions, dtype=float) for name in CONTRASTS}
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
        if seed_frequency.sum() == 0:
            continue
        for name, values in estimates.items():
            bootstrap[name][valid] = np.average(values, weights=seed_frequency)
        valid += 1

    if valid < repetitions * 0.95:
        raise RuntimeError(f"Only {valid}/{repetitions} bootstrap replicates were valid")
    return observed, {name: values[:valid] for name, values in bootstrap.items()}


def tost_from_bootstrap(
    estimate: float,
    bootstrap: np.ndarray,
    margin: float,
) -> dict[str, float | bool]:
    bootstrap_se = float(np.std(bootstrap, ddof=1))
    if bootstrap_se <= 0 or not np.isfinite(bootstrap_se):
        raise ValueError("Bootstrap standard error is not finite and positive")
    p_above_lower = float(norm.sf((estimate + margin) / bootstrap_se))
    p_below_upper = float(norm.sf((margin - estimate) / bootstrap_se))
    p_tost = max(p_above_lower, p_below_upper)
    ci90_low, ci90_high = np.quantile(bootstrap, [0.05, 0.95])
    return {
        "bootstrap_se": bootstrap_se,
        "ci90_low": float(ci90_low),
        "ci90_high": float(ci90_high),
        "p_lower_bound": p_above_lower,
        "p_upper_bound": p_below_upper,
        "p_tost": p_tost,
        "ci90_within_margin": bool(ci90_low > -margin and ci90_high < margin),
    }


def analyze(args: argparse.Namespace) -> pd.DataFrame:
    predictions = load_predictions(args.matched_root, args.models)
    requested = list(dict.fromkeys(args.datasets))
    available = set(predictions.dataset.unique())
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"Requested datasets are absent from predictions: {missing}")
    predictions = predictions.loc[predictions.dataset.isin(requested)].copy()
    groups = list(predictions.groupby(["dataset", "task"], sort=True))
    rows: list[dict[str, object]] = []
    progress = tqdm(
        total=len(groups),
        desc="Factorial SESOI bootstrap",
        unit="dataset-task",
        dynamic_ncols=True,
    )
    for dataset_task, group in groups:
        dataset, task = dataset_task
        merged = merged_exposure_predictions(group)
        observed, bootstrap = cluster_bootstrap_effects(
            merged,
            list(args.models),
            args.bootstrap_repetitions,
            np.random.default_rng(stable_analysis_seed(args.seed, dataset, task, "sesoi")),
        )
        for contrast in CONTRASTS:
            for margin in sorted(set(args.sesoi)):
                result = tost_from_bootstrap(observed[contrast], bootstrap[contrast], margin)
                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "contrast": contrast,
                        "sesoi_margin": margin,
                        "balanced_accuracy_effect": observed[contrast],
                        "bootstrap_valid": len(bootstrap[contrast]),
                        **result,
                    }
                )
        progress.update(1)
    progress.close()

    output = pd.DataFrame(rows)
    output["p_tost_holm_within_contrast_margin"] = np.nan
    for _, indices in output.groupby(["contrast", "sesoi_margin"]).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        output.loc[positions, "p_tost_holm_within_contrast_margin"] = holm_adjust(
            output.loc[positions, "p_tost"].to_numpy(float)
        )
    output["equivalent_unadjusted"] = output.p_tost < 0.05
    output["equivalent_holm"] = output.p_tost_holm_within_contrast_margin < 0.05
    return output


def main() -> None:
    args = parse_args()
    if any(margin <= 0 for margin in args.sesoi):
        raise ValueError("Every SESOI margin must be positive")
    output = analyze(args)
    output_path = args.matched_root / "factorial_effect_equivalence_crossed_valid.csv"
    output.to_csv(output_path, index=False)
    manifest = {
        "input_root": str(args.matched_root),
        "admitted_datasets": list(dict.fromkeys(args.datasets)),
        "models": list(args.models),
        "sesoi_margins": sorted(set(args.sesoi)),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "bootstrap_seed_scheme": "sha256(base_seed|dataset|task|sesoi)",
        "multiplicity_family": (
            f"{output[['dataset', 'task']].drop_duplicates().shape[0]} dataset-task "
            "comparisons, separately within each factorial contrast and SESOI margin"
        ),
        "exclusion_note": (
            "EPPVR is excluded because physical stimulus identity cannot be recovered "
            "from the available records."
        ),
    }
    manifest_path = args.matched_root / "factorial_effect_equivalence_crossed_valid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stimulus = output.loc[output.contrast == "stimulus_exposure_main_effect"]
    print(stimulus.to_string(index=False))
    print(f"\nSaved: {output_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
