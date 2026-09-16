from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import run_protocol_model_benchmark as benchmark
from stimulus_identity_contract import (
    CROSSED_UNIFIED_DATASETS,
    require_crossed_datasets,
    stable_analysis_seed,
)


DEFAULT_DATASETS = CROSSED_UNIFIED_DATASETS
DEFAULT_MODELS = (
    "linear_logistic",
    "rbf_svm",
    "extra_trees",
    "hist_gradient_boosting",
)
DEFAULT_SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)
COMPARISONS = (("random", "dual"), ("subject", "dual"), ("stimulus", "dual"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split-seed sensitivity analysis for the protocol benchmark.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--feature-prefix", default="stimulus__")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--existing-run-root",
        type=Path,
        default=Path("outputs/unified_protocol_model_benchmark"),
        help="Compatible single-seed run that may be reused.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/unified_protocol_model_benchmark_multiseed_crossed_valid"),
    )
    parser.add_argument("--force", action="store_true", help="Refit seeds already present under output-root.")
    return parser.parse_args()


def expected_rows_by_dataset(dataset_specs: list[str]) -> dict[str, int]:
    counts = {}
    for specification in dataset_specs:
        dataset, path = specification.split("=", 1)
        counts[dataset] = len(pd.read_csv(path, usecols=["subject_id"]))
    return counts


def validate_predictions(
    predictions: pd.DataFrame,
    seed: int,
    dataset_specs: list[str],
    tasks: list[str],
    models: list[str],
) -> None:
    required = {
        "dataset",
        "task",
        "protocol",
        "model",
        "row_id",
        "subject_id",
        "trial_id",
        "target",
        "probability",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Seed {seed} predictions lack columns: {sorted(missing)}")
    if not np.isfinite(predictions.probability.to_numpy(float)).all():
        raise ValueError(f"Seed {seed} contains non-finite probabilities")
    if predictions.duplicated(["dataset", "task", "protocol", "model", "row_id"]).any():
        raise ValueError(f"Seed {seed} contains duplicate OOF rows")
    expected_datasets = {item.split("=", 1)[0] for item in dataset_specs}
    observed_datasets = set(predictions.dataset.unique())
    if observed_datasets != expected_datasets:
        raise ValueError(
            f"Seed {seed} dataset mismatch: expected {sorted(expected_datasets)}, "
            f"observed {sorted(observed_datasets)}. Cached predictions are incompatible."
        )
    expected_models = set(models) | {"global_prevalence", "stimulus_prior"}
    observed_models = set(predictions.model)
    if observed_models != expected_models:
        raise ValueError(f"Seed {seed} model mismatch: expected {expected_models}, observed {observed_models}")
    expected_counts = expected_rows_by_dataset(dataset_specs)
    group_counts = predictions.groupby(["dataset", "task", "protocol", "model"]).size()
    for (dataset, task, protocol, model), count in group_counts.items():
        if task not in tasks or protocol not in benchmark.PROTOCOLS:
            raise ValueError(f"Unexpected prediction group for seed {seed}: {dataset}/{task}/{protocol}/{model}")
        if count != expected_counts[dataset]:
            raise ValueError(
                f"Incomplete OOF predictions for seed {seed}: {dataset}/{task}/{protocol}/{model} "
                f"has {count}, expected {expected_counts[dataset]}"
            )
    expected_groups = len(expected_counts) * len(tasks) * len(benchmark.PROTOCOLS) * len(expected_models)
    if len(group_counts) != expected_groups:
        raise ValueError(f"Seed {seed} has {len(group_counts)} groups, expected {expected_groups}")


def compatible_existing_run(root: Path, args: argparse.Namespace, seed: int) -> Path | None:
    manifest_path = root / "run_manifest.json"
    prediction_path = root / "predictions.csv"
    if not manifest_path.exists() or not prediction_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = (
        manifest.get("seed") == seed,
        manifest.get("datasets") == args.datasets,
        manifest.get("tasks") == args.tasks,
        manifest.get("protocols") == list(benchmark.PROTOCOLS),
        manifest.get("models") == args.models,
        manifest.get("folds") == args.folds,
        manifest.get("feature_prefix") == args.feature_prefix,
    )
    return prediction_path if all(checks) else None


def load_or_fit_seed(args: argparse.Namespace, seed: int) -> tuple[pd.DataFrame, str]:
    seed_root = args.output_root / f"seed_{seed}"
    prediction_path = seed_root / "predictions.csv"
    source = ""
    if prediction_path.exists() and not args.force:
        predictions = pd.read_csv(prediction_path, low_memory=False)
        source = str(prediction_path)
    else:
        existing = None if args.force else compatible_existing_run(args.existing_run_root, args, seed)
        if existing is not None:
            predictions = pd.read_csv(existing, low_memory=False)
            source = str(existing)
        else:
            run_args = argparse.Namespace(
                datasets=args.datasets,
                tasks=args.tasks,
                protocols=list(benchmark.PROTOCOLS),
                models=args.models,
                feature_prefix=args.feature_prefix,
                folds=args.folds,
                seed=seed,
                n_jobs=args.n_jobs,
            )
            predictions = benchmark.run_benchmark(run_args)
            seed_root.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(prediction_path, index=False)
            source = str(prediction_path)
    validate_predictions(predictions, seed, args.datasets, args.tasks, args.models)
    return predictions, source


def per_seed_panel_effects(summary: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    candidate = summary.loc[summary.model.isin(models)].copy()
    rows = []
    for (seed, dataset, task), group in candidate.groupby(["split_seed", "dataset", "task"]):
        table = group.pivot(index="model", columns="protocol", values="pooled_balanced_accuracy")
        for left_protocol, right_protocol in COMPARISONS:
            deltas = table[left_protocol] - table[right_protocol]
            rows.append(
                {
                    "split_seed": seed,
                    "dataset": dataset,
                    "task": task,
                    "left_protocol": left_protocol,
                    "right_protocol": right_protocol,
                    "mean_balanced_accuracy_delta": deltas.mean(),
                    "min_model_delta": deltas.min(),
                    "max_model_delta": deltas.max(),
                    "positive_model_fraction": np.mean(deltas > 0),
                }
            )
    return pd.DataFrame(rows)


def add_one_two_sided_p(values: np.ndarray) -> float:
    lower = (np.sum(values <= 0) + 1) / (len(values) + 1)
    upper = (np.sum(values >= 0) + 1) / (len(values) + 1)
    return float(min(1.0, 2 * min(lower, upper)))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[position])
        adjusted[position] = min(running, 1.0)
    return adjusted


def crossed_seed_panel_bootstrap(
    predictions: pd.DataFrame,
    models: list[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    groups = list(predictions.groupby(["dataset", "task"], sort=True))
    progress = tqdm(
        total=len(groups) * len(COMPARISONS),
        desc="Subject-stimulus-seed bootstrap",
        unit="contrast",
        dynamic_ncols=True,
    )
    for dataset_task, group in groups:
        dataset, task = dataset_task
        reference = group.loc[(group.protocol == "dual") & group.model.isin(models)]
        for comparison_index, (left_protocol, right_protocol) in enumerate(COMPARISONS):
            left = group.loc[(group.protocol == left_protocol) & group.model.isin(models)]
            merged = left.merge(
                reference,
                on=["split_seed", "model", "row_id", "subject_id", "trial_id", "target"],
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            subject_codes, subjects = pd.factorize(merged.subject_id, sort=True)
            stimulus_codes, stimuli = pd.factorize(merged.trial_id, sort=True)
            seeds = np.sort(merged.split_seed.unique())
            truth = merged.target.to_numpy(int)
            left_probability = merged.probability_left.to_numpy(float)
            right_probability = merged.probability_right.to_numpy(float)
            seed_model_indices = {
                (split_seed, model): np.flatnonzero(
                    (merged.split_seed.to_numpy() == split_seed) & (merged.model.to_numpy() == model)
                )
                for split_seed in seeds
                for model in models
            }

            def seed_panel_deltas(weights: np.ndarray | None = None) -> np.ndarray:
                estimates = np.empty(len(seeds), dtype=float)
                for seed_index, split_seed in enumerate(seeds):
                    model_deltas = []
                    for model in models:
                        indices = seed_model_indices[(split_seed, model)]
                        model_weights = None if weights is None else weights[indices]
                        left_score = benchmark.balanced_accuracy(
                            truth[indices], left_probability[indices], model_weights
                        )
                        right_score = benchmark.balanced_accuracy(
                            truth[indices], right_probability[indices], model_weights
                        )
                        model_deltas.append(left_score - right_score)
                    estimates[seed_index] = np.mean(model_deltas)
                return estimates

            observed_by_seed = seed_panel_deltas()
            observed = float(np.mean(observed_by_seed))
            rng = np.random.default_rng(
                stable_analysis_seed(seed, dataset, task, left_protocol, right_protocol)
            )
            bootstrap = np.empty(repetitions, dtype=float)
            valid = 0
            for _ in range(repetitions):
                subject_frequency = np.bincount(
                    rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)
                )
                stimulus_frequency = np.bincount(
                    rng.integers(0, len(stimuli), len(stimuli)), minlength=len(stimuli)
                )
                weights = subject_frequency[subject_codes] * stimulus_frequency[stimulus_codes]
                estimates = seed_panel_deltas(weights)
                if not np.isfinite(estimates).all():
                    continue
                seed_frequency = np.bincount(
                    rng.integers(0, len(seeds), len(seeds)), minlength=len(seeds)
                )
                bootstrap[valid] = np.average(estimates, weights=seed_frequency)
                valid += 1
            if valid < repetitions * 0.95:
                raise RuntimeError(
                    f"Only {valid}/{repetitions} valid replicates for {dataset}/{task}/{left_protocol}"
                )
            bootstrap = bootstrap[:valid]
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "left_protocol": left_protocol,
                    "right_protocol": right_protocol,
                    "n_models": len(models),
                    "n_split_seeds": len(seeds),
                    "mean_balanced_accuracy_delta": observed,
                    "split_seed_sd": np.std(observed_by_seed, ddof=1),
                    "split_seed_min": np.min(observed_by_seed),
                    "split_seed_max": np.max(observed_by_seed),
                    "positive_seed_fraction": np.mean(observed_by_seed > 0),
                    "ci_low": np.quantile(bootstrap, 0.025),
                    "ci_high": np.quantile(bootstrap, 0.975),
                    "p_bootstrap": add_one_two_sided_p(bootstrap),
                    "bootstrap_valid": valid,
                }
            )
            progress.update(1)
    progress.close()
    output = pd.DataFrame(rows)
    output["multiplicity_family"] = np.where(
        output.left_protocol == "random", "primary_random_vs_dual", "secondary_grouped_vs_dual"
    )
    output["p_holm_family"] = np.nan
    for _, indices in output.groupby("multiplicity_family").groups.items():
        positions = np.asarray(list(indices), dtype=int)
        output.loc[positions, "p_holm_family"] = holm_adjust(output.loc[positions, "p_bootstrap"].to_numpy())
    return output


def winner_outputs(summary: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate = summary.loc[summary.model.isin(models)].copy()
    winner_rows = []
    reversal_rows = []
    for (seed, dataset, task), group in candidate.groupby(["split_seed", "dataset", "task"]):
        table = group.pivot(index="model", columns="protocol", values="subject_mean_balanced_accuracy")
        winners = {protocol: table[protocol].idxmax() for protocol in benchmark.PROTOCOLS}
        for protocol, winner in winners.items():
            ordered = table[protocol].sort_values(ascending=False)
            winner_rows.append(
                {
                    "split_seed": seed,
                    "dataset": dataset,
                    "task": task,
                    "protocol": protocol,
                    "winner": winner,
                    "winner_score": ordered.iloc[0],
                    "winner_margin": ordered.iloc[0] - ordered.iloc[1],
                }
            )
        dual_rank = table.dual.rank(ascending=False, method="average")
        for left_protocol, _ in COMPARISONS:
            left_rank = table[left_protocol].rank(ascending=False, method="average")
            tau = left_rank.corr(dual_rank, method="kendall")
            reversal_rows.append(
                {
                    "split_seed": seed,
                    "dataset": dataset,
                    "task": task,
                    "left_protocol": left_protocol,
                    "right_protocol": "dual",
                    "left_winner": winners[left_protocol],
                    "dual_winner": winners["dual"],
                    "winner_changed": winners[left_protocol] != winners["dual"],
                    "kendall_tau": tau,
                }
            )
    per_seed_winners = pd.DataFrame(winner_rows)
    winner_frequency = (
        per_seed_winners.groupby(["dataset", "task", "protocol", "winner"])
        .agg(winner_count=("split_seed", "size"), mean_winner_margin=("winner_margin", "mean"))
        .reset_index()
    )
    winner_frequency["winner_frequency"] = winner_frequency.winner_count / per_seed_winners.split_seed.nunique()
    reversals = pd.DataFrame(reversal_rows)
    reversal_summary = (
        reversals.groupby(["dataset", "task", "left_protocol", "right_protocol"])
        .agg(
            winner_reversal_count=("winner_changed", "sum"),
            winner_reversal_frequency=("winner_changed", "mean"),
            mean_kendall_tau=("kendall_tau", "mean"),
            min_kendall_tau=("kendall_tau", "min"),
            max_kendall_tau=("kendall_tau", "max"),
        )
        .reset_index()
    )
    return per_seed_winners, winner_frequency, reversals.merge(
        reversal_summary,
        on=["dataset", "task", "left_protocol", "right_protocol"],
        how="left",
        validate="many_to_one",
    )


def write_report(
    output_path: Path,
    contrasts: pd.DataFrame,
    winner_frequency: pd.DataFrame,
    reversals: pd.DataFrame,
) -> None:
    primary = contrasts.loc[contrasts.left_protocol == "random"].sort_values(["dataset", "task"])
    lines = [
        "# Multi-seed protocol sensitivity",
        "",
        "Primary estimand: model-panel mean balanced-accuracy difference between random and joint-unseen ",
        "protocols, averaged over five split seeds. Confidence intervals resample subjects, stimuli, and ",
        f"split seeds. Holm correction is restricted to the {primary[['dataset', 'task']].drop_duplicates().shape[0]} "
        "dataset-task primary contrasts.",
        "",
        "| Dataset | Task | Delta | 95% CI | Positive seeds | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.task} | {row.mean_balanced_accuracy_delta:+.3f} | "
            f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}] | {row.positive_seed_fraction:.0%} | "
            f"{row.p_holm_family:.4f} |"
        )
    random_reversals = (
        reversals.loc[reversals.left_protocol == "random"]
        .drop_duplicates(["dataset", "task", "left_protocol"])
        .sort_values(["dataset", "task"])
    )
    lines.extend(
        [
            "",
            "## Model-selection sensitivity",
            "",
            "| Dataset | Task | Random-vs-dual winner reversal frequency | Mean Kendall tau |",
            "|---|---|---:|---:|",
        ]
    )
    for row in random_reversals.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.task} | {row.winner_reversal_frequency:.0%} | "
            f"{row.mean_kendall_tau:+.3f} |"
        )
    stable = (
        winner_frequency.sort_values(
            ["dataset", "task", "protocol", "winner_frequency"], ascending=[True, True, True, False]
        )
        .groupby(["dataset", "task", "protocol"], as_index=False)
        .first()
    )
    lines.extend(["", "## Most frequent model winner", "", stable.to_csv(index=False)])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    admitted_datasets = require_crossed_datasets(
        args.datasets,
        context="Multiseed protocol sensitivity",
    )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Split seeds must be unique")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    model_predictions = []
    sources = {}
    seed_progress = tqdm(args.seeds, desc="Split seeds", unit="seed", dynamic_ncols=True)
    for split_seed in seed_progress:
        seed_progress.set_postfix_str(str(split_seed))
        predictions, source = load_or_fit_seed(args, split_seed)
        predictions["split_seed"] = split_seed
        summary = benchmark.summarize(predictions)
        summary["split_seed"] = split_seed
        summaries.append(summary)
        model_predictions.append(predictions.loc[predictions.model.isin(args.models)].copy())
        sources[str(split_seed)] = source
        seed_root = args.output_root / f"seed_{split_seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(seed_root / "summary.csv", index=False)
    seed_progress.close()

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(args.output_root / "per_seed_summary.csv", index=False)
    per_seed_effects = per_seed_panel_effects(all_summary, args.models)
    per_seed_effects.to_csv(args.output_root / "per_seed_panel_effects.csv", index=False)
    all_predictions = pd.concat(model_predictions, ignore_index=True)
    contrasts = crossed_seed_panel_bootstrap(
        all_predictions,
        args.models,
        args.bootstrap_repetitions,
        min(args.seeds) + 7919,
    )
    contrasts.to_csv(args.output_root / "multiseed_panel_contrasts.csv", index=False)
    per_seed_winners, winner_frequency, reversals = winner_outputs(all_summary, args.models)
    per_seed_winners.to_csv(args.output_root / "per_seed_winners.csv", index=False)
    winner_frequency.to_csv(args.output_root / "winner_frequency.csv", index=False)
    reversals.to_csv(args.output_root / "rank_reversal_sensitivity.csv", index=False)
    write_report(args.output_root / "sensitivity_summary.md", contrasts, winner_frequency, reversals)

    manifest = {
        "datasets": args.datasets,
        "tasks": args.tasks,
        "protocols": list(benchmark.PROTOCOLS),
        "models": args.models,
        "feature_prefix": args.feature_prefix,
        "folds": args.folds,
        "split_seeds": args.seeds,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_clusters": ["subject_id", "trial_id", "split_seed"],
        "bootstrap_seed_scheme": "sha256(base_seed|dataset|task|left_protocol|right_protocol)",
        "primary_family": (
            f"{len(admitted_datasets) * len(args.tasks)} random-vs-dual dataset-task panel contrasts"
        ),
        "secondary_family": (
            f"{2 * len(admitted_datasets) * len(args.tasks)} "
            "subject/stimulus-vs-dual dataset-task panel contrasts"
        ),
        "stimulus_identity_contract": {
            "admitted_datasets": admitted_datasets,
            "eppvr_excluded": True,
        },
        "prediction_sources": sources,
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(contrasts.to_string(index=False))
    print(f"\nResults written to {args.output_root}")


if __name__ == "__main__":
    main()
