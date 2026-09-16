from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm

import run_protocol_model_benchmark as benchmark
from run_multiseed_protocol_sensitivity import add_one_two_sided_p, holm_adjust
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
EXPOSURES = ("unseen_both", "seen_subject", "seen_stimulus", "seen_both")
FITTED_EXPOSURES = EXPOSURES[1:]
COMPARISONS = (
    ("seen_stimulus", "unseen_both"),
    ("seen_subject", "unseen_both"),
    ("seen_both", "unseen_both"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training-size-matched 2x2 identity-exposure benchmark.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--feature-prefix", default="stimulus__")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("outputs/unified_protocol_model_benchmark_multiseed_crossed_valid"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/matched_identity_exposure_benchmark_crossed_valid"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def load_reference_manifest(root: Path) -> dict:
    path = root / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing multi-seed manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dual_reference(
    manifest: dict,
    seed: int,
    datasets: list[str],
    tasks: list[str],
    models: list[str],
) -> pd.DataFrame:
    source = manifest.get("prediction_sources", {}).get(str(seed))
    if source is None:
        raise ValueError(f"No reference prediction source for seed {seed}")
    predictions = pd.read_csv(source, low_memory=False)
    names = [item.split("=", 1)[0] for item in datasets]
    reference = predictions.loc[
        (predictions.protocol == "dual")
        & predictions.dataset.isin(names)
        & predictions.task.isin(tasks)
        & predictions.model.isin(models)
    ].copy()
    expected_groups = len(names) * len(tasks) * len(models)
    groups = reference.groupby(["dataset", "task", "model"])
    if groups.ngroups != expected_groups:
        raise ValueError(f"Seed {seed} dual reference has {groups.ngroups} groups, expected {expected_groups}")
    if reference.duplicated(["dataset", "task", "model", "row_id"]).any():
        raise ValueError(f"Seed {seed} dual reference contains duplicate OOF rows")
    reference = reference.drop(columns=["protocol", "stimulus_prior_available"], errors="ignore")
    reference["subject_id"] = reference.subject_id.astype(str)
    reference["trial_id"] = reference.trial_id.astype(int)
    reference["exposure"] = "unseen_both"
    return reference


def fold_assignments(frame: pd.DataFrame, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    subject_map = benchmark.shuffled_fold_map(frame.subject_id.to_numpy(), folds, seed)
    stimulus_map = benchmark.shuffled_fold_map(frame.trial_id.to_numpy(), folds, seed + 1)
    return (
        frame.subject_id.map(subject_map).to_numpy(int),
        frame.trial_id.map(stimulus_map).to_numpy(int),
    )


def matched_training_indices(
    candidate: np.ndarray,
    labels: np.ndarray,
    reference_train: np.ndarray,
    frame: pd.DataFrame,
    required_subjects: set[str],
    required_stimuli: set[int],
    seed: int,
) -> np.ndarray:
    reference_counts = np.bincount(labels[reference_train], minlength=2)
    candidate_by_class = {
        label: candidate[labels[candidate] == label] for label in (0, 1)
    }
    if any(len(candidate_by_class[label]) < reference_counts[label] for label in (0, 1)):
        raise ValueError("Candidate pool cannot match the joint-unseen class counts")
    subjects = frame.subject_id.astype(str).to_numpy()
    stimuli = frame.trial_id.to_numpy(int)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        selected = np.concatenate(
            [
                rng.choice(candidate_by_class[label], size=reference_counts[label], replace=False)
                for label in (0, 1)
            ]
        )
        if required_subjects - set(subjects[selected]):
            continue
        if required_stimuli - set(stimuli[selected]):
            continue
        rng.shuffle(selected)
        return selected
    raise RuntimeError("Could not draw a class-matched training set with the required identity coverage")


def train_sets_for_fold(
    frame: pd.DataFrame,
    labels: np.ndarray,
    subject_fold: np.ndarray,
    stimulus_fold: np.ndarray,
    subject_test: int,
    stimulus_test: int,
    dataset: str,
    task: str,
    split_seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict]]:
    test_mask = (subject_fold == subject_test) & (stimulus_fold == stimulus_test)
    test = np.flatnonzero(test_mask)
    unseen_both = np.flatnonzero((subject_fold != subject_test) & (stimulus_fold != stimulus_test))
    test_subjects = set(frame.iloc[test].subject_id.astype(str))
    test_stimuli = set(frame.iloc[test].trial_id.astype(int))
    candidates = {
        "seen_subject": np.flatnonzero(stimulus_fold != stimulus_test),
        "seen_stimulus": np.flatnonzero(subject_fold != subject_test),
        "seen_both": np.flatnonzero(~test_mask),
    }
    requirements = {
        "seen_subject": (test_subjects, set()),
        "seen_stimulus": (set(), test_stimuli),
        "seen_both": (test_subjects, test_stimuli),
    }
    selected = {"unseen_both": unseen_both}
    for exposure in FITTED_EXPOSURES:
        required_subjects, required_stimuli = requirements[exposure]
        selected[exposure] = matched_training_indices(
            candidates[exposure],
            labels,
            unseen_both,
            frame,
            required_subjects,
            required_stimuli,
            stable_seed(dataset, task, split_seed, subject_test, stimulus_test, exposure),
        )
    audit_rows = []
    reference_counts = np.bincount(labels[unseen_both], minlength=2)
    for exposure, train in selected.items():
        train_subjects = set(frame.iloc[train].subject_id.astype(str))
        train_stimuli = set(frame.iloc[train].trial_id.astype(int))
        counts = np.bincount(labels[train], minlength=2)
        audit_rows.append(
            {
                "dataset": dataset,
                "task": task,
                "split_seed": split_seed,
                "fold": f"s{subject_test}_t{stimulus_test}",
                "exposure": exposure,
                "n_train": len(train),
                "n_test": len(test),
                "class_0_train": counts[0],
                "class_1_train": counts[1],
                "matches_reference_size": len(train) == len(unseen_both),
                "matches_reference_class_counts": np.array_equal(counts, reference_counts),
                "row_overlap_with_test": len(set(train) & set(test)),
                "test_subject_coverage": len(test_subjects & train_subjects) / len(test_subjects),
                "test_stimulus_coverage": len(test_stimuli & train_stimuli) / len(test_stimuli),
                "candidate_size": len(unseen_both) if exposure == "unseen_both" else len(candidates[exposure]),
            }
        )
    return test, selected, audit_rows


def fit_seed(
    args: argparse.Namespace,
    split_seed: int,
    reference: pd.DataFrame,
    progress: tqdm,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_templates = {
        name: model for name, model in benchmark.make_models(split_seed, args.n_jobs).items() if name in args.models
    }
    prediction_rows = []
    audit_rows = []
    for specification in args.datasets:
        dataset, path = specification.split("=", 1)
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["row_id"] = np.arange(len(frame))
        feature_columns = [column for column in frame if column.startswith(args.feature_prefix)]
        if not feature_columns:
            raise ValueError(f"No {args.feature_prefix!r} columns for {dataset}")
        features = frame[feature_columns].to_numpy(float)
        subject_fold, stimulus_fold = fold_assignments(frame, args.folds, split_seed)
        for task in args.tasks:
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
            for subject_test in range(args.folds):
                for stimulus_test in range(args.folds):
                    test, train_sets, fold_audit = train_sets_for_fold(
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
                    audit_rows.extend(fold_audit)
                    fold = f"s{subject_test}_t{stimulus_test}"
                    for exposure in FITTED_EXPOSURES:
                        train = train_sets[exposure]
                        for model_name, template in model_templates.items():
                            model = clone(template).fit(features[train], labels[train])
                            probability = benchmark.model_probability(model, features[test])
                            for index, estimate in zip(test, probability, strict=True):
                                prediction_rows.append(
                                    {
                                        "dataset": dataset,
                                        "task": task,
                                        "exposure": exposure,
                                        "model": model_name,
                                        "fold": fold,
                                        "row_id": int(frame.iloc[index].row_id),
                                        "subject_id": str(frame.iloc[index].subject_id),
                                        "trial_id": int(frame.iloc[index].trial_id),
                                        "target": int(labels[index]),
                                        "probability": float(estimate),
                                    }
                                )
                            progress.update(1)
    fitted = pd.DataFrame(prediction_rows)
    combined = pd.concat([reference, fitted], ignore_index=True)
    combined["split_seed"] = split_seed
    return combined, pd.DataFrame(audit_rows)


def validate_seed(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
    args: argparse.Namespace,
    split_seed: int,
) -> None:
    if not np.isfinite(predictions.probability.to_numpy(float)).all():
        raise ValueError(f"Seed {split_seed} has non-finite probabilities")
    key = ["dataset", "task", "exposure", "model", "row_id"]
    if predictions.duplicated(key).any():
        raise ValueError(f"Seed {split_seed} has duplicate OOF predictions")
    dataset_counts = {
        dataset: len(pd.read_csv(path, usecols=["subject_id"]))
        for dataset, path in (item.split("=", 1) for item in args.datasets)
    }
    observed_datasets = set(predictions.dataset.unique())
    expected_datasets = set(dataset_counts)
    if observed_datasets != expected_datasets:
        raise ValueError(
            f"Seed {split_seed} dataset mismatch: expected {sorted(expected_datasets)}, "
            f"observed {sorted(observed_datasets)}. Remove incompatible cached outputs or use "
            "a new --output-root."
        )
    groups = predictions.groupby(["dataset", "task", "exposure", "model"]).size()
    expected_groups = len(args.datasets) * len(args.tasks) * len(EXPOSURES) * len(args.models)
    if len(groups) != expected_groups:
        raise ValueError(f"Seed {split_seed} has {len(groups)} prediction groups, expected {expected_groups}")
    for (dataset, _, _, _), count in groups.items():
        if count != dataset_counts[dataset]:
            raise ValueError(f"Seed {split_seed} incomplete OOF group for {dataset}: {count}")
    if not audit.matches_reference_size.all() or not audit.matches_reference_class_counts.all():
        raise ValueError(f"Seed {split_seed} violates matched-training contracts")
    if (audit.row_overlap_with_test != 0).any():
        raise ValueError(f"Seed {split_seed} contains train-test row overlap")
    expected_coverage = {
        "unseen_both": (0.0, 0.0),
        "seen_subject": (1.0, 0.0),
        "seen_stimulus": (0.0, 1.0),
        "seen_both": (1.0, 1.0),
    }
    for exposure, (subject_coverage, stimulus_coverage) in expected_coverage.items():
        subset = audit.loc[audit.exposure == exposure]
        if not np.allclose(subset.test_subject_coverage, subject_coverage):
            raise ValueError(f"Seed {split_seed} invalid subject coverage for {exposure}")
        if not np.allclose(subset.test_stimulus_coverage, stimulus_coverage):
            raise ValueError(f"Seed {split_seed} invalid stimulus coverage for {exposure}")


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    protocol_frame = predictions.rename(columns={"exposure": "protocol"}).copy()
    summary = benchmark.summarize(protocol_frame)
    return summary.rename(columns={"protocol": "exposure"})


def per_seed_effects(summary: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for (split_seed, dataset, task), group in summary.groupby(["split_seed", "dataset", "task"]):
        table = group.loc[group.model.isin(models)].pivot(
            index="model", columns="exposure", values="pooled_balanced_accuracy"
        )
        for left, right in COMPARISONS:
            deltas = table[left] - table[right]
            rows.append(
                {
                    "split_seed": split_seed,
                    "dataset": dataset,
                    "task": task,
                    "left_exposure": left,
                    "right_exposure": right,
                    "mean_balanced_accuracy_delta": deltas.mean(),
                    "min_model_delta": deltas.min(),
                    "max_model_delta": deltas.max(),
                    "positive_model_fraction": np.mean(deltas > 0),
                }
            )
    return pd.DataFrame(rows)


def triple_cluster_bootstrap(
    predictions: pd.DataFrame,
    models: list[str],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    groups = list(predictions.groupby(["dataset", "task"], sort=True))
    progress = tqdm(
        total=len(groups) * len(COMPARISONS),
        desc="Matched exposure bootstrap",
        unit="contrast",
        dynamic_ncols=True,
    )
    for dataset_task, group in groups:
        dataset, task = dataset_task
        reference = group.loc[(group.exposure == "unseen_both") & group.model.isin(models)]
        for comparison_index, (left_exposure, right_exposure) in enumerate(COMPARISONS):
            left = group.loc[(group.exposure == left_exposure) & group.model.isin(models)]
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
            rng = np.random.default_rng(
                stable_analysis_seed(seed, dataset, task, left_exposure, right_exposure)
            )
            bootstrap_values = np.empty(repetitions, dtype=float)
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
                bootstrap_values[valid] = np.average(estimates, weights=seed_frequency)
                valid += 1
            if valid < repetitions * 0.95:
                raise RuntimeError(f"Only {valid}/{repetitions} valid replicates for {dataset}/{task}")
            bootstrap_values = bootstrap_values[:valid]
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "left_exposure": left_exposure,
                    "right_exposure": right_exposure,
                    "n_models": len(models),
                    "n_split_seeds": len(seeds),
                    "mean_balanced_accuracy_delta": np.mean(observed_by_seed),
                    "split_seed_sd": np.std(observed_by_seed, ddof=1) if len(seeds) > 1 else 0.0,
                    "split_seed_min": np.min(observed_by_seed),
                    "split_seed_max": np.max(observed_by_seed),
                    "positive_seed_fraction": np.mean(observed_by_seed > 0),
                    "ci_low": np.quantile(bootstrap_values, 0.025),
                    "ci_high": np.quantile(bootstrap_values, 0.975),
                    "p_bootstrap": add_one_two_sided_p(bootstrap_values),
                    "bootstrap_valid": valid,
                }
            )
            progress.update(1)
    progress.close()
    output = pd.DataFrame(rows)
    output["multiplicity_family"] = np.select(
        [
            output.left_exposure == "seen_stimulus",
            output.left_exposure == "seen_subject",
        ],
        ["primary_stimulus_exposure", "secondary_subject_exposure"],
        default="secondary_combined_exposure",
    )
    output["p_holm_family"] = np.nan
    for _, indices in output.groupby("multiplicity_family").groups.items():
        positions = np.asarray(list(indices), dtype=int)
        output.loc[positions, "p_holm_family"] = holm_adjust(output.loc[positions, "p_bootstrap"].to_numpy())
    return output


def rank_sensitivity(summary: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for (split_seed, dataset, task), group in summary.groupby(["split_seed", "dataset", "task"]):
        table = group.loc[group.model.isin(models)].pivot(
            index="model", columns="exposure", values="subject_mean_balanced_accuracy"
        )
        reference_rank = table.unseen_both.rank(ascending=False, method="average")
        reference_winner = table.unseen_both.idxmax()
        for exposure in FITTED_EXPOSURES:
            exposure_rank = table[exposure].rank(ascending=False, method="average")
            rows.append(
                {
                    "split_seed": split_seed,
                    "dataset": dataset,
                    "task": task,
                    "left_exposure": exposure,
                    "right_exposure": "unseen_both",
                    "left_winner": table[exposure].idxmax(),
                    "right_winner": reference_winner,
                    "winner_changed": table[exposure].idxmax() != reference_winner,
                    "kendall_tau": exposure_rank.corr(reference_rank, method="kendall"),
                }
            )
    output = pd.DataFrame(rows)
    summary_frame = (
        output.groupby(["dataset", "task", "left_exposure", "right_exposure"])
        .agg(
            winner_reversal_frequency=("winner_changed", "mean"),
            mean_kendall_tau=("kendall_tau", "mean"),
        )
        .reset_index()
    )
    return output.merge(
        summary_frame,
        on=["dataset", "task", "left_exposure", "right_exposure"],
        validate="many_to_one",
    )


def write_report(path: Path, contrasts: pd.DataFrame, ranks: pd.DataFrame) -> None:
    primary = contrasts.loc[contrasts.left_exposure == "seen_stimulus"].sort_values(["dataset", "task"])
    lines = [
        "# Matched identity-exposure benchmark",
        "",
        "All four exposure conditions use the same joint subject-stimulus test cells. Training sample size ",
        "and binary class counts are exactly matched within each fold. The primary estimand isolates the ",
        "effect of making test stimuli available through other subjects while test subjects remain unseen.",
        "",
        "| Dataset | Task | Stimulus-exposure delta | 95% CI | Positive seeds | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.task} | {row.mean_balanced_accuracy_delta:+.3f} | "
            f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}] | {row.positive_seed_fraction:.0%} | "
            f"{row.p_holm_family:.4f} |"
        )
    rank_summary = (
        ranks.loc[ranks.left_exposure == "seen_stimulus"]
        .drop_duplicates(["dataset", "task", "left_exposure"])
        .sort_values(["dataset", "task"])
    )
    lines.extend(
        [
            "",
            "## Model-selection sensitivity",
            "",
            "| Dataset | Task | Winner reversal frequency | Mean Kendall tau |",
            "|---|---|---:|---:|",
        ]
    )
    for row in rank_summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.task} | {row.winner_reversal_frequency:.0%} | "
            f"{row.mean_kendall_tau:+.3f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    admitted_datasets = require_crossed_datasets(
        args.datasets,
        context="Matched identity-exposure benchmark",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_reference_manifest(args.reference_root)
    manifest["stimulus_identity_contract"] = {
        "admitted_datasets": admitted_datasets,
        "eppvr_excluded": True,
    }
    completed = []
    pending = []
    for split_seed in args.seeds:
        seed_root = args.output_root / f"seed_{split_seed}"
        prediction_path = seed_root / "predictions.csv"
        audit_path = seed_root / "split_audit.csv"
        if prediction_path.exists() and audit_path.exists() and not args.force:
            completed.append(split_seed)
        else:
            pending.append(split_seed)
    total_fits = (
        len(pending)
        * len(args.datasets)
        * len(args.tasks)
        * args.folds**2
        * len(FITTED_EXPOSURES)
        * len(args.models)
    )
    progress = tqdm(total=total_fits, desc="Matched exposure benchmark", unit="fit", dynamic_ncols=True)
    all_predictions = []
    all_audits = []
    all_summaries = []
    for split_seed in args.seeds:
        seed_root = args.output_root / f"seed_{split_seed}"
        prediction_path = seed_root / "predictions.csv"
        audit_path = seed_root / "split_audit.csv"
        if split_seed in completed:
            predictions = pd.read_csv(prediction_path, low_memory=False)
            predictions["subject_id"] = predictions.subject_id.astype(str)
            predictions["trial_id"] = predictions.trial_id.astype(int)
            audit = pd.read_csv(audit_path)
        else:
            reference = load_dual_reference(manifest, split_seed, args.datasets, args.tasks, args.models)
            predictions, audit = fit_seed(args, split_seed, reference, progress)
            seed_root.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(prediction_path, index=False)
            audit.to_csv(audit_path, index=False)
        validate_seed(predictions, audit, args, split_seed)
        summary = summarize(predictions)
        summary["split_seed"] = split_seed
        summary.to_csv(seed_root / "summary.csv", index=False)
        all_predictions.append(predictions)
        all_audits.append(audit)
        all_summaries.append(summary)
    progress.close()

    predictions = pd.concat(all_predictions, ignore_index=True)
    audits = pd.concat(all_audits, ignore_index=True)
    summaries = pd.concat(all_summaries, ignore_index=True)
    audits.to_csv(args.output_root / "split_audit.csv", index=False)
    summaries.to_csv(args.output_root / "per_seed_summary.csv", index=False)
    seed_effects = per_seed_effects(summaries, args.models)
    seed_effects.to_csv(args.output_root / "per_seed_effects.csv", index=False)
    contrasts = triple_cluster_bootstrap(
        predictions,
        args.models,
        args.bootstrap_repetitions,
        min(args.seeds) + 104729,
    )
    contrasts.to_csv(args.output_root / "matched_exposure_contrasts.csv", index=False)
    ranks = rank_sensitivity(summaries, args.models)
    ranks.to_csv(args.output_root / "rank_sensitivity.csv", index=False)
    write_report(args.output_root / "matched_exposure_summary.md", contrasts, ranks)
    output_manifest = {
        "datasets": args.datasets,
        "tasks": args.tasks,
        "models": args.models,
        "split_seeds": args.seeds,
        "folds": args.folds,
        "exposures": list(EXPOSURES),
        "matched_on": ["test rows", "training sample size", "binary class counts"],
        "primary_estimand": "seen_stimulus minus unseen_both",
        "bootstrap_clusters": ["subject_id", "trial_id", "split_seed"],
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed_scheme": "sha256(base_seed|dataset|task|left_exposure|right_exposure)",
        "reference_manifest": str(args.reference_root / "run_manifest.json"),
        "stimulus_identity_contract": {
            "admitted_datasets": admitted_datasets,
            "eppvr_excluded": True,
        },
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    print(contrasts.to_string(index=False))
    print(f"\nResults written to {args.output_root}")


if __name__ == "__main__":
    main()
