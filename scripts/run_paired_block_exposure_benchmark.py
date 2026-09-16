from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm

import run_matched_exposure_benchmark as matched
import run_protocol_model_benchmark as benchmark
from stimulus_identity_contract import require_crossed_datasets


DEFAULT_DATASETS = matched.DEFAULT_DATASETS
DEFAULT_MODELS = matched.DEFAULT_MODELS
DEFAULT_SEEDS = matched.DEFAULT_SEEDS
EXPOSURES = matched.EXPOSURES
FITTED_EXPOSURES = matched.FITTED_EXPOSURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired block-replacement 2x2 identity-exposure benchmark.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--feature-prefix", default="stimulus__")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("outputs/unified_protocol_model_benchmark_multiseed_crossed_valid"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def allocate_blocks(control_count: int, subject_count: int, stimulus_count: int) -> tuple[int, int]:
    if subject_count + stimulus_count <= control_count:
        return subject_count, stimulus_count
    total = subject_count + stimulus_count
    raw_subject = control_count * subject_count / total
    subject_block = min(subject_count, int(np.floor(raw_subject)))
    stimulus_block = min(stimulus_count, control_count - subject_block)
    remaining = control_count - subject_block - stimulus_block
    residuals = [
        (raw_subject - np.floor(raw_subject), "subject"),
        (control_count * stimulus_count / total - np.floor(control_count * stimulus_count / total), "stimulus"),
    ]
    for _, identity in sorted(residuals, reverse=True):
        if remaining == 0:
            break
        if identity == "subject" and subject_block < subject_count:
            subject_block += 1
            remaining -= 1
        elif identity == "stimulus" and stimulus_block < stimulus_count:
            stimulus_block += 1
            remaining -= 1
    if remaining:
        subject_room = subject_count - subject_block
        take = min(subject_room, remaining)
        subject_block += take
        remaining -= take
        stimulus_block += remaining
    return subject_block, stimulus_block


def select_identity_block(
    candidates: np.ndarray,
    labels: np.ndarray,
    class_counts: np.ndarray,
    identities: np.ndarray,
    required_identities: set[str],
    seed: int,
) -> np.ndarray:
    if class_counts.sum() < len(required_identities):
        raise ValueError("Exposure block is too small to cover every required identity")
    rng = np.random.default_rng(seed)
    candidates_by_identity = {
        identity: candidates[identities[candidates] == identity] for identity in required_identities
    }
    for _ in range(2000):
        remaining = class_counts.astype(int).copy()
        selected = []
        valid = True
        identity_order = list(required_identities)
        rng.shuffle(identity_order)
        for identity in identity_order:
            available = candidates_by_identity[identity]
            available = available[remaining[labels[available]] > 0]
            if len(available) == 0:
                valid = False
                break
            index = int(rng.choice(available))
            selected.append(index)
            remaining[labels[index]] -= 1
        if not valid:
            continue
        selected_set = set(selected)
        for label in (0, 1):
            pool = candidates[(labels[candidates] == label) & ~np.isin(candidates, list(selected_set))]
            if len(pool) < remaining[label]:
                valid = False
                break
            if remaining[label]:
                selected.extend(rng.choice(pool, size=remaining[label], replace=False).tolist())
        if valid:
            output = np.asarray(selected, dtype=int)
            rng.shuffle(output)
            return output
    raise RuntimeError("Could not construct an identity-complete exposure block with fixed class counts")


def select_control_blocks(
    control: np.ndarray,
    labels: np.ndarray,
    subject_counts: np.ndarray,
    stimulus_counts: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    subject_control = []
    stimulus_control = []
    for label in (0, 1):
        pool = control[labels[control] == label].copy()
        rng.shuffle(pool)
        first = int(subject_counts[label])
        second = first + int(stimulus_counts[label])
        if second > len(pool):
            raise ValueError("Control quadrant cannot supply label-matched replacement blocks")
        subject_control.extend(pool[:first].tolist())
        stimulus_control.extend(pool[first:second].tolist())
    subject_control = np.asarray(subject_control, dtype=int)
    stimulus_control = np.asarray(stimulus_control, dtype=int)
    core = np.setdiff1d(control, np.concatenate([subject_control, stimulus_control]), assume_unique=True)
    return subject_control, stimulus_control, core


def paired_train_sets(
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
    masks = {
        "control": (subject_fold != subject_test) & (stimulus_fold != stimulus_test),
        "subject_block": (subject_fold == subject_test) & (stimulus_fold != stimulus_test),
        "stimulus_block": (subject_fold != subject_test) & (stimulus_fold == stimulus_test),
        "test": (subject_fold == subject_test) & (stimulus_fold == stimulus_test),
    }
    indices = {name: np.flatnonzero(mask) for name, mask in masks.items()}
    control_counts = np.bincount(labels[indices["control"]], minlength=2)
    subject_available = np.bincount(labels[indices["subject_block"]], minlength=2)
    stimulus_available = np.bincount(labels[indices["stimulus_block"]], minlength=2)
    subject_counts = np.zeros(2, dtype=int)
    stimulus_counts = np.zeros(2, dtype=int)
    for label in (0, 1):
        subject_counts[label], stimulus_counts[label] = allocate_blocks(
            int(control_counts[label]),
            int(subject_available[label]),
            int(stimulus_available[label]),
        )
    subjects = frame.subject_id.astype(str).to_numpy()
    stimuli = frame.trial_id.astype(int).astype(str).to_numpy()
    test_subjects = set(subjects[indices["test"]])
    test_stimuli = set(stimuli[indices["test"]])
    subject_block = select_identity_block(
        indices["subject_block"],
        labels,
        subject_counts,
        subjects,
        test_subjects,
        matched.stable_seed(dataset, task, split_seed, subject_test, stimulus_test, "paired_subject_block"),
    )
    stimulus_block = select_identity_block(
        indices["stimulus_block"],
        labels,
        stimulus_counts,
        stimuli,
        test_stimuli,
        matched.stable_seed(dataset, task, split_seed, subject_test, stimulus_test, "paired_stimulus_block"),
    )
    subject_control, stimulus_control, core = select_control_blocks(
        indices["control"],
        labels,
        subject_counts,
        stimulus_counts,
        matched.stable_seed(dataset, task, split_seed, subject_test, stimulus_test, "paired_controls"),
    )
    train_sets = {
        "unseen_both": indices["control"],
        "seen_subject": np.concatenate([core, subject_block, stimulus_control]),
        "seen_stimulus": np.concatenate([core, subject_control, stimulus_block]),
        "seen_both": np.concatenate([core, subject_block, stimulus_block]),
    }
    audit_rows = []
    reference_counts = np.bincount(labels[indices["control"]], minlength=2)
    control_set = set(indices["control"])
    for exposure, train in train_sets.items():
        train_subjects = set(subjects[train])
        train_stimuli = set(stimuli[train])
        counts = np.bincount(labels[train], minlength=2)
        audit_rows.append(
            {
                "dataset": dataset,
                "task": task,
                "split_seed": split_seed,
                "fold": f"s{subject_test}_t{stimulus_test}",
                "exposure": exposure,
                "n_train": len(train),
                "n_test": len(indices["test"]),
                "class_0_train": counts[0],
                "class_1_train": counts[1],
                "matches_reference_size": len(train) == len(indices["control"]),
                "matches_reference_class_counts": np.array_equal(counts, reference_counts),
                "row_overlap_with_test": len(set(train) & set(indices["test"])),
                "test_subject_coverage": len(test_subjects & train_subjects) / len(test_subjects),
                "test_stimulus_coverage": len(test_stimuli & train_stimuli) / len(test_stimuli),
                "shared_core_fraction": len(core) / len(train),
                "overlap_with_unseen_fraction": len(set(train) & control_set) / len(train),
                "subject_block_size": len(subject_block),
                "stimulus_block_size": len(stimulus_block),
            }
        )
    return indices["test"], train_sets, audit_rows


def fit_seed(
    args: argparse.Namespace,
    split_seed: int,
    reference: pd.DataFrame,
    progress: tqdm,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    templates = {
        name: model for name, model in benchmark.make_models(split_seed, args.n_jobs).items() if name in args.models
    }
    rows = []
    audit_rows = []
    for specification in args.datasets:
        dataset, path = specification.split("=", 1)
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["row_id"] = np.arange(len(frame))
        feature_columns = [column for column in frame if column.startswith(args.feature_prefix)]
        features = frame[feature_columns].to_numpy(float)
        subject_fold, stimulus_fold = matched.fold_assignments(frame, args.folds, split_seed)
        for task in args.tasks:
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
            for subject_test in range(args.folds):
                for stimulus_test in range(args.folds):
                    test, train_sets, audit = paired_train_sets(
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
                    audit_rows.extend(audit)
                    fold = f"s{subject_test}_t{stimulus_test}"
                    for exposure in FITTED_EXPOSURES:
                        train = train_sets[exposure]
                        for model_name, template in templates.items():
                            model = clone(template).fit(features[train], labels[train])
                            probability = benchmark.model_probability(model, features[test])
                            for index, estimate in zip(test, probability, strict=True):
                                rows.append(
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
    fitted = pd.DataFrame(rows)
    combined = pd.concat([reference, fitted], ignore_index=True)
    combined["split_seed"] = split_seed
    return combined, pd.DataFrame(audit_rows)


def validate_seed(predictions: pd.DataFrame, audit: pd.DataFrame, args: argparse.Namespace, seed: int) -> None:
    matched.validate_seed(predictions, audit, args, seed)
    if (audit.shared_core_fraction <= 0).any():
        raise ValueError(f"Seed {seed} has an empty shared training core")
    reference_overlap = audit.loc[audit.exposure == "unseen_both", "overlap_with_unseen_fraction"]
    if not np.allclose(reference_overlap, 1.0):
        raise ValueError(f"Seed {seed} invalid unseen-both overlap")
    for exposure in FITTED_EXPOSURES:
        subset = audit.loc[audit.exposure == exposure]
        if not np.allclose(
            subset.overlap_with_unseen_fraction,
            subset.shared_core_fraction
            + np.where(
                exposure == "seen_subject",
                subset.stimulus_block_size / subset.n_train,
                np.where(exposure == "seen_stimulus", subset.subject_block_size / subset.n_train, 0),
            ),
        ):
            raise ValueError(f"Seed {seed} invalid paired overlap accounting for {exposure}")


def main() -> None:
    args = parse_args()
    admitted_datasets = require_crossed_datasets(
        args.datasets,
        context="Paired block identity-exposure benchmark",
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    reference_manifest = matched.load_reference_manifest(args.reference_root)
    completed = []
    for split_seed in args.seeds:
        root = args.output_root / f"seed_{split_seed}"
        if (root / "predictions.csv").exists() and (root / "split_audit.csv").exists() and not args.force:
            completed.append(split_seed)
    total_fits = (
        (len(args.seeds) - len(completed))
        * len(args.datasets)
        * len(args.tasks)
        * args.folds**2
        * len(FITTED_EXPOSURES)
        * len(args.models)
    )
    progress = tqdm(total=total_fits, desc="Paired block exposure", unit="fit", dynamic_ncols=True)
    all_audits = []
    all_summaries = []
    for split_seed in args.seeds:
        root = args.output_root / f"seed_{split_seed}"
        prediction_path = root / "predictions.csv"
        audit_path = root / "split_audit.csv"
        if split_seed in completed:
            predictions = pd.read_csv(prediction_path, low_memory=False)
            predictions["subject_id"] = predictions.subject_id.astype(str)
            predictions["trial_id"] = predictions.trial_id.astype(int)
            audit = pd.read_csv(audit_path)
        else:
            reference = matched.load_dual_reference(
                reference_manifest,
                split_seed,
                args.datasets,
                args.tasks,
                args.models,
            )
            predictions, audit = fit_seed(args, split_seed, reference, progress)
            root.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(prediction_path, index=False)
            audit.to_csv(audit_path, index=False)
        validate_seed(predictions, audit, args, split_seed)
        summary = matched.summarize(predictions)
        summary["split_seed"] = split_seed
        summary.to_csv(root / "summary.csv", index=False)
        all_audits.append(audit)
        all_summaries.append(summary)
    progress.close()
    audits = pd.concat(all_audits, ignore_index=True)
    summaries = pd.concat(all_summaries, ignore_index=True)
    audits.to_csv(args.output_root / "split_audit.csv", index=False)
    summaries.to_csv(args.output_root / "per_seed_summary.csv", index=False)
    matched.per_seed_effects(summaries, args.models).to_csv(
        args.output_root / "per_seed_effects.csv", index=False
    )
    manifest = {
        "datasets": args.datasets,
        "tasks": args.tasks,
        "models": args.models,
        "split_seeds": args.seeds,
        "folds": args.folds,
        "exposures": list(EXPOSURES),
        "design": "paired label-matched block replacement with a shared training core",
        "matched_on": ["test rows", "training sample size", "binary class counts", "shared core"],
        "reference_manifest": str(args.reference_root / "run_manifest.json"),
        "stimulus_identity_contract": {
            "admitted_datasets": admitted_datasets,
            "eppvr_excluded": True,
        },
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summaries.to_string(index=False))
    print(f"\nResults written to {args.output_root}")


if __name__ == "__main__":
    main()
