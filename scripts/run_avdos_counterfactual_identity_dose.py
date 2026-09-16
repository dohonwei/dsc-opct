from __future__ import annotations

import argparse
from hashlib import sha256 as hashlib_sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from identity_shortcut.core import metadata_prior_opportunity, stable_seed  # noqa: E402
from identity_shortcut.models import make_shortcut_models  # noqa: E402
import run_counterfactual_identity_dose as dose  # noqa: E402
import run_crossed_identity_probes as probes  # noqa: E402
import run_protocol_model_benchmark as benchmark  # noqa: E402
import run_paired_block_exposure_benchmark as paired  # noqa: E402


RESERVATION = ROOT / "docs/avdos_v7_external_confirmation_reservation.json"
AMENDMENT = ROOT / "docs/avdos_v11_external_confirmation_amendment.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
IMPLEMENTATION_LOCK = ROOT / "docs/avdos_v11_external_confirmation_implementation_lock.json"
FEATURE_ROOT = ROOT / "outputs/avdos_trial_features"
FEATURES = FEATURE_ROOT / "avdos_trial_features.csv"
FEATURE_MANIFEST = FEATURE_ROOT / "build_manifest.json"
TASKS = ("arousal", "valence")
AXES = ("subject", "stimulus")
MODELS = ("linear_logistic", "gpu_mlp")
SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
SUBJECT_FOLDS = 5
STIMULUS_FOLDS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registered AVDOS identity-dose experiment.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/avdos_counterfactual_identity_dose"),
    )
    parser.add_argument("--gpu-epochs", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib_sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_locks() -> dict[str, object]:
    reservation = json.loads(RESERVATION.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    lock = json.loads(IMPLEMENTATION_LOCK.read_text(encoding="utf-8"))
    if amendment["reservation_sha256"] != file_sha256(RESERVATION):
        raise RuntimeError("AVDOS amendment does not match the reservation")
    if reservation["source"]["sha256"] != lock["source_sha256"]:
        raise RuntimeError("AVDOS implementation lock does not match the source reservation")
    if lock["reservation_amendment_sha256"] != file_sha256(AMENDMENT):
        raise RuntimeError("AVDOS implementation lock does not match the amendment")
    if lock["freeze_sha256"] != file_sha256(FREEZE):
        raise RuntimeError("AVDOS implementation lock does not match the v11 freeze")
    for relative, expected in freeze["locked_artifacts"].items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen v11 artifact changed: {relative}")
    for hash_group in ("analysis_code_sha256", "dependency_code_sha256"):
        for relative, expected in lock[hash_group].items():
            if file_sha256(ROOT / relative) != expected:
                raise RuntimeError(f"Registered AVDOS code changed: {relative}")
    return lock


def fold_assignments(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    subject_map = benchmark.shuffled_fold_map(
        frame.subject_id.to_numpy(), SUBJECT_FOLDS, seed
    )
    stimulus_map = benchmark.shuffled_fold_map(
        frame.trial_id.to_numpy(), STIMULUS_FOLDS, seed + 1
    )
    return (
        frame.subject_id.map(subject_map).to_numpy(int),
        frame.trial_id.map(stimulus_map).to_numpy(int),
    )


def preflight(frame: pd.DataFrame, settings: dict[str, object]) -> dict[tuple, tuple]:
    cache = {}
    total = len(SEEDS) * len(TASKS) * SUBJECT_FOLDS * STIMULUS_FOLDS * len(AXES)
    progress = tqdm(total=total, desc="AVDOS preflight dose plans", unit="plan")
    for split_seed in SEEDS:
        subject_fold, stimulus_fold = fold_assignments(frame, split_seed)
        for task in TASKS:
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5.0).astype(int)
            for subject_test in range(SUBJECT_FOLDS):
                for stimulus_test in range(STIMULUS_FOLDS):
                    test, train_sets, _ = paired.paired_train_sets(
                        frame,
                        labels,
                        subject_fold,
                        stimulus_fold,
                        subject_test,
                        stimulus_test,
                        "AVDOS-VR",
                        task,
                        split_seed,
                    )
                    fold = f"s{subject_test}_t{stimulus_test}"
                    for axis in AXES:
                        cache[(task, split_seed, subject_test, stimulus_test, axis)] = dose.axis_plan(
                            frame,
                            labels,
                            subject_fold,
                            stimulus_fold,
                            subject_test,
                            stimulus_test,
                            train_sets,
                            test,
                            axis,
                            settings["doses"],
                            settings["candidate_draws"],
                            settings["intervention_block_fraction"],
                            stable_seed("AVDOS-VR", task, axis, split_seed, fold),
                        )
                        progress.update(1)
    progress.close()
    return cache


def identity_encoding_table(
    frame: pd.DataFrame,
    feature_sets: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    progress = tqdm(
        total=len(feature_sets) * len(SEEDS) * STIMULUS_FOLDS * 2,
        desc="AVDOS subject-identity probe",
        unit="fit",
    )
    targets = frame.subject_id.astype(str).to_numpy()
    for representation, features in feature_sets.items():
        for split_seed in SEEDS:
            group_map = benchmark.shuffled_fold_map(
                frame.trial_id.to_numpy(), STIMULUS_FOLDS, split_seed
            )
            folds = frame.trial_id.map(group_map).to_numpy(int)
            for model_name, template in probes.make_models(split_seed).items():
                scores = []
                for fold in range(STIMULUS_FOLDS):
                    test = folds == fold
                    fitted = clone(template).fit(features[~test], targets[~test])
                    scores.append(
                        probes.balanced_multiclass_accuracy(
                            targets[test], fitted.predict(features[test])
                        )
                    )
                    progress.update(1)
                rows.append(
                    {
                        "dataset": "AVDOS-VR",
                        "representation": representation,
                        "n_features": features.shape[1],
                        "model": model_name,
                        "split_seed": split_seed,
                        "chance": 1.0 / frame.subject_id.nunique(),
                        "balanced_accuracy": float(np.mean(scores)),
                    }
                )
    progress.close()
    table = pd.DataFrame(rows)
    table["encoding_margin"] = table.balanced_accuracy - table.chance
    return table


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite AVDOS dose output: {args.output_root}")
    if args.gpu_epochs != 40:
        raise ValueError("The registered AVDOS protocol fixes --gpu-epochs at 40")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered AVDOS gpu_mlp experiment")
    verify_locks()
    feature_manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    settings = {
        "tasks": list(TASKS),
        "axes": list(AXES),
        "representations": list(feature_manifest["representations"]),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "subject_folds": SUBJECT_FOLDS,
        "stimulus_folds": STIMULUS_FOLDS,
        "label_threshold": 5.0,
        "doses": list(DOSES),
        "candidate_draws": 256,
        "intervention_block_fraction": 0.5,
    }
    frame = pd.read_csv(FEATURES).reset_index(drop=True)
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["trial_id"] = frame.trial_id.astype(int)
    feature_sets = {
        name: frame[columns].to_numpy(float)
        for name, columns in feature_manifest["representations"].items()
    }
    if any(values.shape[1] == 0 for values in feature_sets.values()):
        raise ValueError("A registered AVDOS representation is empty")
    identity_encoding = identity_encoding_table(frame, feature_sets)
    plans = preflight(frame, settings)
    rows = []
    audit_rows = []
    total = (
        len(TASKS)
        * len(SEEDS)
        * SUBJECT_FOLDS
        * STIMULUS_FOLDS
        * len(feature_sets)
        * len(MODELS)
        * (1 + len(AXES) * len(DOSES))
    )
    progress = tqdm(total=total, desc="AVDOS counterfactual identity dose", unit="fit")
    for split_seed in SEEDS:
        subject_fold, stimulus_fold = fold_assignments(frame, split_seed)
        templates = make_shortcut_models(
            split_seed,
            args.n_jobs,
            MODELS,
            device="cuda",
            gpu_epochs=args.gpu_epochs,
        )
        for task in TASKS:
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5.0).astype(int)
            for subject_test in range(SUBJECT_FOLDS):
                for stimulus_test in range(STIMULUS_FOLDS):
                    test, train_sets, _ = paired.paired_train_sets(
                        frame,
                        labels,
                        subject_fold,
                        stimulus_fold,
                        subject_test,
                        stimulus_test,
                        "AVDOS-VR",
                        task,
                        split_seed,
                    )
                    fold = f"s{subject_test}_t{stimulus_test}"
                    axis_plans = {}
                    for axis in AXES:
                        fixed, identities, blocks, removed, candidates, class_counts = plans[
                            (task, split_seed, subject_test, stimulus_test, axis)
                        ]
                        axis_plans[axis] = (fixed, blocks)
                        reference_counts = np.bincount(
                            labels[train_sets["unseen_both"]], minlength=2
                        )
                        for block in blocks:
                            train = np.concatenate([fixed, block.indices])
                            counts = np.bincount(labels[train], minlength=2)
                            coverage = len(set(identities[test]) & set(identities[train])) / len(
                                set(identities[test])
                            )
                            if len(train) != len(train_sets["unseen_both"]):
                                raise ValueError(f"AVDOS training-size contract failed: {task}/{axis}/{fold}")
                            if not np.array_equal(counts, reference_counts):
                                raise ValueError(f"AVDOS class-count contract failed: {task}/{axis}/{fold}")
                            if set(train) & set(test) or not np.isclose(coverage, 1.0):
                                raise ValueError(f"AVDOS identity-exposure contract failed: {task}/{axis}/{fold}")
                            audit_rows.append(
                                {
                                    "dataset": "AVDOS-VR",
                                    "task": task,
                                    "axis": axis,
                                    "split_seed": split_seed,
                                    "fold": fold,
                                    "nominal_dose": block.nominal_dose,
                                    "achieved_dose": block.achieved_dose,
                                    "normalized_mutual_information": block.opportunity[
                                        "normalized_mutual_information"
                                    ],
                                    "metadata_prior_opportunity": metadata_prior_opportunity(
                                        labels[block.indices],
                                        identities[block.indices],
                                        labels[test],
                                        identities[test],
                                    ),
                                    "conditional_label_entropy": block.opportunity[
                                        "conditional_label_entropy"
                                    ],
                                    "identity_rate_std": block.opportunity["identity_rate_std"],
                                    "identity_rate_range": block.opportunity["identity_rate_range"],
                                    "candidate_min": block.candidate_min,
                                    "candidate_max": block.candidate_max,
                                    "candidate_size": len(candidates),
                                    "intervention_block_size": len(block.indices),
                                    "intervention_class_0": class_counts[0],
                                    "intervention_class_1": class_counts[1],
                                    "removed_control_size": len(removed),
                                    "intervention_block_fraction": 0.5,
                                    "n_train": len(train),
                                    "n_test": len(test),
                                    "class_0_train": counts[0],
                                    "class_1_train": counts[1],
                                    "identity_coverage": coverage,
                                    "row_overlap_with_test": 0,
                                }
                            )
                    for representation, features in feature_sets.items():
                        for model_name, template in templates.items():
                            baseline = clone(template).fit(
                                features[train_sets["unseen_both"]],
                                labels[train_sets["unseen_both"]],
                            )
                            baseline_probability = benchmark.model_probability(
                                baseline, features[test]
                            )
                            progress.update(1)
                            for axis in AXES:
                                rows.extend(
                                    dose.prediction_rows(
                                        frame,
                                        test,
                                        labels,
                                        baseline_probability,
                                        dataset="AVDOS-VR",
                                        task=task,
                                        axis=axis,
                                        representation=representation,
                                        model=model_name,
                                        split_seed=split_seed,
                                        fold=fold,
                                        condition="unseen",
                                        nominal_dose=None,
                                        achieved_dose=None,
                                    )
                                )
                                fixed, blocks = axis_plans[axis]
                                for block in blocks:
                                    train = np.concatenate([fixed, block.indices])
                                    fitted = clone(template).fit(features[train], labels[train])
                                    probability = benchmark.model_probability(
                                        fitted, features[test]
                                    )
                                    rows.extend(
                                        dose.prediction_rows(
                                            frame,
                                            test,
                                            labels,
                                            probability,
                                            dataset="AVDOS-VR",
                                            task=task,
                                            axis=axis,
                                            representation=representation,
                                            model=model_name,
                                            split_seed=split_seed,
                                            fold=fold,
                                            condition="dose",
                                            nominal_dose=block.nominal_dose,
                                            achieved_dose=block.achieved_dose,
                                        )
                                    )
                                    progress.update(1)
    progress.close()
    predictions = pd.DataFrame(rows)
    split_audit = pd.DataFrame(audit_rows)
    summary = dose.summarize(predictions, split_audit)
    args.output_root.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    split_audit.to_csv(args.output_root / "split_audit.csv", index=False)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    identity_encoding.to_csv(args.output_root / "identity_encoding_margin.csv", index=False)
    manifest = {
        "status": "registered_avdos_counterfactual_complete",
        "settings": settings,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_epochs": args.gpu_epochs,
        "reservation_sha256": file_sha256(RESERVATION),
        "reservation_amendment_sha256": file_sha256(AMENDMENT),
        "freeze_sha256": file_sha256(FREEZE),
        "implementation_lock_sha256": file_sha256(IMPLEMENTATION_LOCK),
        "feature_sha256": file_sha256(FEATURES),
        "script_sha256": file_sha256(Path(__file__)),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
