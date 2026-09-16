from __future__ import annotations

import argparse
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

from faced_v11_contract import (  # noqa: E402
    DOSES,
    GPU_EPOCHS,
    MODELS,
    REPRESENTATIONS,
    SEEDS,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
    sha256,
    verify_implementation_lock,
)
from identity_shortcut.core import metadata_prior_opportunity, stable_seed  # noqa: E402
from identity_shortcut.models import make_shortcut_models  # noqa: E402
import run_counterfactual_identity_dose as dose  # noqa: E402
import run_crossed_identity_probes as probes  # noqa: E402
import run_paired_block_exposure_benchmark as paired  # noqa: E402
import run_protocol_model_benchmark as benchmark  # noqa: E402


DATASET = "FACED"
TASK = "polarity"
AXIS = "subject"
FEATURE_ROOT = ROOT / "outputs/faced_v11_post_access_common_montage_features"
OUTPUT_ROOT = ROOT / "outputs/faced_v11_post_access_common_montage_dose"
EXPLORATORY_PROTOCOL = (
    ROOT / "docs/faced_v11_post_access_common_montage_protocol_amendment_001.json"
)
EXPLORATORY_ANALYSIS_LOCK = (
    ROOT / "docs/faced_v11_post_access_common_montage_analysis_lock.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the post-access exploratory FACED common-montage "
            "subject-identity dose experiment on CUDA."
        )
    )
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--gpu-epochs", type=int, default=GPU_EPOCHS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def verify_exploratory_analysis_lock(feature_manifest_path: Path) -> dict[str, object]:
    if not EXPLORATORY_ANALYSIS_LOCK.is_file():
        raise RuntimeError("FACED post-access exploratory analysis lock is absent")
    lock = json.loads(EXPLORATORY_ANALYSIS_LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "post_access_exploratory_analysis_locked_before_dose":
        raise RuntimeError("FACED post-access exploratory analysis lock is invalid")
    expected = {
        EXPLORATORY_PROTOCOL: lock["protocol_sha256"],
        feature_manifest_path: lock["feature_manifest_sha256"],
    }
    for relative, digest in lock["analysis_code_sha256"].items():
        expected[ROOT / relative] = digest
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Locked FACED exploratory artifact changed: {path}")
    if lock.get("confirmatory_claim_permitted") is not False:
        raise RuntimeError("Exploratory lock does not prohibit confirmatory claims")
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


def preflight(frame: pd.DataFrame, labels: np.ndarray) -> dict[tuple, tuple]:
    cache = {}
    total = len(SEEDS) * SUBJECT_FOLDS * STIMULUS_FOLDS
    progress = tqdm(
        total=total, desc="FACED preflight dose plans", unit="plan", dynamic_ncols=True
    )
    for split_seed in SEEDS:
        subject_fold, stimulus_fold = fold_assignments(frame, split_seed)
        for subject_test in range(SUBJECT_FOLDS):
            for stimulus_test in range(STIMULUS_FOLDS):
                test, train_sets, _ = paired.paired_train_sets(
                    frame,
                    labels,
                    subject_fold,
                    stimulus_fold,
                    subject_test,
                    stimulus_test,
                    DATASET,
                    TASK,
                    split_seed,
                )
                fold = f"s{subject_test}_t{stimulus_test}"
                cache[(split_seed, subject_test, stimulus_test)] = dose.axis_plan(
                    frame,
                    labels,
                    subject_fold,
                    stimulus_fold,
                    subject_test,
                    stimulus_test,
                    train_sets,
                    test,
                    AXIS,
                    list(DOSES),
                    256,
                    0.5,
                    stable_seed(DATASET, TASK, AXIS, split_seed, fold),
                )
                progress.update(1)
    progress.close()
    return cache


def identity_encoding_table(
    frame: pd.DataFrame,
    feature_sets: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    total = len(feature_sets) * len(SEEDS) * STIMULUS_FOLDS * 2
    progress = tqdm(
        total=total, desc="FACED subject-identity probe", unit="fit", dynamic_ncols=True
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
                    missing = set(targets[test]) - set(targets[~test])
                    if missing:
                        raise ValueError(
                            f"FACED identity probe has unseen training classes: {sorted(missing)}"
                        )
                    fitted = clone(template).fit(features[~test], targets[~test])
                    scores.append(
                        probes.balanced_multiclass_accuracy(
                            targets[test], fitted.predict(features[test])
                        )
                    )
                    progress.update(1)
                rows.append(
                    {
                        "dataset": DATASET,
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


def audit_rows_for_plan(
    frame: pd.DataFrame,
    labels: np.ndarray,
    test: np.ndarray,
    train_sets: dict[str, np.ndarray],
    plan: tuple,
    split_seed: int,
    fold: str,
) -> tuple[tuple[np.ndarray, list], list[dict[str, object]]]:
    fixed, identities, blocks, removed, candidates, class_counts = plan
    reference_counts = np.bincount(labels[train_sets["unseen_both"]], minlength=2)
    output = []
    for block in blocks:
        train = np.concatenate([fixed, block.indices])
        counts = np.bincount(labels[train], minlength=2)
        coverage = len(set(identities[test]) & set(identities[train])) / len(
            set(identities[test])
        )
        if len(train) != len(train_sets["unseen_both"]) or not np.array_equal(
            counts, reference_counts
        ):
            raise ValueError(f"FACED dose matching failed: seed={split_seed}/{fold}")
        if set(train) & set(test) or not np.isclose(coverage, 1.0):
            raise ValueError(
                f"FACED identity-exposure contract failed: seed={split_seed}/{fold}"
            )
        output.append(
            {
                "dataset": DATASET,
                "task": TASK,
                "axis": AXIS,
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
    return (fixed, blocks), output


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite FACED dose output: {args.output_root}"
        )
    if args.gpu_epochs != GPU_EPOCHS:
        raise ValueError(f"The exploratory FACED protocol fixes --gpu-epochs at {GPU_EPOCHS}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the exploratory FACED gpu_mlp experiment"
        )
    lock = verify_implementation_lock()

    feature_path = args.feature_root / "faced_polarity_trial_features.csv"
    feature_manifest_path = args.feature_root / "build_manifest.json"
    for path in (feature_path, feature_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if (
        feature_manifest.get("status")
        != "faced_post_access_common_montage_trial_features_built"
    ):
        raise RuntimeError("FACED exploratory common-montage feature manifest is incomplete")
    if feature_manifest.get("confirmatory_claim_permitted") is not False:
        raise RuntimeError("FACED feature manifest does not preserve exploratory status")
    exploratory_lock = verify_exploratory_analysis_lock(feature_manifest_path)
    if feature_manifest["feature_sha256"] != sha256(feature_path):
        raise RuntimeError("FACED feature-table checksum mismatch")
    if tuple(feature_manifest["representations"]) != REPRESENTATIONS:
        raise ValueError("FACED feature representations differ from the reservation")

    frame = pd.read_csv(feature_path).reset_index(drop=True)
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["trial_id"] = frame.trial_id.astype(int)
    labels = (frame.polarity_score.to_numpy(float) >= 5.0).astype(int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("FACED polarity task is not binary")
    if frame.groupby("subject_id").trial_id.nunique().nunique() != 1:
        raise ValueError(
            "FACED retained participants do not share a complete stimulus map"
        )
    feature_sets = {
        name: frame[columns].to_numpy(float)
        for name, columns in feature_manifest["representations"].items()
    }
    if any(
        values.shape[1] == 0 or not np.isfinite(values).all()
        for values in feature_sets.values()
    ):
        raise ValueError("A FACED exploratory representation is empty or non-finite")

    identity_encoding = identity_encoding_table(frame, feature_sets)
    plans = preflight(frame, labels)
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    total = (
        len(SEEDS)
        * SUBJECT_FOLDS
        * STIMULUS_FOLDS
        * len(REPRESENTATIONS)
        * len(MODELS)
        * (1 + len(DOSES))
    )
    progress = tqdm(
        total=total,
        desc="FACED counterfactual subject-identity dose",
        unit="fit",
        dynamic_ncols=True,
    )
    for split_seed in SEEDS:
        subject_fold, stimulus_fold = fold_assignments(frame, split_seed)
        templates = make_shortcut_models(
            split_seed,
            args.n_jobs,
            list(MODELS),
            device="cuda",
            gpu_epochs=args.gpu_epochs,
        )
        for subject_test in range(SUBJECT_FOLDS):
            for stimulus_test in range(STIMULUS_FOLDS):
                test, train_sets, _ = paired.paired_train_sets(
                    frame,
                    labels,
                    subject_fold,
                    stimulus_fold,
                    subject_test,
                    stimulus_test,
                    DATASET,
                    TASK,
                    split_seed,
                )
                fold = f"s{subject_test}_t{stimulus_test}"
                axis_plan, plan_audit = audit_rows_for_plan(
                    frame,
                    labels,
                    test,
                    train_sets,
                    plans[(split_seed, subject_test, stimulus_test)],
                    split_seed,
                    fold,
                )
                audit_rows.extend(plan_audit)
                fixed, blocks = axis_plan
                for representation, features in feature_sets.items():
                    for model_name, template in templates.items():
                        baseline = clone(template).fit(
                            features[train_sets["unseen_both"]],
                            labels[train_sets["unseen_both"]],
                        )
                        baseline_probability = benchmark.model_probability(
                            baseline, features[test]
                        )
                        rows.extend(
                            dose.prediction_rows(
                                frame,
                                test,
                                labels,
                                baseline_probability,
                                dataset=DATASET,
                                task=TASK,
                                axis=AXIS,
                                representation=representation,
                                model=model_name,
                                split_seed=split_seed,
                                fold=fold,
                                condition="unseen",
                                nominal_dose=None,
                                achieved_dose=None,
                            )
                        )
                        progress.update(1)
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
                                    dataset=DATASET,
                                    task=TASK,
                                    axis=AXIS,
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
    expected_summary_rows = len(REPRESENTATIONS) * len(MODELS) * len(SEEDS) * len(DOSES)
    if len(summary) != expected_summary_rows:
        raise ValueError(
            f"Expected {expected_summary_rows} FACED summary rows, found {len(summary)}"
        )
    expected_audit_rows = len(SEEDS) * SUBJECT_FOLDS * STIMULUS_FOLDS * len(DOSES)
    if len(split_audit) != expected_audit_rows:
        raise ValueError(
            f"Expected {expected_audit_rows} split-audit rows, found {len(split_audit)}"
        )

    args.output_root.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    split_audit.to_csv(args.output_root / "split_audit.csv", index=False)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    identity_encoding.to_csv(
        args.output_root / "identity_encoding_margin.csv", index=False
    )
    manifest = {
        "status": "faced_post_access_common_montage_counterfactual_complete",
        "evidence_role": "post_access_exploratory_external_stress_test_only",
        "confirmatory_claim_permitted": False,
        "dataset": DATASET,
        "task": TASK,
        "axes": [AXIS],
        "representations": list(REPRESENTATIONS),
        "models": list(MODELS),
        "split_seeds": list(SEEDS),
        "subject_folds": SUBJECT_FOLDS,
        "stimulus_folds": STIMULUS_FOLDS,
        "doses": list(DOSES),
        "candidate_draws": 256,
        "intervention_block_fraction": 0.5,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_epochs": args.gpu_epochs,
        "feature_sha256": sha256(feature_path),
        "feature_manifest_sha256": sha256(feature_manifest_path),
        "implementation_lock_status": lock["status"],
        "exploratory_analysis_lock_sha256": sha256(EXPLORATORY_ANALYSIS_LOCK),
        "exploratory_analysis_lock_status": exploratory_lock["status"],
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
