from __future__ import annotations

import argparse
import hashlib
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
import run_matched_exposure_benchmark as matched  # noqa: E402
import run_paired_block_exposure_benchmark as paired  # noqa: E402
import run_protocol_model_benchmark as benchmark  # noqa: E402


RESERVATION = Path("docs/ceap_v6_external_confirmation_reservation.json")
FREEZE = Path("docs/order_preserving_transport_v6_final_freeze.json")
IMPLEMENTATION_LOCK = Path(
    "docs/ceap_v6_external_confirmation_implementation_lock_amendment_001.json"
)
FEATURES = Path("outputs/ceap_trial_features/ceap_trial_features.csv")
REPRESENTATIONS = {
    "wearable_all": ("eda", "bvp", "skt", "hr"),
    "cardiovascular": ("bvp", "hr"),
    "electrodermal_thermal": ("eda", "skt"),
    "minimal_pair": ("bvp", "eda"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run registered CEAP identity-dose experiment.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/ceap_counterfactual_identity_dose"),
    )
    parser.add_argument("--gpu-epochs", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def representation_columns(frame: pd.DataFrame, name: str) -> list[str]:
    sensors = REPRESENTATIONS[name]
    return [
        column for column in frame.columns
        if column.startswith("feature__") and any(f"__{sensor}_" in column for sensor in sensors)
    ]


def preflight(frame: pd.DataFrame, settings: dict[str, object]) -> dict[tuple, tuple]:
    cache = {}
    total = len(settings["seeds"]) * len(settings["tasks"]) * 25 * len(settings["axes"])
    progress = tqdm(total=total, desc="CEAP preflight dose plans", unit="plan", dynamic_ncols=True)
    for split_seed in settings["seeds"]:
        subject_fold, stimulus_fold = matched.fold_assignments(frame, settings["folds"], split_seed)
        fold_cells = [(subject, stimulus) for subject in range(5) for stimulus in range(5)]
        for task in settings["tasks"]:
            labels = (frame[f"{task}_score"].to_numpy(float) >= settings["label_threshold"]).astype(int)
            for subject_test, stimulus_test in fold_cells:
                test, train_sets, _ = paired.paired_train_sets(
                    frame, labels, subject_fold, stimulus_fold,
                    subject_test, stimulus_test, "CEAP-360VR", task, split_seed,
                )
                fold = f"s{subject_test}_t{stimulus_test}"
                for axis in settings["axes"]:
                    cache[(task, split_seed, subject_test, stimulus_test, axis)] = dose.axis_plan(
                        frame, labels, subject_fold, stimulus_fold,
                        subject_test, stimulus_test, train_sets, test, axis,
                        settings["doses"], settings["candidate_draws"],
                        settings["intervention_block_fraction"],
                        stable_seed("CEAP-360VR", task, axis, split_seed, fold),
                    )
                    progress.update(1)
    progress.close()
    return cache


def identity_encoding_table(
    frame: pd.DataFrame,
    feature_sets: dict[str, np.ndarray],
    seeds: list[int],
    folds: int,
) -> pd.DataFrame:
    rows = []
    progress = tqdm(
        total=len(feature_sets) * len(seeds) * folds * 2,
        desc="CEAP subject-identity probe",
        unit="fit",
        dynamic_ncols=True,
    )
    targets = frame.subject_id.astype(str).to_numpy()
    for representation, features in feature_sets.items():
        for split_seed in seeds:
            group_map = benchmark.shuffled_fold_map(frame.trial_id.to_numpy(), folds, split_seed)
            group_folds = frame.trial_id.map(group_map).to_numpy(int)
            for model_name, template in probes.make_models(split_seed).items():
                scores = []
                for fold in range(folds):
                    test = group_folds == fold
                    fitted = clone(template).fit(features[~test], targets[~test])
                    scores.append(
                        probes.balanced_multiclass_accuracy(targets[test], fitted.predict(features[test]))
                    )
                    progress.update(1)
                rows.append(
                    {
                        "dataset": "CEAP-360VR",
                        "representation": representation,
                        "n_features": features.shape[1],
                        "model": model_name,
                        "split_seed": split_seed,
                        "chance": 1.0 / frame.subject_id.nunique(),
                        "balanced_accuracy": float(np.mean(scores)),
                    }
                )
    progress.close()
    output = pd.DataFrame(rows)
    output["encoding_margin"] = output.balanced_accuracy - output.chance
    return output


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite CEAP dose output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered CEAP gpu_mlp experiment")
    reservation = json.loads(RESERVATION.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    implementation = json.loads(IMPLEMENTATION_LOCK.read_text(encoding="utf-8"))
    if reservation["dataset"] != "CEAP-360VR" or freeze["status"] != "frozen_for_one_shot_external_confirmation":
        raise ValueError("CEAP reservation or OPCT v6 freeze is invalid")
    if implementation["reservation_sha256"] != sha256(RESERVATION):
        raise ValueError("CEAP implementation lock does not match the reservation")
    if implementation["freeze_sha256"] != sha256(FREEZE):
        raise ValueError("CEAP implementation lock does not match the OPCT v6 freeze")
    for relative, expected in implementation["analysis_code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Registered CEAP analysis code changed: {relative}")
    settings = {
        "tasks": ["arousal", "valence"],
        "axes": ["subject", "stimulus"],
        "representations": list(REPRESENTATIONS),
        "models": ["linear_logistic", "gpu_mlp"],
        "seeds": [20260813, 20260829, 20260911, 20260923, 20261007],
        "folds": 5,
        "label_threshold": 5.0,
        "doses": [0.0, 0.25, 0.5, 0.75, 1.0],
        "candidate_draws": 256,
        "intervention_block_fraction": 0.5,
    }
    frame = pd.read_csv(FEATURES).reset_index(drop=True)
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["trial_id"] = frame.trial_id.astype(str)
    feature_sets = {
        name: frame[representation_columns(frame, name)].to_numpy(float)
        for name in settings["representations"]
    }
    if any(values.shape[1] == 0 for values in feature_sets.values()):
        raise ValueError("A registered CEAP representation is empty")
    identity_encoding = identity_encoding_table(frame, feature_sets, settings["seeds"], 5)
    plans = preflight(frame, settings)
    rows = []
    audit_rows = []
    total = 2 * 5 * 25 * 4 * 2 * (1 + 2 * 5)
    progress = tqdm(
        total=total,
        desc="CEAP counterfactual identity dose",
        unit="fit",
        dynamic_ncols=True,
    )
    for split_seed in settings["seeds"]:
        subject_fold, stimulus_fold = matched.fold_assignments(frame, 5, split_seed)
        templates = make_shortcut_models(
            split_seed, args.n_jobs, settings["models"], device="cuda", gpu_epochs=args.gpu_epochs
        )
        for task in settings["tasks"]:
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5.0).astype(int)
            for subject_test in range(5):
                for stimulus_test in range(5):
                    test, train_sets, _ = paired.paired_train_sets(
                        frame, labels, subject_fold, stimulus_fold,
                        subject_test, stimulus_test, "CEAP-360VR", task, split_seed,
                    )
                    fold = f"s{subject_test}_t{stimulus_test}"
                    axis_plans = {}
                    for axis in settings["axes"]:
                        fixed, identities, blocks, removed, candidates, class_counts = plans[
                            (task, split_seed, subject_test, stimulus_test, axis)
                        ]
                        axis_plans[axis] = (fixed, blocks)
                        reference_counts = np.bincount(labels[train_sets["unseen_both"]], minlength=2)
                        for block in blocks:
                            train = np.concatenate([fixed, block.indices])
                            counts = np.bincount(labels[train], minlength=2)
                            coverage = len(set(identities[test]) & set(identities[train])) / len(set(identities[test]))
                            if len(train) != len(train_sets["unseen_both"]) or not np.array_equal(counts, reference_counts):
                                raise ValueError(f"CEAP dose contract failed for {task}/{axis}/{fold}")
                            if set(train) & set(test) or not np.isclose(coverage, 1.0):
                                raise ValueError(f"CEAP identity exposure contract failed for {task}/{axis}/{fold}")
                            audit_rows.append(
                                {
                                    "dataset": "CEAP-360VR", "task": task, "axis": axis,
                                    "split_seed": split_seed, "fold": fold,
                                    "nominal_dose": block.nominal_dose,
                                    "achieved_dose": block.achieved_dose,
                                    "normalized_mutual_information": block.opportunity["normalized_mutual_information"],
                                    "metadata_prior_opportunity": metadata_prior_opportunity(
                                        labels[block.indices], identities[block.indices], labels[test], identities[test]
                                    ),
                                    "conditional_label_entropy": block.opportunity["conditional_label_entropy"],
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
                                    "n_train": len(train), "n_test": len(test),
                                    "class_0_train": counts[0], "class_1_train": counts[1],
                                    "identity_coverage": coverage, "row_overlap_with_test": 0,
                                }
                            )
                    for representation, features in feature_sets.items():
                        for model_name, template in templates.items():
                            baseline = clone(template).fit(
                                features[train_sets["unseen_both"]], labels[train_sets["unseen_both"]]
                            )
                            baseline_probability = benchmark.model_probability(baseline, features[test])
                            progress.update(1)
                            for axis in settings["axes"]:
                                rows.extend(
                                    dose.prediction_rows(
                                        frame, test, labels, baseline_probability,
                                        dataset="CEAP-360VR", task=task, axis=axis,
                                        representation=representation, model=model_name,
                                        split_seed=split_seed, fold=fold, condition="unseen",
                                        nominal_dose=None, achieved_dose=None,
                                    )
                                )
                                fixed, blocks = axis_plans[axis]
                                for block in blocks:
                                    fitted = clone(template).fit(
                                        features[np.concatenate([fixed, block.indices])],
                                        labels[np.concatenate([fixed, block.indices])],
                                    )
                                    probability = benchmark.model_probability(fitted, features[test])
                                    rows.extend(
                                        dose.prediction_rows(
                                            frame, test, labels, probability,
                                            dataset="CEAP-360VR", task=task, axis=axis,
                                            representation=representation, model=model_name,
                                            split_seed=split_seed, fold=fold, condition="dose",
                                            nominal_dose=block.nominal_dose,
                                            achieved_dose=block.achieved_dose,
                                        )
                                    )
                                    progress.update(1)
    progress.close()
    predictions = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    summary = dose.summarize(predictions, audit)
    args.output_root.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    audit.to_csv(args.output_root / "split_audit.csv", index=False)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    identity_encoding.to_csv(args.output_root / "identity_encoding_margin.csv", index=False)
    manifest = {
        "status": "registered_ceap_counterfactual_complete",
        "settings": settings,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_epochs": args.gpu_epochs,
        "reservation_sha256": sha256(RESERVATION),
        "freeze_sha256": sha256(FREEZE),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "feature_sha256": sha256(FEATURES),
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
