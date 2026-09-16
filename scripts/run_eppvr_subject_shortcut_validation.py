from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from identity_shortcut.core import (  # noqa: E402
    REPRESENTATIONS,
    build_dose_blocks,
    class_matched_control_split,
    metadata_prior_opportunity,
    representation_columns,
    stable_seed,
)
from identity_shortcut.models import make_shortcut_models  # noqa: E402
import run_protocol_model_benchmark as benchmark  # noqa: E402
import run_crossed_identity_probes as probes  # noqa: E402


MODEL_NAMES = (
    "linear_logistic",
    "rbf_svm",
    "extra_trees",
    "hist_gradient_boosting",
    "gpu_mlp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen external validation of subject-identity shortcut risk in EPPVR."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/unified_frontal_features/eppvr.csv"),
    )
    parser.add_argument(
        "--risk-root",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier"),
        help="Admissible public-data freeze that must exist before this external run.",
    )
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["all", "relative_power", "normalized_asymmetry", "baseline_delta"],
        choices=list(REPRESENTATIONS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear_logistic", "gpu_mlp"],
        choices=list(MODEL_NAMES),
        help="Frozen external validation defaults to the same sentinel capacity contrast as the dose experiment.",
    )
    parser.add_argument("--doses", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260813, 20260829, 20260911, 20260923, 20261007])
    parser.add_argument("--subject-folds", type=int, default=5)
    parser.add_argument("--record-folds", type=int, default=2)
    parser.add_argument(
        "--intervention-block-fraction",
        "--exposure-block-fraction",
        dest="intervention_block_fraction",
        type=float,
        default=0.5,
        help="Frozen fraction shared with the public-dataset counterfactual intervention.",
    )
    parser.add_argument("--candidate-draws", type=int, default=256)
    parser.add_argument("--gpu-epochs", type=int, default=40)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-fold-cells", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/eppvr_subject_shortcut_external_validation"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_fold_assignments(frame: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    assignments = np.empty(len(frame), dtype=int)
    for subject, indices in frame.groupby("subject_id").groups.items():
        indices = np.asarray(list(indices), dtype=int)
        rng = np.random.default_rng(stable_seed(seed, subject, "record-fold"))
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        assignments[shuffled] = np.arange(len(shuffled)) % folds
    return assignments


def block_counts(labels: np.ndarray, candidates: np.ndarray, fraction: float, n_identities: int) -> np.ndarray:
    available = np.bincount(labels[candidates], minlength=2)
    counts = np.floor(available * fraction).astype(int)
    if counts.sum() < n_identities:
        raise ValueError(
            "EPPVR intervention block is too small to cover every test subject identity"
        )
    return counts


def append_predictions(
    rows: list[dict],
    frame: pd.DataFrame,
    test: np.ndarray,
    labels: np.ndarray,
    probability: np.ndarray,
    **metadata,
) -> None:
    for index, estimate in zip(test, probability, strict=True):
        rows.append(
            {
                **metadata,
                "row_id": int(index),
                "subject_id": str(frame.iloc[index].subject_id),
                "record_id": str(frame.iloc[index].trial_id),
                "target": int(labels[index]),
                "probability": float(estimate),
            }
        )


def identity_probe_summary(
    feature_sets: dict[str, np.ndarray],
    identities: np.ndarray,
    frame: pd.DataFrame,
    seeds: list[int],
    record_folds: int,
) -> pd.DataFrame:
    rows = []
    total = len(feature_sets) * len(seeds) * record_folds * 2
    progress = tqdm(total=total, desc="EPPVR identity encoding", unit="fit", dynamic_ncols=True)
    for split_seed in seeds:
        assignments = record_fold_assignments(frame, record_folds, split_seed)
        templates = probes.make_models(split_seed)
        for representation, features in feature_sets.items():
            predictions = {name: np.empty(len(frame), dtype=object) for name in templates}
            for fold in range(record_folds):
                test = assignments == fold
                train = ~test
                for name, template in templates.items():
                    fitted = clone(template).fit(features[train], identities[train])
                    predictions[name][test] = fitted.predict(features[test])
                    progress.update(1)
            for name, estimate in predictions.items():
                rows.append(
                    {
                        "dataset": "EPPVR",
                        "representation": representation,
                        "n_features": features.shape[1],
                        "axis": "subject",
                        "probe_model": name,
                        "split_seed": split_seed,
                        "n_classes": len(np.unique(identities)),
                        "chance": 1 / len(np.unique(identities)),
                        "balanced_accuracy": probes.balanced_multiclass_accuracy(identities, estimate),
                    }
                )
    progress.close()
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "representation", "n_features", "model", "split_seed"]
    rows = []
    for key_values, group in predictions.groupby(keys, sort=True):
        unseen = group.loc[group.condition == "unseen"]
        unseen_score = benchmark.balanced_accuracy(
            unseen.target.to_numpy(int), unseen.probability.to_numpy(float)
        )
        for dose, dose_group in group.loc[group.condition == "dose"].groupby("nominal_dose"):
            score = benchmark.balanced_accuracy(
                dose_group.target.to_numpy(int), dose_group.probability.to_numpy(float)
            )
            rows.append(
                dict(
                    zip(keys, key_values, strict=True),
                    nominal_dose=float(dose),
                    unseen_balanced_accuracy=unseen_score,
                    dose_balanced_accuracy=score,
                    subject_exposure_effect=score - unseen_score,
                    achieved_dose=float(dose_group.achieved_dose.mean()),
                )
            )
    result = pd.DataFrame(rows)
    opportunities = (
        audit.groupby(["task", "split_seed", "nominal_dose"])
        .agg(
            normalized_mutual_information=("normalized_mutual_information", "mean"),
            metadata_prior_opportunity=("metadata_prior_opportunity", "mean"),
        )
        .reset_index()
    )
    return result.merge(
        opportunities,
        on=["task", "split_seed", "nominal_dose"],
        validate="many_to_one",
    )


def main() -> None:
    args = parse_args()
    freeze_path = args.risk_root / "freeze_manifest.json"
    frozen_model_path = args.risk_root / "frozen_risk_model.joblib"
    if not freeze_path.is_file() or not frozen_model_path.is_file():
        raise FileNotFoundError(
            "EPPVR is locked until an admissible public-data risk freeze exists"
        )
    risk_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if risk_freeze.get("admissible") is not True:
        raise ValueError("EPPVR cannot run against a rejected risk-model candidate")
    if sha256(frozen_model_path) != risk_freeze.get("model_sha256"):
        raise ValueError("Risk-model hash does not match the public-data freeze manifest")
    if risk_freeze.get("axis") != "subject":
        raise ValueError("EPPVR external validation requires a subject-axis risk freeze")
    if not np.isclose(
        args.intervention_block_fraction,
        float(risk_freeze["intervention_block_fraction"]),
    ):
        raise ValueError(
            "EPPVR intervention fraction differs from the frozen public-data estimand"
        )
    if not 0.0 < args.intervention_block_fraction < 1.0:
        raise ValueError("--intervention-block-fraction must lie strictly between 0 and 1")
    required_outputs = [args.output_root / name for name in ("predictions.csv", "split_audit.csv", "summary.csv")]
    if any(path.exists() for path in required_outputs) and not args.force:
        raise FileExistsError(f"Output exists under {args.output_root}; pass --force to replace it")
    frame = pd.read_csv(args.input).reset_index(drop=True)
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["trial_id"] = frame.trial_id.astype(str)
    if frame.subject_id.nunique() != 30 or len(frame) != 420:
        raise ValueError("EPPVR external-validation contract requires 30 subjects and 420 trial rows")
    feature_sets = {
        name: frame[representation_columns(frame, name)].to_numpy(float)
        for name in args.representations
    }
    encoding = identity_probe_summary(
        feature_sets,
        frame.subject_id.to_numpy(),
        frame,
        args.seeds,
        args.record_folds,
    )
    cells = args.subject_folds * args.record_folds
    if args.max_fold_cells is not None:
        cells = min(cells, args.max_fold_cells)
    total = (
        len(args.tasks)
        * len(args.seeds)
        * cells
        * len(args.representations)
        * len(args.models)
        * (1 + len(args.doses))
    )
    progress = tqdm(total=total, desc="EPPVR subject-shortcut validation", unit="fit", dynamic_ncols=True)
    prediction_rows: list[dict] = []
    audit_rows: list[dict] = []
    for split_seed in args.seeds:
        subject_map = benchmark.shuffled_fold_map(
            frame.subject_id.to_numpy(), args.subject_folds, split_seed
        )
        subject_fold = frame.subject_id.map(subject_map).to_numpy(int)
        record_fold = record_fold_assignments(frame, args.record_folds, split_seed)
        templates = make_shortcut_models(
            split_seed,
            args.n_jobs,
            args.models,
            device=args.device,
            gpu_epochs=args.gpu_epochs,
        )
        fold_cells = [
            (subject_test, record_test)
            for subject_test in range(args.subject_folds)
            for record_test in range(args.record_folds)
        ]
        if args.max_fold_cells is not None:
            fold_cells = fold_cells[: args.max_fold_cells]
        for task in args.tasks:
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
            identities = frame.subject_id.to_numpy()
            for subject_test, record_test in fold_cells:
                test = np.flatnonzero(
                    (subject_fold == subject_test) & (record_fold == record_test)
                )
                exposure_candidates = np.flatnonzero(
                    (subject_fold == subject_test) & (record_fold != record_test)
                )
                control = np.flatnonzero(subject_fold != subject_test)
                required_identities = identities[test]
                counts = block_counts(
                    labels,
                    exposure_candidates,
                    args.intervention_block_fraction,
                    len(np.unique(required_identities)),
                )
                core, control_block = class_matched_control_split(
                    control,
                    labels,
                    counts,
                    stable_seed(split_seed, task, subject_test, record_test, "control"),
                )
                blocks = build_dose_blocks(
                    exposure_candidates,
                    labels,
                    identities,
                    required_identities,
                    counts,
                    args.doses,
                    args.candidate_draws,
                    stable_seed(split_seed, task, subject_test, record_test, "dose"),
                )
                fold = f"s{subject_test}_r{record_test}"
                for block in blocks:
                    train = np.concatenate([core, block.indices])
                    if len(train) != len(control):
                        raise ValueError(f"EPPVR training-size contract failed in {fold}")
                    if not np.array_equal(
                        np.bincount(labels[train], minlength=2),
                        np.bincount(labels[control], minlength=2),
                    ):
                        raise ValueError(f"EPPVR class-count contract failed in {fold}")
                    if set(train) & set(test):
                        raise ValueError(f"EPPVR train-test overlap detected in {fold}")
                    coverage = len(set(identities[test]) & set(identities[train])) / len(
                        set(identities[test])
                    )
                    if not np.isclose(coverage, 1.0):
                        raise ValueError(f"EPPVR subject exposure is incomplete in {fold}")
                    audit_rows.append(
                        {
                            "task": task,
                            "split_seed": split_seed,
                            "fold": fold,
                            "nominal_dose": block.nominal_dose,
                            "achieved_dose": block.achieved_dose,
                            "normalized_mutual_information": block.opportunity["normalized_mutual_information"],
                            "metadata_prior_opportunity": metadata_prior_opportunity(
                                labels[block.indices],
                                identities[block.indices],
                                labels[test],
                                identities[test],
                            ),
                            "candidate_min": block.candidate_min,
                            "candidate_max": block.candidate_max,
                            "candidate_size": len(exposure_candidates),
                            "intervention_block_size": len(block.indices),
                            "intervention_class_0": counts[0],
                            "intervention_class_1": counts[1],
                            "removed_control_size": len(control_block),
                            "intervention_block_fraction": args.intervention_block_fraction,
                            "n_train": len(train),
                            "n_test": len(test),
                            "class_0_train": int(np.sum(labels[train] == 0)),
                            "class_1_train": int(np.sum(labels[train] == 1)),
                            "subject_coverage": coverage,
                            "row_overlap_with_test": 0,
                        }
                    )
                for representation, features in feature_sets.items():
                    for model_name, template in templates.items():
                        unseen_model = clone(template).fit(features[control], labels[control])
                        unseen_probability = benchmark.model_probability(unseen_model, features[test])
                        append_predictions(
                            prediction_rows,
                            frame,
                            test,
                            labels,
                            unseen_probability,
                            task=task,
                            representation=representation,
                            n_features=features.shape[1],
                            model=model_name,
                            split_seed=split_seed,
                            fold=fold,
                            condition="unseen",
                            nominal_dose=None,
                            achieved_dose=None,
                        )
                        progress.update(1)
                        for block in blocks:
                            train = np.concatenate([core, block.indices])
                            dose_model = clone(template).fit(features[train], labels[train])
                            probability = benchmark.model_probability(dose_model, features[test])
                            append_predictions(
                                prediction_rows,
                                frame,
                                test,
                                labels,
                                probability,
                                task=task,
                                representation=representation,
                                n_features=features.shape[1],
                                model=model_name,
                                split_seed=split_seed,
                                fold=fold,
                                condition="dose",
                                nominal_dose=block.nominal_dose,
                                achieved_dose=block.achieved_dose,
                            )
                            progress.update(1)
    progress.close()
    predictions = pd.DataFrame(prediction_rows)
    audit = pd.DataFrame(audit_rows)
    result = summarize(predictions, audit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_root / "predictions.csv", index=False)
    audit.to_csv(args.output_root / "split_audit.csv", index=False)
    result.to_csv(args.output_root / "summary.csv", index=False)
    encoding.to_csv(args.output_root / "identity_probe_summary.csv", index=False)
    manifest = {
        "dataset": "EPPVR",
        "input": str(args.input),
        "validation_axis": "subject identity only",
        "record_partition": "within-subject shuffled record folds; record IDs are not treated as shared physical stimuli",
        "forbidden_claims": [
            "stimulus exposure",
            "subject-by-stimulus interaction",
            "cross-session generalization",
        ],
        "tasks": args.tasks,
        "representations": args.representations,
        "models": args.models,
        "doses": args.doses,
        "intervention_block_fraction": args.intervention_block_fraction,
        "split_seeds": args.seeds,
        "device": args.device,
        "gpu_epochs": args.gpu_epochs,
        "risk_freeze_manifest": str(freeze_path),
        "risk_model_sha256_frozen_before_external_run": risk_freeze["model_sha256"],
        "risk_target_definition": risk_freeze["target_definition"],
        "risk_external_acceptance_rule": risk_freeze["external_acceptance_rule"],
        "external_freeze_rule": "Risk-model coefficients and feature definitions must be frozen before interpreting EPPVR outcomes.",
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Results written to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
