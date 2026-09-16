from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from identity_shortcut.core import identity_opportunity  # noqa: E402
import run_matched_exposure_benchmark as matched  # noqa: E402
import run_paired_block_exposure_benchmark as paired  # noqa: E402


MODEL_CAPACITY = {
    "linear_logistic": 1.0,
    "rbf_svm": 2.0,
    "extra_trees": 3.0,
    "hist_gradient_boosting": 3.0,
    "gpu_mlp": 4.0,
}
AXIS_EXPOSURE = {"subject": "seen_subject", "stimulus": "seen_stimulus"}
AXIS_PROBE = {
    "subject": "subject_across_stimuli",
    "stimulus": "stimulus_across_subjects",
}
AXIS_EFFECT = {
    "subject": "subject_exposure_effect",
    "stimulus": "stimulus_exposure_effect",
}
ALL_EFFECT = {
    "subject_exposure_main_effect": "subject",
    "stimulus_exposure_main_effect": "stimulus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and cross-validate a pre-training identity-shortcut risk model."
    )
    parser.add_argument(
        "--opportunity",
        type=Path,
        default=Path(
            "outputs/paired_block_identity_exposure_benchmark_crossed_valid/identity_opportunity_per_seed.csv"
        ),
    )
    parser.add_argument(
        "--identity-probes",
        type=Path,
        default=Path("outputs/representation_identity_audit_crossed_valid/per_seed_summary.csv"),
    )
    parser.add_argument(
        "--representation-effects",
        type=Path,
        default=Path(
            "outputs/representation_exposure_gate_crossed_valid/factorial_effects_per_seed_model.csv"
        ),
    )
    parser.add_argument(
        "--all-effects",
        type=Path,
        default=Path(
            "outputs/paired_block_identity_exposure_benchmark_crossed_valid/factorial_effects_per_classifier_seed.csv"
        ),
    )
    parser.add_argument(
        "--counterfactual-summary",
        type=Path,
        default=Path("outputs/counterfactual_identity_dose_crossed_valid/summary.csv"),
        help="Optional completed dose-response summary; omitted automatically when absent.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=[
            "DEAP=outputs/unified_frontal_features/deap.csv",
            "MAHNOB-HCI=outputs/unified_frontal_features/mahnob-hci.csv",
        ],
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge penalty for the retained continuous-effect falsification analysis.",
    )
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--material-effect-threshold", type=float, default=0.02)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    parser.add_argument("--minimum-auroc", type=float, default=0.65)
    parser.add_argument("--minimum-average-precision-lift", type=float, default=1.5)
    parser.add_argument("--minimum-brier-skill", type=float, default=0.0)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.60)
    parser.add_argument("--minimum-events-per-dataset", type=int, default=20)
    parser.add_argument(
        "--training-source",
        choices=["counterfactual_only", "all"],
        default="counterfactual_only",
        help=(
            "Freeze on the estimand-matched counterfactual dose rows by default; "
            "the mixed-source option is a sensitivity analysis only."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_descriptors(specifications: list[str]) -> pd.DataFrame:
    rows = []
    for specification in specifications:
        dataset, path = specification.split("=", 1)
        frame = pd.read_csv(path)
        subjects = frame.subject_id.nunique()
        stimuli = frame.trial_id.nunique()
        for task in ("arousal", "valence"):
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "n_rows": len(frame),
                    "n_subjects": subjects,
                    "n_repeated_units": stimuli,
                    "repeated_measure_completeness": len(frame) / (subjects * stimuli),
                    "positive_fraction": labels.mean(),
                    "class_balance": 1.0 - abs(labels.mean() - 0.5) * 2.0,
                }
            )
    return pd.DataFrame(rows)


def observational_nmi(specifications: list[str], seeds: list[int], folds: int = 5) -> pd.DataFrame:
    rows = []
    total = len(specifications) * len(seeds) * 2 * folds**2 * 2
    progress = tqdm(total=total, desc="Observational opportunity NMI", unit="block", dynamic_ncols=True)
    for specification in specifications:
        dataset, path = specification.split("=", 1)
        frame = pd.read_csv(path).reset_index(drop=True)
        frame["subject_id"] = frame.subject_id.astype(str)
        frame["trial_id"] = frame.trial_id.astype(str)
        subject_identities = frame.subject_id.to_numpy()
        stimulus_identities = frame.trial_id.to_numpy()
        for split_seed in seeds:
            subject_fold, stimulus_fold = matched.fold_assignments(frame, folds, split_seed)
            for task in ("arousal", "valence"):
                labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
                for subject_test in range(folds):
                    for stimulus_test in range(folds):
                        _, train_sets, _ = paired.paired_train_sets(
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
                        unseen = train_sets["unseen_both"]
                        for axis, exposure, identities in (
                            ("subject", "seen_subject", subject_identities),
                            ("stimulus", "seen_stimulus", stimulus_identities),
                        ):
                            block = np.setdiff1d(train_sets[exposure], unseen, assume_unique=False)
                            metrics = identity_opportunity(labels[block], identities[block])
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "task": task,
                                    "split_seed": split_seed,
                                    "axis": axis,
                                    "fold": f"s{subject_test}_t{stimulus_test}",
                                    "label_opportunity": metrics["normalized_mutual_information"],
                                    "conditional_label_entropy": metrics["conditional_label_entropy"],
                                    "identity_rate_std": metrics["identity_rate_std"],
                                }
                            )
                            progress.update(1)
    progress.close()
    return (
        pd.DataFrame(rows)
        .groupby(["dataset", "task", "split_seed", "axis"])
        .agg(
            label_opportunity=("label_opportunity", "mean"),
            conditional_label_entropy=("conditional_label_entropy", "mean"),
            identity_rate_std=("identity_rate_std", "mean"),
        )
        .reset_index()
    )


def load_effects(args: argparse.Namespace) -> pd.DataFrame:
    representation = pd.read_csv(args.representation_effects)
    rows = []
    for axis, column in AXIS_EFFECT.items():
        subset = representation[
            ["dataset", "task", "representation", "n_features", "model", "split_seed", column]
        ].rename(columns={column: "exposure_effect"})
        subset["axis"] = axis
        rows.append(subset)
    all_effects = pd.read_csv(args.all_effects)
    all_effects = all_effects.loc[all_effects.contrast.isin(ALL_EFFECT)].copy()
    all_effects["axis"] = all_effects.contrast.map(ALL_EFFECT)
    all_effects["representation"] = "all"
    all_effects["n_features"] = 72
    all_effects = all_effects.rename(columns={"balanced_accuracy_effect": "exposure_effect"})
    rows.append(
        all_effects[
            ["dataset", "task", "representation", "n_features", "model", "split_seed", "axis", "exposure_effect"]
        ]
    )
    return pd.concat(rows, ignore_index=True)


def build_learning_table(args: argparse.Namespace) -> pd.DataFrame:
    effects = load_effects(args)
    opportunity = pd.read_csv(args.opportunity)
    opportunity["axis"] = opportunity.exposure.map(
        {value: key for key, value in AXIS_EXPOSURE.items()}
    )
    opportunity = opportunity.rename(
        columns={"identity_prior_opportunity": "metadata_prior_opportunity"}
    )[
        ["dataset", "task", "split_seed", "axis", "metadata_prior_opportunity"]
    ]
    nmi = observational_nmi(args.dataset, sorted(opportunity.split_seed.unique()))
    opportunity = opportunity.merge(
        nmi,
        on=["dataset", "task", "split_seed", "axis"],
        validate="one_to_one",
    )
    probes = pd.read_csv(args.identity_probes)
    probes["axis"] = probes.axis.map({value: key for key, value in AXIS_PROBE.items()})
    probes["encoding_margin"] = probes.balanced_accuracy - probes.chance
    probes = (
        probes.groupby(["dataset", "representation", "n_features", "split_seed", "axis"])
        .agg(
            identity_probe_balanced_accuracy=("balanced_accuracy", "mean"),
            identity_probe_chance=("chance", "mean"),
            encoding_margin=("encoding_margin", "mean"),
        )
        .reset_index()
    )
    descriptors = dataset_descriptors(args.dataset)
    observational = effects.merge(
        opportunity,
        on=["dataset", "task", "split_seed", "axis"],
        validate="many_to_one",
    ).merge(
        probes,
        on=["dataset", "representation", "n_features", "split_seed", "axis"],
        validate="many_to_one",
    ).merge(descriptors, on=["dataset", "task"], validate="many_to_one")
    observational["source"] = "observational_factorial"
    tables = [observational]
    if args.counterfactual_summary.is_file():
        dose = pd.read_csv(args.counterfactual_summary)
        required = {
            "dataset",
            "task",
            "axis",
            "representation",
            "model",
            "split_seed",
            "exposure_effect",
            "metadata_prior_opportunity",
        }
        missing = required - set(dose.columns)
        if missing:
            raise ValueError(f"Counterfactual summary is missing columns: {sorted(missing)}")
        dose = dose.rename(columns={"achieved_opportunity": "label_opportunity"})
        dose = dose.merge(
            probes,
            on=["dataset", "representation", "split_seed", "axis"],
            validate="many_to_one",
        ).merge(descriptors, on=["dataset", "task"], validate="many_to_one")
        dose["source"] = "counterfactual_dose"
        tables.append(dose)
    table = pd.concat(tables, ignore_index=True, sort=False)
    table["opportunity_encoding_interaction"] = table.label_opportunity * table.encoding_margin
    table["model_capacity"] = table.model.map(MODEL_CAPACITY)
    table["log_n_rows"] = np.log(table.n_rows)
    if table.model_capacity.isna().any():
        missing = sorted(table.loc[table.model_capacity.isna(), "model"].unique())
        raise ValueError(f"Missing model-capacity mapping for {missing}")
    return table.sort_values(
        ["dataset", "task", "axis", "representation", "model", "split_seed"]
    ).reset_index(drop=True)


def make_regression_pipeline(alpha: float) -> tuple[Pipeline, list[str], list[str]]:
    numeric = [
        "label_opportunity",
        "metadata_prior_opportunity",
        "encoding_margin",
        "opportunity_encoding_interaction",
        "model_capacity",
        "n_features",
        "log_n_rows",
        "n_subjects",
        "n_repeated_units",
        "repeated_measure_completeness",
        "class_balance",
    ]
    categorical = ["axis", "representation", "model"]
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=alpha))]), numeric, categorical


def regression_grouped_validation(
    table: pd.DataFrame, alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    metrics = []
    schemes = {
        "leave_dataset_out": "dataset",
        "leave_task_out": "task",
        "leave_representation_out": "representation",
        "leave_model_out": "model",
    }
    total = sum(table[column].nunique() for column in schemes.values())
    progress = tqdm(total=total, desc="Shortcut-risk validation", unit="holdout", dynamic_ncols=True)
    for scheme, column in schemes.items():
        for held_out in sorted(table[column].unique()):
            train = table.loc[table[column] != held_out]
            test = table.loc[table[column] == held_out]
            pipeline, numeric, categorical = make_regression_pipeline(alpha)
            features = numeric + categorical
            pipeline.fit(train[features], train.exposure_effect)
            estimate = pipeline.predict(test[features])
            for index, prediction in zip(test.index, estimate, strict=True):
                predictions.append(
                    {
                        "scheme": scheme,
                        "held_out": held_out,
                        "row_index": index,
                        "observed_effect": table.loc[index, "exposure_effect"],
                        "predicted_effect": float(prediction),
                    }
                )
            observed = test.exposure_effect.to_numpy(float)
            rho = stats.spearmanr(observed, estimate).statistic if len(np.unique(observed)) > 1 else np.nan
            metrics.append(
                {
                    "scheme": scheme,
                    "held_out": held_out,
                    "n_test": len(test),
                    "mae": mean_absolute_error(observed, estimate),
                    "rmse": mean_squared_error(observed, estimate) ** 0.5,
                    "r2": r2_score(observed, estimate) if len(test) > 1 else np.nan,
                    "spearman_rho": rho,
                    "direction_accuracy": np.mean(np.sign(observed) == np.sign(estimate)),
                }
            )
            progress.update(1)
    progress.close()
    return pd.DataFrame(predictions), pd.DataFrame(metrics)


def anchored_subject_risk_table(
    table: pd.DataFrame, material_effect_threshold: float
) -> pd.DataFrame:
    keys = ["dataset", "task", "axis", "representation", "model", "split_seed"]
    subject = table.loc[table.axis == "subject"].copy()
    baseline = subject.loc[subject.nominal_dose == 0, keys + [
        "exposure_effect",
        "label_opportunity",
        "metadata_prior_opportunity",
    ]].rename(
        columns={
            "exposure_effect": "dose_zero_exposure_effect",
            "label_opportunity": "dose_zero_label_opportunity",
            "metadata_prior_opportunity": "dose_zero_metadata_prior_opportunity",
        }
    )
    if baseline.duplicated(keys).any():
        raise ValueError("Dose-zero anchors are not unique within a subject-risk setting")
    risk = subject.merge(baseline, on=keys, validate="many_to_one")
    risk = risk.loc[risk.nominal_dose > 0].copy()
    risk["dose_induced_amplification"] = (
        risk.exposure_effect - risk.dose_zero_exposure_effect
    )
    risk["opportunity_delta"] = (
        risk.label_opportunity - risk.dose_zero_label_opportunity
    )
    risk["metadata_prior_delta"] = (
        risk.metadata_prior_opportunity - risk.dose_zero_metadata_prior_opportunity
    )
    risk["dose_encoding_interaction"] = risk.opportunity_delta * risk.encoding_margin
    risk["material_optimism_event"] = (
        risk.dose_induced_amplification >= material_effect_threshold
    ).astype(int)
    risk["risk_row_id"] = np.arange(len(risk))
    if not np.isfinite(risk.dose_induced_amplification).all():
        raise ValueError("Anchored risk target contains non-finite values")
    return risk.reset_index(drop=True)


def classifier_features() -> list[str]:
    return [
        "opportunity_delta",
        "metadata_prior_opportunity",
        "encoding_margin",
        "dose_encoding_interaction",
        "model_capacity",
    ]


def make_classifier_pipeline(logistic_c: float) -> tuple[Pipeline, list[str]]:
    numeric = classifier_features()
    preprocess = ColumnTransformer([("numeric", StandardScaler(), numeric)])
    model = LogisticRegression(C=logistic_c, max_iter=5000, solver="lbfgs")
    return Pipeline([("preprocess", preprocess), ("model", model)]), numeric


def select_operating_threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], probability, [1.0])))
    scores = np.asarray(
        [balanced_accuracy_score(target, probability >= threshold) for threshold in candidates]
    )
    best = float(scores.max())
    tied = candidates[np.isclose(scores, best)]
    threshold = float(tied[np.argmin(np.abs(tied - 0.5))])
    return threshold, best


def training_only_threshold(train: pd.DataFrame, logistic_c: float) -> tuple[float, float]:
    pipeline, features = make_classifier_pipeline(logistic_c)
    groups = train.dataset.astype(str) + ":" + train.split_seed.astype(str)
    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError("At least two dataset-seed groups are required to set a threshold")
    probability = cross_val_predict(
        pipeline,
        train[features],
        train.material_optimism_event,
        groups=groups,
        cv=GroupKFold(n_splits=n_groups),
        method="predict_proba",
    )[:, 1]
    return select_operating_threshold(
        train.material_optimism_event.to_numpy(int), probability
    )


def classification_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    null_probability: float,
) -> dict[str, float | int]:
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    prevalence = float(np.mean(target))
    average_precision = float(average_precision_score(target, probability))
    brier = float(brier_score_loss(target, probability))
    null_brier = float(
        brier_score_loss(target, np.full(len(target), null_probability, dtype=float))
    )
    return {
        "n_test": len(target),
        "n_events": int(np.sum(target)),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": average_precision,
        "average_precision_lift": average_precision / prevalence,
        "brier_score": brier,
        "null_brier_score": null_brier,
        "brier_skill": 1.0 - brier / null_brier,
        "decision_threshold": threshold,
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "predicted_positive_fraction": float(np.mean(prediction)),
        "mean_predicted_probability": float(np.mean(probability)),
    }


def bootstrap_classification_metrics(
    test: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    null_probability: float,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    cluster_columns = ["task", "representation", "model", "split_seed"]
    cluster_ids = test[cluster_columns].astype(str).agg("|".join, axis=1).to_numpy()
    clusters = [np.flatnonzero(cluster_ids == value) for value in np.unique(cluster_ids)]
    rng = np.random.default_rng(seed)
    metric_names = [
        "roc_auc",
        "average_precision",
        "average_precision_lift",
        "brier_skill",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    ]
    sampled = {name: [] for name in metric_names}
    target = test.material_optimism_event.to_numpy(int)
    for _ in range(repetitions):
        chosen = rng.integers(0, len(clusters), len(clusters))
        indices = np.concatenate([clusters[index] for index in chosen])
        bootstrap_target = target[indices]
        if len(np.unique(bootstrap_target)) < 2:
            continue
        values = classification_metrics(
            bootstrap_target,
            probability[indices],
            threshold,
            null_probability,
        )
        for name in metric_names:
            sampled[name].append(float(values[name]))
    intervals = {}
    for name, values in sampled.items():
        if not values:
            intervals[f"{name}_ci_low"] = np.nan
            intervals[f"{name}_ci_high"] = np.nan
            continue
        intervals[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
        intervals[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
    return intervals


def classification_grouped_validation(
    table: pd.DataFrame,
    logistic_c: float,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    metric_rows = []
    coefficient_rows = []
    schemes = {
        "leave_dataset_out": "dataset",
        "leave_task_out": "task",
        "leave_representation_out": "representation",
        "leave_model_out": "model",
    }
    total = sum(table[column].nunique() for column in schemes.values())
    progress = tqdm(
        total=total,
        desc="Material shortcut-risk validation",
        unit="holdout",
        dynamic_ncols=True,
    )
    for scheme, column in schemes.items():
        for held_out in sorted(table[column].unique()):
            train = table.loc[table[column] != held_out].copy()
            test = table.loc[table[column] == held_out].copy()
            threshold, training_oof_balanced_accuracy = training_only_threshold(
                train, logistic_c
            )
            pipeline, features = make_classifier_pipeline(logistic_c)
            pipeline.fit(train[features], train.material_optimism_event)
            probability = pipeline.predict_proba(test[features])[:, 1]
            null_probability = float(train.material_optimism_event.mean())
            values = classification_metrics(
                test.material_optimism_event.to_numpy(int),
                probability,
                threshold,
                null_probability,
            )
            values.update(
                {
                    "scheme": scheme,
                    "held_out": held_out,
                    "training_prevalence": null_probability,
                    "training_oof_balanced_accuracy": training_oof_balanced_accuracy,
                }
            )
            if scheme == "leave_dataset_out":
                values.update(
                    bootstrap_classification_metrics(
                        test,
                        probability,
                        threshold,
                        null_probability,
                        bootstrap_repetitions,
                        bootstrap_seed + sum(ord(char) for char in str(held_out)),
                    )
                )
            metric_rows.append(values)
            for row, estimate in zip(test.itertuples(index=False), probability, strict=True):
                prediction_rows.append(
                    {
                        "scheme": scheme,
                        "held_out": held_out,
                        "risk_row_id": row.risk_row_id,
                        "dataset": row.dataset,
                        "task": row.task,
                        "representation": row.representation,
                        "model": row.model,
                        "split_seed": row.split_seed,
                        "nominal_dose": row.nominal_dose,
                        "dose_induced_amplification": row.dose_induced_amplification,
                        "material_optimism_event": row.material_optimism_event,
                        "predicted_probability": float(estimate),
                        "decision_threshold": threshold,
                        "predicted_event": int(estimate >= threshold),
                    }
                )
            coefficients = coefficient_table(pipeline)
            coefficients.insert(0, "held_out", held_out)
            coefficients.insert(0, "scheme", scheme)
            coefficient_rows.append(coefficients)
            progress.update(1)
    progress.close()
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.concat(coefficient_rows, ignore_index=True),
    )


def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    values = np.asarray(pipeline.named_steps["model"].coef_).reshape(-1)
    return pd.DataFrame({"feature": names, "coefficient": values}).sort_values(
        "coefficient", key=np.abs, ascending=False
    )


def acceptance_gate(
    metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    transfer = metrics.loc[metrics.scheme == "leave_dataset_out"].copy()
    criteria = []

    def add(name: str, passed: bool, observed, required: str) -> None:
        criteria.append(
            {
                "criterion": name,
                "passed": bool(passed),
                "observed": observed,
                "required": required,
            }
        )

    add(
        "complete_leave_dataset_out",
        set(transfer.held_out) == {"DEAP", "MAHNOB-HCI"},
        sorted(transfer.held_out.astype(str).tolist()),
        "Both DEAP and MAHNOB-HCI must be held out once",
    )
    add(
        "minimum_event_count",
        bool((transfer.n_events >= args.minimum_events_per_dataset).all()),
        int(transfer.n_events.min()),
        f">= {args.minimum_events_per_dataset} events in every held-out dataset",
    )
    add(
        "cross_dataset_discrimination",
        bool((transfer.roc_auc >= args.minimum_auroc).all()),
        float(transfer.roc_auc.min()),
        f"AUROC >= {args.minimum_auroc:.2f} in every held-out dataset",
    )
    add(
        "cross_dataset_precision_recall_lift",
        bool(
            (
                transfer.average_precision_lift
                >= args.minimum_average_precision_lift
            ).all()
        ),
        float(transfer.average_precision_lift.min()),
        (
            "average precision / event prevalence >= "
            f"{args.minimum_average_precision_lift:.2f} in every held-out dataset"
        ),
    )
    add(
        "cross_dataset_probability_skill",
        bool((transfer.brier_skill > args.minimum_brier_skill).all()),
        float(transfer.brier_skill.min()),
        f"Brier skill > {args.minimum_brier_skill:.2f} in every held-out dataset",
    )
    add(
        "cross_dataset_operating_point",
        bool(
            (transfer.balanced_accuracy >= args.minimum_balanced_accuracy).all()
        ),
        float(transfer.balanced_accuracy.min()),
        (
            "balanced accuracy >= "
            f"{args.minimum_balanced_accuracy:.2f} using a training-only threshold"
        ),
    )
    transfer_coefficients = coefficients.loc[
        coefficients.scheme == "leave_dataset_out"
    ].copy()
    mechanism_features = [
        "opportunity_delta",
        "encoding_margin",
        "dose_encoding_interaction",
    ]
    signs = {}
    for feature in mechanism_features:
        values = transfer_coefficients.loc[
            transfer_coefficients.feature.str.endswith(feature), "coefficient"
        ]
        signs[feature] = values.astype(float).tolist()
    coherent = all(values and min(values) > 0 for values in signs.values())
    add(
        "mechanism_direction",
        coherent,
        signs,
        "Opportunity, encoding, and their interaction coefficients must be positive in both transfer fits",
    )
    return {
        "admissible": all(item["passed"] for item in criteria),
        "criteria": criteria,
        "claim_boundary": (
            "Passing permits an EPPVR subject-axis external test of material-risk "
            "classification; it does not rescue exact effect-size prediction or any stimulus claim."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.training_source != "counterfactual_only":
        raise ValueError(
            "The admissible primary model is restricted to counterfactual dose rows; "
            "mixed-source fitting is sensitivity-only."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_sources = build_learning_table(args)
    counterfactual = all_sources.loc[
        all_sources.source == "counterfactual_dose"
    ].copy()
    if counterfactual.empty:
        raise ValueError(
            "The primary risk classifier requires a completed counterfactual dose summary"
        )

    regression_predictions, regression_metrics = regression_grouped_validation(
        counterfactual, args.alpha
    )
    regression_pipeline, regression_numeric, regression_categorical = (
        make_regression_pipeline(args.alpha)
    )
    regression_features = regression_numeric + regression_categorical
    regression_pipeline.fit(
        counterfactual[regression_features], counterfactual.exposure_effect
    )

    risk = anchored_subject_risk_table(
        counterfactual, args.material_effect_threshold
    )
    predictions, metrics, fold_coefficients = classification_grouped_validation(
        risk,
        args.logistic_c,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    gate = acceptance_gate(metrics, fold_coefficients, args)

    pipeline, numeric = make_classifier_pipeline(args.logistic_c)
    pipeline.fit(risk[numeric], risk.material_optimism_event)
    candidate_model_path = args.output_root / "candidate_risk_model.joblib"
    joblib.dump(pipeline, candidate_model_path)
    primary_oof = predictions.loc[predictions.scheme == "leave_dataset_out"]
    final_threshold, public_oof_balanced_accuracy = select_operating_threshold(
        primary_oof.material_optimism_event.to_numpy(int),
        primary_oof.predicted_probability.to_numpy(float),
    )

    all_sources.to_csv(args.output_root / "risk_learning_table_all_sources.csv", index=False)
    risk.to_csv(args.output_root / "risk_learning_table.csv", index=False)
    predictions.to_csv(args.output_root / "grouped_cv_predictions.csv", index=False)
    metrics.to_csv(args.output_root / "grouped_cv_metrics.csv", index=False)
    fold_coefficients.to_csv(args.output_root / "grouped_cv_coefficients.csv", index=False)
    coefficient_table(pipeline).to_csv(args.output_root / "coefficients.csv", index=False)
    regression_predictions.to_csv(
        args.output_root / "continuous_effect_grouped_cv_predictions.csv", index=False
    )
    regression_metrics.to_csv(
        args.output_root / "continuous_effect_grouped_cv_metrics.csv", index=False
    )
    coefficient_table(regression_pipeline).to_csv(
        args.output_root / "continuous_effect_coefficients.csv", index=False
    )
    (args.output_root / "admissibility_report.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )

    counterfactual_manifest_path = args.counterfactual_summary.parent / "run_manifest.json"
    counterfactual_manifest = json.loads(
        counterfactual_manifest_path.read_text(encoding="utf-8")
    )
    manifest = {
        "status": "admissible_freeze" if gate["admissible"] else "rejected_candidate",
        "admissible": gate["admissible"],
        "model": "l2_logistic_regression",
        "logistic_c": args.logistic_c,
        "training_source": args.training_source,
        "axis": "subject",
        "target": "material dose-induced optimism relative to the matched dose-zero exposure block",
        "target_definition": (
            "1 when (exposure_effect_at_dose - exposure_effect_at_dose_zero) "
            f">= {args.material_effect_threshold:.6f} balanced accuracy; dose-zero rows are anchors only"
        ),
        "material_effect_threshold": args.material_effect_threshold,
        "intervention_block_fraction": float(
            counterfactual_manifest["intervention_block_fraction"]
        ),
        "counterfactual_doses": counterfactual_manifest["doses"],
        "numeric_features": numeric,
        "categorical_features": [],
        "decision_threshold": final_threshold,
        "threshold_source": "leave-dataset-out public-data predictions only",
       "public_oof_balanced_accuracy_at_frozen_threshold": public_oof_balanced_accuracy,
        "public_training_event_prevalence": float(risk.material_optimism_event.mean()),
        "external_acceptance_rule": {
            "minimum_roc_auc": args.minimum_auroc,
            "minimum_average_precision_lift": args.minimum_average_precision_lift,
            "minimum_brier_skill": args.minimum_brier_skill,
            "minimum_balanced_accuracy": args.minimum_balanced_accuracy,
            "fixed_material_effect_threshold": args.material_effect_threshold,
            "fixed_decision_threshold": final_threshold,
            "note": (
                "These EPPVR criteria and both thresholds are frozen before the "
                "external outcomes are generated or inspected."
            ),
        },
        "forbidden_predictors": [
            "dataset identity",
            "dataset row count",
            "dataset subject count",
            "dataset repeated-unit count",
            "task identity",
            "representation identity",
            "model identity",
        ],
        "training_datasets": sorted(risk.dataset.unique()),
        "training_tasks": sorted(risk.task.unique()),
        "training_representations": sorted(risk.representation.unique()),
        "training_models": sorted(risk.model.unique()),
        "counterfactual_summary": (
            str(args.counterfactual_summary) if args.counterfactual_summary.is_file() else None
        ),
        "observational_role": (
            "Natural-exposure rows are retained in risk_learning_table_all_sources.csv for "
            "sensitivity analysis but excluded from the primary frozen model."
        ),
        "continuous_effect_regression": (
            "Retained as a falsification analysis; it is not admissible for external "
            "effect-size prediction because leave-dataset-out regression failed."
        ),
        "external_validation": (
            "EPPVR subject-axis classification only; no stimulus claim is admissible "
            "without a shared physical-stimulus map."
        ),
        "candidate_model_sha256": sha256(candidate_model_path),
        "admissibility_report": "admissibility_report.json",
    }

    (args.output_root / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    frozen_model_path = args.output_root / "frozen_risk_model.joblib"
    freeze_manifest_path = args.output_root / "freeze_manifest.json"
    if gate["admissible"]:
        shutil.copyfile(candidate_model_path, frozen_model_path)
        manifest["model_sha256"] = sha256(frozen_model_path)
        freeze_manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    else:
        frozen_model_path.unlink(missing_ok=True)
        freeze_manifest_path.unlink(missing_ok=True)

    print(metrics.to_string(index=False))
    print(json.dumps(gate, indent=2))
    if gate["admissible"]:
        print(f"Admissible frozen model: {frozen_model_path.resolve()}")
    else:
        print(f"Rejected candidate only: {candidate_model_path.resolve()}")


if __name__ == "__main__":
    main()
