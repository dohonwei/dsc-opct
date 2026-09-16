from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_dcs_opct_v11_inductive_target_sensitivity import (  # noqa: E402
    RISK_MODEL, SOURCE_TABLE, apply_opct, coral_fit, quantile_apply, quantile_fit,
)
from analyze_identity_shortcut_risk import MODEL_CAPACITY, classifier_features  # noqa: E402
from develop_consensus_order_preserving_transport_v7 import consensus_diagnostics  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    cluster_geometry, partition_frame, select_distribution_covered_audit,
    stratified_certificate,
)
from develop_order_preserving_transport_v6 import RULE, fit_opct  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import witness_statistics  # noqa: E402
from identity_shortcut.core import (  # noqa: E402
    REPRESENTATIONS, build_dose_blocks, class_matched_control_split,
    metadata_prior_opportunity, representation_columns, stable_seed,
)
from identity_shortcut.models import make_shortcut_models  # noqa: E402
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402
import run_crossed_identity_probes as probes  # noqa: E402
import run_protocol_model_benchmark as benchmark  # noqa: E402

MODEL_NAMES = ("linear_logistic", "gpu_mlp")
PARTITIONS = ("audit_population", "evaluation_population")
CONFIG = ["task", "axis", "representation", "model", "split_seed", "nominal_dose"]
MATERIAL_THRESHOLD = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw-population-disjoint EPPVR DCS-OPCT sensitivity")
    parser.add_argument("--input", type=Path, default=Path("outputs/unified_frontal_features/eppvr.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914"))
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"])
    parser.add_argument("--representations", nargs="+", default=["all", "relative_power", "normalized_asymmetry", "baseline_delta"], choices=list(REPRESENTATIONS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    parser.add_argument("--doses", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260813, 20260829, 20260911, 20260923, 20261007])
    parser.add_argument("--subject-folds", type=int, default=5)
    parser.add_argument("--record-folds", type=int, default=2)
    parser.add_argument("--intervention-block-fraction", type=float, default=0.5)
    parser.add_argument("--candidate-draws", type=int, default=256)
    parser.add_argument("--gpu-epochs", type=int, default=40)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-fold-cells", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_hash(values: np.ndarray) -> str:
    text = "|".join(map(str, sorted(np.asarray(values).tolist())))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def participant_partition(frame: pd.DataFrame) -> pd.DataFrame:
    subjects = sorted(frame.subject_id.astype(str).unique().tolist())
    if len(subjects) != 30:
        raise ValueError(f"Expected 30 EPPVR participants, observed {len(subjects)}")
    assignment = pd.DataFrame([
        {
            "subject_id": subject,
            "population": PARTITIONS[position % 2],
            "assignment_position": position,
            "assignment_basis": "lexicographic_subject_id_parity_only",
        }
        for position, subject in enumerate(subjects)
    ])
    expected = {"audit_population": 15, "evaluation_population": 15}
    if assignment.population.value_counts().to_dict() != expected:
        raise RuntimeError("Participant partition is not 15/15")
    return assignment


def record_fold_assignments(frame: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    assignments = np.empty(len(frame), dtype=int)
    for subject, indices in frame.groupby("subject_id").groups.items():
        indices = np.asarray(list(indices), dtype=int)
        rng = np.random.default_rng(stable_seed(seed, subject, "raw-partition-first-record-fold"))
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        assignments[shuffled] = np.arange(len(shuffled)) % folds
    return assignments


def block_counts(labels: np.ndarray, candidates: np.ndarray, fraction: float, identities: int) -> np.ndarray:
    available = np.bincount(labels[candidates], minlength=2)
    counts = np.floor(available * fraction).astype(int)
    if counts.sum() < identities:
        raise ValueError("Intervention block cannot cover every test identity")
    return counts


def provenance_row(
    population: str,
    fit_kind: str,
    train: np.ndarray,
    test: np.ndarray,
    frame: pd.DataFrame,
    **metadata: object,
) -> dict[str, object]:
    train_rows = frame.iloc[train].global_row_id.to_numpy(int)
    test_rows = frame.iloc[test].global_row_id.to_numpy(int)
    return {
        "population": population,
        "fit_kind": fit_kind,
        **metadata,
        "n_train": len(train),
        "n_test": len(test),
        "train_subjects": ";".join(sorted(frame.iloc[train].subject_id.astype(str).unique())),
        "test_subjects": ";".join(sorted(frame.iloc[test].subject_id.astype(str).unique())),
        "train_global_rows_sha256": value_hash(train_rows),
        "test_global_rows_sha256": value_hash(test_rows),
        "row_overlap": int(len(set(train_rows) & set(test_rows))),
    }


def identity_probe_summary(
    frame: pd.DataFrame,
    feature_sets: dict[str, np.ndarray],
    seeds: list[int],
    record_folds: int,
    population: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    identities = frame.subject_id.astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    total = len(feature_sets) * len(seeds) * record_folds * 2
    progress = tqdm(total=total, desc=f"{population} identity probes", unit="fit", dynamic_ncols=True)
    for split_seed in seeds:
        assignments = record_fold_assignments(frame, record_folds, split_seed)
        templates = probes.make_models(split_seed)
        for representation, features in feature_sets.items():
            predictions = {name: np.empty(len(frame), dtype=object) for name in templates}
            for fold in range(record_folds):
                test = np.flatnonzero(assignments == fold)
                train = np.flatnonzero(assignments != fold)
                for name, template in templates.items():
                    fitted = clone(template).fit(features[train], identities[train])
                    predictions[name][test] = fitted.predict(features[test])
                    provenance.append(provenance_row(
                        population, "identity_probe", train, test, frame,
                        task="identity", representation=representation, model=name,
                        split_seed=split_seed, fold=f"r{fold}", nominal_dose=np.nan,
                    ))
                    progress.update(1)
            for name, estimate in predictions.items():
                rows.append({
                    "dataset": "EPPVR", "population": population,
                    "representation": representation, "n_features": features.shape[1],
                    "axis": "subject", "probe_model": name, "split_seed": split_seed,
                    "n_classes": len(np.unique(identities)),
                    "chance": 1 / len(np.unique(identities)),
                    "balanced_accuracy": probes.balanced_multiclass_accuracy(identities, estimate),
                })
    progress.close()
    return pd.DataFrame(rows), provenance


def append_predictions(
    rows: list[dict[str, object]], frame: pd.DataFrame, test: np.ndarray,
    labels: np.ndarray, probability: np.ndarray, **metadata: object,
) -> None:
    for index, estimate in zip(test, probability, strict=True):
        row = frame.iloc[index]
        rows.append({
            **metadata,
            "row_id": int(row.global_row_id),
            "subject_id": str(row.subject_id),
            "trial_id": str(row.trial_id),
            "target": int(labels[index]),
            "probability": float(estimate),
        })


def summarize(predictions: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "representation", "n_features", "model", "split_seed"]
    rows = []
    for key_values, group in predictions.groupby(keys, sort=True):
        unseen = group.loc[group.condition.eq("unseen")]
        unseen_score = benchmark.balanced_accuracy(
            unseen.target.to_numpy(int), unseen.probability.to_numpy(float)
        )
        for dose, dose_group in group.loc[group.condition.eq("dose")].groupby("nominal_dose"):
            score = benchmark.balanced_accuracy(
                dose_group.target.to_numpy(int), dose_group.probability.to_numpy(float)
            )
            rows.append({
                **dict(zip(keys, key_values, strict=True)),
                "nominal_dose": float(dose),
                "unseen_balanced_accuracy": unseen_score,
                "dose_balanced_accuracy": score,
                "subject_exposure_effect": score - unseen_score,
                "achieved_dose": float(dose_group.achieved_dose.mean()),
            })
    opportunities = (
        audit.groupby(["task", "split_seed", "nominal_dose"])
        .agg(
            normalized_mutual_information=("normalized_mutual_information", "mean"),
            metadata_prior_opportunity=("metadata_prior_opportunity", "mean"),
        ).reset_index()
    )
    return pd.DataFrame(rows).merge(
        opportunities, on=["task", "split_seed", "nominal_dose"], validate="many_to_one"
    )


def run_population(
    frame: pd.DataFrame, population: str, args: argparse.Namespace, progress: tqdm,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.reset_index(drop=True)
    feature_sets = {
        name: frame[representation_columns(frame, name)].to_numpy(float)
        for name in args.representations
    }
    encoding, provenance = identity_probe_summary(
        frame, feature_sets, args.seeds, args.record_folds, population
    )
    prediction_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    cells = [
        (subject_fold, record_fold)
        for subject_fold in range(args.subject_folds)
        for record_fold in range(args.record_folds)
    ]
    if args.max_fold_cells is not None:
        cells = cells[:args.max_fold_cells]

    for split_seed in args.seeds:
        subject_map = benchmark.shuffled_fold_map(
            frame.subject_id.to_numpy(), args.subject_folds, split_seed
        )
        subject_fold = frame.subject_id.map(subject_map).to_numpy(int)
        record_fold = record_fold_assignments(frame, args.record_folds, split_seed)
        templates = make_shortcut_models(
            split_seed, args.n_jobs, args.models,
            device=args.device, gpu_epochs=args.gpu_epochs,
        )
        for task in args.tasks:
            labels = (frame[f"{task}_score"].to_numpy(float) >= 5).astype(int)
            identities = frame.subject_id.astype(str).to_numpy()
            for subject_test, record_test in cells:
                test = np.flatnonzero((subject_fold == subject_test) & (record_fold == record_test))
                candidates = np.flatnonzero((subject_fold == subject_test) & (record_fold != record_test))
                control = np.flatnonzero(subject_fold != subject_test)
                counts = block_counts(
                    labels, candidates, args.intervention_block_fraction,
                    len(np.unique(identities[test])),
                )
                core, removed_control = class_matched_control_split(
                    control, labels, counts,
                    stable_seed(population, split_seed, task, subject_test, record_test, "control"),
                )
                blocks = build_dose_blocks(
                    candidates, labels, identities, identities[test], counts,
                    args.doses, args.candidate_draws,
                    stable_seed(population, split_seed, task, subject_test, record_test, "dose"),
                )
                fold = f"s{subject_test}_r{record_test}"
                for block in blocks:
                    train = np.concatenate([core, block.indices])
                    if len(train) != len(control):
                        raise RuntimeError(f"{population}/{fold}: training-size mismatch")
                    if set(train) & set(test):
                        raise RuntimeError(f"{population}/{fold}: train-test overlap")
                    if not np.array_equal(
                        np.bincount(labels[train], minlength=2),
                        np.bincount(labels[control], minlength=2),
                    ):
                        raise RuntimeError(f"{population}/{fold}: class-count mismatch")
                    coverage = len(set(identities[test]) & set(identities[train])) / len(set(identities[test]))
                    if not np.isclose(coverage, 1.0):
                        raise RuntimeError(f"{population}/{fold}: incomplete identity coverage")
                    audit_rows.append({
                        "population": population,
                        "task": task,
                        "split_seed": split_seed,
                        "fold": fold,
                        "nominal_dose": block.nominal_dose,
                        "achieved_dose": block.achieved_dose,
                        "normalized_mutual_information": block.opportunity["normalized_mutual_information"],
                        "metadata_prior_opportunity": metadata_prior_opportunity(
                            labels[block.indices], identities[block.indices], labels[test], identities[test]
                        ),
                        "candidate_min": block.candidate_min,
                        "candidate_max": block.candidate_max,
                        "candidate_size": len(candidates),
                        "intervention_block_size": len(block.indices),
                        "removed_control_size": len(removed_control),
                        "n_train": len(train),
                        "n_test": len(test),
                        "subject_coverage": coverage,
                        "row_overlap_with_test": 0,
                    })
                for representation, features in feature_sets.items():
                    for model_name, template in templates.items():
                        fitted = clone(template).fit(features[control], labels[control])
                        probability = benchmark.model_probability(fitted, features[test])
                        append_predictions(
                            prediction_rows, frame, test, labels, probability,
                            population=population, task=task, representation=representation,
                            n_features=features.shape[1], model=model_name,
                            split_seed=split_seed, fold=fold, condition="unseen",
                            nominal_dose=np.nan, achieved_dose=np.nan,
                        )
                        provenance.append(provenance_row(
                            population, "emotion_unseen", control, test, frame,
                            task=task, representation=representation, model=model_name,
                            split_seed=split_seed, fold=fold, nominal_dose=np.nan,
                        ))
                        progress.update(1)
                        for block in blocks:
                            train = np.concatenate([core, block.indices])
                            fitted = clone(template).fit(features[train], labels[train])
                            probability = benchmark.model_probability(fitted, features[test])
                            append_predictions(
                                prediction_rows, frame, test, labels, probability,
                                population=population, task=task, representation=representation,
                                n_features=features.shape[1], model=model_name,
                                split_seed=split_seed, fold=fold, condition="dose",
                                nominal_dose=block.nominal_dose, achieved_dose=block.achieved_dose,
                            )
                            provenance.append(provenance_row(
                                population, "emotion_dose", train, test, frame,
                                task=task, representation=representation, model=model_name,
                                split_seed=split_seed, fold=fold,
                                nominal_dose=block.nominal_dose,
                            ))
                            progress.update(1)
    predictions = pd.DataFrame(prediction_rows)
    split_audit = pd.DataFrame(audit_rows)
    summary = summarize(predictions, split_audit)
    return predictions, split_audit, summary, pd.DataFrame(provenance), encoding


def build_risk_table(
    summary: pd.DataFrame,
    encoding: pd.DataFrame,
    population: str,
    frozen_model: object,
) -> pd.DataFrame:
    probe = encoding.copy()
    probe["encoding_margin"] = probe.balanced_accuracy - probe.chance
    probe = probe.groupby(
        ["representation", "n_features", "split_seed"]
    ).encoding_margin.mean().reset_index()
    table = summary.merge(
        probe, on=["representation", "n_features", "split_seed"], validate="many_to_one"
    )
    table["dataset"] = "EPPVR"
    table["population"] = population
    table["axis"] = "subject"
    keys = ["task", "representation", "n_features", "model", "split_seed"]
    anchors = table.loc[
        table.nominal_dose.eq(0.0),
        keys + ["subject_exposure_effect", "normalized_mutual_information", "metadata_prior_opportunity"],
    ].rename(columns={
        "subject_exposure_effect": "dose_zero_exposure_effect",
        "normalized_mutual_information": "dose_zero_label_opportunity",
        "metadata_prior_opportunity": "dose_zero_metadata_prior_opportunity",
    })
    if anchors.duplicated(keys).any():
        raise RuntimeError(f"{population}: non-unique dose-zero anchors")
    table = table.merge(anchors, on=keys, validate="many_to_one")
    table = table.loc[table.nominal_dose.gt(0.0)].copy()
    table["dose_induced_amplification"] = table.subject_exposure_effect - table.dose_zero_exposure_effect
    table["opportunity_delta"] = table.normalized_mutual_information - table.dose_zero_label_opportunity
    table["metadata_prior_delta"] = table.metadata_prior_opportunity - table.dose_zero_metadata_prior_opportunity
    table["dose_encoding_interaction"] = table.opportunity_delta * table.encoding_margin
    table["model_capacity"] = table.model.map(MODEL_CAPACITY)
    table["material_optimism_event"] = (table.dose_induced_amplification >= MATERIAL_THRESHOLD).astype(int)
    features = classifier_features()
    table["probability_identity"] = frozen_model.predict_proba(table[features])[:, 1]
    return table.sort_values(CONFIG).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.output_root}; pass --force")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.bootstrap_repetitions < 5000 and args.max_fold_cells is None:
        raise ValueError("The full analysis requires at least 5,000 bootstrap repetitions")
    if not 0.0 < args.intervention_block_fraction < 1.0:
        raise ValueError("Intervention block fraction must lie in (0, 1)")

    frame = pd.read_csv(args.input).reset_index(drop=True)
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["trial_id"] = frame.trial_id.astype(str)
    frame["global_row_id"] = np.arange(len(frame), dtype=int)
    if len(frame) != 420 or frame.subject_id.nunique() != 30:
        raise ValueError("EPPVR contract requires 420 rows from 30 participants")
    assignment = participant_partition(frame)
    joined = frame.merge(
        assignment[["subject_id", "population"]], on="subject_id", validate="many_to_one"
    )

    cells = args.subject_folds * args.record_folds
    if args.max_fold_cells is not None:
        cells = min(cells, args.max_fold_cells)
    emotion_fits = (
        len(PARTITIONS) * len(args.tasks) * len(args.seeds) * cells
        * len(args.representations) * len(args.models) * (1 + len(args.doses))
    )
    progress = tqdm(
        total=emotion_fits, desc="Raw-partition-first emotion models",
        unit="fit", dynamic_ncols=True,
    )
    all_predictions = []
    all_split_audits = []
    all_summaries = []
    all_provenance = []
    all_encoding = []
    for population in PARTITIONS:
        subset = joined.loc[joined.population.eq(population)].drop(columns="population").copy()
        predictions, split_audit, summary, provenance, encoding = run_population(
            subset, population, args, progress
        )
        all_predictions.append(predictions)
        all_split_audits.append(split_audit)
        all_summaries.append(summary.assign(population=population))
        all_provenance.append(provenance)
        all_encoding.append(encoding)
    progress.close()

    predictions = pd.concat(all_predictions, ignore_index=True)
    split_audits = pd.concat(all_split_audits, ignore_index=True)
    summaries = pd.concat(all_summaries, ignore_index=True)
    provenance = pd.concat(all_provenance, ignore_index=True)
    encoding = pd.concat(all_encoding, ignore_index=True)
    audit_subjects = set(assignment.loc[assignment.population.eq(PARTITIONS[0]), "subject_id"])
    evaluation_subjects = set(assignment.loc[assignment.population.eq(PARTITIONS[1]), "subject_id"])
    if audit_subjects & evaluation_subjects:
        raise RuntimeError("Participant populations overlap")
    for population, allowed in ((PARTITIONS[0], audit_subjects), (PARTITIONS[1], evaluation_subjects)):
        observed = set(predictions.loc[predictions.population.eq(population), "subject_id"])
        if not observed <= allowed:
            raise RuntimeError(f"{population}: prediction rows escaped their raw population")
    if provenance.row_overlap.ne(0).any():
        raise RuntimeError("At least one fitted model has train-test row overlap")

    risk_model = joblib.load(RISK_MODEL)
    risk_tables = {}
    for population in PARTITIONS:
        risk_tables[population] = build_risk_table(
            summaries.loc[summaries.population.eq(population)].drop(columns="population"),
            encoding.loc[encoding.population.eq(population)],
            population,
            risk_model,
        )

    features = classifier_features()
    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    source_frame = pd.read_csv(SOURCE_TABLE, usecols=features)
    source = scaler.transform(source_frame[features]).astype(np.float64)
    audit_frame = risk_tables[PARTITIONS[0]].copy()
    evaluation_frame = risk_tables[PARTITIONS[1]].copy()
    audit_target = scaler.transform(audit_frame[features]).astype(np.float64)
    evaluation_target = scaler.transform(evaluation_frame[features]).astype(np.float64)
    audit_identity = logistic.predict_proba(audit_target)[:, 1]
    evaluation_identity = logistic.predict_proba(evaluation_target)[:, 1]
    if np.max(np.abs(audit_identity - audit_frame.probability_identity)) > 1e-6:
        raise RuntimeError("Audit-population risk probability reconstruction failed")
    if np.max(np.abs(evaluation_identity - evaluation_frame.probability_identity)) > 1e-6:
        raise RuntimeError("Evaluation-population risk probability reconstruction failed")

    matrix, offset = coral_fit(source, audit_target)
    qmaps = quantile_fit(source, audit_target)
    component_bases = {
        "coral": (
            logistic.predict_proba(audit_target @ matrix + offset)[:, 1],
            logistic.predict_proba(evaluation_target @ matrix + offset)[:, 1],
        ),
        "quantile_mapping": (
            logistic.predict_proba(quantile_apply(audit_target, qmaps))[:, 1],
            logistic.predict_proba(quantile_apply(evaluation_target, qmaps))[:, 1],
        ),
    }
    transform_progress = tqdm(
        total=len(component_bases) * RULE.epochs,
        desc="Audit-population CUDA OPCT", unit="epoch", dynamic_ncols=True,
    )
    projected_audit = {}
    projected_evaluation = {}
    diagnostics = {}
    component_rows = []
    for component_index, (name, (audit_base, evaluation_base)) in enumerate(component_bases.items()):
        _, fitted = fit_opct(
            audit_identity, audit_base, RULE,
            args.seed + component_index * 1009, "cuda", transform_progress,
        )
        projected_audit[name] = apply_opct(audit_identity, fitted)
        projected_evaluation[name] = apply_opct(evaluation_identity, fitted)
        diagnostics[name] = consensus_diagnostics(audit_identity, projected_audit[name])
        component_rows.append({
            "component": name,
            "fit_population": PARTITIONS[0],
            "application_population": PARTITIONS[1],
            **fitted,
            **{f"consensus_{key}": value for key, value in diagnostics[name].items()},
        })
    transform_progress.close()
    components_pass = all(bool(diagnostics[name]["applicable"]) for name in diagnostics)
    witness = (
        witness_statistics(
            audit_identity, projected_audit["coral"], projected_audit["quantile_mapping"]
        ) if components_pass else {"witness_pass": False}
    )
    applicable = bool(components_pass and witness["witness_pass"])
    audit_frame["probability_wg_opct"] = projected_audit["coral"] if applicable else audit_identity
    evaluation_frame["probability_wg_opct"] = (
        projected_evaluation["coral"] if applicable else evaluation_identity
    )

    geometry = cluster_geometry(audit_frame)
    audit_assignment = select_distribution_covered_audit(geometry)
    certificate_frame, unused_audit_frame = partition_frame(audit_frame, audit_assignment)
    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )
    certificate = stratified_certificate(
        certificate_frame, config, args.seed + RULE.configuration_budget * 1009,
        applicable, float(diagnostics["coral"]["probability_rank"]),
        int(diagnostics["coral"]["order_inversions"]),
    )
    selected = "wg_opct" if bool(certificate["certified"]) else "identity"
    outcome = evaluate_action(evaluation_frame, selected)
    result = {
        "dataset": "EPPVR",
        "protocol": "raw_partition_first_participant_disjoint_end_to_end",
        "audit_population_participants": len(audit_subjects),
        "evaluation_population_participants": len(evaluation_subjects),
        "audit_configurations": len(audit_frame),
        "certificate_configurations": len(certificate_frame),
        "unused_audit_configurations": len(unused_audit_frame),
        "evaluation_configurations": len(evaluation_frame),
        "audit_event_prevalence": float(audit_frame.material_optimism_event.mean()),
        "evaluation_event_prevalence": float(evaluation_frame.material_optimism_event.mean()),
        "components_pass": components_pass,
        "witness_pass": bool(witness["witness_pass"]),
        "certificate_pass": bool(certificate["certified"]),
        "selected_method": selected,
        "certificate_brier_gain": certificate["brier_gain"],
        "certificate_brier_gain_lcb": certificate["brier_gain_lcb"],
        "evaluation_brier_gain": outcome["brier_gain"],
        "evaluation_auc_delta": outcome["auc_delta"],
        "negative_transfer": outcome["negative_transfer"],
        "material_negative_transfer": outcome["material_negative_transfer"],
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(args.output_root / "participant_population_assignment.csv", index=False)
    predictions.to_csv(args.output_root / "population_predictions.csv", index=False)
    split_audits.to_csv(args.output_root / "population_split_audit.csv", index=False)
    summaries.to_csv(args.output_root / "population_summary.csv", index=False)
    encoding.to_csv(args.output_root / "population_identity_probe_summary.csv", index=False)
    provenance.to_csv(args.output_root / "fit_provenance.csv", index=False)
    pd.concat(risk_tables.values(), ignore_index=True).to_csv(
        args.output_root / "population_risk_tables.csv", index=False
    )
    geometry.to_csv(args.output_root / "audit_population_geometry.csv", index=False)
    audit_assignment.to_csv(
        args.output_root / "audit_population_certificate_assignment.csv", index=False
    )
    pd.DataFrame(component_rows).to_csv(
        args.output_root / "component_fit_diagnostics.csv", index=False
    )
    pd.DataFrame([{**witness, "fit_population": PARTITIONS[0]}]).to_csv(
        args.output_root / "witness_diagnostics.csv", index=False
    )
    pd.DataFrame([certificate]).to_csv(args.output_root / "certificate.csv", index=False)
    pd.DataFrame([result]).to_csv(args.output_root / "evaluation_outcome.csv", index=False)
    evaluation_frame[[
        *CONFIG, "material_optimism_event", "dose_induced_amplification",
        "probability_identity", "probability_wg_opct",
    ]].to_csv(args.output_root / "evaluation_population_action_predictions.csv", index=False)
    manifest = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "risk_model": str(RISK_MODEL),
        "risk_model_sha256": sha256(RISK_MODEL),
        "source_table": str(SOURCE_TABLE),
        "source_table_sha256": sha256(SOURCE_TABLE),
        "participant_assignment": "lexicographic subject_id parity; outcomes and covariates unused",
        "tasks": args.tasks,
        "representations": args.representations,
        "models": args.models,
        "doses": args.doses,
        "seeds": args.seeds,
        "subject_folds": args.subject_folds,
        "record_folds": args.record_folds,
        "fold_cells_per_seed": cells,
        "gpu_epochs": args.gpu_epochs,
        "candidate_draws": args.candidate_draws,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "expected_emotion_model_fits": emotion_fits,
        "observed_emotion_model_fits": int(provenance.fit_kind.str.startswith("emotion_").sum()),
        "observed_identity_probe_fits": int(provenance.fit_kind.eq("identity_probe").sum()),
        "raw_independence_contract": (
            "No participant, global raw row, emotion-model training row, identity-probe training row, "
            "or fitted target-domain model crosses the audit/evaluation population boundary."
        ),
        "action_fit_contract": (
            "CORAL, quantile maps, OPCT parameters, applicability gates, audit geometry, and the "
            "certificate use the audit population only. The frozen action is applied once to every "
            "evaluation-population configuration."
        ),
        "claim_boundary": (
            "Retrospective fully participant- and training-row-disjoint EPPVR sensitivity. "
            "It is not a new-dataset, prospective, cross-session, or stimulus-independent validation."
        ),
    }
    (args.output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"Results written to {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
