from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_dcs_opct_v11_inductive_target_sensitivity import (  # noqa: E402
    ARCHIVED,
    RISK_MODEL,
    SOURCE_TABLE,
    apply_opct,
    coral_fit,
    fixed_assignment,
    quantile_apply,
    quantile_fit,
)
from analyze_dcs_opct_v11_target_action_endpoint_uncertainty import (  # noqa: E402
    BASE_CONFIG,
    CONFIG,
    PREDICTION_PATHS,
    normalize_predictions,
    verify_configuration_coverage,
)
from analyze_identity_shortcut_risk import classifier_features  # noqa: E402
from develop_consensus_order_preserving_transport_v7 import (  # noqa: E402
    consensus_diagnostics,
    load_development,
)
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    CLUSTER_COLUMNS,
    stratified_certificate,
)
from develop_order_preserving_transport_v6 import RULE, fit_opct  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import witness_statistics  # noqa: E402
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def alternating_sets(values: pd.Series) -> tuple[set, set]:
    ordered = sorted(values.drop_duplicates().tolist())
    return set(ordered[::2]), set(ordered[1::2])


def restricted_amplification(
    dataset: str,
    configurations: pd.DataFrame,
    predictions: pd.DataFrame,
    progress: tqdm,
) -> tuple[pd.DataFrame, dict[str, object]]:
    dose = predictions.loc[predictions.condition.eq("dose")].copy()
    audit_subjects, evaluation_subjects = alternating_sets(dose.subject_id)
    if dataset == "EPPVR":
        audit_stimuli = evaluation_stimuli = None
        audit_mask = dose.subject_id.isin(audit_subjects)
        evaluation_mask = dose.subject_id.isin(evaluation_subjects)
        structure = "participant_disjoint_test_rows"
    else:
        audit_stimuli, evaluation_stimuli = alternating_sets(dose.trial_id)
        audit_mask = dose.subject_id.isin(audit_subjects) & dose.trial_id.isin(audit_stimuli)
        evaluation_mask = (
            dose.subject_id.isin(evaluation_subjects)
            & dose.trial_id.isin(evaluation_stimuli)
        )
        structure = "participant_and_physical_stimulus_disjoint_test_rows"

    rows = []
    for config in configurations[CONFIG].sort_values(CONFIG).itertuples(index=False):
        base = dose.nominal_dose.eq(0.0).to_numpy()
        treated = np.ones(len(dose), dtype=bool)
        for column in BASE_CONFIG:
            value = getattr(config, column)
            base &= dose[column].to_numpy() == value
            treated &= dose[column].to_numpy() == value
        treated &= dose.nominal_dose.to_numpy() == config.nominal_dose
        values = {}
        for partition, raw_mask in (
            ("audit", audit_mask.to_numpy()),
            ("evaluation", evaluation_mask.to_numpy()),
        ):
            baseline = dose.loc[base & raw_mask].sort_values("row_id")
            exposed = dose.loc[treated & raw_mask].sort_values("row_id")
            if baseline.empty or len(baseline) != len(exposed):
                raise ValueError(f"{dataset}: incomplete {partition} rows for {config}")
            if not np.array_equal(baseline.row_id, exposed.row_id):
                raise ValueError(f"{dataset}: unpaired {partition} row IDs for {config}")
            if baseline.target.nunique() != 2:
                raise ValueError(f"{dataset}: single-class {partition} endpoint for {config}")
            target = baseline.target.to_numpy(int)
            baseline_class = baseline.probability.to_numpy(float) >= 0.5
            exposed_class = exposed.probability.to_numpy(float) >= 0.5
            amplification = balanced_accuracy_score(target, exposed_class) - balanced_accuracy_score(
                target, baseline_class
            )
            values[f"{partition}_amplification"] = float(amplification)
            values[f"{partition}_event"] = int(amplification >= 0.02)
            values[f"{partition}_test_rows"] = len(baseline)
        rows.append({
            **{column: getattr(config, column) for column in CONFIG},
            **values,
        })
        progress.update(1)
    metadata = {
        "raw_independence_structure": structure,
        "audit_subjects": len(audit_subjects),
        "evaluation_subjects": len(evaluation_subjects),
        "audit_stimuli": None if audit_stimuli is None else len(audit_stimuli),
        "evaluation_stimuli": None if evaluation_stimuli is None else len(evaluation_stimuli),
    }
    return pd.DataFrame(rows), metadata


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.bootstrap_repetitions < 5000:
        raise ValueError("At least 5,000 bootstrap repetitions are required")

    features = classifier_features()
    model = joblib.load(RISK_MODEL)
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = model.named_steps["model"]
    source_frame = pd.read_csv(SOURCE_TABLE, usecols=features)
    source = scaler.transform(source_frame[features]).astype(np.float64)
    development = load_development()
    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )

    endpoint_progress = tqdm(
        total=sum(len(raw) for raw, _ in development.values()),
        desc="Raw-group-disjoint endpoint reconstruction", unit="config", dynamic_ncols=True,
    )
    endpoints = {}
    metadata_rows = []
    predictions = {}
    for dataset, (raw, _) in sorted(development.items()):
        prediction = normalize_predictions(dataset, PREDICTION_PATHS[dataset])
        verify_configuration_coverage(dataset, raw, prediction)
        endpoint, metadata = restricted_amplification(
            dataset, raw, prediction, endpoint_progress
        )
        endpoints[dataset] = endpoint
        predictions[dataset] = prediction
        metadata_rows.append({"dataset": dataset, **metadata})
    endpoint_progress.close()

    transform_progress = tqdm(
        total=len(development) * 2 * RULE.epochs,
        desc="Audit-only CUDA transforms", unit="epoch", dynamic_ncols=True,
    )
    result_rows = []
    endpoint_rows = []
    for dataset_index, (dataset, (raw, _)) in enumerate(sorted(development.items())):
        frame = raw.copy().reset_index(drop=True)
        frame = frame.merge(endpoints[dataset], on=CONFIG, validate="one_to_one")
        frame["dataset"] = dataset
        assignment = fixed_assignment(frame)
        frame = frame.merge(assignment, on=list(CLUSTER_COLUMNS), validate="many_to_one")
        audit_mask = frame.partition.eq("audit").to_numpy()
        heldout_mask = ~audit_mask
        target = scaler.transform(frame[features]).astype(np.float64)
        identity = logistic.predict_proba(target)[:, 1]
        if np.max(np.abs(identity - frame.probability_identity.to_numpy(float))) > 1e-6:
            raise RuntimeError(f"{dataset}: frozen risk probability reconstruction failed")

        matrix, offset = coral_fit(source, target[audit_mask])
        qmaps = quantile_fit(source, target[audit_mask])
        bases = {
            "coral": logistic.predict_proba(target @ matrix + offset)[:, 1],
            "quantile_mapping": logistic.predict_proba(quantile_apply(target, qmaps))[:, 1],
        }
        projected = {}
        diagnostics = {}
        for component_index, (name, probability) in enumerate(bases.items()):
            _, fitted = fit_opct(
                identity[audit_mask], probability[audit_mask], RULE,
                args.seed + dataset_index * 10007 + component_index * 1009,
                "cuda", transform_progress,
            )
            projected[name] = apply_opct(identity, fitted)
            diagnostics[name] = consensus_diagnostics(
                identity[audit_mask], projected[name][audit_mask]
            )
        components_pass = all(bool(diagnostics[name]["applicable"]) for name in projected)
        witness = witness_statistics(
            identity[audit_mask], projected["coral"][audit_mask],
            projected["quantile_mapping"][audit_mask],
        ) if components_pass else {"witness_pass": False}
        applicable = bool(components_pass and witness["witness_pass"])
        action = projected["coral"] if applicable else identity
        frame["probability_wg_opct"] = action

        audit = frame.loc[audit_mask].copy()
        audit["material_optimism_event"] = audit.audit_event
        heldout = frame.loc[heldout_mask].copy()
        heldout["material_optimism_event"] = heldout.evaluation_event
        certificate = stratified_certificate(
            audit.reset_index(drop=True), config,
            args.seed + dataset_index * 7919, applicable,
            float(diagnostics["coral"]["probability_rank"]),
            int(diagnostics["coral"]["order_inversions"]),
        )
        selected = "wg_opct" if certificate["certified"] else "identity"
        outcome = evaluate_action(heldout.reset_index(drop=True), selected)
        result_rows.append({
            "dataset": dataset,
            "components_pass": components_pass,
            "witness_pass": bool(witness["witness_pass"]),
            "certificate_pass": bool(certificate["certified"]),
            "selected_method": selected,
            "audit_event_prevalence": float(audit.audit_event.mean()),
            "evaluation_event_prevalence": float(heldout.evaluation_event.mean()),
            "audit_brier_gain": certificate["brier_gain"],
            "audit_brier_gain_lcb": certificate["brier_gain_lcb"],
            "heldout_brier_gain": outcome["brier_gain"],
            "heldout_auc_delta": outcome["auc_delta"],
            "material_negative_transfer": outcome["material_negative_transfer"],
        })
        endpoint_rows.append(frame[[
            "dataset", *CONFIG, "partition", "audit_amplification", "audit_event",
            "evaluation_amplification", "evaluation_event", "audit_test_rows",
            "evaluation_test_rows",
        ]])
        transform_progress.set_postfix(dataset=dataset, action=selected, refresh=False)
    transform_progress.close()

    results = pd.DataFrame(result_rows)
    archived = pd.read_csv(ARCHIVED / "primary_heldout_results.csv")[["dataset", "brier_gain"]]
    archived = archived.rename(columns={"brier_gain": "transductive_original_endpoint_gain"})
    results = results.merge(archived, on="dataset", validate="one_to_one")
    args.output_root.mkdir(parents=True, exist_ok=False)
    results.to_csv(args.output_root / "raw_group_independent_action_outcomes.csv", index=False)
    pd.concat(endpoint_rows, ignore_index=True).to_csv(
        args.output_root / "raw_group_independent_endpoints.csv", index=False
    )
    pd.DataFrame(metadata_rows).to_csv(args.output_root / "raw_partition_metadata.csv", index=False)
    report = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "claim_boundary": (
            "Audit and evaluation endpoints use disjoint raw test participants and, where verified, "
            "disjoint physical stimuli. Emotion-model training rows and split-seed fits can still be shared."
        ),
    }
    (args.output_root / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
