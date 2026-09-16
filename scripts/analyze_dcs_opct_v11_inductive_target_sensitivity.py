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
from scipy.special import expit, logit
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_identity_shortcut_risk import classifier_features  # noqa: E402
from develop_consensus_order_preserving_transport_v7 import (  # noqa: E402
    consensus_diagnostics,
    load_development,
)
from develop_order_preserving_transport_v6 import RULE, fit_opct  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import witness_statistics  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    CLUSTER_COLUMNS,
    STRATUM_COLUMNS,
    stratified_certificate,
)
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402


SOURCE_TABLE = Path("outputs/identity_shortcut_risk_classifier/risk_learning_table.csv")
RISK_MODEL = Path("outputs/identity_shortcut_risk_classifier/frozen_risk_model.joblib")
ARCHIVED = Path("outputs/distribution_covered_stratified_opct_v11_development")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Covariate-unseen inductive target-transform sensitivity for DCS-OPCT v11."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dcs_opct_v11_inductive_target_sensitivity_20260914"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def fixed_assignment(frame: pd.DataFrame) -> pd.DataFrame:
    """Split clusters without reading any target feature or outcome value."""
    clusters = frame[list(CLUSTER_COLUMNS)].drop_duplicates().sort_values(list(CLUSTER_COLUMNS))
    rows: list[dict[str, object]] = []
    for stratum_index, (_, group) in enumerate(
        clusters.groupby(list(STRATUM_COLUMNS), sort=True, dropna=False)
    ):
        group = group.sort_values("split_seed").reset_index(drop=True)
        if len(group) != 5:
            raise ValueError("Expected five split seeds per stratum")
        audit_positions = {0, 2, 4} if stratum_index % 2 == 0 else {1, 3}
        for position, row in group.iterrows():
            rows.append(
                {
                    **{column: row[column] for column in CLUSTER_COLUMNS},
                    "partition": "audit" if position in audit_positions else "heldout",
                    "assignment_basis": "stratum_order_and_seed_only",
                }
            )
    result = pd.DataFrame(rows)
    if result.partition.value_counts().to_dict() != {"audit": len(result) // 2, "heldout": len(result) // 2}:
        raise ValueError("Fixed assignment is not balanced")
    return result


def coral_fit(source: np.ndarray, target_fit: np.ndarray, ridge: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    def sym_power(matrix: np.ndarray, power: float) -> np.ndarray:
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
        values = np.maximum(values, ridge)
        return (vectors * values**power) @ vectors.T

    source_cov = np.cov(source - source.mean(axis=0), rowvar=False) + ridge * np.eye(source.shape[1])
    target_cov = np.cov(target_fit - target_fit.mean(axis=0), rowvar=False) + ridge * np.eye(source.shape[1])
    transform = sym_power(target_cov, -0.5) @ sym_power(source_cov, 0.5)
    offset = source.mean(axis=0) - target_fit.mean(axis=0) @ transform
    return transform, offset


def quantile_fit(source: np.ndarray, target_fit: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    grid = np.linspace(0.0, 1.0, max(len(source), len(target_fit)))
    return [
        (np.quantile(target_fit[:, j], grid), np.quantile(source[:, j], grid))
        for j in range(source.shape[1])
    ]


def quantile_apply(values: np.ndarray, maps: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    for j, (target_q, source_q) in enumerate(maps):
        output[:, j] = np.interp(values[:, j], target_q, source_q)
    return output


def apply_opct(identity: np.ndarray, diagnostics: dict[str, float | bool]) -> np.ndarray:
    clipped = np.clip(identity, 1e-6, 1 - 1e-6)
    return expit(float(diagnostics["scale"]) * logit(clipped) + float(diagnostics["shift"]))


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
    datasets = load_development()
    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )

    result_rows = []
    assignment_rows = []
    prediction_rows = []
    progress = tqdm(
        total=len(datasets) * 2 * RULE.epochs,
        desc="Inductive audit-only CUDA transforms",
        unit="epoch",
        dynamic_ncols=True,
    )
    for dataset_index, (dataset, (raw, _)) in enumerate(sorted(datasets.items())):
        frame = raw.copy().reset_index(drop=True)
        assignment = fixed_assignment(frame)
        frame = frame.merge(assignment, on=list(CLUSTER_COLUMNS), validate="many_to_one")
        assignment.insert(0, "dataset", dataset)
        assignment_rows.append(assignment)
        audit_mask = frame.partition.eq("audit").to_numpy()
        heldout_mask = ~audit_mask
        target = scaler.transform(frame[features]).astype(np.float64)
        identity = logistic.predict_proba(target)[:, 1]
        identity_error = float(np.max(np.abs(identity - frame.probability_identity.to_numpy(float))))
        if identity_error > 1e-6:
            raise RuntimeError(f"{dataset}: frozen identity probabilities do not reconstruct ({identity_error})")

        matrix, offset = coral_fit(source, target[audit_mask])
        coral_values = target @ matrix + offset
        qmaps = quantile_fit(source, target[audit_mask])
        quantile_values = quantile_apply(target, qmaps)
        component_probabilities = {
            "coral": logistic.predict_proba(coral_values)[:, 1],
            "quantile_mapping": logistic.predict_proba(quantile_values)[:, 1],
        }
        projected = {}
        component_diagnostics = {}
        for component_index, (name, probability) in enumerate(component_probabilities.items()):
            _, fit_diagnostics = fit_opct(
                identity[audit_mask],
                probability[audit_mask],
                RULE,
                args.seed + dataset_index * 10007 + component_index * 1009,
                "cuda",
                progress,
            )
            projected[name] = apply_opct(identity, fit_diagnostics)
            component_diagnostics[name] = consensus_diagnostics(
                identity[audit_mask], projected[name][audit_mask]
            )

        both_pass = all(bool(component_diagnostics[name]["applicable"]) for name in projected)
        witness = (
            witness_statistics(
                identity[audit_mask],
                projected["coral"][audit_mask],
                projected["quantile_mapping"][audit_mask],
            )
            if both_pass
            else {"witness_pass": False}
        )
        applicable = bool(both_pass and witness["witness_pass"])
        frame["probability_wg_opct"] = projected["coral"] if applicable else identity
        audit = frame.loc[audit_mask].reset_index(drop=True)
        heldout = frame.loc[heldout_mask].reset_index(drop=True)
        certificate = stratified_certificate(
            audit,
            config,
            args.seed + dataset_index * 7919,
            applicable,
            float(component_diagnostics["coral"]["probability_rank"]),
            int(component_diagnostics["coral"]["order_inversions"]),
        )
        selected = "wg_opct" if bool(certificate["certified"]) else "identity"
        outcome = evaluate_action(heldout, selected)
        result_rows.append(
            {
                "dataset": dataset,
                "protocol": "audit_covariate_fit_heldout_covariate_unseen",
                "identity_reconstruction_max_abs_error": identity_error,
                "audit_configurations": int(audit_mask.sum()),
                "heldout_configurations": int(heldout_mask.sum()),
                "component_gate_pass": both_pass,
                "witness_gate_pass": bool(witness["witness_pass"]),
                "certificate_pass": bool(certificate["certified"]),
                "selected_method": selected,
                "audit_brier_gain": certificate["brier_gain"],
                "audit_brier_gain_lcb": certificate["brier_gain_lcb"],
                "heldout_brier_gain": outcome["brier_gain"],
                "heldout_auc_delta": outcome["auc_delta"],
                "material_negative_transfer": outcome["material_negative_transfer"],
            }
        )
        for row_index, row in frame.iterrows():
            prediction_rows.append(
                {
                    "dataset": dataset,
                    "row_index": row_index,
                    "partition": row.partition,
                    **{column: row[column] for column in CLUSTER_COLUMNS},
                    "material_optimism_event": int(row.material_optimism_event),
                    "probability_identity": float(identity[row_index]),
                    "probability_inductive_action": float(frame.loc[row_index, "probability_wg_opct"]),
                }
            )
        progress.set_postfix(dataset=dataset, action=selected, refresh=False)
    progress.close()

    results = pd.DataFrame(result_rows)
    assignments = pd.concat(assignment_rows, ignore_index=True)
    predictions = pd.DataFrame(prediction_rows)
    archived = pd.read_csv(ARCHIVED / "primary_heldout_results.csv")[["dataset", "selected_method", "brier_gain"]]
    archived = archived.rename(columns={"selected_method": "transductive_method", "brier_gain": "transductive_heldout_brier_gain"})
    comparison = results.merge(archived, on="dataset", how="left", validate="one_to_one")
    comparison["gain_delta_inductive_minus_transductive"] = (
        comparison.heldout_brier_gain - comparison.transductive_heldout_brier_gain
    )

    args.output_root.mkdir(parents=True, exist_ok=False)
    results.to_csv(args.output_root / "inductive_action_outcomes.csv", index=False)
    assignments.to_csv(args.output_root / "covariate_blind_assignments.csv", index=False)
    predictions.to_csv(args.output_root / "inductive_heldout_predictions.csv", index=False)
    comparison.to_csv(args.output_root / "inductive_vs_transductive.csv", index=False)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "device": torch.cuda.get_device_name(0),
        "protocol": "Assignment uses stratum labels and split-seed identifiers only. CORAL, quantile maps, OPCT parameters, applicability gates, and the certificate use audit rows only; held-out covariates and outcomes are unseen until final evaluation.",
        "datasets": int(len(results)),
        "released_actions": int(results.selected_method.ne("identity").sum()),
        "positive_released_heldout_gains": int(
            ((results.selected_method.ne("identity")) & (results.heldout_brier_gain > 0)).sum()
        ),
        "material_negative_releases": int(results.material_negative_transfer.sum()),
        "claim_boundary": "Configuration-level inductive sensitivity; split-seed configurations may still reuse underlying participant or stimulus observations.",
    }
    (args.output_root / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
