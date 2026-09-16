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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_dcs_opct_v11_inductive_target_sensitivity import (  # noqa: E402
    RISK_MODEL,
    SOURCE_TABLE,
    fixed_assignment,
)
from analyze_identity_shortcut_risk import classifier_features  # noqa: E402
from develop_consensus_order_preserving_transport_v7 import load_development  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    CLUSTER_COLUMNS,
    stratified_certificate,
)
from develop_order_preserving_transport_v6 import RULE  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402


INDUCTIVE = Path(
    "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/inductive_action_outcomes.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914"),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260914)
    return parser.parse_args()


def importance_weights(source: np.ndarray, target: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    values = np.vstack([source, target])
    domain = np.r_[np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
    classifier = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed)
    classifier.fit(values, domain)
    probability_target = np.clip(classifier.predict_proba(source)[:, 1], 1e-4, 1 - 1e-4)
    odds = probability_target / (1.0 - probability_target)
    prior_correction = len(source) / len(target)
    raw = odds * prior_correction
    weights = np.clip(raw, 0.05, 20.0)
    weights /= weights.mean()
    ess = float(weights.sum() ** 2 / np.sum(weights**2))
    auc = float(roc_auc_score(domain, classifier.predict_proba(values)[:, 1]))
    return weights, {
        "domain_auc": auc,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_ess": ess,
    }


def fit_positive_weighted_platt(
    probability: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    epochs: int,
    seed: int,
    progress: tqdm,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.as_tensor(logit(np.clip(probability, 1e-6, 1 - 1e-6)), dtype=torch.float32, device="cuda")
    y = torch.as_tensor(labels, dtype=torch.float32, device="cuda")
    w = torch.as_tensor(weights, dtype=torch.float32, device="cuda")
    raw_scale = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
    shift = torch.nn.Parameter(torch.tensor(0.0, device="cuda"))
    optimizer = torch.optim.Adam([raw_scale, shift], lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.nn.functional.softplus(raw_scale) + 1e-4
        prediction = torch.sigmoid(scale * x + shift)
        loss = (w * torch.nn.functional.binary_cross_entropy(prediction, y, reduction="none")).mean()
        loss.backward()
        optimizer.step()
        progress.update(1)
    return float((torch.nn.functional.softplus(raw_scale) + 1e-4).detach()), float(shift.detach())


def atc_threshold(probability: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    hard = probability >= 0.5
    source_accuracy = float(accuracy_score(labels, hard))
    confidence = np.maximum(probability, 1.0 - probability)
    candidates = np.unique(np.r_[0.5, confidence, 1.0])
    estimates = np.array([(confidence >= threshold).mean() for threshold in candidates])
    index = int(np.argmin(np.abs(estimates - source_accuracy)))
    return float(candidates[index]), source_accuracy


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.bootstrap_repetitions < 5000:
        raise ValueError("At least 5,000 bootstrap repetitions are required")

    features = classifier_features()
    frozen = joblib.load(RISK_MODEL)
    scaler = frozen.named_steps["preprocess"].named_transformers_["numeric"]
    risk_model = frozen.named_steps["model"]
    source_frame = pd.read_csv(SOURCE_TABLE)
    source_x = scaler.transform(source_frame[features]).astype(np.float64)
    source_y = source_frame.material_optimism_event.to_numpy(int)
    source_probability = risk_model.predict_proba(source_x)[:, 1]
    threshold, source_accuracy = atc_threshold(source_probability, source_y)
    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )

    datasets = load_development()
    progress = tqdm(
        total=len(datasets) * args.epochs,
        desc="CUDA CPCS-style weighted calibration", unit="epoch", dynamic_ncols=True,
    )
    rows = []
    atc_rows = []
    parameter_rows = []
    for dataset_index, (dataset, (raw, _)) in enumerate(sorted(datasets.items())):
        frame = raw.copy().reset_index(drop=True)
        frame["dataset"] = dataset
        assignment = fixed_assignment(frame)
        frame = frame.merge(assignment, on=list(CLUSTER_COLUMNS), validate="many_to_one")
        audit_mask = frame.partition.eq("audit").to_numpy()
        target_x = scaler.transform(frame[features]).astype(np.float64)
        identity = risk_model.predict_proba(target_x)[:, 1]
        weights, diagnostics = importance_weights(source_x, target_x[audit_mask], args.seed + dataset_index)
        scale, shift = fit_positive_weighted_platt(
            source_probability, source_y, weights, args.epochs,
            args.seed + dataset_index * 1009, progress,
        )
        candidate = expit(scale * logit(np.clip(identity, 1e-6, 1 - 1e-6)) + shift)
        frame["probability_identity"] = identity
        frame["probability_wg_opct"] = candidate
        audit = frame.loc[audit_mask].reset_index(drop=True)
        heldout = frame.loc[~audit_mask].reset_index(drop=True)
        certificate = stratified_certificate(
            audit, config, args.seed + dataset_index * 7919,
            True, 1.0, 0,
        )
        selected = "wg_opct" if certificate["certified"] else "identity"
        outcome = evaluate_action(heldout, selected)
        rows.append({
            "dataset": dataset,
            "method": "cpcs_style_iw_platt_certified",
            "target_labels_for_candidate_fit": 0,
            "audit_labels_for_certificate": len(audit),
            "released": selected != "identity",
            "audit_brier_gain": certificate["brier_gain"],
            "audit_brier_gain_lcb": certificate["brier_gain_lcb"],
            "heldout_brier_gain": outcome["brier_gain"],
            "heldout_auc_delta": outcome["auc_delta"],
            "material_negative_transfer": outcome["material_negative_transfer"],
        })
        parameter_rows.append({
            "dataset": dataset, "scale": scale, "shift": shift, **diagnostics
        })

        heldout_probability = identity[~audit_mask]
        heldout_y = heldout.material_optimism_event.to_numpy(int)
        target_confidence = np.maximum(heldout_probability, 1.0 - heldout_probability)
        estimate = float((target_confidence >= threshold).mean())
        actual = float(accuracy_score(heldout_y, heldout_probability >= 0.5))
        atc_rows.append({
            "dataset": dataset,
            "method": "atc_style_accuracy_estimator",
            "source_confidence_threshold": threshold,
            "source_accuracy": source_accuracy,
            "estimated_target_accuracy": estimate,
            "observed_heldout_accuracy": actual,
            "absolute_estimation_error": abs(estimate - actual),
            "intervention_released": False,
            "brier_gain": np.nan,
            "material_negative_transfer": np.nan,
        })
        progress.set_postfix(dataset=dataset, released=selected != "identity", refresh=False)
    progress.close()

    cpcs = pd.DataFrame(rows)
    dcs = pd.read_csv(INDUCTIVE)[[
        "dataset", "selected_method", "audit_configurations", "heldout_brier_gain",
        "heldout_auc_delta", "material_negative_transfer",
    ]].rename(columns={"selected_method": "dcs_selected_method"})
    comparison = cpcs.merge(dcs, on="dataset", suffixes=("_cpcs", "_dcs"), validate="one_to_one")
    comparison["paired_gain_dcs_minus_cpcs"] = (
        comparison.heldout_brier_gain_dcs - comparison.heldout_brier_gain_cpcs
    )

    args.output_root.mkdir(parents=True, exist_ok=False)
    cpcs.to_csv(args.output_root / "cpcs_style_certified_outcomes.csv", index=False)
    comparison.to_csv(args.output_root / "dcs_vs_cpcs_style.csv", index=False)
    pd.DataFrame(atc_rows).to_csv(args.output_root / "atc_style_accuracy_estimation.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_root / "cpcs_style_parameters.csv", index=False)
    report = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "candidate_fit_resources": "source configuration labels plus audit-side unlabeled target features",
        "release_resources": "same full inductive audit-label certificate used for DCS-OPCT",
        "atc_output": "aggregate target hard-classification accuracy estimate; not a probability action",
        "claim_boundary": (
            "CPCS-style and ATC-style adaptations respect the configuration-level task but are not claimed "
            "as verbatim reproductions of instance-level implementations. TransCal is not instantiated because "
            "the study has no adapted target classifier logits or target task model required by its primary formulation."
        ),
    }
    (args.output_root / "analysis_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(pd.DataFrame(atc_rows).to_string(index=False))


if __name__ == "__main__":
    main()
