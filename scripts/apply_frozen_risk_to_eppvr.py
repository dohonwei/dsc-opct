from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_identity_shortcut_risk import (  # noqa: E402
    MODEL_CAPACITY,
    bootstrap_classification_metrics,
    classification_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a frozen public-dataset shortcut-risk model to EPPVR outcomes."
    )
    parser.add_argument(
        "--risk-root",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier"),
    )
    parser.add_argument(
        "--eppvr-root",
        type=Path,
        default=Path("outputs/eppvr_subject_shortcut_external_validation"),
    )
    parser.add_argument(
        "--eppvr-features",
        type=Path,
        default=Path("outputs/unified_frontal_features/eppvr.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/eppvr_frozen_risk_validation"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260904)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    freeze_path = args.risk_root / "freeze_manifest.json"
    model_path = args.risk_root / "frozen_risk_model.joblib"
    if not freeze_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            "No admissible frozen risk classifier is available; run the public-data "
            "risk analysis and satisfy its acceptance gate before EPPVR application"
        )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("admissible") is not True or freeze.get("status") != "admissible_freeze":
        raise ValueError("The supplied risk manifest is not an admissible freeze")
    if freeze.get("axis") != "subject":
        raise ValueError("EPPVR accepts only a subject-axis frozen risk classifier")
    if sha256(model_path) != freeze["model_sha256"]:
        raise ValueError("Frozen risk-model hash does not match its manifest")
    if set(freeze["training_datasets"]) != {"DEAP", "MAHNOB-HCI"}:
        raise ValueError("External validation requires a model frozen on DEAP and MAHNOB-HCI only")
    summary = pd.read_csv(args.eppvr_root / "summary.csv")
    probes = pd.read_csv(args.eppvr_root / "identity_probe_summary.csv")
    probes["encoding_margin"] = probes.balanced_accuracy - probes.chance
    probes = (
        probes.groupby(["representation", "n_features", "split_seed"])
        .encoding_margin.mean()
        .reset_index()
    )
    table = summary.merge(
        probes,
        on=["representation", "n_features", "split_seed"],
        validate="many_to_one",
    )
    table["dataset"] = "EPPVR"
    table["axis"] = "subject"
    keys = ["task", "representation", "n_features", "model", "split_seed"]
    anchors = table.loc[
        table.nominal_dose == 0,
        keys
        + [
            "subject_exposure_effect",
            "normalized_mutual_information",
            "metadata_prior_opportunity",
        ],
    ].rename(
        columns={
            "subject_exposure_effect": "dose_zero_exposure_effect",
            "normalized_mutual_information": "dose_zero_label_opportunity",
            "metadata_prior_opportunity": "dose_zero_metadata_prior_opportunity",
        }
    )
    if anchors.duplicated(keys).any():
        raise ValueError("EPPVR dose-zero anchors are not unique")
    table = table.merge(anchors, on=keys, validate="many_to_one")
    table = table.loc[table.nominal_dose > 0].copy()
    table["dose_induced_amplification"] = (
        table.subject_exposure_effect - table.dose_zero_exposure_effect
    )
    table["opportunity_delta"] = (
        table.normalized_mutual_information - table.dose_zero_label_opportunity
    )
    table["metadata_prior_delta"] = (
        table.metadata_prior_opportunity - table.dose_zero_metadata_prior_opportunity
    )
    table["dose_encoding_interaction"] = table.opportunity_delta * table.encoding_margin
    table["model_capacity"] = table.model.map(MODEL_CAPACITY)
    material_threshold = float(freeze["material_effect_threshold"])
    table["material_optimism_event"] = (
        table.dose_induced_amplification >= material_threshold
    ).astype(int)
    if table.model_capacity.isna().any():
        raise ValueError("EPPVR summary contains a model absent from the frozen capacity mapping")
    feature_names = freeze["numeric_features"] + freeze["categorical_features"]
    missing = set(feature_names) - set(table.columns)
    if missing:
        raise ValueError(f"EPPVR risk table is missing frozen features: {sorted(missing)}")
    pipeline = joblib.load(model_path)
    table["predicted_material_risk"] = pipeline.predict_proba(table[feature_names])[:, 1]
    decision_threshold = float(freeze["decision_threshold"])
    table["predicted_material_optimism_event"] = (
        table.predicted_material_risk >= decision_threshold
    ).astype(int)
    observed = table.material_optimism_event.to_numpy(int)
    predicted = table.predicted_material_risk.to_numpy(float)
    if len(np.unique(observed)) < 2:
        raise ValueError("EPPVR external target contains only one class")
    public_prevalence = float(freeze["public_training_event_prevalence"])
    metrics = classification_metrics(
        observed,
        predicted,
        decision_threshold,
        public_prevalence,
    )
    metrics.update(
        bootstrap_classification_metrics(
            table,
            predicted,
            decision_threshold,
            public_prevalence,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        )
    )
    metrics.update({
        "n_rows": len(table),
        "event_prevalence": metrics["prevalence"],
        "material_effect_threshold": material_threshold,
        "null_probability_source": "frozen public training prevalence",
        "risk_model_sha256": freeze["model_sha256"],
        "claim_boundary": (
            "Subject-axis external validation of material dose-induced optimism only; "
            "EPPVR record IDs are not physical stimulus identities."
        ),
    })
    rule = freeze["external_acceptance_rule"]
    criteria = [
        {
            "criterion": "external_discrimination",
            "passed": metrics["roc_auc"] >= rule["minimum_roc_auc"],
            "observed": metrics["roc_auc"],
            "required": f">= {rule['minimum_roc_auc']}",
        },
        {
            "criterion": "external_precision_recall_lift",
            "passed": (
                metrics["average_precision_lift"]
                >= rule["minimum_average_precision_lift"]
            ),
            "observed": metrics["average_precision_lift"],
            "required": f">= {rule['minimum_average_precision_lift']}",
        },
        {
            "criterion": "external_probability_skill",
            "passed": metrics["brier_skill"] > rule["minimum_brier_skill"],
            "observed": metrics["brier_skill"],
            "required": f"> {rule['minimum_brier_skill']}",
        },
        {
            "criterion": "external_operating_point",
            "passed": (
                metrics["balanced_accuracy"]
                >= rule["minimum_balanced_accuracy"]
            ),
            "observed": metrics["balanced_accuracy"],
            "required": f">= {rule['minimum_balanced_accuracy']}",
        },
    ]
    external_gate = {
        "admissible": all(item["passed"] for item in criteria),
        "criteria": criteria,
        "frozen_rule": rule,
        "model_sha256": freeze["model_sha256"],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_root / "eppvr_risk_predictions.csv", index=False)
    (args.output_root / "external_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.output_root / "external_validation_gate.json").write_text(
        json.dumps(external_gate, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(json.dumps(external_gate, indent=2))


if __name__ == "__main__":
    main()
