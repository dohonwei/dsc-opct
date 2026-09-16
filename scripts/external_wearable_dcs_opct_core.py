from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_identity_shortcut_risk import MODEL_CAPACITY  # noqa: E402
from apply_frozen_transport_policy_to_seediv import probability  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    cluster_geometry,
    partition_frame,
    select_distribution_covered_audit,
    stratified_certificate,
)
from develop_order_preserving_transport_v6 import RULE, fit_opct  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import witness_statistics  # noqa: E402
from identity_shortcut.domain_transport import coral, quantile_map  # noqa: E402
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402


KEYS = ("dataset", "task", "axis", "representation", "model", "split_seed")
PREDICTORS = (
    "opportunity_delta",
    "metadata_prior_opportunity",
    "encoding_margin",
    "dose_encoding_interaction",
    "model_capacity",
)
FORBIDDEN_OUTCOME_COLUMNS = frozenset(
    {
        "exposure_effect",
        "dose_zero_exposure_effect",
        "dose_induced_amplification",
        "material_optimism_event",
        "target",
    }
)


@dataclass(frozen=True)
class ExternalApplicationConfig:
    dataset: str
    tasks: tuple[str, ...]
    representations: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    doses: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    material_threshold: float = 0.02
    axis: str = "subject"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_config(config: ExternalApplicationConfig) -> None:
    named = {
        "tasks": config.tasks,
        "representations": config.representations,
        "models": config.models,
        "seeds": config.seeds,
    }
    if not config.dataset.strip() or config.axis != "subject":
        raise ValueError("External application requires a named dataset and subject axis")
    for name, values in named.items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be non-empty and unique")
    if (
        config.doses != tuple(sorted(config.doses))
        or len(config.doses) != len(set(config.doses))
        or not config.doses
        or config.doses[0] != 0.0
        or any(value < 0.0 or value > 1.0 for value in config.doses)
    ):
        raise ValueError("Registered doses must be unique, ordered, in [0, 1], and start at zero")
    if not np.isclose(config.material_threshold, 0.02):
        raise ValueError("The frozen material-event threshold is 0.02")
    unknown_models = set(config.models) - set(MODEL_CAPACITY)
    if unknown_models:
        raise ValueError(f"Unknown model-capacity levels: {sorted(unknown_models)}")


def _require_exact_levels(table: pd.DataFrame, config: ExternalApplicationConfig) -> None:
    expected = {
        "dataset": {config.dataset},
        "task": set(config.tasks),
        "axis": {config.axis},
        "representation": set(config.representations),
        "model": set(config.models),
        "split_seed": set(config.seeds),
        "nominal_dose": set(config.doses),
    }
    for column, levels in expected.items():
        observed = set(table[column].drop_duplicates())
        if observed != levels:
            raise ValueError(
                f"Unexpected {column} levels: expected {sorted(levels, key=str)}, "
                f"observed {sorted(observed, key=str)}"
            )


def load_unlabeled_configuration_table(
    summary_path: Path,
    encoding_path: Path,
    config: ExternalApplicationConfig,
) -> pd.DataFrame:
    _validate_config(config)
    mechanism_columns = list(KEYS) + [
        "nominal_dose",
        "achieved_opportunity",
        "metadata_prior_opportunity",
    ]
    table = pd.read_csv(summary_path, usecols=mechanism_columns)
    _require_exact_levels(table, config)
    unique_keys = list(KEYS) + ["nominal_dose"]
    if table.duplicated(unique_keys).any():
        raise ValueError("Dose summary contains duplicate configuration-dose rows")
    expected_rows = (
        len(config.tasks)
        * len(config.representations)
        * len(config.models)
        * len(config.seeds)
        * len(config.doses)
    )
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} dose rows, observed {len(table)}")
    complete_doses = table.groupby(list(KEYS), dropna=False).nominal_dose.apply(
        lambda values: np.allclose(values.sort_values().to_numpy(), config.doses)
    )
    if not complete_doses.all():
        raise ValueError("At least one configuration lacks the complete registered dose grid")

    encoding = pd.read_csv(
        encoding_path,
        usecols=[
            "dataset",
            "representation",
            "split_seed",
            "n_features",
            "encoding_margin",
        ],
    )
    if set(encoding.dataset) != {config.dataset}:
        raise ValueError("Identity-encoding table contains an unexpected dataset")
    encoding = encoding.groupby(
        ["dataset", "representation", "split_seed"], as_index=False
    ).agg(n_features=("n_features", "first"), encoding_margin=("encoding_margin", "mean"))
    expected_encoding = len(config.representations) * len(config.seeds)
    if (
        len(encoding) != expected_encoding
        or set(encoding.representation) != set(config.representations)
        or set(encoding.split_seed) != set(config.seeds)
    ):
        raise ValueError("Identity-encoding table does not cover the registered grid")

    table = table.rename(columns={"achieved_opportunity": "label_opportunity"})
    table = table.merge(
        encoding,
        on=["dataset", "representation", "split_seed"],
        validate="many_to_one",
    )
    anchors = table.loc[
        table.nominal_dose.eq(0.0),
        list(KEYS) + ["label_opportunity", "metadata_prior_opportunity"],
    ].rename(
        columns={
            "label_opportunity": "dose_zero_label_opportunity",
            "metadata_prior_opportunity": "dose_zero_metadata_prior_opportunity",
        }
    )
    if len(anchors) * len(config.doses) != len(table) or anchors.duplicated(KEYS).any():
        raise ValueError("Dose-zero anchors are incomplete or non-unique")
    table = table.merge(anchors, on=list(KEYS), validate="many_to_one")
    table = table.loc[table.nominal_dose.gt(0.0)].copy()
    table["opportunity_delta"] = (
        table.label_opportunity - table.dose_zero_label_opportunity
    )
    table["metadata_prior_delta"] = (
        table.metadata_prior_opportunity - table.dose_zero_metadata_prior_opportunity
    )
    table["dose_encoding_interaction"] = table.opportunity_delta * table.encoding_margin
    table["model_capacity"] = table.model.map(MODEL_CAPACITY)
    numeric = table.select_dtypes(include=[np.number]).to_numpy(float)
    if not np.isfinite(numeric).all() or table.model_capacity.isna().any():
        raise ValueError("Unlabeled mechanism table contains invalid numeric values")
    return table.sort_values(list(KEYS) + ["nominal_dose"]).reset_index(drop=True)


def risk_control_config(freeze: dict[str, Any]) -> RiskControlledConfig:
    certificate = freeze["certification"]
    return RiskControlledConfig(
        configuration_budgets=(int(certificate["audit_configuration_budget"]),),
        bootstrap_repetitions=int(certificate["bootstrap_repetitions"]),
        auc_noninferiority_margin=float(certificate["auc_noninferiority_margin"]),
        minimum_brier_gain=float(certificate["minimum_brier_gain_lcb"]),
        minimum_valid_auc_bootstraps=int(certificate["minimum_valid_auc_bootstraps"]),
    )


def validate_frozen_contract(
    risk_manifest: dict[str, Any], freeze: dict[str, Any], config: ExternalApplicationConfig
) -> None:
    _validate_config(config)
    if not risk_manifest.get("admissible") or risk_manifest.get("axis") != "subject":
        raise RuntimeError("Frozen risk model is not admissible for the subject axis")
    if tuple(risk_manifest.get("numeric_features", ())) != PREDICTORS:
        raise RuntimeError("Frozen risk predictors differ from the registered five-predictor model")
    if not np.isclose(risk_manifest.get("material_effect_threshold"), config.material_threshold):
        raise RuntimeError("Risk-model and external material thresholds differ")
    action = freeze["probability_action"]
    checks = {
        "epochs": RULE.epochs,
        "learning_rate": RULE.learning_rate,
        "regularization": RULE.regularization,
        "minimum_scale": RULE.minimum_scale,
        "maximum_scale": RULE.maximum_scale,
        "maximum_absolute_shift": RULE.maximum_absolute_shift,
        "maximum_probability_mean_shift": RULE.maximum_probability_mean_shift,
        "minimum_probability_rank": RULE.minimum_rank,
        "maximum_order_inversions": 0,
    }
    for key, expected in checks.items():
        if not np.isclose(float(action[key]), float(expected)):
            raise RuntimeError(f"Frozen OPCT setting changed: {key}")


def fit_unlabeled_action(
    target_table: pd.DataFrame,
    public_risk_table: pd.DataFrame,
    risk_model: Any,
    risk_manifest: dict[str, Any],
    freeze: dict[str, Any],
    config: ExternalApplicationConfig,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    validate_frozen_contract(risk_manifest, freeze, config)
    leaked = FORBIDDEN_OUTCOME_COLUMNS & set(target_table)
    if leaked:
        raise ValueError(f"Outcome columns reached the unlabeled stage: {sorted(leaked)}")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The registered external OPCT stage requires CUDA")
    if len(target_table) != (
        len(config.tasks)
        * len(config.representations)
        * len(config.models)
        * len(config.seeds)
        * (len(config.doses) - 1)
    ):
        raise ValueError("Unlabeled configuration count differs from the registered grid")

    scaler = risk_model.named_steps["preprocess"].named_transformers_["numeric"]
    logistic = risk_model.named_steps["model"]
    source = scaler.transform(public_risk_table[list(PREDICTORS)]).astype(np.float32)
    target = scaler.transform(target_table[list(PREDICTORS)]).astype(np.float32)
    transformed = {
        "identity": target.copy(),
        "coral": coral(source, target),
        "quantile_mapping": quantile_map(source, target),
    }
    mechanism = target_table.copy()
    for method, values in transformed.items():
        mechanism[f"probability_{method}"] = probability(logistic, values)

    identity = mechanism.probability_identity.to_numpy(float)
    projections: dict[str, np.ndarray] = {}
    component_rows = []
    progress = tqdm(
        total=2 * RULE.epochs,
        desc=f"{config.dataset} CUDA OPCT components",
        unit="epoch",
        dynamic_ncols=True,
    )
    for index, method in enumerate(("coral", "quantile_mapping")):
        projected, diagnostics = fit_opct(
            identity,
            mechanism[f"probability_{method}"].to_numpy(float),
            RULE,
            int(freeze["certification"]["primary_audit_seed"]) + index * 1009,
            device,
            progress,
        )
        projections[method] = projected
        component_rows.append({"method": method, **diagnostics})
    progress.close()
    components = pd.DataFrame(component_rows)
    components_pass = bool(components.applicable.eq(True).all())
    witness = (
        witness_statistics(identity, projections["coral"], projections["quantile_mapping"])
        if components_pass
        else {
            "directional_agreement": np.nan,
            "displacement_cosine": np.nan,
            "normalized_disagreement": np.nan,
            "witness_pass": False,
        }
    )
    candidate_applicable = bool(components_pass and witness["witness_pass"])
    mechanism["probability_wg_opct"] = (
        projections["coral"] if candidate_applicable else identity
    )
    probability_columns = [column for column in mechanism if column.startswith("probability_")]
    probability_values = mechanism[probability_columns].to_numpy(float)
    if not np.isfinite(probability_values).all() or not (
        (probability_values >= 0.0) & (probability_values <= 1.0)
    ).all():
        raise ValueError("The unlabeled application produced invalid probabilities")

    geometry = cluster_geometry(mechanism)
    assignment = select_distribution_covered_audit(geometry)
    expected_clusters = (
        len(config.tasks)
        * len(config.representations)
        * len(config.models)
        * len(config.seeds)
    )
    if len(geometry) != expected_clusters or len(assignment) != expected_clusters:
        raise ValueError("Distribution-covered assignment has an unexpected cluster count")
    counts = assignment.partition.value_counts().to_dict()
    if counts != {"audit": expected_clusters // 2, "heldout": expected_clusters // 2}:
        raise ValueError("Distribution-covered assignment is not an exact family half-split")
    return {
        "mechanism": mechanism,
        "components": components,
        "witness": witness,
        "candidate_applicable": candidate_applicable,
        "candidate_method": "wg_opct" if candidate_applicable else "identity",
        "geometry": geometry,
        "assignment": assignment,
    }


def write_blinded_action_bundle(
    output_root: Path,
    prepared: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[Path, str]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite external action bundle: {output_root}")
    mechanism = prepared["mechanism"]
    leaked = FORBIDDEN_OUTCOME_COLUMNS & set(mechanism)
    if leaked:
        raise ValueError(f"Cannot lock a bundle containing outcomes: {sorted(leaked)}")
    output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "components": output_root / "component_projection_diagnostics.csv",
        "geometry": output_root / "unlabeled_cluster_geometry.csv",
        "assignment": output_root / "distribution_covered_assignments.csv",
        "mechanism": output_root / "mechanism_probabilities_blinded.csv",
    }
    prepared["components"].to_csv(paths["components"], index=False)
    prepared["geometry"].to_csv(paths["geometry"], index=False)
    prepared["assignment"].to_csv(paths["assignment"], index=False)
    mechanism.to_csv(paths["mechanism"], index=False)
    payload = {
        "status": "external_action_and_audit_locked_before_outcomes",
        "candidate_method": prepared["candidate_method"],
        "candidate_applicable": bool(prepared["candidate_applicable"]),
        "component_diagnostics": prepared["components"].to_dict(orient="records"),
        "witness": prepared["witness"],
        "artifact_sha256": {name: sha256(path) for name, path in paths.items()},
        "provenance": provenance,
        "claim_boundary": "This lock contains no material-event outcome or effectiveness result.",
    }
    lock_path = output_root / "primary_action_and_audit_lock.json"
    lock_path.write_text(json.dumps(json_native(payload), indent=2), encoding="utf-8")
    return lock_path, sha256(lock_path)


def attach_registered_outcomes(
    mechanism: pd.DataFrame,
    summary_path: Path,
    config: ExternalApplicationConfig,
    lock_path: Path,
    expected_lock_sha256: str,
) -> pd.DataFrame:
    if sha256(lock_path) != expected_lock_sha256:
        raise RuntimeError("The blinded action lock changed before outcome access")
    outcome_columns = list(KEYS) + ["nominal_dose", "exposure_effect"]
    summary = pd.read_csv(summary_path, usecols=outcome_columns)
    _require_exact_levels(summary, config)
    if summary.duplicated(list(KEYS) + ["nominal_dose"]).any():
        raise ValueError("Outcome summary contains duplicate configuration-dose rows")
    anchors = summary.loc[
        summary.nominal_dose.eq(0.0), list(KEYS) + ["exposure_effect"]
    ].rename(columns={"exposure_effect": "dose_zero_exposure_effect"})
    outcomes = summary.merge(anchors, on=list(KEYS), validate="many_to_one")
    outcomes = outcomes.loc[outcomes.nominal_dose.gt(0.0)].copy()
    outcomes["dose_induced_amplification"] = (
        outcomes.exposure_effect - outcomes.dose_zero_exposure_effect
    )
    outcomes["material_optimism_event"] = (
        outcomes.dose_induced_amplification >= config.material_threshold
    ).astype(int)
    evaluation = mechanism.merge(
        outcomes,
        on=list(KEYS) + ["nominal_dose"],
        validate="one_to_one",
    )
    if len(evaluation) != len(mechanism):
        raise ValueError("Outcome attachment did not preserve all locked configurations")
    if sha256(lock_path) != expected_lock_sha256:
        raise RuntimeError("The blinded action lock changed during outcome attachment")
    return evaluation


def evaluate_locked_endpoint(
    evaluation: pd.DataFrame,
    assignment: pd.DataFrame,
    components: pd.DataFrame,
    candidate_applicable: bool,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    audit, heldout = partition_frame(evaluation, assignment)
    audit_classes = sorted(audit.material_optimism_event.unique().tolist())
    heldout_classes = sorted(heldout.material_optimism_event.unique().tolist())
    endpoint_estimable = audit_classes == [0, 1] and heldout_classes == [0, 1]
    coral = components.loc[components.method.eq("coral")]
    if len(coral) != 1:
        raise ValueError("Exactly one CORAL component diagnostic is required")
    coral = coral.iloc[0]
    if not endpoint_estimable:
        certificate = {
            "certified": False,
            "failure_reason": "endpoint_non_estimable",
            "audit_event_classes": audit_classes,
            "heldout_event_classes": heldout_classes,
        }
        selected = "identity"
        heldout_metrics = None
        effectiveness_pass = False
        released_action_nonharm_pass = None
    else:
        certificate = stratified_certificate(
            audit,
            risk_control_config(freeze),
            int(freeze["certification"]["primary_audit_seed"])
            + RULE.configuration_budget * 1009,
            candidate_applicable,
            float(coral.probability_rank),
            int(coral.order_inversions),
        )
        selected = "wg_opct" if bool(certificate["certified"]) else "identity"
        heldout_metrics = evaluate_action(heldout, selected)
        effectiveness_pass = bool(
            selected == "wg_opct" and float(heldout_metrics["brier_gain"]) > 0.001
        )
        released_action_nonharm_pass = (
            bool(
                float(heldout_metrics["brier_gain"]) >= 0.0
                and not heldout_metrics["negative_transfer"]
                and heldout_metrics["auc_noninferior"]
                and float(coral.probability_rank) >= RULE.minimum_rank
                and int(coral.order_inversions) == 0
            )
            if selected == "wg_opct"
            else None
        )
    claim_supported = bool(effectiveness_pass and released_action_nonharm_pass is True)
    return {
        "status": "passed" if claim_supported else "failed",
        "claim_supported": claim_supported,
        "endpoint_estimable": endpoint_estimable,
        "audit_event_classes": audit_classes,
        "heldout_event_classes": heldout_classes,
        "candidate_method": "wg_opct" if candidate_applicable else "identity",
        "selected_method": selected,
        "certificate_pass": bool(certificate.get("certified", False)),
        "certificate": certificate,
        "effectiveness_pass": effectiveness_pass,
        "released_action_nonharm_pass": released_action_nonharm_pass,
        "safety_success": claim_supported,
        "heldout_metrics": heldout_metrics,
        "interpretation": (
            "prospective external effectiveness with bounded released-action non-harm"
            if claim_supported
            else (
                "effectiveness non-estimable; this is not a safety success"
                if not endpoint_estimable
                else (
                    "identity non-intervention; this is not effectiveness or a safety success"
                    if selected == "identity"
                    else "the released action failed at least one frozen external gate"
                )
            )
        ),
        "audit": audit,
        "heldout": heldout,
    }
