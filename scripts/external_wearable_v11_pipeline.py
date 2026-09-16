from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from external_wearable_dcs_opct_core import (
    FORBIDDEN_OUTCOME_COLUMNS,
    ExternalApplicationConfig,
    attach_registered_outcomes,
    evaluate_locked_endpoint,
    fit_unlabeled_action,
    json_native,
    load_unlabeled_configuration_table,
    sha256,
    write_blinded_action_bundle,
)
from external_wearable_dose_core import run_prepared_dose_outcomes


PREOUTCOME_DIR = "preoutcome"
ACTION_DIR = "action"
OUTCOME_DIR = "outcome"
MODEL_OUTCOME_DIR = "model_outcomes"
PREOUTCOME_LOCK = "preoutcome_design_lock.json"


def _config_from_dict(values: dict[str, Any]) -> ExternalApplicationConfig:
    return ExternalApplicationConfig(
        dataset=str(values["dataset"]),
        tasks=tuple(values["tasks"]),
        representations=tuple(values["representations"]),
        models=tuple(values["models"]),
        seeds=tuple(int(value) for value in values["seeds"]),
        doses=tuple(float(value) for value in values["doses"]),
        material_threshold=float(values["material_threshold"]),
        axis=str(values["axis"]),
    )


def _assert_frame_equivalent(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: list[str],
    name: str,
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{name} columns differ from the prepared dose design")
    left = observed.sort_values(keys).reset_index(drop=True)[sorted(observed.columns)]
    right = expected.sort_values(keys).reset_index(drop=True)[sorted(expected.columns)]
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"{name} differs from the prepared dose design") from error


def _validate_prepared_payload(
    prepared_payload: Any,
    mechanism_summary: pd.DataFrame,
    identity_encoding: pd.DataFrame,
    split_audit: pd.DataFrame,
    config: ExternalApplicationConfig,
) -> None:
    if not isinstance(prepared_payload, dict) or set(prepared_payload) != set(config.tasks):
        raise ValueError("Prepared payload must contain exactly one design per registered task")
    mechanisms = []
    audits = []
    encodings = []
    for task in config.tasks:
        prepared = prepared_payload[task]
        required = {
            "frame",
            "labels",
            "feature_sets",
            "config",
            "cells",
            "split_audit",
            "mechanism_summary",
            "identity_encoding_margin",
        }
        if not isinstance(prepared, dict) or not required.issubset(prepared):
            raise ValueError(f"Prepared dose design is incomplete for task {task}")
        dose_config = prepared["config"]
        observed_contract = (
            dose_config.dataset,
            dose_config.task,
            dose_config.axis,
            tuple(dose_config.representations),
            tuple(dose_config.models),
            tuple(dose_config.seeds),
            tuple(dose_config.doses),
        )
        expected_contract = (
            config.dataset,
            task,
            config.axis,
            config.representations,
            config.models,
            config.seeds,
            config.doses,
        )
        if observed_contract != expected_contract:
            raise ValueError(f"Prepared dose configuration differs for task {task}")
        expected_cells = (
            len(config.seeds) * dose_config.subject_folds * dose_config.stimulus_folds
        )
        expected_audit = expected_cells * len(config.doses)
        if len(prepared["cells"]) != expected_cells or len(prepared["split_audit"]) != expected_audit:
            raise ValueError(f"Prepared fold or split-audit count differs for task {task}")
        if set(np.unique(np.asarray(prepared["labels"], dtype=int))) != {0, 1}:
            raise ValueError(f"Prepared labels are not binary for task {task}")
        mechanisms.append(prepared["mechanism_summary"])
        audits.append(prepared["split_audit"])
        encodings.append(prepared["identity_encoding_margin"])

    payload_mechanism = pd.concat(mechanisms, ignore_index=True)
    payload_audit = pd.concat(audits, ignore_index=True)
    _assert_frame_equivalent(
        mechanism_summary,
        payload_mechanism,
        keys=[
            "dataset",
            "task",
            "axis",
            "representation",
            "model",
            "split_seed",
            "nominal_dose",
        ],
        name="Mechanism summary",
    )
    _assert_frame_equivalent(
        split_audit,
        payload_audit,
        keys=["dataset", "task", "axis", "split_seed", "fold", "nominal_dose"],
        name="Split audit",
    )
    encoding_keys = ["dataset", "representation", "model", "split_seed"]
    for task_encoding in encodings[1:]:
        _assert_frame_equivalent(
            task_encoding,
            encodings[0],
            keys=encoding_keys,
            name="Cross-task identity encoding",
        )
    _assert_frame_equivalent(
        identity_encoding,
        encodings[0],
        keys=encoding_keys,
        name="Identity encoding",
    )


def write_preoutcome_design_bundle(
    output_root: Path,
    *,
    mechanism_summary: pd.DataFrame,
    identity_encoding: pd.DataFrame,
    split_audit: pd.DataFrame,
    prepared_payload: Any,
    config: ExternalApplicationConfig,
    provenance: dict[str, Any],
    adapter_audits: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, str]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite external pipeline: {output_root}")
    leaked = FORBIDDEN_OUTCOME_COLUMNS & set(mechanism_summary)
    if leaked:
        raise ValueError(f"Pre-outcome mechanism summary contains outcomes: {sorted(leaked)}")
    forbidden_audit = {
        "probability",
        "exposure_effect",
        "dose_induced_amplification",
        "material_optimism_event",
    }
    leaked_audit = forbidden_audit & set(split_audit)
    if leaked_audit:
        raise ValueError(f"Pre-outcome split audit contains outcomes: {sorted(leaked_audit)}")
    _validate_prepared_payload(
        prepared_payload,
        mechanism_summary,
        identity_encoding,
        split_audit,
        config,
    )
    if adapter_audits is not None:
        if set(adapter_audits) != set(config.tasks):
            raise ValueError("Adapter audits must cover exactly the registered tasks")
        for task, audit in adapter_audits.items():
            if (
                not isinstance(audit, dict)
                or audit.get("status") != "adapter_output_contract_passed"
                or audit.get("dataset") != config.dataset
                or audit.get("task") != task
            ):
                raise ValueError(f"Adapter audit is invalid for task {task}")

    stage = output_root / PREOUTCOME_DIR
    stage.mkdir(parents=True, exist_ok=False)
    paths = {
        "mechanism_summary": stage / "mechanism_summary.csv",
        "identity_encoding": stage / "identity_encoding_margin.csv",
        "split_audit": stage / "split_audit.csv",
        "prepared_payload": stage / "prepared_dose_design.joblib",
    }
    mechanism_summary.to_csv(paths["mechanism_summary"], index=False)
    identity_encoding.to_csv(paths["identity_encoding"], index=False)
    split_audit.to_csv(paths["split_audit"], index=False)
    joblib.dump(prepared_payload, paths["prepared_payload"])
    if adapter_audits is not None:
        paths["adapter_audits"] = stage / "adapter_output_audits.json"
        paths["adapter_audits"].write_text(
            json.dumps(json_native(adapter_audits), indent=2), encoding="utf-8"
        )

    validated = load_unlabeled_configuration_table(
        paths["mechanism_summary"], paths["identity_encoding"], config
    )
    expected_nonzero = (
        len(config.tasks)
        * len(config.representations)
        * len(config.models)
        * len(config.seeds)
        * (len(config.doses) - 1)
    )
    if len(validated) != expected_nonzero:
        raise ValueError("Pre-outcome bundle does not cover the registered configuration grid")
    payload = {
        "status": "external_preoutcome_design_locked",
        "date": "2026-09-09",
        "config": asdict(config),
        "artifact_sha256": {name: sha256(path) for name, path in paths.items()},
        "provenance": provenance,
        "execution_boundary": (
            "No emotion-classifier probability, exposure effect, dose-induced "
            "amplification, or material event existed in this bundle at lock time."
        ),
        "claim_boundary": "This design lock supplies no external empirical result.",
    }
    lock_path = stage / PREOUTCOME_LOCK
    lock_path.write_text(json.dumps(json_native(payload), indent=2), encoding="utf-8")
    return lock_path, sha256(lock_path)


def verify_preoutcome_design_bundle(
    output_root: Path, expected_lock_sha256: str | None = None
) -> tuple[dict[str, Any], ExternalApplicationConfig]:
    stage = output_root / PREOUTCOME_DIR
    lock_path = stage / PREOUTCOME_LOCK
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    if expected_lock_sha256 is not None and sha256(lock_path) != expected_lock_sha256:
        raise RuntimeError("Pre-outcome design lock checksum changed")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "external_preoutcome_design_locked":
        raise RuntimeError("Pre-outcome design lock has an invalid status")
    for name, expected in lock["artifact_sha256"].items():
        filenames = {
            "mechanism_summary": "mechanism_summary.csv",
            "identity_encoding": "identity_encoding_margin.csv",
            "split_audit": "split_audit.csv",
            "prepared_payload": "prepared_dose_design.joblib",
            "adapter_audits": "adapter_output_audits.json",
        }
        if name not in filenames:
            raise RuntimeError(f"Unknown pre-outcome artifact in lock: {name}")
        path = stage / filenames[name]
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Pre-outcome artifact checksum changed: {name}")
    config = _config_from_dict(lock["config"])
    load_unlabeled_configuration_table(
        stage / "mechanism_summary.csv",
        stage / "identity_encoding_margin.csv",
        config,
    )
    return lock, config


def load_prepared_payload(
    output_root: Path, expected_lock_sha256: str
) -> Any:
    verify_preoutcome_design_bundle(output_root, expected_lock_sha256)
    return joblib.load(output_root / PREOUTCOME_DIR / "prepared_dose_design.joblib")


def lock_unlabeled_action_from_bundle(
    output_root: Path,
    *,
    preoutcome_lock_sha256: str,
    public_risk_table: pd.DataFrame,
    risk_model: Any,
    risk_manifest: dict[str, Any],
    freeze: dict[str, Any],
    freeze_path: Path,
    device: str = "cuda",
    provenance: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    lock, config = verify_preoutcome_design_bundle(
        output_root, preoutcome_lock_sha256
    )
    stage = output_root / PREOUTCOME_DIR
    target = load_unlabeled_configuration_table(
        stage / "mechanism_summary.csv",
        stage / "identity_encoding_margin.csv",
        config,
    )
    prepared = fit_unlabeled_action(
        target,
        public_risk_table,
        risk_model,
        risk_manifest,
        freeze,
        config,
        device=device,
    )
    return write_blinded_action_bundle(
        output_root / ACTION_DIR,
        prepared,
        {
            **(provenance or {}),
            "preoutcome_lock_sha256": preoutcome_lock_sha256,
            "freeze_sha256": sha256(freeze_path),
            "preoutcome_artifact_sha256": lock["artifact_sha256"],
        },
    )


def _verify_action_lock(
    output_root: Path,
    *,
    preoutcome_lock_sha256: str,
    action_lock_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    action_lock_path = output_root / ACTION_DIR / "primary_action_and_audit_lock.json"
    if not action_lock_path.is_file() or sha256(action_lock_path) != action_lock_sha256:
        raise RuntimeError("Unlabeled action lock checksum changed")
    action_lock = json.loads(action_lock_path.read_text(encoding="utf-8"))
    if (
        action_lock.get("status")
        != "external_action_and_audit_locked_before_outcomes"
        or action_lock.get("provenance", {}).get("preoutcome_lock_sha256")
        != preoutcome_lock_sha256
    ):
        raise RuntimeError("Action lock does not belong to the pre-outcome design lock")
    filenames = {
        "components": "component_projection_diagnostics.csv",
        "geometry": "unlabeled_cluster_geometry.csv",
        "assignment": "distribution_covered_assignments.csv",
        "mechanism": "mechanism_probabilities_blinded.csv",
    }
    artifact_hashes = action_lock.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(filenames):
        raise RuntimeError("Action lock artifact inventory is incomplete")
    action_root = output_root / ACTION_DIR
    for name, filename in filenames.items():
        path = action_root / filename
        if not path.is_file() or sha256(path) != artifact_hashes[name]:
            raise RuntimeError(f"Locked action artifact checksum changed: {name}")
    if sha256(action_lock_path) != action_lock_sha256:
        raise RuntimeError("Unlabeled action lock changed during verification")
    return action_lock_path, action_lock


def run_locked_prepared_outcomes(
    output_root: Path,
    *,
    preoutcome_lock_sha256: str,
    action_lock_sha256: str,
    n_jobs: int = -1,
    device: str = "cuda",
) -> Path:
    _, config = verify_preoutcome_design_bundle(
        output_root, preoutcome_lock_sha256
    )
    _verify_action_lock(
        output_root,
        preoutcome_lock_sha256=preoutcome_lock_sha256,
        action_lock_sha256=action_lock_sha256,
    )
    model_root = output_root / MODEL_OUTCOME_DIR
    if model_root.exists():
        raise FileExistsError(f"Refusing to overwrite model outcomes: {model_root}")

    prepared_payload = load_prepared_payload(output_root, preoutcome_lock_sha256)
    task_outputs = {}
    for task in config.tasks:
        _verify_action_lock(
            output_root,
            preoutcome_lock_sha256=preoutcome_lock_sha256,
            action_lock_sha256=action_lock_sha256,
        )
        task_outputs[task] = run_prepared_dose_outcomes(
            prepared_payload[task], n_jobs=n_jobs, device=device
        )
    _verify_action_lock(
        output_root,
        preoutcome_lock_sha256=preoutcome_lock_sha256,
        action_lock_sha256=action_lock_sha256,
    )

    predictions = pd.concat(
        [task_outputs[task]["predictions"] for task in config.tasks],
        ignore_index=True,
    )
    split_audit = pd.concat(
        [task_outputs[task]["split_audit"] for task in config.tasks],
        ignore_index=True,
    )
    summary = pd.concat(
        [task_outputs[task]["summary"] for task in config.tasks],
        ignore_index=True,
    )
    encoding = task_outputs[config.tasks[0]]["identity_encoding_margin"]
    for task in config.tasks[1:]:
        _assert_frame_equivalent(
            task_outputs[task]["identity_encoding_margin"],
            encoding,
            keys=["dataset", "representation", "model", "split_seed"],
            name="Post-training cross-task identity encoding",
        )
    mechanism = pd.concat(
        [task_outputs[task]["mechanism_summary"] for task in config.tasks],
        ignore_index=True,
    )
    _assert_frame_equivalent(
        summary[mechanism.columns],
        mechanism,
        keys=[
            "dataset",
            "task",
            "axis",
            "representation",
            "model",
            "split_seed",
            "nominal_dose",
        ],
        name="Post-training mechanism summary",
    )

    model_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "predictions": model_root / "predictions.csv",
        "split_audit": model_root / "split_audit.csv",
        "summary": model_root / "summary.csv",
        "identity_encoding": model_root / "identity_encoding_margin.csv",
    }
    predictions.to_csv(paths["predictions"], index=False)
    split_audit.to_csv(paths["split_audit"], index=False)
    summary.to_csv(paths["summary"], index=False)
    encoding.to_csv(paths["identity_encoding"], index=False)
    _verify_mechanism_unchanged(output_root, paths["summary"], config)
    manifest = {
        "status": "registered_model_outcomes_complete_after_action_lock",
        "date": "2026-09-09",
        "dataset": config.dataset,
        "tasks": list(config.tasks),
        "preoutcome_lock_sha256": preoutcome_lock_sha256,
        "action_lock_sha256": action_lock_sha256,
        "artifact_sha256": {name: sha256(path) for name, path in paths.items()},
        "execution_boundary": (
            "Both immutable locks were verified before every task and again after "
            "all registered model fits completed."
        ),
    }
    manifest_path = model_root / "model_outcome_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths["summary"]


def _verify_mechanism_unchanged(
    output_root: Path,
    full_summary_path: Path,
    config: ExternalApplicationConfig,
) -> None:
    stage = output_root / PREOUTCOME_DIR
    locked = load_unlabeled_configuration_table(
        stage / "mechanism_summary.csv",
        stage / "identity_encoding_margin.csv",
        config,
    )
    observed = load_unlabeled_configuration_table(
        full_summary_path,
        stage / "identity_encoding_margin.csv",
        config,
    )
    keys = [
        "dataset",
        "task",
        "axis",
        "representation",
        "model",
        "split_seed",
        "nominal_dose",
    ]
    locked = locked.set_index(keys).sort_index()
    observed = observed.set_index(keys).sort_index()
    columns = [column for column in locked if column in observed]
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(locked[column])]
    nonnumeric = [column for column in columns if column not in numeric]
    if not locked.index.equals(observed.index):
        raise RuntimeError("Post-outcome configuration keys differ from the design lock")
    if numeric and not np.allclose(
        locked[numeric].to_numpy(float),
        observed[numeric].to_numpy(float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Post-outcome mechanism values differ from the design lock")
    if nonnumeric and not locked[nonnumeric].equals(observed[nonnumeric]):
        raise RuntimeError("Post-outcome mechanism identifiers differ from the design lock")


def finalize_locked_outcomes(
    output_root: Path,
    *,
    preoutcome_lock_sha256: str,
    action_lock_sha256: str,
    full_summary_path: Path,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    _, config = verify_preoutcome_design_bundle(
        output_root, preoutcome_lock_sha256
    )
    action_root = output_root / ACTION_DIR
    action_lock_path, action_lock = _verify_action_lock(
        output_root,
        preoutcome_lock_sha256=preoutcome_lock_sha256,
        action_lock_sha256=action_lock_sha256,
    )
    _verify_mechanism_unchanged(output_root, full_summary_path, config)

    mechanism = pd.read_csv(action_root / "mechanism_probabilities_blinded.csv")
    assignment = pd.read_csv(action_root / "distribution_covered_assignments.csv")
    components = pd.read_csv(action_root / "component_projection_diagnostics.csv")
    _verify_action_lock(
        output_root,
        preoutcome_lock_sha256=preoutcome_lock_sha256,
        action_lock_sha256=action_lock_sha256,
    )
    evaluation = attach_registered_outcomes(
        mechanism,
        full_summary_path,
        config,
        action_lock_path,
        action_lock_sha256,
    )
    result = evaluate_locked_endpoint(
        evaluation,
        assignment,
        components,
        bool(action_lock["candidate_applicable"]),
        freeze,
    )
    _verify_action_lock(
        output_root,
        preoutcome_lock_sha256=preoutcome_lock_sha256,
        action_lock_sha256=action_lock_sha256,
    )
    outcome_root = output_root / OUTCOME_DIR
    if outcome_root.exists():
        raise FileExistsError(f"Refusing to overwrite external outcomes: {outcome_root}")
    outcome_root.mkdir(parents=True, exist_ok=False)
    result["audit"].to_csv(outcome_root / "primary_audit_labeled.csv", index=False)
    result["heldout"].to_csv(outcome_root / "primary_heldout_results.csv", index=False)
    certificate = result["certificate"]
    gate = {key: value for key, value in result.items() if key not in {"audit", "heldout", "certificate"}}
    (outcome_root / "primary_candidate_certificate.json").write_text(
        json.dumps(json_native(certificate), indent=2), encoding="utf-8"
    )
    (outcome_root / "external_confirmation_gate.json").write_text(
        json.dumps(json_native(gate), indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "external_one_shot_pipeline_complete",
        "date": "2026-09-09",
        "dataset": config.dataset,
        "claim_supported": bool(result["claim_supported"]),
        "preoutcome_lock_sha256": preoutcome_lock_sha256,
        "action_lock_sha256": action_lock_sha256,
        "full_summary_sha256": sha256(full_summary_path),
        "gate_sha256": sha256(outcome_root / "external_confirmation_gate.json"),
        "identity_boundary": (
            "Identity is non-intervention and is never counted as effectiveness "
            "or safety success."
        ),
    }
    (outcome_root / "external_confirmation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return gate
