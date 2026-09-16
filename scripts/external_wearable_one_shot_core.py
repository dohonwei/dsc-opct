from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from external_wearable_dcs_opct_core import ExternalApplicationConfig
from external_wearable_dose_core import DoseRunConfig, prepare_dose_design
from external_wearable_implementation_lock import sha256, verify_implementation_lock
from external_wearable_v11_pipeline import (
    finalize_locked_outcomes,
    lock_unlabeled_action_from_bundle,
    run_locked_prepared_outcomes,
    write_preoutcome_design_bundle,
)


@dataclass(frozen=True)
class RegisteredTaskBundle:
    frame: pd.DataFrame
    labels: np.ndarray
    feature_sets: dict[str, np.ndarray]


TaskValidator = Callable[
    [pd.DataFrame, np.ndarray, dict[str, np.ndarray], str], dict[str, object]
]


def _validate_application_lock_contract(
    application_config: ExternalApplicationConfig,
    implementation_lock: dict[str, Any],
) -> None:
    if implementation_lock.get("dataset") != application_config.dataset:
        raise ValueError("Application dataset differs from the implementation lock")
    registered = implementation_lock.get("registered_execution", {})
    comparisons = {
        "tasks": list(application_config.tasks),
        "representations": list(application_config.representations),
        "models": list(application_config.models),
        "split_seeds": list(application_config.seeds),
        "doses": list(application_config.doses),
    }
    for key, observed in comparisons.items():
        if key in registered and registered[key] != observed:
            raise ValueError(f"Application {key} differ from the implementation lock")
    if "material_threshold" in registered and not np.isclose(
        float(registered["material_threshold"]), application_config.material_threshold
    ):
        raise ValueError("Application material threshold differs from the implementation lock")


def _validate_dose_config(
    task: str,
    dose_config: DoseRunConfig,
    application_config: ExternalApplicationConfig,
) -> None:
    observed = (
        dose_config.dataset,
        dose_config.task,
        dose_config.axis,
        tuple(dose_config.representations),
        tuple(dose_config.models),
        tuple(dose_config.seeds),
        tuple(dose_config.doses),
    )
    expected = (
        application_config.dataset,
        task,
        application_config.axis,
        application_config.representations,
        application_config.models,
        application_config.seeds,
        application_config.doses,
    )
    if observed != expected:
        raise ValueError(f"Dose configuration differs from the application contract for {task}")


def prepare_and_lock_preoutcome(
    output_root: Path,
    *,
    task_bundles: dict[str, RegisteredTaskBundle],
    task_validators: dict[str, TaskValidator],
    dose_configs: dict[str, DoseRunConfig],
    application_config: ExternalApplicationConfig,
    implementation_lock_path: Path,
    implementation_lock_sha256: str,
    project_root: Path,
    provenance: dict[str, Any],
) -> tuple[Path, str]:
    implementation_lock = verify_implementation_lock(
        implementation_lock_path,
        root=project_root,
        expected_sha256=implementation_lock_sha256,
    )
    _validate_application_lock_contract(application_config, implementation_lock)
    registered = set(application_config.tasks)
    for name, mapping in {
        "task bundles": task_bundles,
        "task validators": task_validators,
        "dose configurations": dose_configs,
    }.items():
        if set(mapping) != registered:
            raise ValueError(f"{name} must cover exactly the registered tasks")

    prepared = {}
    adapter_audits = {}
    progress = tqdm(
        application_config.tasks,
        desc=f"{application_config.dataset} pre-outcome task preparation",
        unit="task",
        dynamic_ncols=True,
    )
    for task in progress:
        verify_implementation_lock(
            implementation_lock_path,
            root=project_root,
            expected_sha256=implementation_lock_sha256,
        )
        bundle = task_bundles[task]
        audit = task_validators[task](
            bundle.frame,
            bundle.labels,
            bundle.feature_sets,
            task,
        )
        if (
            audit.get("status") != "adapter_output_contract_passed"
            or audit.get("dataset") != application_config.dataset
            or audit.get("task") != task
        ):
            raise ValueError(f"Adapter validator returned an invalid audit for {task}")
        _validate_dose_config(task, dose_configs[task], application_config)
        adapter_audits[task] = audit
        prepared[task] = prepare_dose_design(
            bundle.frame,
            bundle.labels,
            bundle.feature_sets,
            dose_configs[task],
        )
    progress.close()
    verify_implementation_lock(
        implementation_lock_path,
        root=project_root,
        expected_sha256=implementation_lock_sha256,
    )

    mechanism_summary = pd.concat(
        [prepared[task]["mechanism_summary"] for task in application_config.tasks],
        ignore_index=True,
    )
    split_audit = pd.concat(
        [prepared[task]["split_audit"] for task in application_config.tasks],
        ignore_index=True,
    )
    identity_encoding = prepared[application_config.tasks[0]][
        "identity_encoding_margin"
    ]
    lock_path, lock_sha256 = write_preoutcome_design_bundle(
        output_root,
        mechanism_summary=mechanism_summary,
        identity_encoding=identity_encoding,
        split_audit=split_audit,
        prepared_payload=prepared,
        config=application_config,
        provenance={
            **provenance,
            "implementation_lock_path": implementation_lock_path.resolve().as_posix(),
            "implementation_lock_sha256": implementation_lock_sha256,
            "schema_manifest_path": implementation_lock["schema_manifest_path"],
            "schema_manifest_sha256": implementation_lock["schema_manifest_sha256"],
        },
        adapter_audits=adapter_audits,
    )
    verify_implementation_lock(
        implementation_lock_path,
        root=project_root,
        expected_sha256=implementation_lock_sha256,
    )
    return lock_path, lock_sha256


def lock_registered_action(
    output_root: Path,
    *,
    preoutcome_lock_sha256: str,
    implementation_lock_path: Path,
    implementation_lock_sha256: str,
    project_root: Path,
    risk_root: Path,
    device: str = "cuda",
) -> tuple[Path, str]:
    implementation_lock = verify_implementation_lock(
        implementation_lock_path,
        root=project_root,
        expected_sha256=implementation_lock_sha256,
    )
    freeze_path = project_root / implementation_lock["freeze_path"]
    frozen_inventory = implementation_lock["frozen_artifact_sha256"]
    paths = {
        "risk_manifest": risk_root / "freeze_manifest.json",
        "risk_model": risk_root / "frozen_risk_model.joblib",
        "public_risk_table": risk_root / "risk_learning_table.csv",
    }
    for name, path in paths.items():
        relative = path.resolve().relative_to(project_root.resolve()).as_posix()
        if relative not in frozen_inventory or sha256(path) != frozen_inventory[relative]:
            raise RuntimeError(f"Registered frozen risk artifact is not locked: {name}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    risk_manifest = json.loads(paths["risk_manifest"].read_text(encoding="utf-8"))
    if risk_manifest.get("model_sha256") != sha256(paths["risk_model"]):
        raise RuntimeError("Frozen risk manifest and model checksum differ")
    action_path, action_sha256 = lock_unlabeled_action_from_bundle(
        output_root,
        preoutcome_lock_sha256=preoutcome_lock_sha256,
        public_risk_table=pd.read_csv(paths["public_risk_table"]),
        risk_model=joblib.load(paths["risk_model"]),
        risk_manifest=risk_manifest,
        freeze=freeze,
        freeze_path=freeze_path,
        device=device,
        provenance={
            "implementation_lock_path": implementation_lock_path.resolve().as_posix(),
            "implementation_lock_sha256": implementation_lock_sha256,
            "frozen_risk_artifact_sha256": {
                name: sha256(path) for name, path in paths.items()
            },
        },
    )
    verify_implementation_lock(
        implementation_lock_path,
        root=project_root,
        expected_sha256=implementation_lock_sha256,
    )
    return action_path, action_sha256


def run_registered_outcomes(
    output_root: Path,
    *,
    preoutcome_lock_sha256: str,
    action_lock_sha256: str,
    implementation_lock_path: Path,
    implementation_lock_sha256: str,
    project_root: Path,
    n_jobs: int = -1,
    device: str = "cuda",
) -> Path:
    verify_implementation_lock(
        implementation_lock_path,
        root=project_root,
        expected_sha256=implementation_lock_sha256,
    )
    summary_path = run_locked_prepared_outcomes(
        output_root,
        preoutcome_lock_sha256=preoutcome_lock_sha256,
        action_lock_sha256=action_lock_sha256,
        n_jobs=n_jobs,
        device=device,
    )
    verify_implementation_lock(
        implementation_lock_path,
        root=project_root,
        expected_sha256=implementation_lock_sha256,
    )
    return summary_path


def finalize_registered_outcomes(
    output_root: Path,
    *,
    preoutcome_lock_sha256: str,
    action_lock_sha256: str,
    full_summary_path: Path,
    implementation_lock_path: Path,
    implementation_lock_sha256: str,
    project_root: Path,
) -> dict[str, Any]:
    implementation_lock = verify_implementation_lock(
        implementation_lock_path,
        root=project_root,
        expected_sha256=implementation_lock_sha256,
    )
    freeze_path = project_root / implementation_lock["freeze_path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    gate = finalize_locked_outcomes(
        output_root,
        preoutcome_lock_sha256=preoutcome_lock_sha256,
        action_lock_sha256=action_lock_sha256,
        full_summary_path=full_summary_path,
        freeze=freeze,
    )
    verify_implementation_lock(
        implementation_lock_path,
        root=project_root,
        expected_sha256=implementation_lock_sha256,
    )
    return gate
