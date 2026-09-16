from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ImplementationLockSpec:
    dataset: str
    lock_id: str
    reservation_path: Path
    schema_manifest_path: Path
    freeze_path: Path
    output_path: Path
    expected_schema_status: str
    analysis_code: tuple[Path, ...]
    dependency_code: tuple[Path, ...]
    tests: tuple[Path, ...]
    registered_execution: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Lock input must remain under the project root: {path}") from error


def hash_inventory(paths: tuple[Path, ...], root: Path) -> dict[str, str]:
    inventory = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required lock input is missing: {path}")
        inventory[relative_name(path, root)] = sha256(path)
    return inventory


def verify_frozen_inventory(freeze: dict[str, Any], root: Path) -> dict[str, str]:
    locked = freeze.get("locked_artifacts")
    if not isinstance(locked, dict) or not locked:
        raise RuntimeError("Frozen v11 manifest has no locked artifact inventory")
    verified = {}
    for relative, expected in locked.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Frozen v11 artifact changed or is missing: {relative}")
        verified[str(relative).replace("\\", "/")] = str(expected)
    return verified


def runtime_versions() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "tqdm", "joblib"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run_lock_tests(paths: tuple[Path, ...], root: Path) -> list[dict[str, Any]]:
    results = []
    progress = tqdm(paths, desc="Implementation-lock tests", unit="test", dynamic_ncols=True)
    for path in progress:
        if not path.is_file():
            raise FileNotFoundError(f"Required pre-lock test is missing: {path}")
        test_sha256 = sha256(path)
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        result = {
            "path": relative_name(path, root),
            "sha256": test_sha256,
            "passed": completed.returncode == 0,
            "stdout": completed.stdout.strip()[-2000:],
            "stderr_tail": completed.stderr.strip()[-2000:],
        }
        results.append(result)
        if sha256(path) != test_sha256:
            raise RuntimeError(f"Pre-lock test changed while it was executing: {path}")
        if not result["passed"]:
            raise RuntimeError(f"Pre-lock test failed: {path}")
    progress.close()
    return results


def validate_schema_manifest(
    manifest: dict[str, Any], reservation: dict[str, Any], spec: ImplementationLockSpec
) -> None:
    if manifest.get("status") != spec.expected_schema_status:
        raise RuntimeError("Schema-only acquisition manifest has an invalid status")
    if manifest.get("reservation_id") != reservation.get("registration_id"):
        raise RuntimeError("Schema manifest does not belong to the registered reservation")
    forbidden = {
        "data",
        "values",
        "samples",
        "ratings",
        "labels",
        "label_values",
        "eeg_values",
        "participant_values",
        "features",
        "outcomes",
    }

    def nested_keys(value: Any):
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key).lower()
                yield from nested_keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from nested_keys(item)

    leaked = forbidden & set(nested_keys(manifest))
    if leaked:
        raise RuntimeError(
            f"Schema manifest contains participant-value fields: {sorted(leaked)}"
        )


def build_implementation_lock(
    spec: ImplementationLockSpec,
    *,
    root: Path,
    operator_attests_no_participant_values_accessed: bool,
) -> tuple[Path, str]:
    if spec.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite implementation lock: {spec.output_path}")
    if not operator_attests_no_participant_values_accessed:
        raise RuntimeError(
            "Implementation locking requires an explicit attestation that participant "
            "values have not been opened, parsed, extracted, or inspected"
        )
    for path in (spec.reservation_path, spec.schema_manifest_path, spec.freeze_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    reservation = read_json(spec.reservation_path)
    manifest = read_json(spec.schema_manifest_path)
    freeze = read_json(spec.freeze_path)
    if reservation.get("status") != "reserved_before_participant_value_access":
        raise RuntimeError("External reservation status is invalid")
    if reservation.get("dataset") != spec.dataset:
        raise RuntimeError("Implementation-lock dataset differs from the reservation")
    validate_schema_manifest(manifest, reservation, spec)
    expected_freeze = reservation.get("dcs_opct_v11_contract", {}).get("freeze_sha256")
    if not expected_freeze or sha256(spec.freeze_path) != expected_freeze:
        raise RuntimeError("Implementation lock does not match the reserved v11 freeze")
    frozen_artifacts = verify_frozen_inventory(freeze, root)

    analysis_hashes = hash_inventory(spec.analysis_code, root)
    dependency_hashes = hash_inventory(spec.dependency_code, root)
    test_results = run_lock_tests(spec.tests, root)
    runtime = runtime_versions()
    if not runtime["cuda_available"]:
        raise RuntimeError("CUDA must be available when the external implementation is locked")

    payload = {
        "status": "implementation_locked_before_participant_value_access",
        "lock_id": spec.lock_id,
        "locked_at_local": date.today().isoformat(),
        "dataset": spec.dataset,
        "reservation_path": relative_name(spec.reservation_path, root),
        "reservation_sha256": sha256(spec.reservation_path),
        "schema_manifest_path": relative_name(spec.schema_manifest_path, root),
        "schema_manifest_sha256": sha256(spec.schema_manifest_path),
        "schema_manifest_status": manifest["status"],
        "freeze_path": relative_name(spec.freeze_path, root),
        "freeze_sha256": sha256(spec.freeze_path),
        "frozen_artifact_sha256": frozen_artifacts,
        "analysis_code_sha256": analysis_hashes,
        "dependency_code_sha256": dependency_hashes,
        "pre_lock_tests": test_results,
        "runtime": runtime,
        "registered_execution": spec.registered_execution,
        "participant_value_access_attestation": {
            "operator_attested": True,
            "scope": (
                "No participant file content, label value, signal sample, derived feature, "
                "risk probability, material event, or model outcome was opened or inspected "
                "before this lock."
            ),
        },
        "change_policy": (
            "After this lock, no algorithm, feature, label, quality rule, threshold, split, "
            "seed, dose, candidate, audit assignment, model setting, or analysis-code change "
            "is permitted before the one-shot gate. Execution defects require a numbered "
            "failure record and a non-overwriting amendment."
        ),
        "claim_boundary": (
            "This lock establishes schema-bound implementation provenance only. It supplies "
            "no structural eligibility, endpoint, effectiveness, non-harm, safety, or "
            "replication evidence."
        ),
    }
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    spec.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    verify_implementation_lock(spec.output_path, root=root)
    return spec.output_path, sha256(spec.output_path)


def verify_implementation_lock(
    lock_path: Path, *, root: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    if expected_sha256 is not None and sha256(lock_path) != expected_sha256:
        raise RuntimeError("Implementation-lock checksum changed")
    lock = read_json(lock_path)
    if lock.get("status") != "implementation_locked_before_participant_value_access":
        raise RuntimeError("Implementation lock has an invalid status")
    single_files = {
        lock["reservation_path"]: lock["reservation_sha256"],
        lock["schema_manifest_path"]: lock["schema_manifest_sha256"],
        lock["freeze_path"]: lock["freeze_sha256"],
    }
    inventories = (
        single_files,
        lock.get("analysis_code_sha256", {}),
        lock.get("dependency_code_sha256", {}),
        lock.get("frozen_artifact_sha256", {}),
        {
            item["path"]: item["sha256"]
            for item in lock.get("pre_lock_tests", [])
            if item.get("passed") is True and item.get("path") and item.get("sha256")
        },
    )
    for inventory in inventories:
        if not inventory:
            raise RuntimeError("Implementation lock contains an empty artifact inventory")
        for relative, expected in inventory.items():
            path = root / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"Implementation-lock artifact changed: {relative}")
    if not lock.get("participant_value_access_attestation", {}).get("operator_attested"):
        raise RuntimeError("Implementation lock lacks the required access attestation")
    if expected_sha256 is not None and sha256(lock_path) != expected_sha256:
        raise RuntimeError("Implementation lock changed during verification")
    return lock
