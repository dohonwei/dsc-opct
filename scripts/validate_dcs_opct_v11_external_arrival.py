from __future__ import annotations

import tempfile
from pathlib import Path

from detect_dcs_opct_v11_external_arrival import (
    DatasetArrivalSpec,
    assess_dataset,
)


def main() -> None:
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        acquisition = root / "dataset"
        schema = root / "schema.json"
        lock = root / "lock.json"
        runner = root / "runner.py"
        spec = DatasetArrivalSpec(
            dataset="SYNTHETIC",
            acquisition_path=acquisition,
            acquisition_kind="directory",
            schema_manifest=schema,
            schema_command=("inspect.py", "--data-root", str(acquisition)),
            implementation_lock=lock,
            runner=runner,
        )

        absent = assess_dataset(spec, "python")
        checks.append(("absent_is_blocked", absent["status"] == "BLOCKED_EXTERNAL_ACCESS"))
        checks.append(("absent_has_no_command", absent["next_command"] is None))

        acquisition.mkdir()
        (acquisition / "opaque-authorized-archive.bin").touch()
        arrived = assess_dataset(spec, "python")
        checks.append(
            (
                "arrival_requires_schema_only_step",
                arrived["status"] == "ARRIVED_SCHEMA_ONLY_INSPECTION_REQUIRED"
                and "inspect.py" in arrived["next_command"],
            )
        )
        checks.append(("arrival_does_not_emit_gpu", arrived["gpu_one_shot_command"] is None))

        schema.write_text("{}", encoding="utf-8")
        unlocked = assess_dataset(spec, "python")
        checks.append(
            (
                "schema_without_lock_refuses_run",
                unlocked["status"] == "SCHEMA_RECORDED_IMPLEMENTATION_LOCK_REQUIRED",
            )
        )

        lock.write_text("locked", encoding="utf-8")
        runner.write_text("# locked runner\n", encoding="utf-8")
        ready = assess_dataset(spec, "python")
        command = ready["gpu_one_shot_command"] or ""
        checks.append(
            (
                "ready_emits_cuda_one_shot",
                ready["status"] == "READY_FOR_OPERATOR_AUTHORIZED_GPU_ONE_SHOT"
                and " all " in f" {command} "
                and "--device cuda" in command
                and "--implementation-lock-sha256" in command,
            )
        )
        checks.append(("detector_never_claims_value_access", ready["participant_values_accessed"] is False))

    failed = [name for name, passed in checks if not passed]
    if failed:
        raise RuntimeError(f"External-arrival validation failed: {failed}")
    print(f"External-arrival validation passed: {len(checks)}/{len(checks)}")


if __name__ == "__main__":
    main()
