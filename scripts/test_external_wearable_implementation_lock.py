from __future__ import annotations

import json
from pathlib import Path
import tempfile

from external_wearable_implementation_lock import (
    ImplementationLockSpec,
    build_implementation_lock,
    sha256,
    verify_implementation_lock,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESERVATION = ROOT / "docs/amigos_v11_external_confirmation_reservation.json"


def expect_error(error_type, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__} was not raised")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        docs = root / "docs"
        scripts = root / "scripts"
        outputs = root / "outputs"
        docs.mkdir()
        scripts.mkdir()
        outputs.mkdir()
        freeze = docs / "freeze.json"
        reservation = docs / "reservation.json"
        schema = outputs / "schema.json"
        code = scripts / "adapter.py"
        dependency = scripts / "dependency.py"
        passing_test = scripts / "test_adapter.py"
        frozen_artifact = root / "frozen_model.bin"
        frozen_artifact.write_bytes(b"synthetic frozen model")
        freeze.write_text(
            json.dumps(
                {
                    "locked_artifacts": {
                        "frozen_model.bin": sha256(frozen_artifact),
                    }
                }
            ),
            encoding="utf-8",
        )
        source_reservation = json.loads(SOURCE_RESERVATION.read_text(encoding="utf-8"))
        source_reservation["dataset"] = "SYNTHETIC-WEARABLE"
        source_reservation["dcs_opct_v11_contract"]["freeze_sha256"] = sha256(freeze)
        reservation.write_text(json.dumps(source_reservation), encoding="utf-8")
        schema.write_text(
            json.dumps(
                {
                    "status": "synthetic_schema_only_complete",
                    "reservation_id": source_reservation["registration_id"],
                    "files": [{"path": "synthetic.mat", "shape": [1, 2]}],
                }
            ),
            encoding="utf-8",
        )
        code.write_text("VALUE = 1\n", encoding="utf-8")
        dependency.write_text("DEPENDENCY = True\n", encoding="utf-8")
        passing_test.write_text("print('synthetic adapter test passed')\n", encoding="utf-8")
        lock_path = docs / "implementation_lock.json"
        spec = ImplementationLockSpec(
            dataset="SYNTHETIC-WEARABLE",
            lock_id="synthetic-lock",
            reservation_path=reservation,
            schema_manifest_path=schema,
            freeze_path=freeze,
            output_path=lock_path,
            expected_schema_status="synthetic_schema_only_complete",
            analysis_code=(code, passing_test),
            dependency_code=(dependency,),
            tests=(passing_test,),
            registered_execution={"device": "cuda", "material_threshold": 0.02},
        )
        wrong_dataset_spec = ImplementationLockSpec(
            **{
                **spec.__dict__,
                "dataset": "WRONG-DATASET",
                "output_path": docs / "wrong_dataset_lock.json",
            }
        )
        expect_error(
            RuntimeError,
            build_implementation_lock,
            wrong_dataset_spec,
            root=root,
            operator_attests_no_participant_values_accessed=True,
        )
        expect_error(
            RuntimeError,
            build_implementation_lock,
            spec,
            root=root,
            operator_attests_no_participant_values_accessed=False,
        )
        path, digest = build_implementation_lock(
            spec,
            root=root,
            operator_attests_no_participant_values_accessed=True,
        )
        assert path == lock_path
        assert sha256(path) == digest
        verified = verify_implementation_lock(path, root=root, expected_sha256=digest)
        assert verified["schema_manifest_sha256"] == sha256(schema)
        assert verified["runtime"]["cuda_available"] is True
        assert verified["frozen_artifact_sha256"] == {
            "frozen_model.bin": sha256(frozen_artifact)
        }
        expect_error(
            FileExistsError,
            build_implementation_lock,
            spec,
            root=root,
            operator_attests_no_participant_values_accessed=True,
        )
        original_code = code.read_bytes()
        code.write_bytes(original_code + b"\n")
        expect_error(
            RuntimeError,
            verify_implementation_lock,
            path,
            root=root,
            expected_sha256=digest,
        )
        code.write_bytes(original_code)
        verify_implementation_lock(path, root=root, expected_sha256=digest)

        original_frozen_artifact = frozen_artifact.read_bytes()
        frozen_artifact.write_bytes(original_frozen_artifact + b"changed")
        expect_error(
            RuntimeError,
            verify_implementation_lock,
            path,
            root=root,
            expected_sha256=digest,
        )
        frozen_artifact.write_bytes(original_frozen_artifact)

        leaked_schema = outputs / "leaked_schema.json"
        leaked_schema.write_text(
            json.dumps(
                {
                    "status": "synthetic_schema_only_complete",
                    "reservation_id": source_reservation["registration_id"],
                    "files": [{"path": "synthetic.mat", "payload": {"values": [1, 2]}}],
                }
            ),
            encoding="utf-8",
        )
        leaked_spec = ImplementationLockSpec(
            **{
                **spec.__dict__,
                "schema_manifest_path": leaked_schema,
                "output_path": docs / "leaked_lock.json",
            }
        )
        expect_error(
            RuntimeError,
            build_implementation_lock,
            leaked_spec,
            root=root,
            operator_attests_no_participant_values_accessed=True,
        )

        mutating_test = scripts / "test_mutating.py"
        mutating_test.write_text(
            "from pathlib import Path\n"
            "path = Path(__file__)\n"
            "path.write_text(path.read_text(encoding='utf-8') + '\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )
        mutating_spec = ImplementationLockSpec(
            **{
                **spec.__dict__,
                "output_path": docs / "mutating_lock.json",
                "tests": (mutating_test,),
            }
        )
        expect_error(
            RuntimeError,
            build_implementation_lock,
            mutating_spec,
            root=root,
            operator_attests_no_participant_values_accessed=True,
        )

    print(
        "External wearable implementation-lock test passed: attestation, tests, "
        "non-overwrite, and artifact tamper rejection"
    )


if __name__ == "__main__":
    main()
