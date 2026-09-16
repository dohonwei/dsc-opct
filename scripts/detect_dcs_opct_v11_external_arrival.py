from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_external_arrival_readiness"
REPORT_JSON = OUT / "report.json"
REPORT_MD = OUT / "report.md"
EXPECTED_FREEZE_SHA256 = (
    "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
)


@dataclass(frozen=True)
class DatasetArrivalSpec:
    dataset: str
    acquisition_path: Path
    acquisition_kind: str
    schema_manifest: Path
    schema_command: tuple[str, ...]
    implementation_lock: Path
    runner: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quote_command(parts: list[str]) -> str:
    if sys.platform == "win32":
        return " ".join(f'"{part}"' if " " in part else part for part in parts)
    return shlex.join(parts)


def acquisition_present(spec: DatasetArrivalSpec) -> bool:
    if spec.acquisition_kind == "directory":
        return spec.acquisition_path.is_dir() and next(
            spec.acquisition_path.iterdir(), None
        ) is not None
    if spec.acquisition_kind == "file":
        return spec.acquisition_path.is_file()
    raise ValueError(f"Unsupported acquisition kind: {spec.acquisition_kind}")


def assess_dataset(spec: DatasetArrivalSpec, python_executable: str) -> dict:
    acquired = acquisition_present(spec)
    schema_ready = spec.schema_manifest.is_file()
    lock_ready = spec.implementation_lock.is_file()
    runner_ready = spec.runner.is_file()

    if not acquired:
        return {
            "dataset": spec.dataset,
            "status": "BLOCKED_EXTERNAL_ACCESS",
            "acquisition_path": str(spec.acquisition_path),
            "participant_values_accessed": False,
            "next_command": None,
            "gpu_one_shot_command": None,
            "interpretation": "The authorized dataset is not present; no analysis can start.",
        }

    if not schema_ready:
        command = [python_executable, *spec.schema_command]
        return {
            "dataset": spec.dataset,
            "status": "ARRIVED_SCHEMA_ONLY_INSPECTION_REQUIRED",
            "acquisition_path": str(spec.acquisition_path),
            "participant_values_accessed": False,
            "next_command": quote_command(command),
            "gpu_one_shot_command": None,
            "interpretation": (
                "Authorized bytes appear present. Run only the registered schema-only "
                "inspection before opening participant values."
            ),
        }

    if not (lock_ready and runner_ready):
        missing = [
            str(path)
            for path, present in (
                (spec.implementation_lock, lock_ready),
                (spec.runner, runner_ready),
            )
            if not present
        ]
        return {
            "dataset": spec.dataset,
            "status": "SCHEMA_RECORDED_IMPLEMENTATION_LOCK_REQUIRED",
            "acquisition_path": str(spec.acquisition_path),
            "participant_values_accessed": False,
            "missing_locked_components": missing,
            "next_command": None,
            "gpu_one_shot_command": None,
            "interpretation": (
                "The schema manifest exists, but the schema-bound runner and implementation "
                "lock are not both available. Do not access participant values."
            ),
        }

    lock_hash = sha256(spec.implementation_lock)
    command = [
        python_executable,
        str(spec.runner),
        "all",
        "--implementation-lock-sha256",
        lock_hash,
        "--device",
        "cuda",
        "--n-jobs",
        "1",
    ]
    rendered = quote_command(command)
    return {
        "dataset": spec.dataset,
        "status": "READY_FOR_OPERATOR_AUTHORIZED_GPU_ONE_SHOT",
        "acquisition_path": str(spec.acquisition_path),
        "participant_values_accessed": False,
        "implementation_lock_sha256": lock_hash,
        "next_command": rendered,
        "gpu_one_shot_command": rendered,
        "interpretation": (
            "All pre-outcome artifacts are present. The command is emitted for explicit "
            "operator execution and is not launched by this detector."
        ),
    }


def build_report(
    amigos_root: Path,
    emognition_archive: Path,
    python_executable: str = sys.executable,
) -> dict:
    freeze = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
    freeze_hash = sha256(freeze)
    if freeze_hash != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("Frozen v11 hash changed; refusing to emit execution commands")

    specs = (
        DatasetArrivalSpec(
            dataset="AMIGOS",
            acquisition_path=amigos_root,
            acquisition_kind="directory",
            schema_manifest=ROOT
            / "outputs/amigos_v11_preaccess/schema_only_acquisition_manifest.json",
            schema_command=(
                "scripts/inspect_amigos_v11_schema.py",
                "--data-root",
                str(amigos_root),
            ),
            implementation_lock=ROOT
            / "docs/amigos_v11_external_confirmation_implementation_lock.json",
            runner=ROOT / "scripts/run_amigos_v11_external_confirmation.py",
        ),
        DatasetArrivalSpec(
            dataset="Emognition",
            acquisition_path=emognition_archive,
            acquisition_kind="file",
            schema_manifest=ROOT
            / "outputs/emognition_v11_preaccess/schema_only_acquisition_manifest.json",
            schema_command=(
                "scripts/inspect_emognition_v11_archive_schema.py",
                "--archive",
                str(emognition_archive),
            ),
            implementation_lock=ROOT
            / "docs/emognition_v11_external_confirmation_implementation_lock.json",
            runner=ROOT / "scripts/run_emognition_v11_external_confirmation.py",
        ),
    )
    datasets = [assess_dataset(spec, python_executable) for spec in specs]
    return {
        "status": (
            "operator_action_available"
            if any(item["next_command"] for item in datasets)
            else "blocked_external_access"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "metadata-only external-dataset arrival detection",
        "freeze_sha256": freeze_hash,
        "automatic_execution": False,
        "participant_values_accessed": False,
        "datasets": datasets,
        "claim_boundary": (
            "This detector checks path and lock-file availability only. It neither opens "
            "participant records nor supplies structural, endpoint, effectiveness, non-harm, "
            "replication, or safety evidence."
        ),
    }


def write_idempotent(report: dict, report_json: Path, report_md: Path) -> None:
    if report_json.is_file():
        previous = json.loads(report_json.read_text(encoding="utf-8"))
        old = {key: value for key, value in previous.items() if key != "generated_at_utc"}
        new = {key: value for key, value in report.items() if key != "generated_at_utc"}
        if old == new:
            report["generated_at_utc"] = previous.get("generated_at_utc")

    lines = [
        "# DCS-OPCT v11 External Arrival Readiness",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "| Dataset | Status | Next command |",
        "|---|---|---|",
    ]
    for item in report["datasets"]:
        command = f"`{item['next_command']}`" if item["next_command"] else "None"
        lines.append(f"| {item['dataset']} | {item['status']} | {command} |")
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect authorized external-data arrival without reading participant values"
    )
    parser.add_argument(
        "--amigos-root",
        type=Path,
        default=Path(r"E:\AA发表论文的数据\dataset\AMIGOS"),
    )
    parser.add_argument(
        "--emognition-archive",
        type=Path,
        default=Path(r"E:\AA发表论文的数据\dataset\Emognition\study_data.zip"),
    )
    parser.add_argument("--output-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--output-md", type=Path, default=REPORT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.amigos_root, args.emognition_archive)
    write_idempotent(report, args.output_json, args.output_md)
    for item in report["datasets"]:
        print(f"{item['dataset']}: {item['status']}")
        if item["next_command"]:
            print(f"  next: {item['next_command']}")


if __name__ == "__main__":
    main()
