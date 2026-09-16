from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v14"
OUT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"

REPLACED = {
    "submission_artifact_manifest.json",
    "independent_validation_report.json",
    "v14_package_builder.py",
    "v14_package_validator.py",
    "amigos_preaccess_stack_report.json",
    "emognition_preaccess_stack_report.json",
    "v11_release_readiness_report.json",
    "v11_reproducibility_inventory.json",
    "v11_reproducibility_validation_report.json",
}

EXTRAS = {
    "v14_base_submission_artifact_manifest.json": "outputs/dcs_opct_v11_submission_artifacts_crossfit_v14/submission_artifact_manifest.json",
    "v14_base_independent_validation_report.json": "outputs/dcs_opct_v11_submission_artifacts_crossfit_v14/independent_validation_report.json",
    "v14_package_superseded.json": "docs/dcs_opct_v14_submission_package_superseded.json",
    "v12_release_readiness_failure_003.json": "docs/dcs_opct_v12_release_readiness_failure_003.json",
    "amigos_preaccess_stack_report.json": "outputs/amigos_v11_preaccess/stack_validation_report.json",
    "emognition_preaccess_stack_report.json": "outputs/emognition_v11_preaccess/stack_validation_report.json",
    "v11_release_readiness_report.json": "outputs/dcs_opct_v11_release_readiness/report.json",
    "v11_reproducibility_inventory.json": "outputs/dcs_opct_v11_reproducibility_inventory/inventory.json",
    "v11_reproducibility_validation_report.json": "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json",
    "v15_package_builder.py": "scripts/make_dcs_opct_v15_submission_artifacts.py",
    "v15_package_validator.py": "scripts/validate_dcs_opct_v15_submission_artifacts.py",
}

VOLATILE_SOURCE_SNAPSHOTS = {
    "outputs/amigos_v11_preaccess/stack_validation_report.json",
    "outputs/emognition_v11_preaccess/stack_validation_report.json",
    "outputs/dcs_opct_v11_release_readiness/report.json",
    "outputs/dcs_opct_v11_reproducibility_inventory/inventory.json",
    "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Immutable v15 package already exists: {OUT}")
    if sha256(FREEZE) != EXPECTED_FREEZE_HASH:
        raise RuntimeError("Frozen v11 manifest changed; refusing to package evidence")

    sources = {
        path.name: path
        for path in sorted(BASE.iterdir())
        if path.is_file() and path.name not in REPLACED
    }
    for output_name, relative in EXTRAS.items():
        if output_name in sources:
            raise RuntimeError(f"Package filename collision: {output_name}")
        sources[output_name] = ROOT / relative
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot build v15 package; missing sources: {missing}")

    OUT.mkdir(parents=True)
    source_hashes = {}
    output_hashes = {}
    progress = tqdm(
        sorted(sources.items()),
        desc="Package DCS-OPCT v15 evidence",
        unit="file",
        dynamic_ncols=True,
    )
    for output_name, source in progress:
        target = OUT / output_name
        shutil.copy2(source, target)
        source_hashes[source.relative_to(ROOT).as_posix()] = sha256(source)
        output_hashes[target.relative_to(ROOT).as_posix()] = sha256(target)
    progress.close()

    gate = json.loads(
        (OUT / "v12_analysis_retrospective_robustness_gate.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = {
        "status": "submission_artifacts_generated_pending_independent_validation",
        "date": "2026-09-09",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "DCS-OPCT v11 with preserved v12 failure evidence and explicit runtime-report snapshot semantics",
        "frozen_v11_sha256": EXPECTED_FREEZE_HASH,
        "v12_gate_status": gate.get("status"),
        "volatile_source_snapshots": sorted(VOLATILE_SOURCE_SNAPSHOTS),
        "claim_boundary": (
            "The seven-domain retrospective robustness gate failed and is retained. "
            "AMIGOS and Emognition evidence establishes pre-access software readiness only. "
            "This package does not establish prospective external effectiveness, universal "
            "safety, non-harm, or transferable operating points."
        ),
        "anti_cycle_rule": (
            "Volatile runtime reports are immutable package snapshots whose current execution "
            "state is checked by the post-package v12 release audit. That audit and the final "
            "v12 inventory are not package members."
        ),
        "restricted_participant_data_included": False,
        "base_package": BASE.relative_to(ROOT).as_posix(),
        "file_count_excluding_manifest_and_validation_report": len(output_hashes),
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
    }
    (OUT / "submission_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"DCS-OPCT v15 submission package built: {len(output_hashes)} evidence files")
    print(f"Package: {OUT}")


if __name__ == "__main__":
    main()
