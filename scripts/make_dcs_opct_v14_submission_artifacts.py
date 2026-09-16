from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v13"
OUT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v14"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"

REPLACED_BASE_FILES = {
    "submission_artifact_manifest.json",
    "independent_validation_report.json",
    "manuscript.tex",
    "manuscript.pdf",
    "supplementary.tex",
    "supplementary.pdf",
    "v12_revision_log.md",
}

EXTRA_SOURCES = {
    "v13_base_submission_artifact_manifest.json": "outputs/dcs_opct_v11_submission_artifacts_crossfit_v13/submission_artifact_manifest.json",
    "v13_base_independent_validation_report.json": "outputs/dcs_opct_v11_submission_artifacts_crossfit_v13/independent_validation_report.json",
    "manuscript.tex": "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
    "manuscript.pdf": "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.pdf",
    "supplementary.tex": "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex",
    "supplementary.pdf": "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.pdf",
    "v14_revision_log.md": "docs/elsarticle/dcs_opct_v11_revision_log.md",
    "v11_release_readiness_report.json": "outputs/dcs_opct_v11_release_readiness/report.json",
    "v11_reproducibility_inventory.json": "outputs/dcs_opct_v11_reproducibility_inventory/inventory.json",
    "v11_reproducibility_validation_report.json": "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json",
    "external_access_v5_manifest.json": "outputs/dcs_opct_v11_external_access_evidence_v5/manifest.json",
    "external_access_v5_validation_report.json": "outputs/dcs_opct_v11_external_access_evidence_v5/independent_validation_report.json",
    "amigos_preaccess_stack_report.json": "outputs/amigos_v11_preaccess/stack_validation_report.json",
    "emognition_preaccess_stack_report.json": "outputs/emognition_v11_preaccess/stack_validation_report.json",
    "dataset_availability_matrix.md": "docs/dcs_opct_v11_dataset_availability_matrix.md",
    "claim_evidence_matrix.md": "docs/dcs_opct_v11_claim_evidence_matrix.md",
    "external_access_v4_superseded.json": "docs/dcs_opct_v11_external_access_evidence_v4_superseded.json",
    "v14_packaging_failure_001.json": "docs/dcs_opct_v14_packaging_failure_001.json",
    "v14_package_builder.py": "scripts/make_dcs_opct_v14_submission_artifacts.py",
    "v14_package_validator.py": "scripts/validate_dcs_opct_v14_submission_artifacts.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Immutable v14 package already exists: {OUT}")
    if sha256(FREEZE) != EXPECTED_FREEZE_HASH:
        raise RuntimeError("Frozen v11 manifest changed; refusing to package evidence")

    sources: dict[str, Path] = {}
    for path in sorted(BASE.iterdir()):
        if path.is_file() and path.name not in REPLACED_BASE_FILES:
            sources[path.name] = path
    for output_name, relative in EXTRA_SOURCES.items():
        if output_name in sources:
            raise RuntimeError(f"Package filename collision: {output_name}")
        sources[output_name] = ROOT / relative

    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot build v14 package; missing sources: {missing}")

    OUT.mkdir(parents=True)
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    progress = tqdm(
        sorted(sources.items()),
        desc="Package DCS-OPCT v14 evidence",
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
        "method": "DCS-OPCT v11 with preserved v12 retrospective robustness and v5 prospective-access readiness evidence",
        "frozen_v11_sha256": EXPECTED_FREEZE_HASH,
        "v12_gate_status": gate.get("status"),
        "claim_boundary": (
            "The seven-domain retrospective robustness gate failed and is retained. "
            "AMIGOS and Emognition evidence establishes pre-access software readiness only. "
            "This package does not establish prospective external effectiveness, universal "
            "safety, non-harm, or transferable operating points."
        ),
        "anti_cycle_rule": (
            "The v12 top-level release report and v12 reproducibility inventory are generated "
            "after this immutable package and are not package members."
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
    print(f"DCS-OPCT v14 submission package built: {len(output_hashes)} evidence files")
    print(f"Package: {OUT}")


if __name__ == "__main__":
    main()
