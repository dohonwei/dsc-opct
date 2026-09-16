from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v23"
MANIFEST = OUT / "submission_artifact_manifest.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"

TOP_LEVEL_FILES = (
    "README.md",
    "requirements.txt",
    "requirements-experiment-lock.txt",
)

SUBMISSION_FILES = {
    "manuscript.tex": "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
    "manuscript.pdf": "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.pdf",
    "supplementary.tex": "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex",
    "supplementary.pdf": "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.pdf",
    "references.bib": "docs/elsarticle/dcs_opct_v11_bspc_refs.bib",
    "highlights.txt": "docs/elsarticle/dcs_opct_v11_bspc_highlights.txt",
    "cover_letter.md": "docs/elsarticle/dcs_opct_v11_bspc_cover_letter.md",
    "author_confirmation_checklist.md": "docs/elsarticle/dcs_opct_v11_author_confirmation_checklist.md",
}

REPOSITORY_FILES = {
    "docs/dcs_opct_v11_anonymous_repository_readme.md",
    "docs/dcs_opct_v11_public_repository_readme.md",
    "docs/dcs_opct_v11_public_repository_receipt_20260916.json",
    "docs/dcs_opct_v11_claim_evidence_matrix.md",
    "docs/dcs_opct_v11_eppvr_metadata_provenance_20260916.md",
    "docs/dcs_opct_v11_reviewer_task_traceability_20260916.md",
    "docs/dcs_opct_v11_dataset_availability_matrix.md",
    "docs/dcs_opct_v11_closest_method_capability_matrix.md",
    "docs/dcs_opct_v11_manuscript_blueprint.md",
    "docs/elsarticle/dcs_opct_v11_revision_log.md",
    "docs/distribution_covered_stratified_opct_v11_final_freeze.json",
    "docs/external_wearable_v11_standardized_analysis_contract.md",
    "docs/amigos_v11_external_confirmation_reservation.json",
    "docs/amigos_v11_external_confirmation_implementation_protocol.md",
    "docs/emognition_v11_external_confirmation_reservation.json",
    "docs/emognition_v11_external_confirmation_protocol.md",
    "docs/ekmed_v11_external_confirmation_reservation.json",
    "docs/ekmed_v11_external_confirmation_reservation_amendment_001.json",
    "docs/ekmed_v11_external_confirmation_protocol.md",
    "docs/ekmed_v11_external_confirmation_implementation_lock.json",
    "docs/ekmed_v11_preaccess_history_audit.json",
    "docs/ekmed_v11_public_key_moment_receipt_20260916.json",
    "docs/ekmed_v11_zenodo_metadata_receipt_20260916.json",
}

EVIDENCE_DIRECTORIES = (
    "outputs/identity_shortcut_risk_classifier",
    "outputs/distribution_covered_stratified_opct_v11_development",
    "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914",
    "outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914",
    "outputs/dcs_opct_v11_probabilistic_event_risk_20260914",
    "outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914",
    "outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914",
    "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914",
    "outputs/dcs_opct_v11_reviewer_closure_20260914",
    "outputs/dcs_opct_v11_reviewer_task_traceability_20260916",
    "outputs/dcs_opct_v11_reproducibility_inventory",
    "outputs/dcs_opct_v11_release_readiness",
    "outputs/dcs_opct_v11_external_arrival_readiness",
    "outputs/dcs_opct_v11_nature_review_20260914_final",
    "outputs/amigos_v11_preaccess",
    "outputs/emognition_v11_preaccess",
    "outputs/ekmed_v11_preaccess",
    "outputs/ekmed_v11_archive_schema",
    "outputs/ekmed_v11_external_confirmation",
)

EXCLUDED_EVIDENCE_NAMES = {
    "participant_population_assignment.csv",
    "population_predictions.csv",
    "fit_provenance.csv",
}

FORBIDDEN_SUFFIXES = {
    ".mat",
    ".set",
    ".fdt",
    ".edf",
    ".bdf",
    ".vhdr",
    ".eeg",
    ".npy",
    ".npz",
}

AUTHOR_BLOCKERS: tuple[str, ...] = ()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_tree(sources: dict[str, Path], relative: str, destination_root: str) -> None:
    root = ROOT / relative
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.name in EXCLUDED_EVIDENCE_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        destination = Path(destination_root) / path.relative_to(root)
        sources[destination.as_posix()] = path


def collect_sources() -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for relative in TOP_LEVEL_FILES:
        sources[f"repository/{relative}"] = ROOT / relative
    for destination, relative in SUBMISSION_FILES.items():
        sources[f"submission/{destination}"] = ROOT / relative
    for relative in REPOSITORY_FILES:
        sources[f"repository/{relative}"] = ROOT / relative

    add_tree(sources, "scripts", "repository/scripts")
    add_tree(sources, "src", "repository/src")
    add_tree(sources, "configs", "repository/configs")
    add_tree(sources, "docs/elsarticle/figures", "submission/figures")
    add_tree(sources, "docs/elsarticle/tables", "submission/tables")

    for relative in EVIDENCE_DIRECTORIES:
        add_tree(sources, relative, f"repository/{relative}")

    for name in ("elsarticle.cls", "elsarticle-num.bst"):
        sources[f"submission/{name}"] = ROOT / "docs/elsarticle" / name

    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing v23 package sources: {missing}")
    return dict(sorted(sources.items()))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Immutable v23 package already exists: {OUT}")
    if sha256(FREEZE) != EXPECTED_FREEZE_HASH:
        raise RuntimeError("Frozen DCS-OPCT v11 manifest changed")

    sources = collect_sources()
    OUT.mkdir(parents=True)
    hashes: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    progress = tqdm(sources.items(), desc="Build DCS-OPCT v23 package", unit="file", dynamic_ncols=True)
    for destination, source in progress:
        target = OUT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[destination] = sha256(target)
        source_hashes[source.relative_to(ROOT).as_posix()] = sha256(source)
    progress.close()

    manifest = {
        "schema_version": "dcs-opct-v11-bspc-submission-package-v23",
        "status": "built_pending_independent_validation",
        "date": "2026-09-16",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_v11_sha256": EXPECTED_FREEZE_HASH,
        "file_count_excluding_manifest_and_validation_report": len(hashes),
        "packaged_sha256": hashes,
        "source_sha256": source_hashes,
        "restricted_participant_data_included": False,
        "explicitly_excluded_sensitive_or_governance_review_files": sorted(EXCLUDED_EVIDENCE_NAMES),
        "author_dependent_blockers": list(AUTHOR_BLOCKERS),
        "external_scientific_blocker": (
            "The prospectively reserved EKM-ED attempt failed structural endpoint eligibility before "
            "model fitting, and no compatible external dataset has passed all frozen gates and produced "
            "a non-identity release. This package therefore does not claim prospective external "
            "effectiveness, future-domain non-harm, or universal safety."
        ),
        "claim_boundary": (
            "The package supports retrospective selective configuration-level audit-risk calibration "
            "within evaluated development support and verifies fail-closed execution, including a "
            "prospective endpoint-transport failure. It does not "
            "establish positive fully training-data-independent calibration, prospective external "
            "effectiveness, future-domain non-harm, or direct improvement of EEG emotion predictions."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DCS-OPCT v23 package built: {len(hashes)} files")
    print(f"Package: {OUT}")


if __name__ == "__main__":
    main()
