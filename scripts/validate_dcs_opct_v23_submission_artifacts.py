from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v23"
MANIFEST = OUT / "submission_artifact_manifest.json"
REPORT = OUT / "independent_validation_report.json"
MAIN_TEX = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex"
MAIN_PDF = ROOT / "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.pdf"
SUPP_TEX = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex"
SUPP_PDF = ROOT / "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.pdf"
MAIN_LOG = ROOT / "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.log"
SUPP_LOG = ROOT / "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.log"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EXPECTED_EKMED_LOCK_HASH = "ec729f31627033819af91368995089780c2825dac3cee3f0ae233d73bf380235"
EXPECTED_EKMED_FAILURE_HASH = "8f621780673a87e455594ec46e2bb38ef1538470f355471b46de0f64633cb070"
EXPECTED_AUTHOR_BLOCKERS: set[str] = set()
PUBLIC_REPOSITORY_URL = "https://github.com/dohonwei/dsc-opct"
PUBLIC_RECEIPT = ROOT / "docs/dcs_opct_v11_public_repository_receipt_20260916.json"
EXPECTED_PUBLIC_V22_SHA256 = "442BFB517EA8DD88E0693080780946EF9A321AFC66CE8F756DE4243A6E19AC66"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_latex_log(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    banned = (
        r"Overfull \\[hv]box",
        r"LaTeX Warning: There were undefined references",
        r"LaTeX Warning: Citation .* undefined",
        r"LaTeX Warning: Reference .* undefined",
        r"! LaTeX Error:",
        r"Fatal error occurred",
    )
    return not any(re.search(pattern, text) for pattern in banned)


def pdf_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 10_000 and path.read_bytes()[:4] == b"%PDF"


def main() -> None:
    manifest = read_json(MANIFEST)
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record("schema_version", manifest.get("schema_version") == "dcs-opct-v11-bspc-submission-package-v23", manifest.get("schema_version"))
    record("freeze_hash", manifest.get("frozen_v11_sha256") == EXPECTED_FREEZE_HASH, manifest.get("frozen_v11_sha256"))

    mismatches = {}
    packaged = manifest.get("packaged_sha256", {})
    progress = tqdm(packaged.items(), desc="Validate DCS-OPCT v23 hashes", unit="file", dynamic_ncols=True)
    for relative, expected in progress:
        path = OUT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    progress.close()
    record("all_packaged_hashes_match", not mismatches, mismatches)
    record("declared_file_count", manifest.get("file_count_excluding_manifest_and_validation_report") == len(packaged), len(packaged))

    source_mismatches = {}
    for relative, expected in manifest.get("source_sha256", {}).items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            source_mismatches[relative] = {"expected": expected, "actual": actual}
    record("all_source_hashes_current", not source_mismatches, source_mismatches)

    record("main_pdf_valid_and_fresh", pdf_valid(MAIN_PDF) and MAIN_PDF.stat().st_mtime >= MAIN_TEX.stat().st_mtime, MAIN_PDF.stat().st_mtime)
    record("supp_pdf_valid_and_fresh", pdf_valid(SUPP_PDF) and SUPP_PDF.stat().st_mtime >= SUPP_TEX.stat().st_mtime, SUPP_PDF.stat().st_mtime)
    record("latex_logs_clean", clean_latex_log(MAIN_LOG) and clean_latex_log(SUPP_LOG), [MAIN_LOG.name, SUPP_LOG.name])

    closure = read_json(ROOT / "outputs/dcs_opct_v11_reviewer_closure_20260914/validation_report.json")
    partition = read_json(ROOT / "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/independent_validation_report.json")
    inventory = read_json(ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json")
    traceability = read_json(ROOT / "outputs/dcs_opct_v11_reviewer_task_traceability_20260916/independent_validation_report.json")
    record("reviewer_closure_14_of_14", closure.get("status") == "passed" and closure.get("checks_passed") == closure.get("checks_total") == 14, [closure.get("checks_passed"), closure.get("checks_total")])
    record("raw_partition_first_21_of_21", partition.get("status") == "passed" and partition.get("checks_passed") == partition.get("checks_total") == 21, [partition.get("checks_passed"), partition.get("checks_total")])
    record("reproducibility_inventory_23_of_23", inventory.get("status") == "passed" and inventory.get("checks_passed") == inventory.get("checks_total") == 23, [inventory.get("checks_passed"), inventory.get("checks_total")])
    record("reviewer_task_traceability_16_of_16", traceability.get("status") == "passed" and traceability.get("checks_passed") == traceability.get("checks_total") == 16, [traceability.get("checks_passed"), traceability.get("checks_total")])

    ekmed = read_json(ROOT / "outputs/ekmed_v11_external_confirmation/independent_validation_report.json")
    ekmed_lock = ROOT / "docs/ekmed_v11_external_confirmation_implementation_lock.json"
    ekmed_failure = ROOT / "outputs/ekmed_v11_external_confirmation/failure_001_structural_ineligibility.json"
    record("ekmed_confirmatory_failure_19_of_19", ekmed.get("status") == "passed" and ekmed.get("passed") == ekmed.get("total") == 19, [ekmed.get("passed"), ekmed.get("total")])
    record("ekmed_implementation_lock_immutable", sha256(ekmed_lock) == EXPECTED_EKMED_LOCK_HASH, sha256(ekmed_lock))
    record("ekmed_failure_record_immutable", sha256(ekmed_failure) == EXPECTED_EKMED_FAILURE_HASH, sha256(ekmed_failure))

    manuscript = MAIN_TEX.read_text(encoding="utf-8")
    cover = (ROOT / "docs/elsarticle/dcs_opct_v11_bspc_cover_letter.md").read_text(encoding="utf-8")
    highlights = (ROOT / "docs/elsarticle/dcs_opct_v11_bspc_highlights.txt").read_text(encoding="utf-8")
    combined = (manuscript + "\n" + cover + "\n" + highlights).lower()
    prohibited = [
        phrase
        for phrase in (
            "universally safe transfer",
            "externally validated safety",
            "proven safe cross-domain transfer",
            "independent unlabeled transformations",
        )
        if phrase in combined
    ]
    record("claim_overstatement_guard", not prohibited, prohibited)
    required_boundaries = (
        "no positive fully training-data-independent release",
        "does not establish prospective external effectiveness",
        "does not establish universal",
        "fail-closed",
    )
    record("claim_boundaries_present", all(term in combined for term in required_boundaries), required_boundaries)
    record("highlights_match_manuscript", all(line.strip() in manuscript for line in highlights.splitlines() if line.strip()), highlights.splitlines())
    record("funding_statement_verified", "This work was supported by the National Natural Science Foundation of China (No. 62172081)." in manuscript, "NSFC 62172081")
    record("ethical_committee_wording_verified", "Ethical Committee of the University of Electronic Science and Technology of China" in manuscript, "Ethical Committee wording")
    record("funding_todo_removed", "add confirmed Funding statement" not in manuscript, "no Funding TODO")
    record(
        "public_repository_link_verified",
        PUBLIC_REPOSITORY_URL in manuscript
        and "prepared for deposition" not in manuscript
        and "TODO(author): add anonymized repository URL" not in manuscript,
        PUBLIC_REPOSITORY_URL,
    )
    receipt = read_json(PUBLIC_RECEIPT)
    record(
        "public_repository_receipt_verified",
        receipt.get("status") == "verified_public_repository"
        and receipt.get("repository_url") == PUBLIC_REPOSITORY_URL
        and receipt.get("visibility") == "public"
        and receipt.get("package_sha256") == EXPECTED_PUBLIC_V22_SHA256,
        receipt,
    )

    excluded = set(manifest.get("explicitly_excluded_sensitive_or_governance_review_files", []))
    record("sensitive_outputs_excluded", {"participant_population_assignment.csv", "population_predictions.csv", "fit_provenance.csv"}.issubset(excluded), sorted(excluded))
    forbidden_suffixes = {".mat", ".set", ".fdt", ".edf", ".bdf", ".vhdr", ".eeg", ".npy", ".npz"}
    forbidden_files = [relative for relative in packaged if Path(relative).suffix.lower() in forbidden_suffixes]
    record("no_raw_signal_files_packaged", not forbidden_files, forbidden_files)
    record("restricted_data_flag_false", manifest.get("restricted_participant_data_included") is False, manifest.get("restricted_participant_data_included"))
    blockers = set(manifest.get("author_dependent_blockers", []))
    record("author_blockers_exact", blockers == EXPECTED_AUTHOR_BLOCKERS, sorted(blockers))

    required_files = {
        "submission/manuscript.pdf",
        "submission/supplementary.pdf",
        "submission/highlights.txt",
        "submission/cover_letter.md",
        "submission/author_confirmation_checklist.md",
        "repository/docs/dcs_opct_v11_anonymous_repository_readme.md",
        "repository/docs/dcs_opct_v11_public_repository_readme.md",
        "repository/docs/dcs_opct_v11_public_repository_receipt_20260916.json",
        "repository/docs/dcs_opct_v11_reviewer_task_traceability_20260916.md",
        "repository/docs/dcs_opct_v11_eppvr_metadata_provenance_20260916.md",
        "repository/outputs/dcs_opct_v11_reviewer_task_traceability_20260916/reviewer_task_traceability.json",
        "repository/outputs/dcs_opct_v11_reviewer_task_traceability_20260916/independent_validation_report.json",
        "repository/docs/ekmed_v11_external_confirmation_reservation.json",
        "repository/docs/ekmed_v11_external_confirmation_reservation_amendment_001.json",
        "repository/docs/ekmed_v11_external_confirmation_implementation_lock.json",
        "repository/outputs/ekmed_v11_archive_schema/central_directory_manifest.json",
        "repository/outputs/ekmed_v11_archive_schema/header_schema_receipt.json",
        "repository/outputs/ekmed_v11_external_confirmation/failure_001_structural_ineligibility.json",
        "repository/outputs/ekmed_v11_external_confirmation/independent_validation_report.json",
    }
    record("submission_attachments_complete", required_files.issubset(packaged), sorted(required_files - set(packaged)))

    passed = sum(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(MANIFEST),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "author_dependent_blockers": sorted(blockers),
        "external_scientific_blocker": manifest.get("external_scientific_blocker"),
        "claim_boundary": manifest.get("claim_boundary"),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"DCS-OPCT v23 package validation failed: {failed}")
    print(f"DCS-OPCT v23 package validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
