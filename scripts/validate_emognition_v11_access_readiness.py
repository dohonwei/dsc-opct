from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from pypdf import PdfReader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
METADATA_URL = (
    "https://dataverse.harvard.edu/api/datasets/:persistentId/"
    "?persistentId=doi:10.7910/DVN/R9WAF4"
)
RECEIPT = ROOT / "docs/emognition_v11_dataverse_metadata_receipt_20260910.json"
EULA = ROOT / "docs/evidence/emognition/Emognition_EULA_Dataverse_v6.0.pdf"
REQUEST_EMAIL = ROOT / "docs/emognition_v11_official_access_request_email.md"
RESERVATION = ROOT / "docs/emognition_v11_external_confirmation_reservation.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
DATA_ROOT = Path(r"E:\AA发表论文的数据\dataset\Emognition")
ARCHIVE = DATA_ROOT / "study_data.zip"
OUT = ROOT / "outputs/emognition_v11_access_readiness/report.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EXPECTED_EULA_MD5 = "e5ee78ae040b2753fd07bd288c6929d8"
EXPECTED_EULA_SHA256 = "cbb89107841e5c1dccaa7c1d7e865c2af5e87874417ce4333b87ccf86fbdc62f"
EXPECTED_ARCHIVE_MD5 = "28422d18399dc5befab8a6c3d3eddb4a"


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_metadata() -> dict:
    request = Request(METADATA_URL, headers={"User-Agent": "DCS-OPCT-v11-metadata-audit/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    metadata = fetch_metadata()
    version = metadata["data"]["latestVersion"]
    files = version["files"]
    by_name = {item["label"].strip(): item for item in files}
    receipt = read_json(RECEIPT)
    reservation = read_json(RESERVATION)
    email = REQUEST_EMAIL.read_text(encoding="utf-8")
    checks: list[dict] = []
    progress = tqdm(total=13, desc="Validate Emognition access readiness", unit="check", dynamic_ncols=True)

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        progress.update(1)

    record("official_api_status", metadata.get("status") == "OK", metadata.get("status"))
    record(
        "official_dataset_identity",
        metadata["data"].get("persistentUrl") == "https://doi.org/10.7910/DVN/R9WAF4"
        and metadata["data"].get("id") == 4302097,
        {"id": metadata["data"].get("id"), "persistentUrl": metadata["data"].get("persistentUrl")},
    )
    record(
        "official_version_6_0",
        version.get("versionNumber") == 6 and version.get("versionMinorNumber") == 0
        and version.get("versionState") == "RELEASED",
        {key: version.get(key) for key in ["versionNumber", "versionMinorNumber", "versionState", "releaseTime"]},
    )
    restricted_count = sum(bool(item.get("restricted")) for item in files)
    record("file_access_counts", len(files) == 1256 and restricted_count == 1255, {"files": len(files), "restricted": restricted_count})

    archive = by_name.get("study_data.zip", {})
    record(
        "registered_archive_metadata",
        archive.get("restricted") is True
        and archive.get("dataFile", {}).get("id") == 5741602
        and archive.get("dataFile", {}).get("filesize") == 1036435463
        and archive.get("dataFile", {}).get("checksum", {}).get("value") == EXPECTED_ARCHIVE_MD5,
        archive.get("dataFile", {}),
    )
    official_eula = by_name.get("!EULA - Emognition Wearable Dataset 2020.pdf", {})
    record(
        "public_eula_metadata",
        official_eula.get("restricted") is False
        and official_eula.get("dataFile", {}).get("id") == 7662096
        and official_eula.get("dataFile", {}).get("checksum", {}).get("value") == EXPECTED_EULA_MD5,
        official_eula.get("dataFile", {}),
    )
    record(
        "local_eula_integrity",
        EULA.is_file() and EULA.stat().st_size == 70363
        and digest(EULA, "md5") == EXPECTED_EULA_MD5
        and digest(EULA) == EXPECTED_EULA_SHA256,
        {"bytes": EULA.stat().st_size, "md5": digest(EULA, "md5"), "sha256": digest(EULA)},
    )
    record("local_eula_pdf_structure", len(PdfReader(EULA).pages) == 2 and EULA.read_bytes()[:4] == b"%PDF", "two-page PDF")
    record(
        "metadata_receipt_matches_live_record",
        receipt.get("version") == "6.0"
        and receipt.get("dataset_file_count") == len(files)
        and receipt.get("restricted_file_count") == restricted_count
        and receipt.get("registered_primary_archive", {}).get("official_md5") == EXPECTED_ARCHIVE_MD5
        and receipt.get("participant_values_accessed") is False,
        receipt.get("access_state"),
    )
    record("frozen_v11_unchanged", digest(FREEZE) == EXPECTED_FREEZE_HASH, digest(FREEZE))
    record(
        "prospective_reservation_unchanged_in_role",
        reservation.get("status") == "reserved_before_participant_value_access"
        and reservation.get("dataset_role", "").startswith("prospectively reserved one-shot"),
        reservation.get("dataset_role"),
    )
    record(
        "restricted_archive_absent",
        not ARCHIVE.exists(),
        {"path": str(ARCHIVE), "present": ARCHIVE.exists(), "data_root_present": DATA_ROOT.exists()},
    )
    required_email_tokens = [
        "emotions@pwr.edu.pl",
        "Dongyi Chen",
        "Daohong Wei",
        "Tian Li",
        "Zhiqi Huang",
        "dychen@uestc.edu.cn",
        "28422d18399dc5befab8a6c3d3eddb4a",
        "simulated or delegated signature",
    ]
    record("request_package_complete_but_unsigned", all(token in email for token in required_email_tokens), required_email_tokens)
    progress.close()

    passed = sum(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Emognition official-access readiness before restricted data acquisition",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "evidence_sha256": {
            str(RECEIPT.relative_to(ROOT)): digest(RECEIPT),
            str(EULA.relative_to(ROOT)): digest(EULA),
            str(REQUEST_EMAIL.relative_to(ROOT)): digest(REQUEST_EMAIL),
            str(RESERVATION.relative_to(ROOT)): digest(RESERVATION),
            str(FREEZE.relative_to(ROOT)): digest(FREEZE),
            str(Path(__file__).relative_to(ROOT)): digest(Path(__file__)),
        },
        "restricted_file_download_attempted": False,
        "participant_values_accessed": False,
        "claim_boundary": "Access readiness and metadata integrity are not external effectiveness, non-harm, safety, or structural-eligibility evidence.",
        "remaining_human_action": "A qualifying faculty signatory must provide the linked Harvard Dataverse username, sign and date the EULA, and send it from the linked academic email account.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file():
        previous = read_json(OUT)
        old = {key: value for key, value in previous.items() if key != "generated_at_utc"}
        new = {key: value for key, value in report.items() if key != "generated_at_utc"}
        if json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True):
            report["generated_at_utc"] = previous.get("generated_at_utc")
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"Emognition access-readiness validation failed: {failed}")
    print(f"Emognition access-readiness validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {OUT}")


if __name__ == "__main__":
    main()
