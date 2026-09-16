from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from ekmed_v11_contract import (
    DEFAULT_ARCHIVE,
    REGISTRATION_ID,
    ROOT,
    STIMULUS_NAMES,
    hash_file,
    verify_preaccess_contract,
)


CENTRAL_MANIFEST = (
    ROOT / "outputs/ekmed_v11_archive_schema/central_directory_manifest.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/ekmed_v11_archive_schema/header_schema_receipt.json"
MAX_HEADER_BYTES = 65_536
CLEAN_SIGNAL_PREFIX = (
    "EmoKey Moments EEG Dataset (EKM-ED)/muse_wearable_data/"
    "preprocessed/clean-signals/0.0078125S/"
)


def normalize_name(name: str) -> str:
    return name.replace("\\", "/")


def select_members(names: list[str]) -> tuple[str, str]:
    normalized = sorted(normalize_name(name) for name in names)
    questionnaire = [
        name
        for name in normalized
        if not name.startswith("__MACOSX/")
        if name.lower().endswith("ficha_evaluacion_participante_sam_refactored.csv")
    ]
    signals = [
        name
        for name in normalized
        if not name.startswith("__MACOSX/")
        and name.startswith(CLEAN_SIGNAL_PREFIX)
        and Path(name).suffix.lower() == ".csv"
        and Path(name).stem.upper() in STIMULUS_NAMES
        and len(Path(name).parts) == len(Path(CLEAN_SIGNAL_PREFIX).parts) + 2
    ]
    if len(questionnaire) != 1:
        raise RuntimeError(
            "Expected exactly one registered SAM questionnaire CSV in the central directory"
        )
    if not signals:
        raise RuntimeError(
            "No registered clean 128-Hz Muse signal CSV exists in the central directory"
        )
    return questionnaire[0], signals[0]


def decode_header(raw: bytes) -> tuple[str, str, list[str]]:
    if not raw or len(raw) > MAX_HEADER_BYTES:
        raise RuntimeError("Header is empty or exceeds the registered byte limit")
    if not raw.endswith((b"\n", b"\r")):
        raise RuntimeError("Header line terminator was not found within the registered limit")
    encoding = "utf-8-sig"
    try:
        text = raw.decode(encoding).rstrip("\r\n")
    except UnicodeDecodeError:
        encoding = "latin-1"
        text = raw.decode(encoding).rstrip("\r\n")
    if not text.strip():
        raise RuntimeError("Decoded header is empty")
    dialect = csv.Sniffer().sniff(text, delimiters=",;\t")
    columns = next(csv.reader([text], dialect))
    normalized = [column.strip() for column in columns]
    if normalized and not normalized[0]:
        normalized[0] = "__index__"
    if len(normalized) < 2 or any(not column for column in normalized):
        raise RuntimeError("Header does not contain an unambiguous tabular schema")
    return encoding, dialect.delimiter, normalized


def read_header_only(archive: zipfile.ZipFile, member: str) -> dict[str, object]:
    with archive.open(member, "r") as handle:
        raw = handle.readline(MAX_HEADER_BYTES + 1)
    encoding, delimiter, columns = decode_header(raw)
    return {
        "member": member,
        "header_bytes": len(raw),
        "header_sha256": hashlib.sha256(raw).hexdigest(),
        "encoding": encoding,
        "delimiter": delimiter,
        "columns": columns,
        "data_rows_read": 0,
    }


def build_header_receipt(
    archive_path: Path,
    central_manifest_path: Path,
) -> dict[str, object]:
    central = json.loads(central_manifest_path.read_text(encoding="utf-8"))
    if central.get("status") != "central_directory_recorded_without_member_reads":
        raise RuntimeError("EKM-ED central-directory manifest status is invalid")
    if central.get("archive_sha256") != hash_file(archive_path):
        raise RuntimeError("EKM-ED archive differs from the central-directory manifest")
    members = central["central_directory"]["members"]
    questionnaire, signal = select_members([row["name"] for row in members])
    with zipfile.ZipFile(archive_path, "r") as archive:
        questionnaire_header = read_header_only(archive, questionnaire)
        signal_header = read_header_only(archive, signal)
    return {
        "status": "header_schema_recorded_without_data_rows",
        "reservation_id": REGISTRATION_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": central["archive_sha256"],
        "central_manifest_sha256": hash_file(central_manifest_path),
        "selection_rule": "lexicographically first registered matching member",
        "maximum_uncompressed_header_bytes_per_member": MAX_HEADER_BYTES,
        "questionnaire": questionnaire_header,
        "signal": signal_header,
        "participant_value_accessed": False,
        "authorized_next_action": (
            "Implement and synthetic-test the adapter against these exact column names, "
            "then write the implementation lock before reading any data row."
        ),
        "claim_boundary": (
            "This receipt records two CSV header lines only. It supplies no participant, "
            "endpoint, effectiveness, non-harm, replication, or safety evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read only two deterministically selected EKM-ED CSV header lines."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--central-manifest", type=Path, default=CENTRAL_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_preaccess_contract(require_archive_absent=False)
    receipt = build_header_receipt(args.archive, args.central_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"EKM-ED header-only schema receipt: {args.output}")


if __name__ == "__main__":
    main()
