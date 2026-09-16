from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

from ekmed_v11_contract import (
    DEFAULT_ARCHIVE,
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_MD5,
    REGISTRATION_ID,
    ROOT,
    verify_preaccess_contract,
)


DEFAULT_OUTPUT = ROOT / "outputs/ekmed_v11_archive_schema/central_directory_manifest.json"


def hash_archive(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    total = path.stat().st_size
    with path.open("rb") as handle, tqdm(
        total=total,
        desc="Verify EKM-ED archive",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
    ) as progress:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
            progress.update(len(block))
    return md5.hexdigest(), sha256.hexdigest()


def inspect_central_directory(path: Path) -> dict[str, object]:
    if not zipfile.is_zipfile(path):
        raise RuntimeError("EKM-ED archive is not a readable ZIP container")
    with zipfile.ZipFile(path, "r") as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
    if not infos:
        raise RuntimeError("EKM-ED ZIP central directory contains no files")

    suffix_counts = Counter(Path(info.filename).suffix.lower() or "<none>" for info in infos)
    top_level_counts = Counter(
        Path(info.filename.replace("\\", "/")).parts[0] for info in infos
    )
    members = [
        {
            "name": info.filename.replace("\\", "/"),
            "compressed_bytes": int(info.compress_size),
            "uncompressed_bytes": int(info.file_size),
            "crc32": f"{info.CRC:08x}",
            "compression_type": int(info.compress_type),
        }
        for info in infos
    ]
    return {
        "member_count": len(members),
        "total_uncompressed_bytes": int(sum(info.file_size for info in infos)),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "top_level_counts": dict(sorted(top_level_counts.items())),
        "members": members,
    }


def build_manifest(
    archive_path: Path,
    expected_bytes: int | None,
    expected_md5: str | None,
    compute_hashes: bool,
) -> dict[str, object]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    actual_bytes = archive_path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise RuntimeError(
            f"EKM-ED byte size mismatch: expected {expected_bytes}, found {actual_bytes}"
        )
    actual_md5 = None
    actual_sha256 = None
    if compute_hashes:
        actual_md5, actual_sha256 = hash_archive(archive_path)
        if expected_md5 is not None and actual_md5 != expected_md5.lower():
            raise RuntimeError(
                f"EKM-ED MD5 mismatch: expected {expected_md5}, found {actual_md5}"
            )
    schema = inspect_central_directory(archive_path)
    return {
        "status": "central_directory_recorded_without_member_reads",
        "reservation_id": REGISTRATION_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive_path),
        "archive_bytes": actual_bytes,
        "archive_md5": actual_md5,
        "archive_sha256": actual_sha256,
        "central_directory": schema,
        "participant_member_opened": False,
        "authorized_next_action": (
            "Use member names and sizes only to implement and synthetic-test the schema "
            "adapter, then write the implementation lock before opening any member."
        ),
        "claim_boundary": (
            "This manifest verifies container identity and records ZIP metadata only. "
            "It supplies no participant, endpoint, effectiveness, non-harm, replication, "
            "or safety evidence."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify EKM-ED and record its ZIP central directory without member reads."
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_preaccess_contract(require_archive_absent=False)
    manifest = build_manifest(
        args.archive,
        expected_bytes=EXPECTED_ARCHIVE_BYTES,
        expected_md5=EXPECTED_ARCHIVE_MD5,
        compute_hashes=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"EKM-ED central-directory manifest: {args.output}")
    print(f"Members recorded: {manifest['central_directory']['member_count']}")


if __name__ == "__main__":
    main()
