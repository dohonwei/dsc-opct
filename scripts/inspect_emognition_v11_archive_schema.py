from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from tqdm.auto import tqdm

from emognition_v11_contract import (
    DEFAULT_ARCHIVE,
    EXPECTED_ARCHIVE_MD5,
    ROOT,
    hash_file,
    verify_preaccess_contract,
)


DEFAULT_OUTPUT = ROOT / "outputs/emognition_v11_preaccess/schema_only_acquisition_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record the authorized Emognition ZIP checksum and central-directory "
            "schema without reading or extracting participant file contents."
        )
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-nonofficial-checksum", action="store_true")
    return parser.parse_args()


def build_manifest(archive: Path, require_official_checksum: bool = True) -> dict:
    reservation, _, _ = verify_preaccess_contract()
    if not archive.is_file():
        raise FileNotFoundError(
            f"Authorized Emognition archive not found: {archive}. Obtain written EULA authorization first."
        )
    archive_md5 = hash_file(archive, "md5")
    if require_official_checksum and archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise RuntimeError(
            "Archive MD5 does not match the official Dataverse v6 metadata; "
            "stop for provenance review before participant-value access."
        )
    entries = []
    suffixes: Counter[str] = Counter()
    top_levels: Counter[str] = Counter()
    with zipfile.ZipFile(archive) as handle:
        infos = sorted(handle.infolist(), key=lambda item: item.filename.lower())
        for info in tqdm(
            infos,
            desc="Emognition ZIP schema-only audit",
            unit="entry",
            dynamic_ncols=True,
        ):
            path = PurePosixPath(info.filename)
            suffixes[path.suffix.lower() or "<none>"] += 1
            if path.parts:
                top_levels[path.parts[0]] += 1
            entries.append(
                {
                    "path": info.filename,
                    "is_directory": info.is_dir(),
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                    "compression_method": info.compress_type,
                }
            )
    return {
        "status": "emognition_schema_only_acquisition_manifest_complete",
        "dataset": "Emognition Wearable Dataset 2020",
        "phase": "pre-participant-value container inspection",
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "archive_md5": archive_md5,
        "archive_sha256": hash_file(archive),
        "official_dataverse_v6_md5_match": archive_md5 == EXPECTED_ARCHIVE_MD5,
        "zip_entry_count": len(entries),
        "suffix_counts": dict(sorted(suffixes.items())),
        "top_level_entry_counts": dict(sorted(top_levels.items())),
        "entries": entries,
        "inspection_method": "ZIP central directory only; no member was opened or extracted",
        "reservation_id": reservation["registration_id"],
        "claim_boundary": (
            "This manifest proves only archive provenance and container structure. "
            "It does not establish participant eligibility, signal quality, endpoint "
            "variation, applicability, effectiveness, non-harm, or safety."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite schema manifest: {args.output}")
    manifest = build_manifest(
        args.archive, require_official_checksum=not args.allow_nonofficial_checksum
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Recorded {manifest['zip_entry_count']} ZIP entries without reading participant files"
    )


if __name__ == "__main__":
    main()
