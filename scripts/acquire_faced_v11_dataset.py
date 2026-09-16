from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time

import requests
from tqdm.auto import tqdm

from faced_v11_contract import (
    ARCHIVED_MANIFEST,
    DATASET_ID,
    DATASET_VERSION,
    DEFAULT_DATA_ROOT,
    IMPLEMENTATION_LOCK,
    MANIFEST_URL,
    sha256,
    verify_implementation_lock,
)


CHUNK_BYTES = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire the complete versioned FACED NEMAR release with resume and checksums."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=ARCHIVED_MANIFEST)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def manifest_core(items: list[dict[str, object]]) -> list[tuple[object, ...]]:
    return sorted(
        (
            item["path"],
            int(item["size"]),
            item["checksum"],
            item["bytes_url"],
        )
        for item in items
    )


def checksum(path: Path, expected: str, expected_size: int, progress=None) -> str:
    if len(expected) == 64:
        digest = hashlib.sha256()
    elif len(expected) == 40:
        digest = hashlib.sha1()
        digest.update(f"blob {expected_size}\0".encode("ascii"))
    else:
        raise ValueError(f"Unsupported checksum length: {len(expected)}")
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
            if progress is not None:
                progress.update(len(block))
    return digest.hexdigest()


def verified_existing(path: Path, item: dict[str, object]) -> bool:
    size = int(item["size"])
    return (
        path.is_file()
        and path.stat().st_size == size
        and checksum(path, str(item["checksum"]), size) == item["checksum"]
    )


def download_one(
    session: requests.Session,
    item: dict[str, object],
    target: Path,
    overall: tqdm,
    retries: int,
    timeout: int,
) -> None:
    expected_size = int(item["size"])
    if verified_existing(target, item):
        overall.update(expected_size)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError(
            f"Oversized partial file requires manual inspection: {partial}"
        )
    offset = partial.stat().st_size if partial.exists() else 0
    overall.update(offset)
    with tqdm(
        total=expected_size,
        initial=offset,
        desc=target.name,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
        dynamic_ncols=True,
    ) as file_progress:
        for attempt in range(1, retries + 1):
            try:
                headers = {"Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                response = session.get(
                    str(item["bytes_url"]),
                    headers=headers,
                    stream=True,
                    timeout=(30, timeout),
                    allow_redirects=True,
                )
                response.raise_for_status()
                if offset and response.status_code != 206:
                    overall.update(-offset)
                    file_progress.update(-offset)
                    offset = 0
                    partial.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    for block in response.iter_content(chunk_size=CHUNK_BYTES):
                        if not block:
                            continue
                        handle.write(block)
                        offset += len(block)
                        overall.update(len(block))
                        file_progress.update(len(block))
                if offset != expected_size:
                    raise IOError(
                        f"Incomplete response for {item['path']}: {offset}/{expected_size} bytes"
                    )
                break
            except (requests.RequestException, OSError) as error:
                if attempt == retries:
                    raise RuntimeError(
                        f"Download failed after {retries} attempts: {item['path']}"
                    ) from error
                time.sleep(min(30, 2**attempt))
                offset = partial.stat().st_size if partial.exists() else 0
        observed = checksum(partial, str(item["checksum"]), expected_size)
        if observed != item["checksum"]:
            raise RuntimeError(
                f"Checksum mismatch for {item['path']}: expected {item['checksum']}, got {observed}"
            )
        partial.replace(target)


def main() -> None:
    args = parse_args()
    local_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(local_manifest, list) or len(local_manifest) != 745:
        raise ValueError("Archived FACED manifest must contain exactly 745 files")
    total_size = sum(int(item["size"]) for item in local_manifest)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "status": "plan_only_no_participant_values_accessed",
                    "dataset_id": DATASET_ID,
                    "version": DATASET_VERSION,
                    "files": len(local_manifest),
                    "bytes": total_size,
                    "gib": total_size / 1024**3,
                    "destination": str(args.data_root),
                    "archived_manifest_sha256": sha256(args.manifest),
                },
                indent=2,
            )
        )
        return

    lock = verify_implementation_lock()
    remote = requests.get(MANIFEST_URL, timeout=(30, args.timeout))
    remote.raise_for_status()
    remote_manifest = remote.json()
    if manifest_core(remote_manifest) != manifest_core(local_manifest):
        raise RuntimeError(
            "Versioned remote manifest no longer matches the archived pre-access manifest"
        )

    existing_bytes = 0
    for item in local_manifest:
        target = args.data_root / PurePosixPath(str(item["path"])).as_posix()
        if target.is_file() and target.stat().st_size == int(item["size"]):
            existing_bytes += int(item["size"])
        else:
            partial = target.with_name(target.name + ".part")
            if partial.is_file():
                existing_bytes += min(partial.stat().st_size, int(item["size"]))
    remaining = total_size - existing_bytes
    free = shutil.disk_usage(args.data_root.parent).free
    if free < remaining + max(5 * 1024**3, int(remaining * 0.1)):
        raise RuntimeError(
            f"Insufficient free space: need {remaining / 1024**3:.2f} GiB plus reserve, "
            f"have {free / 1024**3:.2f} GiB"
        )

    args.data_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "FACED-DCS-OPCT-v11-reproducible-acquisition/1.0"
    with tqdm(
        total=total_size,
        desc="FACED complete release",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    ) as overall:
        for item in local_manifest:
            target = args.data_root / PurePosixPath(str(item["path"])).as_posix()
            download_one(session, item, target, overall, args.retries, args.timeout)

    receipt_rows = []
    for item in tqdm(
        local_manifest,
        desc="FACED final checksum audit",
        unit="file",
        dynamic_ncols=True,
    ):
        target = args.data_root / PurePosixPath(str(item["path"])).as_posix()
        observed = checksum(target, str(item["checksum"]), int(item["size"]))
        if observed != item["checksum"]:
            raise RuntimeError(f"Final checksum audit failed: {item['path']}")
        receipt_rows.append(
            {"path": item["path"], "size": item["size"], "checksum": observed}
        )
    receipt = {
        "status": "complete_release_downloaded_and_verified",
        "dataset_id": DATASET_ID,
        "version": DATASET_VERSION,
        "file_count": len(receipt_rows),
        "total_bytes": total_size,
        "archived_manifest_sha256": sha256(args.manifest),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "implementation_lock_status": lock["status"],
        "files": receipt_rows,
    }
    (args.data_root / "faced_v11_acquisition_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "files"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
