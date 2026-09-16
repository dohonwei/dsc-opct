from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from tqdm.auto import tqdm

from eegemotions27_v11_contract import LOCKED_COMMIT, REGISTERED_EMOTION_IDS, ROOT


RAW_NAME = re.compile(r"^(?P<participant>\d+)_(?P<emotion>\d+)\.0\.txt$")
DEFAULT_DATA_ROOT = Path(r"E:\AA发表论文的数据\dataset\EEGEmotions-27")
DEFAULT_RECEIPT = (
    ROOT / "outputs/eegemotions27_v11_presignal/official_eeg_raw_tree.json"
)
DEFAULT_REPORT = (
    ROOT
    / "outputs/eegemotions27_v11_presignal/exact_blob_materialization_report.json"
)
RECORD_DATE = "2026-09-09"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the registered EEGEmotions-27 files from the local Git "
            "object database without checkout line-ending filters"
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tree-receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def git_output(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_entries(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for entry in receipt.get("tree", []):
        match = RAW_NAME.fullmatch(str(entry.get("path", "")))
        if (
            entry.get("type") == "blob"
            and match
            and int(match.group("emotion")) in REGISTERED_EMOTION_IDS
        ):
            selected.append(entry)
    return sorted(
        selected,
        key=lambda item: (
            int(RAW_NAME.fullmatch(item["path"]).group("participant")),
            int(RAW_NAME.fullmatch(item["path"]).group("emotion")),
        ),
    )


def materialize_blob(repository: Path, entry: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".git-blob", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            completed = subprocess.run(
                ["git", "-C", str(repository), "cat-file", "blob", entry["sha"]],
                check=False,
                stdout=handle,
                stderr=subprocess.PIPE,
            )
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git cat-file failed for {entry['path']}: {error}")
        actual_size = temporary.stat().st_size
        if actual_size != int(entry["size"]):
            raise RuntimeError(
                f"Blob size mismatch for {entry['path']}: {actual_size} != {entry['size']}"
            )
        actual_sha = git_blob_sha1(temporary, actual_size)
        if actual_sha != entry["sha"]:
            raise RuntimeError(
                f"Blob SHA-1 mismatch for {entry['path']}: {actual_sha} != {entry['sha']}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    repository = args.data_root.resolve()
    if git_output(repository, "rev-parse", "HEAD") != LOCKED_COMMIT:
        raise RuntimeError("EEGEmotions-27 repository differs from the locked commit")
    receipt = json.loads(args.tree_receipt.read_text(encoding="utf-8"))
    if receipt.get("truncated") is not False:
        raise RuntimeError("Tree receipt is truncated")
    entries = selected_entries(receipt)
    if not entries:
        raise RuntimeError("No registered EEGEmotions-27 blobs found in tree receipt")

    missing_objects = [
        entry["sha"]
        for entry in entries
        if subprocess.run(
            ["git", "-C", str(repository), "cat-file", "-e", f"{entry['sha']}^{{blob}}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        != 0
    ]
    if missing_objects:
        raise RuntimeError(
            f"Local object database is missing {len(missing_objects)} registered blobs"
        )

    rewritten = 0
    already_exact = 0
    total_bytes = 0
    progress = tqdm(entries, desc="Materializing exact registered Git blobs", unit="trial")
    for entry in progress:
        destination = repository / "eeg_raw" / entry["path"]
        expected_size = int(entry["size"])
        total_bytes += expected_size
        if (
            destination.is_file()
            and destination.stat().st_size == expected_size
            and git_blob_sha1(destination, expected_size) == entry["sha"]
        ):
            already_exact += 1
            continue
        materialize_blob(repository, entry, destination)
        rewritten += 1
        progress.set_postfix(rewritten=rewritten, exact=already_exact)
    progress.close()

    failures = []
    verify = tqdm(entries, desc="Verifying exact registered Git blobs", unit="trial")
    for entry in verify:
        destination = repository / "eeg_raw" / entry["path"]
        size = destination.stat().st_size if destination.is_file() else -1
        actual_sha = git_blob_sha1(destination, size) if size >= 0 else None
        if size != int(entry["size"]) or actual_sha != entry["sha"]:
            failures.append(
                {
                    "path": entry["path"],
                    "expected_size": int(entry["size"]),
                    "actual_size": size,
                    "expected_sha1": entry["sha"],
                    "actual_sha1": actual_sha,
                }
            )
    verify.close()
    if failures:
        raise RuntimeError(f"Exact blob verification failed for {len(failures)} files")

    report = {
        "status": "eegemotions27_registered_exact_blob_materialization_complete",
        "record_date": RECORD_DATE,
        "dataset": "EEGEmotions-27",
        "locked_commit": LOCKED_COMMIT,
        "repository": repository.as_posix(),
        "tree_receipt": args.tree_receipt.resolve().as_posix(),
        "registered_emotion_ids": list(REGISTERED_EMOTION_IDS),
        "registered_blob_count": len(entries),
        "registered_blob_bytes": total_bytes,
        "rewritten_from_git_object_database": rewritten,
        "already_exact": already_exact,
        "verification_failures": failures,
        "materialization_method": (
            "git cat-file blob from the locked local object database, followed by "
            "direct Git blob SHA-1 verification; checkout filters were bypassed"
        ),
        "scientific_method_change": False,
        "claim_boundary": (
            "This report records byte-exact acquisition repair only and supplies no "
            "endpoint, effectiveness, non-harm, safety, or replication evidence."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Exact registered blobs verified: {len(entries)}")
    print(f"Materialization report: {args.report.resolve()}")


if __name__ == "__main__":
    main()
