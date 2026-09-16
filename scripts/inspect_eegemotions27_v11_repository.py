from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from eegemotions27_v11_contract import (
    DEFAULT_DATA_ROOT,
    LOCKED_COMMIT,
    REGISTERED_EMOTION_IDS,
    ROOT,
    normalized_channel,
    verify_presignal_contract,
)


DEFAULT_OUTPUT = ROOT / "outputs/eegemotions27_v11_presignal/repository_metadata_manifest.json"
DEFAULT_TREE_METADATA = ROOT / "outputs/eegemotions27_v11_presignal/official_eeg_raw_tree.json"
CHANNEL_EVIDENCE = ROOT / "docs/evidence/eegemotions27/emotivX_channels_location.ced"
CHANNEL_BLOB_SHA1 = "7a1c263c06725f8bdb17a66fad36a99b4a478540"
RAW_TREE_SHA1 = "9be19b2dbd759121ed8c381bf632c07904ba07b5"
RAW_NAME = re.compile(r"^(?P<participant>\d+)_(?P<emotion>\d+)\.0\.txt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect EEGEmotions-27 repository metadata and channel metadata without "
            "opening raw EEG text files."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tree-metadata", type=Path, default=DEFAULT_TREE_METADATA)
    parser.add_argument("--allow-non-git-fixture", action="store_true")
    return parser.parse_args()


def repository_commit(data_root: Path, allow_fixture: bool) -> str:
    if allow_fixture:
        return LOCKED_COMMIT
    completed = subprocess.run(
        ["git", "-C", str(data_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def parse_channel_metadata(text: str) -> list[str]:
    required = {
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
        "P8", "T8", "FC6", "F4", "F8", "AF4",
    }
    channels = []
    for line in text.lstrip("\ufeff").splitlines():
        fields = re.split(r"[\s,\t]+", line.strip())
        normalized = [normalized_channel(field) for field in fields if field]
        for field in normalized:
            if field in required and field not in channels:
                channels.append(field)
    return channels


def verified_channel_evidence() -> str:
    text = CHANNEL_EVIDENCE.read_text(encoding="utf-8-sig")
    canonical = ("\r\n".join(text.splitlines()) + "\r\n").encode("utf-8")
    header = f"blob {len(canonical)}\0".encode("ascii")
    observed = hashlib.sha1(header + canonical).hexdigest()
    if observed != CHANNEL_BLOB_SHA1:
        raise RuntimeError("Local channel evidence differs from the locked Git blob")
    return text


def official_raw_tree_entries(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sha") != RAW_TREE_SHA1 or payload.get("truncated") is not False:
        raise RuntimeError("Official eeg_raw tree receipt is wrong or truncated")
    entries = payload.get("tree")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("Official eeg_raw tree receipt has no entries")
    validated = []
    for entry in entries:
        if entry.get("type") != "blob" or not isinstance(entry.get("size"), int):
            raise RuntimeError("Unexpected non-blob or sizeless eeg_raw tree entry")
        validated.append(
            {
                "name": str(entry["path"]),
                "bytes": int(entry["size"]),
                "blob": str(entry["sha"]),
            }
        )
    return validated


def build_manifest(
    data_root: Path,
    tree_metadata: Path = DEFAULT_TREE_METADATA,
    allow_fixture: bool = False,
) -> dict:
    reservation, history, _ = verify_presignal_contract()
    if not data_root.is_dir():
        raise FileNotFoundError(f"EEGEmotions-27 repository not found: {data_root}")
    commit = repository_commit(data_root, allow_fixture)
    if commit != LOCKED_COMMIT:
        raise RuntimeError(f"Repository commit {commit} differs from lock {LOCKED_COMMIT}")

    if allow_fixture:
        raw_root = data_root / "eeg_raw"
        channel_file = data_root / "emotivX_channels_location.ced"
        if not raw_root.is_dir() or not channel_file.is_file():
            raise FileNotFoundError("Expected eeg_raw directory or channel metadata is missing")
        raw_entries = [
            {"name": path.name, "bytes": path.stat().st_size, "blob": None}
            for path in sorted(raw_root.glob("*.txt"), key=lambda item: item.name)
        ]
        channel_text = channel_file.read_text(encoding="utf-8-sig")
        inspection_method = (
            "fixture path enumeration and channel-metadata parsing; empty synthetic "
            "signal files were not opened"
        )
    else:
        raw_entries = official_raw_tree_entries(tree_metadata)
        channel_text = verified_channel_evidence()
        inspection_method = (
            "offline parsing of an official GitHub tree-API receipt plus a local channel-"
            "metadata copy verified against its Git blob SHA-1; the inspector invoked no "
            "promisor remote and no eeg_raw blob was opened, checked out, or hashed"
        )

    participant_trials: Counter[str] = Counter()
    emotion_counts: Counter[int] = Counter()
    malformed = []
    total_bytes = 0
    for entry in raw_entries:
        name = str(entry["name"])
        match = RAW_NAME.fullmatch(name)
        if match is None:
            malformed.append(name)
            continue
        participant = match.group("participant")
        emotion_id = int(match.group("emotion"))
        participant_trials[participant] += 1
        emotion_counts[emotion_id] += 1
        total_bytes += int(entry["bytes"])
    if not participant_trials:
        raise ValueError("No valid raw filenames were found")

    registered_counts = {
        str(emotion_id): int(emotion_counts[emotion_id])
        for emotion_id in REGISTERED_EMOTION_IDS
    }
    return {
        "status": "eegemotions27_repository_metadata_manifest_complete",
        "dataset": "EEGEmotions-27",
        "phase": "post-reservation pre-signal metadata inspection",
        "data_root": str(data_root.resolve()),
        "locked_commit": commit,
        "raw_tree_sha1": RAW_TREE_SHA1,
        "raw_tree_metadata_receipt": str(tree_metadata.resolve()),
        "raw_filename_count": len(raw_entries),
        "raw_total_bytes_from_filesystem_metadata": total_bytes,
        "malformed_raw_filenames": malformed,
        "participant_count_encoded_in_filenames": len(participant_trials),
        "trials_per_participant_min_from_filenames": min(participant_trials.values()),
        "trials_per_participant_max_from_filenames": max(participant_trials.values()),
        "registered_emotion_file_counts": registered_counts,
        "channels_from_metadata_file": parse_channel_metadata(channel_text),
        "inspection_method": inspection_method,
        "prior_metadata_disclosure": history["information_revealed_by_prior_metadata_access"],
        "reservation_id": reservation["registration_id"],
        "claim_boundary": (
            "This manifest establishes repository identity and filename/channel metadata "
            "only. It supplies no signal-quality, endpoint, effectiveness, non-harm, "
            "safety, or replication evidence."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite metadata manifest: {args.output}")
    manifest = build_manifest(
        args.data_root,
        tree_metadata=args.tree_metadata,
        allow_fixture=args.allow_non_git_fixture,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Recorded metadata for {manifest['raw_filename_count']} raw filenames "
        "without opening EEG signal files"
    )


if __name__ == "__main__":
    main()
