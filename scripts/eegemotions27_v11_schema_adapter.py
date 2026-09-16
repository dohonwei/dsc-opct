from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from eegemotions27_v11_contract import (
    LOCKED_COMMIT,
    MINIMUM_TRIALS_PER_PARTICIPANT,
    NEGATIVE_EMOTION_IDS,
    POSITIVE_EMOTION_IDS,
    REGISTERED_EMOTION_IDS,
    REQUIRED_CHANNELS,
)
from eegemotions27_v11_feature_core import representation_columns, trial_features

if TYPE_CHECKING:
    from external_wearable_one_shot_core import RegisteredTaskBundle


RAW_NAME = re.compile(r"^(?P<participant>\d+)_(?P<emotion>\d+)\.0\.txt$")


@dataclass(frozen=True)
class EEGEmotions27AdapterProfile:
    signal_encoding: str = "headerless_numeric_text"
    delimiter_policy: str = "whitespace_then_comma"
    orientation_policy: str = "exactly_one_axis_has_14_channels"
    native_value_policy: str = "official_export_values_interpreted_as_microvolts"
    sampling_rate_hz: int = 256
    minimum_samples: int = 2048


LOCKED_PROFILE = EEGEmotions27AdapterProfile()


class EEGEmotions27SchemaError(ValueError):
    pass


class EEGEmotions27TrialExclusion(ValueError):
    pass


def _require_locked_profile(profile: EEGEmotions27AdapterProfile) -> None:
    if profile != LOCKED_PROFILE:
        raise EEGEmotions27SchemaError(
            "Unknown EEGEmotions-27 delimiter, orientation, unit, sampling-rate, or "
            "minimum-duration convention; refusing to guess signal semantics"
        )


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def parse_raw_bytes(
    raw: bytes,
    profile: EEGEmotions27AdapterProfile = LOCKED_PROFILE,
) -> tuple[np.ndarray, dict[str, Any]]:
    _require_locked_profile(profile)
    attempts = []
    matrix = None
    delimiter_used = None
    for delimiter, name in ((None, "whitespace"), (",", "comma")):
        try:
            candidate = np.loadtxt(io.BytesIO(raw), dtype=float, delimiter=delimiter)
        except (UnicodeDecodeError, ValueError) as error:
            attempts.append(f"{name}: {error}")
            continue
        matrix = np.asarray(candidate, dtype=float)
        delimiter_used = name
        break
    if matrix is None:
        raise EEGEmotions27SchemaError(
            "Raw text is not a registered headerless numeric matrix: " + " | ".join(attempts)
        )
    if matrix.ndim != 2:
        raise EEGEmotions27SchemaError(f"Raw signal matrix must be 2-D, found {matrix.shape}")
    if matrix.shape[1] == len(REQUIRED_CHANNELS) and matrix.shape[0] != len(REQUIRED_CHANNELS):
        data = matrix.T
        orientation = "samples_by_channels"
    elif matrix.shape[0] == len(REQUIRED_CHANNELS) and matrix.shape[1] != len(REQUIRED_CHANNELS):
        data = matrix
        orientation = "channels_by_samples"
    else:
        raise EEGEmotions27SchemaError(
            f"Exactly one raw-matrix axis must contain 14 channels, found {matrix.shape}"
        )
    if data.shape[1] < profile.minimum_samples:
        raise EEGEmotions27TrialExclusion(
            f"Raw trial has {data.shape[1]} samples; minimum is {profile.minimum_samples}"
        )
    if not np.isfinite(data).all():
        raise EEGEmotions27TrialExclusion("Raw trial contains nonfinite signal values")
    return data, {
        "samples": int(data.shape[1]),
        "channels": int(data.shape[0]),
        "delimiter": delimiter_used,
        "orientation": orientation,
        "native_value_policy": profile.native_value_policy,
    }


def build_trial_record(
    raw: bytes,
    *,
    subject_id: str,
    emotion_id: int,
    profile: EEGEmotions27AdapterProfile = LOCKED_PROFILE,
) -> tuple[dict[str, object], dict[str, Any]]:
    data, schema = parse_raw_bytes(raw, profile)
    try:
        row = trial_features(
            data,
            profile.sampling_rate_hz,
            subject_id,
            emotion_id,
        )
    except ValueError as error:
        raise EEGEmotions27TrialExclusion(str(error)) from error
    return row, schema


def _repository_commit(data_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(data_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verified_manifest(schema_manifest: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest = json.loads(schema_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "eegemotions27_repository_metadata_manifest_complete":
        raise RuntimeError("EEGEmotions-27 metadata manifest has an invalid status")
    receipt_path = Path(manifest["raw_tree_metadata_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("sha") != manifest.get("raw_tree_sha1") or receipt.get("truncated") is not False:
        raise RuntimeError("EEGEmotions-27 tree receipt differs from the metadata manifest")
    blobs = {
        str(entry["path"]): str(entry["sha"])
        for entry in receipt["tree"]
        if entry.get("type") == "blob"
    }
    return manifest, blobs


def load_eegemotions27_task_bundles(
    data_root: Path,
    schema_manifest: Path,
    profile: EEGEmotions27AdapterProfile = LOCKED_PROFILE,
) -> tuple[dict[str, "RegisteredTaskBundle"], dict[str, Any]]:
    from external_wearable_one_shot_core import RegisteredTaskBundle

    _require_locked_profile(profile)
    if _repository_commit(data_root) != LOCKED_COMMIT:
        raise RuntimeError("EEGEmotions-27 working repository differs from the locked commit")
    manifest, blobs = _verified_manifest(schema_manifest)
    selected = []
    for name, blob in blobs.items():
        match = RAW_NAME.fullmatch(name)
        if match and int(match.group("emotion")) in REGISTERED_EMOTION_IDS:
            selected.append((name, blob, match.group("participant"), int(match.group("emotion"))))
    selected.sort(key=lambda item: (int(item[2]), item[3]))
    if not selected:
        raise ValueError("No registered EEGEmotions-27 raw trials were found")

    records: list[dict[str, object]] = []
    schema_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    progress = tqdm(
        selected,
        desc="EEGEmotions-27 schema-bound feature extraction",
        unit="trial",
        dynamic_ncols=True,
    )
    for name, expected_blob, subject_id, emotion_id in progress:
        path = data_root / "eeg_raw" / name
        if not path.is_file():
            exclusions.append(
                {"subject_id": subject_id, "emotion_id": str(emotion_id), "reason": "missing_checked_out_raw_file"}
            )
            continue
        raw = path.read_bytes()
        if git_blob_sha1(raw) != expected_blob:
            raise RuntimeError(f"Raw file differs from locked Git blob: {name}")
        try:
            row, schema = build_trial_record(
                raw,
                subject_id=subject_id,
                emotion_id=emotion_id,
                profile=profile,
            )
        except EEGEmotions27TrialExclusion as error:
            exclusions.append(
                {"subject_id": subject_id, "emotion_id": str(emotion_id), "reason": str(error)}
            )
            continue
        records.append(row)
        schema_rows.append({"subject_id": subject_id, "emotion_id": emotion_id, **schema})
    progress.close()
    if not records:
        raise ValueError("No EEGEmotions-27 trial survived schema-bound construction")

    frame = pd.DataFrame(records)
    labels = frame.emotion_id.isin(POSITIVE_EMOTION_IDS).to_numpy(dtype=int)
    subject_audit = frame.assign(label=labels).groupby("subject_id").agg(
        trials=("trial_id", "size"), classes=("label", "nunique")
    )
    eligible_subjects = subject_audit.index[
        (subject_audit.trials >= MINIMUM_TRIALS_PER_PARTICIPANT)
        & (subject_audit.classes == 2)
    ]
    eligible = frame.subject_id.isin(eligible_subjects).to_numpy()
    retained_frame = frame.loc[eligible].reset_index(drop=True)
    retained_labels = labels[eligible]
    columns = representation_columns(records[0])
    feature_sets = {
        name: frame.loc[:, selected_columns].to_numpy(dtype=float)[eligible]
        for name, selected_columns in columns.items()
    }
    bundle = RegisteredTaskBundle(retained_frame, retained_labels, feature_sets)

    exclusion_counts = Counter(item["reason"] for item in exclusions)
    schema_frame = pd.DataFrame(schema_rows)
    provenance = {
        "adapter_profile": LOCKED_PROFILE.__dict__,
        "locked_commit": LOCKED_COMMIT,
        "metadata_manifest": str(schema_manifest.resolve()),
        "metadata_manifest_raw_count": manifest["raw_filename_count"],
        "registered_files_selected": len(selected),
        "constructed_trials_before_subject_filter": len(frame),
        "eligible_subjects": int(len(eligible_subjects)),
        "eligible_trials": int(eligible.sum()),
        "class_counts": {
            str(key): int(value)
            for key, value in pd.Series(retained_labels).value_counts().sort_index().items()
        },
        "raw_schema_summary": {
            "delimiters": schema_frame.delimiter.value_counts().to_dict(),
            "orientations": schema_frame.orientation.value_counts().to_dict(),
            "samples_min": int(schema_frame.samples.min()),
            "samples_median": float(schema_frame.samples.median()),
            "samples_max": int(schema_frame.samples.max()),
        },
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "label_rule": {
            "positive_emotion_ids": list(POSITIVE_EMOTION_IDS),
            "negative_emotion_ids": list(NEGATIVE_EMOTION_IDS),
            "construct": "fixed category-derived affective polarity",
        },
    }
    return {"category_polarity": bundle}, provenance
