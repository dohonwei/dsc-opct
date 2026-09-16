from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
import zipfile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from emognition_v11_contract import (
    ALLOWED_HEADBAND_VALUES,
    ALLOWED_HSI_VALUES,
    EXPECTED_ARCHIVE_MD5,
    MINIMUM_TRIALS_PER_PARTICIPANT,
    QUALITY_KEYS,
    RAW_CHANNEL_KEYS,
    RAW_VALUE_POLICY,
    REQUIRED_CHANNELS,
    SAM_HIGH_MINIMUM,
    SAM_LOW_MAXIMUM,
    SAM_MIDPOINT,
    STIMULUS_NAMES,
    USABLE_HSI_VALUES,
    hash_file,
    sam_binary_label,
)
from emognition_v11_feature_core import representation_columns, trial_features

if TYPE_CHECKING:
    from external_wearable_one_shot_core import RegisteredTaskBundle


TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S:%f"
SAM_KEYS = {"valence": "VALENCE", "arousal": "AROUSAL"}
PUBLIC_PARTICIPANT_IDS = tuple(str(value) for value in range(22, 65))


@dataclass(frozen=True)
class EmognitionAdapterProfile:
    signal_encoding: str = "top_level_timestamp_value_pairs"
    questionnaire_encoding: str = "questionnaires_list_movie_and_sam"
    raw_value_policy: str = RAW_VALUE_POLICY
    quality_policy: str = "complete_window_headband_on_and_no_bad_hsi"
    sampling_rate_hz: int = 256


LOCKED_PROFILE = EmognitionAdapterProfile()


class EmognitionSchemaError(ValueError):
    pass


class EmognitionTrialExclusion(ValueError):
    pass


def _require_locked_profile(profile: EmognitionAdapterProfile) -> None:
    if profile != LOCKED_PROFILE:
        raise EmognitionSchemaError(
            "Unknown Emognition encoding, unit basis, quality convention, or sampling rate; "
            "refusing to guess participant-data semantics"
        )


def _read_json_member(handle: zipfile.ZipFile, member: str) -> Any:
    try:
        raw = handle.read(member)
    except KeyError as error:
        raise FileNotFoundError(f"Registered archive member is absent: {member}") from error
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Unreadable UTF-8 JSON member: {member}") from error


def _validate_member_path(member: str, expected: str) -> None:
    normalized = PurePosixPath(member).as_posix()
    if normalized != expected or ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"Unexpected Emognition member path: {member}")


def parse_questionnaire_payload(payload: Any, subject_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not {"metadata", "questionnaires"}.issubset(payload):
        raise EmognitionSchemaError("Unknown Emognition questionnaire top-level structure")
    metadata = payload["metadata"]
    questionnaires = payload["questionnaires"]
    if not isinstance(metadata, dict) or not isinstance(questionnaires, list):
        raise EmognitionSchemaError("Unknown Emognition questionnaire metadata or list structure")
    if str(metadata.get("id")) != str(subject_id):
        raise EmognitionSchemaError("Questionnaire participant identifier differs from its path")
    movie_order = metadata.get("movie_order")
    if not isinstance(movie_order, list):
        raise EmognitionSchemaError("Questionnaire metadata.movie_order is absent or not a list")
    observed_order = [str(value).upper() for value in movie_order]
    if len(observed_order) != len(set(observed_order)):
        raise EmognitionSchemaError("Questionnaire movie_order contains duplicates")
    if set(STIMULUS_NAMES) - set(observed_order):
        raise EmognitionSchemaError("Questionnaire movie_order lacks a registered stimulus")

    ratings: dict[str, dict[str, float]] = {}
    for item in questionnaires:
        if not isinstance(item, dict) or "movie" not in item or "sam" not in item:
            raise EmognitionSchemaError("Unknown questionnaire item nesting")
        movie = str(item["movie"]).upper()
        if movie not in STIMULUS_NAMES:
            continue
        if movie in ratings:
            raise EmognitionSchemaError(f"Duplicate questionnaire rating for {movie}")
        sam = item["sam"]
        if not isinstance(sam, dict):
            raise EmognitionSchemaError(f"Questionnaire SAM field is not a mapping for {movie}")
        task_values = {}
        for task, key in SAM_KEYS.items():
            if key not in sam or sam[key] is None:
                task_values[task] = None
                continue
            value = float(sam[key])
            sam_binary_label(value)
            task_values[task] = value
        ratings[movie] = task_values
    return {"movie_order": observed_order, "ratings": ratings}


def _timestamp_value_series(payload: Any, key: str) -> tuple[list[str], np.ndarray]:
    if key not in payload or not isinstance(payload[key], list) or not payload[key]:
        raise EmognitionSchemaError(f"Signal member lacks non-empty pair series: {key}")
    timestamps: list[str] = []
    values: list[float] = []
    for index, item in enumerate(payload[key]):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise EmognitionSchemaError(
                f"{key} row {index} is not a timestamp-value pair"
            )
        timestamp = str(item[0])
        try:
            datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        except ValueError as error:
            raise EmognitionSchemaError(
                f"{key} contains an unknown timestamp format"
            ) from error
        try:
            value = float(item[1])
        except (TypeError, ValueError) as error:
            raise EmognitionSchemaError(f"{key} contains a nonnumeric value") from error
        timestamps.append(timestamp)
        values.append(value)
    if len(timestamps) != len(set(timestamps)) or timestamps != sorted(timestamps):
        raise EmognitionSchemaError(f"{key} timestamps must be unique and monotonic")
    return timestamps, np.asarray(values, dtype=float)


def parse_muse_payload(
    payload: Any,
    profile: EmognitionAdapterProfile = LOCKED_PROFILE,
) -> tuple[np.ndarray, dict[str, Any]]:
    _require_locked_profile(profile)
    if not isinstance(payload, dict):
        raise EmognitionSchemaError("Unknown Emognition Muse JSON structure")
    required = set(RAW_CHANNEL_KEYS) | set(QUALITY_KEYS)
    missing = required - set(payload)
    if missing:
        raise EmognitionSchemaError(f"Muse JSON lacks registered fields: {sorted(missing)}")

    timestamps: list[str] | None = None
    values: dict[str, np.ndarray] = {}
    for key in (*RAW_CHANNEL_KEYS, *QUALITY_KEYS):
        key_timestamps, key_values = _timestamp_value_series(payload, key)
        if timestamps is None:
            timestamps = key_timestamps
        elif key_timestamps != timestamps:
            raise EmognitionSchemaError(
                "Muse channels and quality indicators are not timestamp aligned"
            )
        values[key] = key_values
    if not all(np.isfinite(values[key]).all() for key in RAW_CHANNEL_KEYS):
        raise EmognitionTrialExclusion("Muse raw signal contains nonfinite values")

    headband = values["HeadBandOn"]
    if not np.equal(headband, np.floor(headband)).all() or not set(
        headband.astype(int)
    ).issubset(ALLOWED_HEADBAND_VALUES):
        raise EmognitionSchemaError("Unknown Muse HeadBandOn coding")
    hsi = np.stack([values[f"HSI_{channel}"] for channel in REQUIRED_CHANNELS], axis=0)
    if not np.equal(hsi, np.floor(hsi)).all() or not set(hsi.astype(int).ravel()).issubset(
        ALLOWED_HSI_VALUES
    ):
        raise EmognitionSchemaError("Unknown Muse HSI coding")

    usable = (headband == 1) & np.all(np.isin(hsi, USABLE_HSI_VALUES), axis=0)
    data = np.stack([values[key] for key in RAW_CHANNEL_KEYS], axis=0)
    data[:, ~usable] = np.nan
    return data, {
        "samples": int(data.shape[1]),
        "usable_samples": int(usable.sum()),
        "usable_fraction": float(usable.mean()),
        "raw_value_policy": profile.raw_value_policy,
        "quality_policy": profile.quality_policy,
    }


def build_trial_record(
    signal_payload: Any,
    *,
    subject_id: str,
    stimulus_name: str,
    stimulus_index: int,
    profile: EmognitionAdapterProfile = LOCKED_PROFILE,
) -> tuple[dict[str, object], dict[str, Any]]:
    data, quality = parse_muse_payload(signal_payload, profile)
    try:
        row = trial_features(
            data,
            profile.sampling_rate_hz,
            subject_id,
            stimulus_index,
            stimulus_name,
        )
    except ValueError as error:
        raise EmognitionTrialExclusion(str(error)) from error
    return row, quality


def _verify_archive_and_manifest(archive: Path, schema_manifest: Path) -> dict[str, Any]:
    if not archive.is_file() or not schema_manifest.is_file():
        raise FileNotFoundError("Authorized archive and schema-only manifest are both required")
    manifest = json.loads(schema_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "emognition_schema_only_acquisition_manifest_complete":
        raise RuntimeError("Emognition schema-only manifest has an invalid status")
    if manifest.get("archive_md5") != EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("Schema manifest does not identify the official archive checksum")
    if hash_file(archive, "md5") != EXPECTED_ARCHIVE_MD5:
        raise RuntimeError("Authorized archive checksum differs from the official record")
    if manifest.get("archive_sha256") != hash_file(archive):
        raise RuntimeError("Authorized archive changed after schema-only inspection")
    return manifest


def load_emognition_task_bundles(
    archive: Path,
    schema_manifest: Path,
    profile: EmognitionAdapterProfile = LOCKED_PROFILE,
) -> tuple[dict[str, "RegisteredTaskBundle"], dict[str, Any]]:
    from external_wearable_one_shot_core import RegisteredTaskBundle

    _require_locked_profile(profile)
    manifest = _verify_archive_and_manifest(archive, schema_manifest)
    manifest_members = {
        PurePosixPath(entry["path"]).as_posix()
        for entry in manifest.get("entries", [])
        if not entry.get("is_directory")
    }
    records: list[dict[str, object]] = []
    ratings: list[dict[str, float]] = []
    exclusions: list[dict[str, str]] = []
    quality_rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive) as handle:
        live_members = {
            PurePosixPath(info.filename).as_posix()
            for info in handle.infolist()
            if not info.is_dir()
        }
        if live_members != manifest_members:
            raise RuntimeError("ZIP member inventory changed after schema-only inspection")
        prefixes = set()
        for subject_id in PUBLIC_PARTICIPANT_IDS:
            suffix = f"{subject_id}/{subject_id}_QUESTIONNAIRES.json"
            prefixes.update(
                member[: -len(suffix)] for member in live_members if member.endswith(suffix)
            )
        if len(prefixes) != 1:
            raise EmognitionSchemaError(
                "Cannot identify a unique participant-root prefix from the ZIP manifest"
            )
        prefix = prefixes.pop()
        progress = tqdm(
            PUBLIC_PARTICIPANT_IDS,
            desc="Emognition schema-bound feature extraction",
            unit="participant",
            dynamic_ncols=True,
        )
        for subject_id in progress:
            questionnaire_member = f"{prefix}{subject_id}/{subject_id}_QUESTIONNAIRES.json"
            _validate_member_path(questionnaire_member, questionnaire_member)
            if questionnaire_member not in live_members:
                exclusions.append({"subject_id": subject_id, "reason": "missing_questionnaire"})
                continue
            try:
                questionnaire = parse_questionnaire_payload(
                    _read_json_member(handle, questionnaire_member), subject_id
                )
            except FileNotFoundError as error:
                exclusions.append({"subject_id": subject_id, "reason": str(error)})
                continue

            for stimulus_index, stimulus_name in enumerate(STIMULUS_NAMES):
                task_ratings = questionnaire["ratings"].get(stimulus_name)
                if task_ratings is None or all(
                    value is None for value in task_ratings.values()
                ):
                    exclusions.append(
                        {
                            "subject_id": subject_id,
                            "stimulus": stimulus_name,
                            "reason": "missing_registered_sam_rating",
                        }
                    )
                    continue
                signal_member = (
                    f"{prefix}{subject_id}/{subject_id}_{stimulus_name}_STIMULUS_MUSE.json"
                )
                _validate_member_path(signal_member, signal_member)
                if signal_member not in live_members:
                    exclusions.append(
                        {
                            "subject_id": subject_id,
                            "stimulus": stimulus_name,
                            "reason": "missing_registered_signal_file",
                        }
                    )
                    continue
                try:
                    row, quality = build_trial_record(
                        _read_json_member(handle, signal_member),
                        subject_id=subject_id,
                        stimulus_name=stimulus_name,
                        stimulus_index=stimulus_index,
                        profile=profile,
                    )
                except (FileNotFoundError, EmognitionTrialExclusion) as error:
                    exclusions.append(
                        {
                            "subject_id": subject_id,
                            "stimulus": stimulus_name,
                            "reason": str(error),
                        }
                    )
                    continue
                records.append(row)
                ratings.append(task_ratings)
                quality_rows.append(
                    {"subject_id": subject_id, "stimulus": stimulus_name, **quality}
                )
        progress.close()

    if not records:
        raise ValueError("No Emognition trial survived schema-bound construction")
    frame = pd.DataFrame(records)
    rating_frame = pd.DataFrame(ratings)
    columns = representation_columns(records[0])
    feature_sets_all = {
        name: frame.loc[:, selected].to_numpy(dtype=float)
        for name, selected in columns.items()
    }
    bundles: dict[str, RegisteredTaskBundle] = {}
    eligibility: dict[str, Any] = {}
    for task in SAM_KEYS:
        task_ratings = rating_frame[task]
        missing_rating = task_ratings.isna()
        midpoint_rating = task_ratings.eq(SAM_MIDPOINT)
        binary = task_ratings.map(
            lambda value: None if pd.isna(value) else sam_binary_label(value)
        )
        keep = binary.notna()
        task_frame = frame.loc[keep].reset_index(drop=True)
        task_labels = binary.loc[keep].astype(int).to_numpy()
        subject_audit = task_frame.assign(label=task_labels).groupby("subject_id").agg(
            trials=("trial_id", "size"), classes=("label", "nunique")
        )
        eligible_subjects = subject_audit.index[
            (subject_audit.trials >= MINIMUM_TRIALS_PER_PARTICIPANT)
            & (subject_audit.classes == 2)
        ]
        eligible_mask = task_frame.subject_id.isin(eligible_subjects).to_numpy()
        original_indices = np.flatnonzero(keep.to_numpy())[eligible_mask]
        bundles[task] = RegisteredTaskBundle(
            frame=task_frame.loc[eligible_mask].reset_index(drop=True),
            labels=task_labels[eligible_mask],
            feature_sets={
                name: values[original_indices] for name, values in feature_sets_all.items()
            },
        )
        eligibility[task] = {
            "eligible_subjects": int(len(eligible_subjects)),
            "eligible_trials": int(eligible_mask.sum()),
            "missing_rating_trials_excluded": int(missing_rating.sum()),
            "midpoint_trials_excluded": int(midpoint_rating.sum()),
            "total_nonbinary_label_trials_excluded": int((~keep).sum()),
        }
    exclusion_counts = Counter(item["reason"] for item in exclusions)
    usable_fractions = np.asarray(
        [row["usable_fraction"] for row in quality_rows], dtype=float
    )
    provenance = {
        "adapter_profile": LOCKED_PROFILE.__dict__,
        "archive_md5": EXPECTED_ARCHIVE_MD5,
        "archive_sha256": hash_file(archive),
        "schema_manifest_sha256": hash_file(schema_manifest),
        "constructed_trials_before_task_filter": len(records),
        "eligibility": eligibility,
        "quality_summary": {
            "retained_trial_count": len(quality_rows),
            "usable_fraction_min": float(usable_fractions.min()),
            "usable_fraction_median": float(np.median(usable_fractions)),
            "usable_fraction_max": float(usable_fractions.max()),
        },
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "label_rule": {
            "high_minimum": SAM_HIGH_MINIMUM,
            "low_maximum": SAM_LOW_MAXIMUM,
            "excluded_midpoint": SAM_MIDPOINT,
        },
    }
    return bundles, provenance
