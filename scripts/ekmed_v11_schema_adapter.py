from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
import zipfile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ekmed_v11_contract import (
    KEY_MOMENTS_SECONDS,
    MINIMUM_TRIALS_PER_PARTICIPANT,
    REQUIRED_CHANNELS,
    SAM_HIGH_MINIMUM,
    SAM_LOW_MAXIMUM,
    SAM_MIDPOINT,
    STIMULUS_NAMES,
    hash_file,
    sam_binary_label,
)
from ekmed_v11_feature_core import representation_columns, trial_features_from_key_windows

if TYPE_CHECKING:
    from external_wearable_one_shot_core import RegisteredTaskBundle


SAM_COLUMNS = {"valence": "VALENCE", "arousal": "AROUSAL"}
RAW_COLUMNS = tuple(f"RAW_{channel}" for channel in REQUIRED_CHANNELS)
QUESTIONNAIRE_COLUMNS = ("__index__", "ID", "VALENCE", "AROUSAL", "EMOTION")
SIGNAL_COLUMNS = (
    "TimeStamp",
    "Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
    "Theta_TP9", "Theta_AF7", "Theta_AF8", "Theta_TP10",
    "Alpha_TP9", "Alpha_AF7", "Alpha_AF8", "Alpha_TP10",
    "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
    "Gamma_TP9", "Gamma_AF7", "Gamma_AF8", "Gamma_TP10",
    "RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10",
    "AUX_RIGHT", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z",
    "Gyro_X", "Gyro_Y", "Gyro_Z", "HeadBandOn",
    "HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10",
    "Battery", "Elements", "Key_Moments",
    "MyDelta_TP9", "MyTheta_TP9", "MyAlpha_TP9", "MyBeta_TP9", "MyGamma_TP9",
    "MyDelta_TP10", "MyTheta_TP10", "MyAlpha_TP10", "MyBeta_TP10", "MyGamma_TP10",
    "MyDelta_AF7", "MyTheta_AF7", "MyAlpha_AF7", "MyBeta_AF7", "MyGamma_AF7",
    "MyDelta_AF8", "MyTheta_AF8", "MyAlpha_AF8", "MyBeta_AF8", "MyGamma_AF8",
)


@dataclass(frozen=True)
class EKMAdapterProfile:
    sampling_rate_hz: int = 128
    window_seconds: float = 4.0
    time_alignment: str = "official_128hz_row_order_from_clip_onset"
    amplitude_policy: str = "archive_native_raw_values_without_unit_rescaling"
    feature_source: str = "RAW_channels_only"
    key_moment_source: str = "frozen_public_seconds_not_archive_Key_Moments_column"


LOCKED_PROFILE = EKMAdapterProfile()


class EKMSchemaError(ValueError):
    pass


class EKMTrialExclusion(ValueError):
    pass


def _require_locked_profile(profile: EKMAdapterProfile) -> None:
    if profile != LOCKED_PROFILE:
        raise EKMSchemaError("Unknown EKM-ED sampling, alignment, amplitude, or feature policy")


def _normalized_columns(columns: list[Any]) -> tuple[str, ...]:
    normalized = [str(column).strip() for column in columns]
    if normalized and (not normalized[0] or normalized[0].lower().startswith("unnamed:")):
        normalized[0] = "__index__"
    return tuple(normalized)


def _subject_token(value: Any) -> str:
    if pd.isna(value):
        raise EKMSchemaError("Questionnaire participant identifier is missing")
    text = str(value).strip()
    if not text:
        raise EKMSchemaError("Questionnaire participant identifier is empty")
    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None and np.isfinite(numeric) and numeric.is_integer():
        text = str(int(numeric))
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise EKMSchemaError("Questionnaire participant identifier is not path-safe")
    return text


def parse_questionnaire_frame(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, float | None]]]:
    if _normalized_columns(frame.columns.tolist()) != QUESTIONNAIRE_COLUMNS:
        raise EKMSchemaError("Questionnaire columns differ from the locked header schema")
    work = frame.copy()
    work.columns = QUESTIONNAIRE_COLUMNS
    ratings: dict[str, dict[str, dict[str, float | None]]] = {}
    observed_pairs: set[tuple[str, str]] = set()
    for row in work.itertuples(index=False):
        subject_id = _subject_token(row.ID)
        stimulus = str(row.EMOTION).strip().upper()
        if stimulus not in STIMULUS_NAMES:
            continue
        pair = (subject_id, stimulus)
        if pair in observed_pairs:
            raise EKMSchemaError(f"Duplicate questionnaire row for {subject_id}/{stimulus}")
        observed_pairs.add(pair)
        task_values: dict[str, float | None] = {}
        for task, column in SAM_COLUMNS.items():
            value = getattr(row, column)
            if pd.isna(value):
                task_values[task] = None
                continue
            try:
                numeric = float(value)
                sam_binary_label(numeric)
            except (TypeError, ValueError) as error:
                raise EKMSchemaError(
                    f"Invalid registered SAM value for {subject_id}/{stimulus}/{task}"
                ) from error
            task_values[task] = numeric
        ratings.setdefault(subject_id, {})[stimulus] = task_values
    if not ratings:
        raise EKMSchemaError("Questionnaire contains no registered participant-film rows")
    return ratings


def parse_signal_frame(
    frame: pd.DataFrame,
    *,
    subject_id: str,
    stimulus_name: str,
    stimulus_index: int,
    profile: EKMAdapterProfile = LOCKED_PROFILE,
) -> tuple[dict[str, object], dict[str, Any]]:
    _require_locked_profile(profile)
    if _normalized_columns(frame.columns.tolist()) != SIGNAL_COLUMNS:
        raise EKMSchemaError("Signal columns differ from the locked clean-signal schema")
    if frame.empty:
        raise EKMTrialExclusion("clean signal file is empty")
    timestamps = frame.loc[:, "TimeStamp"]
    if timestamps.isna().any() or timestamps.astype(str).str.strip().eq("").any():
        raise EKMTrialExclusion("TimeStamp contains missing values")
    if timestamps.astype(str).duplicated().any():
        raise EKMTrialExclusion("TimeStamp contains duplicate sample rows")
    try:
        raw = frame.loc[:, RAW_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float).T
    except (TypeError, ValueError) as error:
        raise EKMTrialExclusion("registered RAW channel contains a nonnumeric value") from error

    sampling_rate = float(profile.sampling_rate_hz)
    window_samples = int(round(profile.window_seconds * sampling_rate))
    half_window = window_samples // 2
    key_windows = []
    for moment_seconds in KEY_MOMENTS_SECONDS[stimulus_name]:
        center = int(round(float(moment_seconds) * sampling_rate))
        start = center - half_window
        stop = start + window_samples
        if start < 0 or stop > raw.shape[1]:
            continue
        window = raw[:, start:stop]
        if window.shape != (len(REQUIRED_CHANNELS), window_samples):
            continue
        if not np.isfinite(window).all():
            continue
        key_windows.append(window)
    if len(key_windows) < 3:
        raise EKMTrialExclusion(
            f"only {len(key_windows)} complete finite registered key-moment windows"
        )
    try:
        row = trial_features_from_key_windows(
            key_windows,
            sampling_rate,
            subject_id,
            stimulus_index,
            stimulus_name,
        )
    except ValueError as error:
        raise EKMTrialExclusion(str(error)) from error
    return row, {
        "samples": int(raw.shape[1]),
        "key_moment_windows": int(len(key_windows)),
        "time_alignment": profile.time_alignment,
        "amplitude_policy": profile.amplitude_policy,
        "feature_source": profile.feature_source,
    }


def _verify_archive_and_receipt(
    archive: Path, schema_receipt: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not archive.is_file() or not schema_receipt.is_file():
        raise FileNotFoundError("Official archive and locked header receipt are required")
    receipt = json.loads(schema_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "header_schema_recorded_without_data_rows":
        raise RuntimeError("EKM-ED header receipt has an invalid status")
    if receipt.get("questionnaire", {}).get("columns") != list(QUESTIONNAIRE_COLUMNS):
        raise RuntimeError("Locked questionnaire header differs from the adapter schema")
    if receipt.get("signal", {}).get("columns") != list(SIGNAL_COLUMNS):
        raise RuntimeError("Locked signal header differs from the adapter schema")
    if hash_file(archive) != receipt.get("archive_sha256"):
        raise RuntimeError("Official archive changed after header-only inspection")
    central_path = schema_receipt.parent / "central_directory_manifest.json"
    if not central_path.is_file() or hash_file(central_path) != receipt.get("central_manifest_sha256"):
        raise RuntimeError("Central-directory manifest changed after header-only inspection")
    central = json.loads(central_path.read_text(encoding="utf-8"))
    if central.get("status") != "central_directory_recorded_without_member_reads":
        raise RuntimeError("Central-directory manifest has an invalid status")
    return receipt, central


def _clean_root(signal_member: str) -> str:
    path = PurePosixPath(signal_member)
    if len(path.parts) < 3 or path.name.upper() != "ANGER.CSV":
        raise EKMSchemaError("Representative signal member does not match the locked branch")
    return PurePosixPath(*path.parts[:-2]).as_posix() + "/"


def _read_csv_member(
    handle: zipfile.ZipFile,
    member: str,
    *,
    delimiter: str,
    encoding: str,
) -> pd.DataFrame:
    try:
        with handle.open(member, "r") as raw:
            return pd.read_csv(raw, sep=delimiter, encoding=encoding, low_memory=False)
    except KeyError as error:
        raise FileNotFoundError(f"Registered archive member is absent: {member}") from error


def load_ekmed_task_bundles(
    archive: Path,
    schema_receipt: Path,
    profile: EKMAdapterProfile = LOCKED_PROFILE,
) -> tuple[dict[str, "RegisteredTaskBundle"], dict[str, Any]]:
    from external_wearable_one_shot_core import RegisteredTaskBundle

    _require_locked_profile(profile)
    receipt, central = _verify_archive_and_receipt(archive, schema_receipt)
    manifest_members = {
        PurePosixPath(item["name"]).as_posix()
        for item in central["central_directory"]["members"]
    }
    questionnaire_member = PurePosixPath(receipt["questionnaire"]["member"]).as_posix()
    signal_root = _clean_root(PurePosixPath(receipt["signal"]["member"]).as_posix())
    records: list[dict[str, object]] = []
    rating_rows: list[dict[str, float | None]] = []
    exclusions: list[dict[str, str]] = []
    quality_rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive, "r") as handle:
        live_members = {
            PurePosixPath(info.filename).as_posix()
            for info in handle.infolist()
            if not info.is_dir()
        }
        if live_members != manifest_members:
            raise RuntimeError("ZIP member inventory changed after central-directory lock")
        questionnaire_frame = _read_csv_member(
            handle,
            questionnaire_member,
            delimiter=receipt["questionnaire"]["delimiter"],
            encoding=receipt["questionnaire"]["encoding"],
        )
        ratings = parse_questionnaire_frame(questionnaire_frame)
        participants = sorted(ratings, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        progress = tqdm(
            participants,
            desc="EKM-ED schema-bound key-moment extraction",
            unit="participant",
            dynamic_ncols=True,
        )
        for subject_id in progress:
            for stimulus_index, stimulus_name in enumerate(STIMULUS_NAMES):
                task_ratings = ratings[subject_id].get(stimulus_name)
                if task_ratings is None or all(value is None for value in task_ratings.values()):
                    exclusions.append({
                        "subject_id": subject_id,
                        "stimulus": stimulus_name,
                        "reason": "missing_registered_sam_rating",
                    })
                    continue
                signal_member = f"{signal_root}{subject_id}/{stimulus_name}.csv"
                if signal_member not in live_members:
                    exclusions.append({
                        "subject_id": subject_id,
                        "stimulus": stimulus_name,
                        "reason": "missing_registered_clean_signal_file",
                    })
                    continue
                try:
                    signal_frame = _read_csv_member(
                        handle,
                        signal_member,
                        delimiter=receipt["signal"]["delimiter"],
                        encoding=receipt["signal"]["encoding"],
                    )
                    row, quality = parse_signal_frame(
                        signal_frame,
                        subject_id=subject_id,
                        stimulus_name=stimulus_name,
                        stimulus_index=stimulus_index,
                        profile=profile,
                    )
                except (FileNotFoundError, EKMTrialExclusion) as error:
                    exclusions.append({
                        "subject_id": subject_id,
                        "stimulus": stimulus_name,
                        "reason": str(error),
                    })
                    continue
                records.append(row)
                rating_rows.append(task_ratings)
                quality_rows.append({"subject_id": subject_id, "stimulus": stimulus_name, **quality})
        progress.close()

    if not records:
        raise ValueError("No EKM-ED trial survived schema-bound construction")
    frame = pd.DataFrame(records)
    rating_frame = pd.DataFrame(rating_rows)
    columns = representation_columns(records[0])
    all_features = {
        name: frame.loc[:, selected].to_numpy(dtype=float)
        for name, selected in columns.items()
    }
    bundles: dict[str, RegisteredTaskBundle] = {}
    eligibility: dict[str, Any] = {}
    for task in SAM_COLUMNS:
        values = rating_frame[task]
        missing = values.isna()
        midpoint = values.eq(SAM_MIDPOINT)
        binary = values.map(lambda value: None if pd.isna(value) else sam_binary_label(value))
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
        source_indices = np.flatnonzero(keep.to_numpy())[eligible_mask]
        bundles[task] = RegisteredTaskBundle(
            frame=task_frame.loc[eligible_mask].reset_index(drop=True),
            labels=task_labels[eligible_mask],
            feature_sets={name: values[source_indices] for name, values in all_features.items()},
        )
        eligibility[task] = {
            "eligible_subjects": int(len(eligible_subjects)),
            "eligible_trials": int(eligible_mask.sum()),
            "missing_rating_trials_excluded": int(missing.sum()),
            "midpoint_trials_excluded": int(midpoint.sum()),
            "total_nonbinary_label_trials_excluded": int((~keep).sum()),
        }
    exclusion_counts = Counter(item["reason"] for item in exclusions)
    return bundles, {
        "adapter_profile": asdict(LOCKED_PROFILE),
        "archive_sha256": hash_file(archive),
        "schema_receipt_sha256": hash_file(schema_receipt),
        "central_manifest_sha256": receipt["central_manifest_sha256"],
        "questionnaire_participants_attempted": int(len(ratings)),
        "constructed_trials_before_task_filter": int(len(records)),
        "eligibility": eligibility,
        "quality_summary": {
            "retained_trial_count": int(len(quality_rows)),
            "minimum_samples": int(min(row["samples"] for row in quality_rows)),
            "maximum_samples": int(max(row["samples"] for row in quality_rows)),
            "key_moment_windows_per_trial": sorted(
                {int(row["key_moment_windows"]) for row in quality_rows}
            ),
        },
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "label_rule": {
            "high_minimum": SAM_HIGH_MINIMUM,
            "low_maximum": SAM_LOW_MAXIMUM,
            "excluded_midpoint": SAM_MIDPOINT,
        },
        "claim_boundary": (
            "Adapter provenance and structural eligibility are not effectiveness, "
            "non-harm, safety, or replicated external-confirmation evidence."
        ),
    }


__all__ = [
    "EKMAdapterProfile",
    "EKMSchemaError",
    "EKMTrialExclusion",
    "LOCKED_PROFILE",
    "load_ekmed_task_bundles",
    "parse_questionnaire_frame",
    "parse_signal_frame",
]
