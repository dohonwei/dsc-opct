from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import mne
import numpy as np
import pandas as pd
from scipy.signal import detrend, welch
from tqdm.auto import tqdm

from faced_v11_contract import (
    BANDS_HZ,
    BILATERAL_PAIRS,
    DEFAULT_DATA_ROOT,
    DEFAULT_FEATURE_ROOT,
    MINIMUM_PARTICIPANTS,
    NON_NEUTRAL_VIDEOS,
    REQUIRED_CHANNELS,
    REPRESENTATIONS,
    VIDEO_EMOTION,
    expected_polarity,
    normalized_channel,
    sha256,
    verify_implementation_lock,
)


BAD_UNIT_FIELD = b"?V      "
GOOD_UNIT_FIELD = b"uV      "
WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0
WELCH_SECONDS = 2.0
MINIMUM_VALID_WINDOWS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one FACED EEG feature row per participant and physical video."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--temp-root", type=Path, default=None)
    return parser.parse_args()


def bdf_with_fixed_units(path: Path, temp_root: Path) -> tuple[Path, bool]:
    with path.open("rb") as handle:
        general = handle.read(256)
        if len(general) != 256:
            raise ValueError("BDF general header is incomplete")
        n_signals = int(general[252:256].strip())
        physical_dimension_start = 256 + n_signals * (16 + 80)
        handle.seek(physical_dimension_start)
        block = handle.read(n_signals * 8)
    if BAD_UNIT_FIELD not in block:
        return path, False
    fixed = b"".join(
        GOOD_UNIT_FIELD
        if block[index : index + 8] == BAD_UNIT_FIELD
        else block[index : index + 8]
        for index in range(0, len(block), 8)
    )
    temp_root.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="faced_fixed_unit_", suffix=".bdf", dir=temp_root
    )
    os.close(descriptor)
    temporary = Path(name)
    shutil.copyfile(path, temporary)
    with temporary.open("r+b") as handle:
        handle.seek(physical_dimension_start)
        handle.write(fixed)
    return temporary, True


def trajectory(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    time = np.linspace(-1.0, 1.0, values.shape[0])
    centered = time - time.mean()
    denominator = float(np.sum(centered**2))
    slope = float(centered @ values / max(denominator, 1e-12))
    return float(values.mean()), float(values.std()), slope


def add_trajectory(row: dict[str, object], prefix: str, values: np.ndarray) -> None:
    mean, standard_deviation, slope = trajectory(values)
    row[f"stimulus__{prefix}_mean"] = mean
    row[f"stimulus__{prefix}_std"] = standard_deviation
    row[f"stimulus__{prefix}_slope"] = slope


def window_band_features(
    data_uv: np.ndarray, sampling_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    window_samples = int(round(WINDOW_SECONDS * sampling_rate))
    step_samples = int(round(STEP_SECONDS * sampling_rate))
    if data_uv.shape[1] < window_samples:
        raise ValueError("Video segment is shorter than one registered analysis window")
    starts = range(0, data_uv.shape[1] - window_samples + 1, step_samples)
    absolute_rows = []
    for start in starts:
        window = data_uv[:, start : start + window_samples]
        if not np.isfinite(window).all():
            continue
        window = detrend(window, axis=1, type="linear")
        if np.any(np.var(window, axis=1) <= 1e-12):
            continue
        nperseg = min(window_samples, int(round(WELCH_SECONDS * sampling_rate)))
        frequencies, density = welch(
            window,
            fs=sampling_rate,
            axis=1,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend=False,
            scaling="density",
        )
        band_values = []
        for low, high in BANDS_HZ.values():
            selected = (frequencies >= low) & (frequencies <= high)
            if selected.sum() < 2:
                raise ValueError(
                    f"Insufficient Welch bins for registered band {low}-{high} Hz"
                )
            band_values.append(
                np.trapz(density[:, selected], frequencies[selected], axis=1)
            )
        absolute_rows.append(np.stack(band_values, axis=1))
    if len(absolute_rows) < MINIMUM_VALID_WINDOWS:
        raise ValueError(
            f"Only {len(absolute_rows)} valid windows; minimum is {MINIMUM_VALID_WINDOWS}"
        )
    absolute = np.stack(absolute_rows, axis=0)
    relative = absolute / np.maximum(absolute.sum(axis=2, keepdims=True), 1e-20)
    return absolute, relative


def trial_features(
    data_uv: np.ndarray,
    sampling_rate: float,
    subject_id: str,
    video_index: int,
) -> dict[str, object]:
    data_uv = data_uv - data_uv.mean(axis=0, keepdims=True)
    absolute, relative = window_band_features(data_uv, sampling_rate)
    log_power = np.log10(np.maximum(absolute, 1e-20))
    row: dict[str, object] = {
        "subject_id": subject_id,
        "trial_id": int(video_index),
        "video_index": int(video_index),
        "emotion_label": VIDEO_EMOTION[int(video_index)],
        "polarity_label": expected_polarity(video_index),
        "polarity_score": 9.0 if expected_polarity(video_index) == "positive" else 1.0,
        "sampling_rate_hz": float(sampling_rate),
        "n_windows": int(absolute.shape[0]),
    }
    channel_lookup = {name: index for index, name in enumerate(REQUIRED_CHANNELS)}
    for band_index, band in enumerate(BANDS_HZ):
        for channel_index, channel in enumerate(REQUIRED_CHANNELS):
            add_trajectory(
                row,
                f"log_power_{channel}_{band}",
                log_power[:, channel_index, band_index],
            )
            add_trajectory(
                row,
                f"relative_power_{channel}_{band}",
                relative[:, channel_index, band_index],
            )
        for left, right in BILATERAL_PAIRS:
            left_index = channel_lookup[left]
            right_index = channel_lookup[right]
            pair = f"{left}_{right}"
            add_trajectory(
                row,
                f"log_difference_{pair}_{band}",
                log_power[:, left_index, band_index]
                - log_power[:, right_index, band_index],
            )
            denominator = np.maximum(
                np.abs(relative[:, left_index, band_index])
                + np.abs(relative[:, right_index, band_index]),
                1e-20,
            )
            add_trajectory(
                row,
                f"normalized_asymmetry_{pair}_{band}",
                (
                    relative[:, left_index, band_index]
                    - relative[:, right_index, band_index]
                )
                / denominator,
            )
    feature_values = np.asarray(
        [value for key, value in row.items() if key.startswith("stimulus__")],
        dtype=float,
    )
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite trial-level feature")
    return row


def event_table(path: Path) -> pd.DataFrame:
    required = {"onset", "duration", "video_index", "emotion_label", "binary_label"}
    events = pd.read_csv(path, sep="\t")
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing registered event columns: {sorted(missing)}")
    numeric_video = pd.to_numeric(events.video_index, errors="coerce")
    videos = events.loc[numeric_video.between(1, 28)].copy()
    videos["video_index"] = numeric_video.loc[videos.index].astype(int)
    if len(videos) != 28 or videos.video_index.nunique() != 28:
        raise ValueError(
            "Participant does not have exactly one event for each physical video 1-28"
        )
    if set(videos.video_index) != set(range(1, 29)):
        raise ValueError("Participant physical-video map is incomplete")
    for item in videos.itertuples(index=False):
        expected_emotion = VIDEO_EMOTION[int(item.video_index)]
        if (
            int(item.video_index) in NON_NEUTRAL_VIDEOS
            and str(item.emotion_label).strip().lower() != expected_emotion
        ):
            raise ValueError(
                f"Event emotion mismatch for video {item.video_index}: {item.emotion_label}"
            )
        if str(item.binary_label).strip().lower() != expected_polarity(
            int(item.video_index)
        ):
            raise ValueError(
                f"Event polarity mismatch for video {item.video_index}: {item.binary_label}"
            )
        if not 20.0 <= float(item.duration) <= 180.0:
            raise ValueError(
                f"Implausible video duration for video {item.video_index}: {item.duration}"
            )
    return videos.sort_values("video_index")


def subject_paths(data_root: Path) -> list[tuple[str, Path, Path, Path]]:
    output = []
    for bdf in sorted(data_root.glob("sub-*/eeg/*_eeg.bdf")):
        subject_id = bdf.parts[-3]
        stem = bdf.name.removesuffix("_eeg.bdf")
        output.append(
            (
                subject_id,
                bdf,
                bdf.with_name(stem + "_events.tsv"),
                bdf.with_name(stem + "_channels.tsv"),
            )
        )
    return output


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite FACED features: {args.output_root}"
        )
    lock = verify_implementation_lock()
    receipt_path = args.data_root / "faced_v11_acquisition_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("Verified FACED acquisition receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete_release_downloaded_and_verified":
        raise RuntimeError("FACED acquisition receipt is incomplete")
    subjects = subject_paths(args.data_root)
    if len(subjects) != 123:
        raise ValueError(f"Expected 123 FACED BDF files, found {len(subjects)}")

    temp_root = args.temp_root or (args.output_root.parent / "faced_v11_bdf_temp")
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    unit_patched = 0
    progress = tqdm(
        subjects,
        desc="FACED participant-video features",
        unit="participant",
        dynamic_ncols=True,
    )
    for subject_id, bdf_path, events_path, channels_path in progress:
        temporary = None
        try:
            for required_path in (events_path, channels_path):
                if not required_path.is_file():
                    raise FileNotFoundError(required_path)
            events = event_table(events_path)
            channel_table = pd.read_csv(channels_path, sep="\t")
            if "name" not in channel_table:
                raise ValueError("channels.tsv lacks the BIDS name column")
            channel_names = tuple(
                normalized_channel(name) for name in channel_table.name
            )
            if set(channel_names) != set(REQUIRED_CHANNELS) or len(channel_names) != 32:
                raise ValueError(
                    "channels.tsv does not match the registered 32-channel montage"
                )

            readable_path, patched = bdf_with_fixed_units(bdf_path, temp_root)
            temporary = readable_path if patched else None
            unit_patched += int(patched)
            raw = mne.io.read_raw_bdf(readable_path, preload=False, verbose="ERROR")
            sampling_rate = float(raw.info["sfreq"])
            if sampling_rate not in (250.0, 1000.0):
                raise ValueError(f"Unexpected sampling rate: {sampling_rate}")
            raw_map = {normalized_channel(name): name for name in raw.ch_names}
            missing_channels = set(REQUIRED_CHANNELS) - set(raw_map)
            if missing_channels:
                raise ValueError(
                    f"BDF is missing registered channels: {sorted(missing_channels)}"
                )
            picks = [raw.ch_names.index(raw_map[name]) for name in REQUIRED_CHANNELS]
            subject_rows = []
            for event in events.itertuples(index=False):
                video_index = int(event.video_index)
                if video_index not in NON_NEUTRAL_VIDEOS:
                    continue
                start = max(0, int(np.floor(float(event.onset) * sampling_rate)))
                stop = min(
                    raw.n_times,
                    int(
                        np.ceil(
                            (float(event.onset) + float(event.duration)) * sampling_rate
                        )
                    ),
                )
                if stop <= start:
                    raise ValueError(f"Empty BDF interval for video {video_index}")
                data_uv = raw.get_data(picks=picks, start=start, stop=stop) * 1e6
                subject_rows.append(
                    trial_features(data_uv, sampling_rate, subject_id, video_index)
                )
            if len(subject_rows) != len(NON_NEUTRAL_VIDEOS):
                raise ValueError(
                    "Participant did not produce all 24 registered non-neutral trials"
                )
            rows.extend(subject_rows)
            progress.set_postfix(
                usable=len({row["subject_id"] for row in rows}),
                failed=len(failures),
            )
        except Exception as error:
            failures.append(
                {
                    "subject_id": subject_id,
                    "bdf_path": str(bdf_path.relative_to(args.data_root)),
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    frame = pd.DataFrame(rows)
    usable_participants = frame.subject_id.nunique() if not frame.empty else 0
    if usable_participants < MINIMUM_PARTICIPANTS:
        args.output_root.mkdir(parents=True, exist_ok=False)
        pd.DataFrame(failures).to_csv(
            args.output_root / "failed_participants.csv", index=False
        )
        failure = {
            "status": "hard_failure_minimum_participants_not_met",
            "usable_participants": int(usable_participants),
            "minimum_participants": MINIMUM_PARTICIPANTS,
            "failed_participants": len(failures),
        }
        (args.output_root / "build_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise RuntimeError(json.dumps(failure))
    expected_rows = usable_participants * len(NON_NEUTRAL_VIDEOS)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} complete participant-video rows, found {len(frame)}"
        )
    for _, group in frame.groupby("subject_id"):
        if set(group.trial_id.astype(int)) != set(NON_NEUTRAL_VIDEOS):
            raise ValueError(
                "Retained participant has an incomplete physical-video map"
            )
    if frame.duplicated(["subject_id", "trial_id"]).any():
        raise ValueError("Duplicate participant-video rows")

    metadata = {
        "subject_id",
        "trial_id",
        "video_index",
        "emotion_label",
        "polarity_label",
        "polarity_score",
        "sampling_rate_hz",
        "n_windows",
    }
    feature_columns = [column for column in frame if column not in metadata]
    representation_columns = {
        "all": feature_columns,
        "relative_power": [
            column
            for column in feature_columns
            if column.startswith("stimulus__relative_power_")
        ],
        "normalized_asymmetry": [
            column
            for column in feature_columns
            if column.startswith("stimulus__normalized_asymmetry_")
        ],
    }
    if tuple(representation_columns) != REPRESENTATIONS or any(
        not columns for columns in representation_columns.values()
    ):
        raise ValueError("A registered FACED representation is unavailable")

    args.output_root.mkdir(parents=True, exist_ok=False)
    output_path = args.output_root / "faced_polarity_trial_features.csv"
    frame.sort_values(["subject_id", "trial_id"]).to_csv(output_path, index=False)
    pd.DataFrame(failures).to_csv(
        args.output_root / "failed_participants.csv", index=False
    )
    manifest = {
        "status": "faced_registered_trial_features_built",
        "dataset": "FACED",
        "source_version": "NEMAR nm000112 v1.1.3",
        "unit": "one participant by one physical video",
        "usable_participants": int(usable_participants),
        "failed_participants": len(failures),
        "rows": len(frame),
        "physical_videos": len(NON_NEUTRAL_VIDEOS),
        "class_counts": frame.polarity_score.value_counts().sort_index().to_dict(),
        "unit_headers_patched": unit_patched,
        "bands_hz": BANDS_HZ,
        "window_seconds": WINDOW_SECONDS,
        "step_seconds": STEP_SECONDS,
        "welch_seconds": WELCH_SECONDS,
        "reference": "within-sample common average across the registered 32 EEG channels",
        "neutral_videos_excluded": [13, 14, 15, 16],
        "representations": representation_columns,
        "feature_sha256": sha256(output_path),
        "acquisition_receipt_sha256": sha256(receipt_path),
        "implementation_lock_status": lock["status"],
    }
    (args.output_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    public_manifest = {
        key: value for key, value in manifest.items() if key != "representations"
    }
    print(json.dumps(public_manifest, indent=2))


if __name__ == "__main__":
    main()
