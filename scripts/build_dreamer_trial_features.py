from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend
from tqdm.auto import tqdm


EXPECTED_SHA256 = "35eed939e6f206a0938b13ba063178e2795ef1739b9cac8e5704a968393f3463"
ELECTRODES = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
CHANNELS = ("AF3", "AF4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build registered DREAMER AF3/AF4 features.")
    parser.add_argument("--mat-path", type=Path, default=Path("tmp/dreamer_original/DREAMER.mat"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dreamer_trial_features"))
    parser.add_argument("--segment-seconds", type=float, default=32.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unwrap(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.flat[0]
    return value


def scalar(value) -> int:
    values = np.asarray(unwrap(value)).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"Expected scalar, observed {values.shape}")
    return int(values[0])


def centered_pair(values: np.ndarray, seconds: float) -> np.ndarray:
    samples = int(round(128 * seconds))
    if values.shape[0] < samples or values.shape[1] != 14:
        raise ValueError(f"Invalid DREAMER EEG shape: {values.shape}")
    start = (values.shape[0] - samples) // 2
    indices = [ELECTRODES.index(channel) for channel in CHANNELS]
    return values[start : start + samples, indices].T


def feature_names() -> list[str]:
    names = []
    for kind in ("log_power", "relative_power"):
        for channel in CHANNELS:
            names.extend(f"{kind}_{channel}_{band}" for band in BANDS)
    names.extend(f"log_difference_{band}" for band in BANDS)
    names.extend(f"normalized_asymmetry_{band}" for band in BANDS)
    return names


def window_features(pair: np.ndarray) -> np.ndarray:
    window, step = 4 * 128, 2 * 128
    starts = np.arange(0, pair.shape[1] - window + 1, step)
    windows = np.stack([pair[:, start : start + window] for start in starts])
    windows = detrend(windows, axis=-1, type="linear")
    spectrum = np.fft.rfft(windows * np.hanning(window), axis=-1)
    power = np.square(np.abs(spectrum))
    frequency = np.fft.rfftfreq(window, d=1.0 / 128)
    band_power = np.stack(
        [power[:, :, (frequency >= low) & (frequency < high)].mean(axis=-1)
         for low, high in BANDS.values()],
        axis=-1,
    )
    band_power = np.maximum(band_power, 1e-12)
    log_power = np.log10(band_power)
    relative = band_power / band_power.sum(axis=-1, keepdims=True)
    difference = log_power[:, 0] - log_power[:, 1]
    asymmetry = (band_power[:, 0] - band_power[:, 1]) / (
        band_power[:, 0] + band_power[:, 1]
    )
    return np.concatenate(
        [log_power.reshape(len(starts), -1), relative.reshape(len(starts), -1), difference, asymmetry],
        axis=1,
    ).astype(np.float32)


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    time = np.linspace(-1.0, 1.0, len(values))
    slopes = time @ values / float(np.sum(time**2))
    row = {}
    for index, name in enumerate(feature_names()):
        row[f"{prefix}__{name}_mean"] = float(values[:, index].mean())
        row[f"{prefix}__{name}_std"] = float(values[:, index].std())
        row[f"{prefix}__{name}_slope"] = float(slopes[index])
    return row


def main() -> None:
    args = parse_args()
    if sha256(args.mat_path) != EXPECTED_SHA256:
        raise ValueError("DREAMER.mat hash differs from registration")
    root = loadmat(args.mat_path, variable_names=["DREAMER"], struct_as_record=True)["DREAMER"][0, 0]
    electrodes = tuple(str(unwrap(value).flat[0]) for value in unwrap(root["EEG_Electrodes"]).flat)
    if electrodes != ELECTRODES or scalar(root["EEG_SamplingRate"]) != 128:
        raise ValueError("DREAMER channel or sampling-rate contract failed")
    if scalar(root["noOfSubjects"]) != 23 or scalar(root["noOfVideoSequences"]) != 18:
        raise ValueError("DREAMER participant/video contract failed")
    subjects = unwrap(root["Data"])
    rows = []
    for subject_id, subject in enumerate(tqdm(subjects.flat, total=23, desc="DREAMER AF3/AF4 features", unit="subject"), start=1):
        eeg = unwrap(subject["EEG"])
        baseline, stimuli = unwrap(eeg["baseline"]), unwrap(eeg["stimuli"])
        scores = {
            "valence_score": np.asarray(unwrap(subject["ScoreValence"]), dtype=int).reshape(-1),
            "arousal_score": np.asarray(unwrap(subject["ScoreArousal"]), dtype=int).reshape(-1),
            "dominance_score": np.asarray(unwrap(subject["ScoreDominance"]), dtype=int).reshape(-1),
        }
        if any(len(values) != 18 for values in scores.values()) or len(baseline) != 18 or len(stimuli) != 18:
            raise ValueError(f"Subject {subject_id} lacks the complete 18-video grid")
        for trial in range(18):
            baseline_raw = np.asarray(unwrap(baseline.flat[trial]), dtype=float)
            stimulus_raw = np.asarray(unwrap(stimuli.flat[trial]), dtype=float)
            baseline_features = window_features(centered_pair(baseline_raw, args.segment_seconds))
            stimulus_features = window_features(centered_pair(stimulus_raw, args.segment_seconds))
            row: dict[str, object] = {
                "subject_id": subject_id,
                "trial_id": trial + 1,
                "analyzed_seconds": args.segment_seconds,
                "n_windows": len(stimulus_features),
                "baseline_samples": len(baseline_raw),
                "stimulus_samples": len(stimulus_raw),
            }
            row.update({name: int(values[trial]) for name, values in scores.items()})
            row.update(summarize(stimulus_features, "stimulus"))
            row.update(summarize(baseline_features, "baseline"))
            stimulus_names = [name for name in row if name.startswith("stimulus__")]
            for stimulus_name in stimulus_names:
                suffix = stimulus_name.removeprefix("stimulus__")
                row[f"delta__{suffix}"] = row[stimulus_name] - row[f"baseline__{suffix}"]
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["subject_id", "trial_id"])
    if len(frame) != 414 or frame.subject_id.nunique() != 23 or frame.trial_id.nunique() != 18:
        raise ValueError("DREAMER crossed trial grid is incomplete")
    if frame.n_windows.nunique() != 1 or int(frame.n_windows.iloc[0]) != 15:
        raise ValueError("DREAMER 32-second window contract failed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "dreamer_primary_pair.csv"
    frame.to_csv(output, index=False)
    manifest = {
        "status": "built_from_registered_source",
        "source_mat_sha256": EXPECTED_SHA256,
        "subjects": 23,
        "shared_physical_stimuli": 18,
        "channels": list(CHANNELS),
        "sampling_rate_hz": 128,
        "segment_seconds": args.segment_seconds,
        "window_seconds": 4,
        "step_seconds": 2,
        "label_contract": {
            "primary": "rating > 3 versus rating <= 3",
            "extreme_sensitivity": "rating > 3 versus rating < 3 after excluding rating 3",
        },
        "output": str(output),
        "output_sha256": sha256(output),
    }
    (args.output_root / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
