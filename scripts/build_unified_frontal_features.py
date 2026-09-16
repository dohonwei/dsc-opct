from __future__ import annotations

import argparse
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend
from tqdm.auto import tqdm


BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified raw FP1/FP2 trial feature contract.")
    parser.add_argument(
        "--eppvr-root", type=Path, default=Path(r"E:\AA发表论文的数据\dataset\EPPVR")
    )
    parser.add_argument(
        "--deap-root", type=Path, default=Path(r"E:\AA发表论文的数据\dataset\DEAP")
    )
    parser.add_argument(
        "--hci-root",
        type=Path,
        default=Path(r"E:\AA发表论文的数据\dataset\hci-tagging-database"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/unified_frontal_features")
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["EPPVR", "DEAP", "MAHNOB-HCI"],
        choices=["EPPVR", "DEAP", "MAHNOB-HCI"],
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=32.0,
        help="Analyze a centered fixed-duration stimulus segment in every trial.",
    )
    return parser.parse_args()


def window_features(pair: np.ndarray, sample_rate: int) -> np.ndarray:
    window_size = 4 * sample_rate
    step_size = 2 * sample_rate
    starts = np.arange(0, pair.shape[1] - window_size + 1, step_size)
    if starts.size == 0:
        raise ValueError(f"Signal has only {pair.shape[1]} samples at {sample_rate} Hz")
    windows = np.stack([pair[:, start : start + window_size] for start in starts], axis=0)
    windows = detrend(windows, axis=-1, type="linear")
    taper = np.hanning(window_size)
    spectrum = np.fft.rfft(windows * taper, axis=-1)
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)
    band_power = []
    for low, high in BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        band_power.append(power[:, :, mask].mean(axis=-1))
    band_power_array = np.stack(band_power, axis=-1)
    band_power_array = np.maximum(band_power_array, 1e-12)
    total = band_power_array.sum(axis=-1, keepdims=True)
    log_power = np.log10(band_power_array)
    relative_power = band_power_array / total
    log_difference = log_power[:, 0] - log_power[:, 1]
    asymmetry = (band_power_array[:, 0] - band_power_array[:, 1]) / (
        band_power_array[:, 0] + band_power_array[:, 1]
    )
    return np.concatenate(
        [
            log_power.reshape(len(starts), -1),
            relative_power.reshape(len(starts), -1),
            log_difference,
            asymmetry,
        ],
        axis=1,
    ).astype(np.float32)


def centered_segment(pair: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    target = int(round(seconds * sample_rate))
    if pair.shape[1] < target:
        raise ValueError(
            f"Signal has {pair.shape[1] / sample_rate:.3f} s, shorter than requested {seconds:.3f} s"
        )
    start = (pair.shape[1] - target) // 2
    return pair[:, start : start + target]


def feature_names() -> list[str]:
    names = []
    for kind in ("log_power", "relative_power"):
        for channel in ("FP1", "FP2"):
            names.extend(f"{kind}_{channel}_{band}" for band in BANDS)
    names.extend(f"log_difference_{band}" for band in BANDS)
    names.extend(f"normalized_asymmetry_{band}" for band in BANDS)
    return names


def summarize_trial(values: np.ndarray, prefix: str = "stimulus") -> dict[str, float]:
    time = np.linspace(-1.0, 1.0, len(values), dtype=float)
    denominator = float(np.sum(time**2))
    slopes = time @ values / denominator if denominator > 0 else np.zeros(values.shape[1])
    row = {}
    for index, name in enumerate(feature_names()):
        row[f"{prefix}__{name}_mean"] = float(np.mean(values[:, index]))
        row[f"{prefix}__{name}_std"] = float(np.std(values[:, index]))
        row[f"{prefix}__{name}_slope"] = float(slopes[index])
    return row


def baseline_delta_features(stimulus: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    output = {}
    for stimulus_name, stimulus_value in stimulus.items():
        suffix = stimulus_name.removeprefix("stimulus__")
        baseline_name = f"baseline__{suffix}"
        output[f"delta__{suffix}"] = stimulus_value - baseline[baseline_name]
    return output


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def build_eppvr(root: Path, segment_seconds: float) -> pd.DataFrame:
    rows = []
    subject_paths = sorted(root.glob("s*.dat"))
    if len(subject_paths) != 30:
        raise ValueError(f"Expected 30 EPPVR files, found {len(subject_paths)}")
    for subject_path in tqdm(subject_paths, desc="Unified EPPVR features", unit="subject"):
        payload = load_pickle(subject_path)
        data = np.asarray(payload["data"], dtype=float)
        labels = np.asarray(payload["label"], dtype=float)
        if data.shape != (14, 10, 7000) or labels.shape != (14, 3):
            raise ValueError(f"Unexpected EPPVR shape in {subject_path}: {data.shape}, {labels.shape}")
        for trial_index in range(14):
            baseline_signal = data[trial_index, :2, :1000]
            signal = centered_segment(data[trial_index, :2, 1000:], 100, segment_seconds)
            baseline_values = window_features(baseline_signal, 100)
            values = window_features(signal, 100)
            row = {
                "subject_id": subject_path.stem,
                "trial_id": trial_index + 1,
                "valence_score": float(labels[trial_index, 0]),
                "arousal_score": float(labels[trial_index, 1]),
                "analyzed_seconds": segment_seconds,
                "n_windows": len(values),
            }
            stimulus_summary = summarize_trial(values)
            baseline_summary = summarize_trial(baseline_values, prefix="baseline")
            row.update(stimulus_summary)
            row.update(baseline_summary)
            row.update(baseline_delta_features(stimulus_summary, baseline_summary))
            rows.append(row)
    frame = pd.DataFrame(rows)
    expected_windows = int(np.floor((segment_seconds - 4) / 2) + 1)
    if len(frame) != 420 or frame.n_windows.nunique() != 1 or frame.n_windows.iloc[0] != expected_windows:
        raise ValueError("EPPVR unified feature contract failed")
    return frame


def build_deap(root: Path, segment_seconds: float) -> pd.DataFrame:
    rows = []
    subject_paths = sorted(root.glob("s*.dat"))
    if len(subject_paths) != 32:
        raise ValueError(f"Expected 32 DEAP files, found {len(subject_paths)}")
    for subject_index, subject_path in enumerate(
        tqdm(subject_paths, desc="Unified DEAP features", unit="subject"), start=1
    ):
        payload = load_pickle(subject_path)
        data = np.asarray(payload["data"], dtype=float)
        labels = np.asarray(payload["labels"], dtype=float)
        if data.shape != (40, 40, 8064) or labels.shape != (40, 4):
            raise ValueError(f"Unexpected DEAP shape in {subject_path}: {data.shape}, {labels.shape}")
        for trial_index in range(40):
            signal = centered_segment(data[trial_index, [0, 16], 384:], 128, segment_seconds)
            values = window_features(signal, 128)
            row = {
                "subject_id": subject_index,
                "trial_id": trial_index,
                "valence_score": float(labels[trial_index, 0]),
                "arousal_score": float(labels[trial_index, 1]),
                "analyzed_seconds": segment_seconds,
                "n_windows": len(values),
            }
            row.update(summarize_trial(values))
            rows.append(row)
    frame = pd.DataFrame(rows)
    expected_windows = int(np.floor((segment_seconds - 4) / 2) + 1)
    if len(frame) != 1280 or frame.n_windows.nunique() != 1 or frame.n_windows.iloc[0] != expected_windows:
        raise ValueError("DEAP unified feature contract failed")
    return frame


def parse_bdf_header(path: Path) -> dict:
    with path.open("rb") as handle:
        fixed = handle.read(256)
        n_signals = int(fixed[252:256].decode().strip())
        header_bytes = int(fixed[184:192].decode().strip())
        n_records = int(fixed[236:244].decode().strip())
        record_seconds = float(fixed[244:252].decode().strip())
        signal_header = handle.read(header_bytes - 256)
    position = 0
    widths = (16, 80, 8, 8, 8, 8, 8, 80, 8, 32)
    fields = []
    for width in widths:
        fields.append(
            [
                signal_header[position + index * width : position + (index + 1) * width]
                .decode("latin1")
                .strip()
                for index in range(n_signals)
            ]
        )
        position += width * n_signals
    return {
        "header_bytes": header_bytes,
        "n_records": n_records,
        "record_seconds": record_seconds,
        "labels": fields[0],
        "physical_min": np.asarray(fields[3], dtype=float),
        "physical_max": np.asarray(fields[4], dtype=float),
        "digital_min": np.asarray(fields[5], dtype=float),
        "digital_max": np.asarray(fields[6], dtype=float),
        "samples_per_record": np.asarray(fields[8], dtype=int),
    }


def decode_24bit(values: np.ndarray) -> np.ndarray:
    decoded = (
        values[..., 0].astype(np.int32)
        + (values[..., 1].astype(np.int32) << 8)
        + (values[..., 2].astype(np.int32) << 16)
    )
    return np.where(decoded & 0x800000, decoded - 0x1000000, decoded)


def read_bdf_fp_pair(path: Path) -> tuple[np.ndarray, int, float, float]:
    header = parse_bdf_header(path)
    labels = [label.upper() for label in header["labels"]]
    indices = [labels.index("FP1"), labels.index("FP2")]
    status_index = labels.index("STATUS")
    samples_per_record = header["samples_per_record"]
    if samples_per_record[indices[0]] != samples_per_record[indices[1]]:
        raise ValueError(f"FP1/FP2 sampling rates differ in {path}")
    if samples_per_record[status_index] != samples_per_record[indices[0]]:
        raise ValueError(f"EEG and status sampling rates differ in {path}")
    sample_rate_float = samples_per_record[indices[0]] / header["record_seconds"]
    sample_rate = int(round(sample_rate_float))
    if not np.isclose(sample_rate_float, sample_rate):
        raise ValueError(f"Non-integer FP sample rate in {path}: {sample_rate_float}")
    bytes_per_record = int(samples_per_record.sum() * 3)
    memory = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=header["header_bytes"],
        shape=(header["n_records"], bytes_per_record),
    )
    cumulative_samples = np.concatenate([[0], np.cumsum(samples_per_record)])

    def digital_channel(index: int) -> np.ndarray:
        start = int(cumulative_samples[index] * 3)
        stop = int(cumulative_samples[index + 1] * 3)
        raw = np.asarray(memory[:, start:stop]).reshape(
            header["n_records"], samples_per_record[index], 3
        )
        return decode_24bit(raw).reshape(-1)

    status = digital_channel(status_index)
    status_values, status_counts = np.unique(status, return_counts=True)
    baseline_status = status_values[np.argmax(status_counts)]
    event_mask = status != baseline_status
    event_starts = np.flatnonzero(event_mask & np.concatenate([[True], ~event_mask[:-1]]))
    if len(event_starts) != 2:
        raise ValueError(f"Expected two MAHNOB-HCI movie triggers in {path}, found {len(event_starts)}")
    stimulus_start, stimulus_stop = event_starts
    if stimulus_stop <= stimulus_start:
        raise ValueError(f"Invalid MAHNOB-HCI stimulus trigger order in {path}")

    channels = []
    for index in indices:
        digital = digital_channel(index).astype(float)
        physical = (
            (digital - header["digital_min"][index])
            * (header["physical_max"][index] - header["physical_min"][index])
            / (header["digital_max"][index] - header["digital_min"][index])
            + header["physical_min"][index]
        )
        channels.append(physical[stimulus_start:stimulus_stop])
    del memory
    recording_seconds = len(status) / sample_rate
    stimulus_seconds = (stimulus_stop - stimulus_start) / sample_rate
    return np.stack(channels), sample_rate, recording_seconds, stimulus_seconds


def load_hci_metadata(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "Sessions").glob("*/session.xml"), key=lambda item: int(item.parent.name)):
        xml = ET.parse(path).getroot()
        subject = xml.find("subject")
        if subject is None:
            raise ValueError(f"Missing subject in {path}")
        attributes = xml.attrib
        bdf_files = list(path.parent.glob("*.bdf"))
        if len(bdf_files) != 1:
            raise ValueError(f"Expected one BDF under {path.parent}, found {len(bdf_files)}")
        rows.append(
            {
                "subject_id": int(subject.attrib["id"]),
                "session_id": int(attributes["sessionId"]),
                "cut_number": int(attributes["cutNr"]),
                "cut_length_seconds": float(attributes["cutLenSec"]),
                "stimulus_name": attributes["mediaFile"].strip().lower(),
                "valence_score": float(attributes["feltVlnc"]),
                "arousal_score": float(attributes["feltArsl"]),
                "bdf_path": bdf_files[0],
            }
        )
    frame = pd.DataFrame(rows)
    stimulus_map = {name: index for index, name in enumerate(sorted(frame.stimulus_name.unique()))}
    frame["trial_id"] = frame.stimulus_name.map(stimulus_map).astype(int)
    return frame


def build_hci(root: Path, segment_seconds: float) -> pd.DataFrame:
    metadata = load_hci_metadata(root)
    rows = []
    for item in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Unified MAHNOB-HCI features", unit="session"):
        pair, sample_rate, bdf_seconds, trigger_seconds = read_bdf_fp_pair(Path(item.bdf_path))
        if sample_rate != 256:
            raise ValueError(f"Expected 256 Hz MAHNOB-HCI EEG, found {sample_rate} in {item.bdf_path}")
        pair = centered_segment(pair, sample_rate, segment_seconds)
        values = window_features(pair, sample_rate)
        row = {
            "subject_id": item.subject_id,
            "trial_id": item.trial_id,
            "stimulus_name": item.stimulus_name,
            "session_id": item.session_id,
            "valence_score": item.valence_score,
            "arousal_score": item.arousal_score,
            "xml_cut_seconds": item.cut_length_seconds,
            "bdf_record_seconds": bdf_seconds,
            "trigger_stimulus_seconds": trigger_seconds,
            "analyzed_seconds": segment_seconds,
            "n_windows": len(values),
        }
        row.update(summarize_trial(values))
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["subject_id", "trial_id"])
    if len(frame) != 527 or frame.subject_id.nunique() != 27 or frame.trial_id.nunique() != 20:
        raise ValueError("MAHNOB-HCI unified feature contract failed")
    expected_windows = int(np.floor((segment_seconds - 4) / 2) + 1)
    if frame.n_windows.nunique() != 1 or frame.n_windows.iloc[0] != expected_windows:
        raise ValueError("MAHNOB-HCI fixed-duration window contract failed")
    return frame


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    builders = {
        "EPPVR": lambda: build_eppvr(args.eppvr_root, args.segment_seconds),
        "DEAP": lambda: build_deap(args.deap_root, args.segment_seconds),
        "MAHNOB-HCI": lambda: build_hci(args.hci_root, args.segment_seconds),
    }
    filenames = {"EPPVR": "eppvr.csv", "DEAP": "deap.csv", "MAHNOB-HCI": "mahnob-hci.csv"}
    for dataset in args.datasets:
        frame = builders[dataset]()
        output = args.output_root / filenames[dataset]
        frame.to_csv(output, index=False)
        print(
            f"Saved {dataset}: {len(frame)} trials, {frame.subject_id.nunique()} subjects, "
            f"{frame.trial_id.nunique()} repeated units, "
            f"{sum(column.startswith(('stimulus__', 'baseline__', 'delta__')) for column in frame)} features "
            f"-> {output.resolve()}"
        )
    manifest = {
        "datasets": args.datasets,
        "segment_seconds": args.segment_seconds,
        "window_seconds": 4,
        "step_seconds": 2,
        "bands_hz": BANDS,
        "eppvr_baseline": "Per-trial first 10 seconds; baseline__ stores summaries and delta__ stores stimulus minus baseline summaries.",
        "deap_source": "preprocessed 128-Hz trial arrays after removal of the 3-s baseline",
        "mahnob_hci_source": "raw 256-Hz BDF FP1/FP2 cropped between the two Status-channel movie triggers",
    }
    (args.output_root / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
