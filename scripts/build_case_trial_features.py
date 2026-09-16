from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


VIDEO_IDS = {
    "amusing-1": 1,
    "amusing-2": 2,
    "boring-1": 3,
    "boring-2": 4,
    "relaxed-1": 5,
    "relaxed-2": 6,
    "scary-1": 7,
    "scary-2": 8,
    "startVid": 10,
    "bluVid": 11,
    "endVid": 12,
}
SENSORS = ("ecg", "bvp", "gsr", "rsp", "skt", "emg_zygo", "emg_coru", "emg_trap")
STATISTICS = ("mean", "std", "median", "iqr", "q05", "q95", "slope", "rms", "diff_std")
REGISTRATION = Path("docs/case_v5_external_confirmation_registration.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build registered CASE subject-video features.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:\AA发表论文的数据\dataset\case_dataset-ver_SciData_0\case_dataset-ver_SciData_0"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/case_trial_features"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_metadata(root: Path) -> tuple[dict[str, float], pd.DataFrame]:
    duration_map = {}
    lines = (root / "metadata/videos_duration.txt").read_text("utf-8").splitlines()
    for line in lines[1:]:
        fields = [field.strip().strip('"') for field in line.split("\t") if field.strip()]
        if len(fields) != 2:
            raise ValueError(f"Malformed CASE duration row: {line!r}")
        duration_map[fields[0]] = float(fields[1])
    sequence = pd.read_csv(root / "metadata/seqs_order.txt", sep="\t", dtype=str)
    sequence.columns = [column.strip('"') for column in sequence.columns]
    return duration_map, sequence.apply(lambda column: column.str.strip('"'))


def registered_hashes() -> dict[str, str]:
    registration = json.loads(REGISTRATION.read_text("utf-8"))
    return {item["path"]: item["sha256"] for item in registration["source_files"]}


def verify_registered_source(root: Path, expected: dict[str, str]) -> None:
    progress = tqdm(expected.items(), desc="Verify CASE registration", unit="file", dynamic_ncols=True)
    for relative, digest in progress:
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"CASE registered source mismatch: {path}")


def calibrated_physiology(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_ms = raw[:, 0] * 1000.0
    values = np.empty((len(raw), 8), dtype=np.float64)
    values[:, 0] = (raw[:, 1] - 2.8) / 50.0 * 1000.0
    values[:, 1] = 58.962 * raw[:, 2] - 115.09
    values[:, 2] = 24.0 * raw[:, 3] - 49.2
    values[:, 3] = 58.923 * raw[:, 4] - 115.01
    values[:, 4] = 21.341 * raw[:, 5] - 32.085
    values[:, 5:] = (raw[:, 6:9] - 2.0) / 4000.0 * 1_000_000.0
    return time_ms, values


def calibrated_annotations(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time_ms = raw[:, 0] * 1000.0
    values = 0.5 + 9.0 * (raw[:, 1:3] + 26225.0) / 52450.0
    return time_ms, values


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("CASE segment is missing, too short, or non-finite")
    centered_time = np.linspace(-1.0, 1.0, len(values))
    slope = float(np.dot(centered_time, values - values.mean()) / np.dot(centered_time, centered_time))
    q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(median),
        "iqr": float(q75 - q25),
        "q05": float(q05),
        "q95": float(q95),
        "slope": slope,
        "rms": float(np.sqrt(np.mean(values**2))),
        "diff_std": float(np.std(np.diff(values))),
    }


def segment_plan(sequence: list[str], durations: dict[str, float]) -> list[dict[str, object]]:
    elapsed = 0.0
    plan = []
    previous = None
    for label in sequence:
        start = elapsed
        end = start + durations[label]
        if label in VIDEO_IDS and 1 <= VIDEO_IDS[label] <= 8:
            if previous is None or previous["label"] != "bluVid":
                raise ValueError(f"Emotion video {label} lacks its registered preceding baseline")
            plan.append(
                {
                    "video": VIDEO_IDS[label],
                    "label": label,
                    "start_ms": start,
                    "end_ms": end,
                    "baseline_start_ms": previous["start_ms"],
                    "baseline_end_ms": previous["end_ms"],
                }
            )
        previous = {"label": label, "start_ms": start, "end_ms": end}
        elapsed = end
    return plan


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite CASE features: {args.output_root}")
    verify_registered_source(args.data_root, registered_hashes())
    durations, sequences = load_metadata(args.data_root)
    rows = []
    progress = tqdm(range(1, 31), desc="CASE trial features", unit="subject", dynamic_ncols=True)
    for subject in progress:
        physiological_raw = np.loadtxt(
            args.data_root / f"data/raw/physiological/sub{subject}_DAQ.txt"
        )
        annotation_raw = np.loadtxt(
            args.data_root / f"data/raw/annotations/sub{subject}_joystick.txt"
        )
        physiological_time, physiological = calibrated_physiology(physiological_raw)
        annotation_time, annotations = calibrated_annotations(annotation_raw)
        plan = segment_plan(sequences[f"seq_sub{subject}"].tolist(), durations)
        if len(plan) != 8:
            raise ValueError(f"Subject {subject} has {len(plan)} emotion videos, expected 8")
        for segment in plan:
            stimulus_mask = (
                (physiological_time >= float(segment["start_ms"]) + 5000.0)
                & (physiological_time <= float(segment["end_ms"]))
            )
            baseline_mask = (
                (physiological_time >= float(segment["baseline_start_ms"]))
                & (physiological_time <= float(segment["baseline_end_ms"]))
            )
            annotation_mask = (
                (annotation_time >= float(segment["start_ms"]) + 5000.0)
                & (annotation_time <= float(segment["end_ms"]))
            )
            row = {
                "subject_id": str(subject),
                "trial_id": str(segment["video"]),
                "video_label": str(segment["label"]),
                "valence_score": float(np.median(annotations[annotation_mask, 0])),
                "arousal_score": float(np.median(annotations[annotation_mask, 1])),
                "n_stimulus_samples": int(stimulus_mask.sum()),
                "n_baseline_samples": int(baseline_mask.sum()),
                "n_annotation_samples": int(annotation_mask.sum()),
            }
            for sensor_index, sensor in enumerate(SENSORS):
                stimulus_stats = summarize(physiological[stimulus_mask, sensor_index])
                baseline_stats = summarize(physiological[baseline_mask, sensor_index])
                for statistic in STATISTICS:
                    row[f"stimulus__abs_{sensor}_{statistic}"] = stimulus_stats[statistic]
                    row[f"delta__{sensor}_{statistic}"] = (
                        stimulus_stats[statistic] - baseline_stats[statistic]
                    )
            rows.append(row)
        del physiological_raw, annotation_raw, physiological, annotations

    frame = pd.DataFrame(rows).sort_values(["subject_id", "trial_id"]).reset_index(drop=True)
    if len(frame) != 240 or frame.subject_id.nunique() != 30 or frame.trial_id.nunique() != 8:
        raise ValueError("CASE subject-video table violates the registered 30 x 8 contract")
    if frame[["valence_score", "arousal_score"]].isna().any().any():
        raise ValueError("CASE registered outcome table contains missing scores")
    args.output_root.mkdir(parents=True, exist_ok=False)
    output = args.output_root / "case_trial_features.csv"
    frame.to_csv(output, index=False)
    manifest = {
        "status": "built_under_registered_case_protocol",
        "dataset": "CASE",
        "rows": len(frame),
        "subjects": int(frame.subject_id.nunique()),
        "physical_stimuli": int(frame.trial_id.nunique()),
        "features": 2 * len(SENSORS) * len(STATISTICS),
        "registration_sha256": sha256(REGISTRATION),
        "output_sha256": sha256(output),
        "builder_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
