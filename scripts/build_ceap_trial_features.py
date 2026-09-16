from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
RESERVATION = ROOT / "docs/ceap_v6_external_confirmation_reservation.json"
RESERVATION_AMENDMENT = (
    ROOT / "docs/ceap_v6_external_confirmation_reservation_amendment_001.json"
)
IMPLEMENTATION_LOCK = (
    ROOT / "docs/ceap_v6_external_confirmation_implementation_lock_amendment_001.json"
)
SENSORS = ("EDA", "BVP", "SKT", "HR")
STATISTICS = ("mean", "std", "median", "iqr", "q05", "q95", "slope", "rms", "diff_std")
EXPECTED_VIDEO_SAMPLES = {1: 1500, **{video: 1800 for video in range(2, 9)}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build registered CEAP subject-video features.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:\AA发表论文的数据\dataset\CEAP-360VR\CEAP-360VR"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ceap_trial_features"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree(root: Path, included: list[str]) -> tuple[int, int, str]:
    files = sorted(
        (path for relative in included for path in (root / relative).glob("*") if path.is_file()),
        key=lambda path: str(path).lower(),
    )
    records = [
        f"{path.relative_to(root.parent)}\t{path.stat().st_size}\t{sha256(path)}"
        for path in files
    ]
    digest = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    return len(files), sum(path.stat().st_size for path in files), digest


def verify_reservation(
    root: Path,
    reservation: dict[str, object],
    amendment: dict[str, object],
) -> None:
    source = reservation["source"]
    count, size, digest = source_tree(root, source["included_directories"])
    observed = (count, size, digest)
    expected = (
        amendment["source_file_count"],
        amendment["source_total_bytes"],
        amendment["canonical_tree_sha256"],
    )
    if observed != expected:
        raise RuntimeError(f"CEAP registered source mismatch: observed={observed}, expected={expected}")


def verify_implementation_lock() -> dict[str, object]:
    if not IMPLEMENTATION_LOCK.is_file():
        raise FileNotFoundError("CEAP implementation lock must exist before feature extraction")
    lock = json.loads(IMPLEMENTATION_LOCK.read_text(encoding="utf-8"))
    if lock["reservation_sha256"] != sha256(RESERVATION):
        raise RuntimeError("CEAP implementation lock does not match the reservation")
    if lock["reservation_amendment_sha256"] != sha256(RESERVATION_AMENDMENT):
        raise RuntimeError("CEAP implementation lock does not match the reservation amendment")
    for relative, expected in lock["analysis_code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Registered CEAP analysis code changed: {relative}")
    return lock


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("CEAP video signal is missing, too short, or non-finite")
    centered_time = np.linspace(-1.0, 1.0, len(values))
    slope = float(
        np.dot(centered_time, values - values.mean())
        / np.dot(centered_time, centered_time)
    )
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


def sensor_values(video: dict[str, object], sensor: str, video_index: int) -> np.ndarray:
    samples = video[f"{sensor}_FrameData"]
    values = np.asarray([sample[sensor] for sample in samples], dtype=np.float64)
    expected = EXPECTED_VIDEO_SAMPLES[video_index]
    if len(values) != expected:
        raise ValueError(
            f"CEAP {sensor} video {video_index} length is {len(values)}, expected {expected}"
        )
    return values


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite CEAP features: {args.output_root}")
    reservation = json.loads(RESERVATION.read_text(encoding="utf-8"))
    amendment = json.loads(RESERVATION_AMENDMENT.read_text(encoding="utf-8"))
    verify_implementation_lock()
    if amendment["reservation_sha256"] != sha256(RESERVATION):
        raise RuntimeError("CEAP hash amendment does not match the original reservation")
    verify_reservation(args.data_root, reservation, amendment)
    rows = []
    progress = tqdm(range(1, 33), desc="CEAP trial features", unit="subject", dynamic_ncols=True)
    for subject in progress:
        physio_path = args.data_root / "5_PhysioData/Frame" / f"P{subject}_Physio_FrameData.json"
        questionnaire_path = (
            args.data_root / "2_QuestionnaireData" / f"P{subject}_Questionnaire_Data.json"
        )
        physiology = json.loads(physio_path.read_text(encoding="utf-8"))
        questionnaire = json.loads(questionnaire_path.read_text(encoding="utf-8"))
        videos = physiology["Physio_FrameData"][0]["Video_Physio_FrameData"]
        ratings = questionnaire["QuestionnaireData"][0]["Video_SAMRating_VideoTime_Data"]
        if len(videos) != 8 or len(ratings) != 8:
            raise ValueError(f"CEAP participant {subject} violates the registered eight-video contract")
        for video_index, (video, rating) in enumerate(zip(videos, ratings, strict=True), start=1):
            row = {
                "subject_id": str(subject),
                "trial_id": str(video_index),
                "valence_score": float(rating["ValenceValue"]),
                "arousal_score": float(rating["ArousalValue"]),
            }
            lengths = []
            for sensor in SENSORS:
                values = sensor_values(video, sensor, video_index)
                lengths.append(len(values))
                for statistic, value in summarize(values).items():
                    row[f"feature__{sensor.lower()}_{statistic}"] = value
            if len(set(lengths)) != 1:
                raise ValueError(f"CEAP participant {subject}, video {video_index} is unsynchronized")
            row["n_samples"] = lengths[0]
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["subject_id", "trial_id"]).reset_index(drop=True)
    if len(frame) != 256 or frame.subject_id.nunique() != 32 or frame.trial_id.nunique() != 8:
        raise ValueError("CEAP feature table violates the registered 32 x 8 contract")
    if frame.isna().any().any():
        raise ValueError("CEAP feature table contains missing values")
    for task in ("arousal", "valence"):
        labels = (frame[f"{task}_score"].to_numpy(float) >= 5.0).astype(int)
        if len(np.unique(labels)) != 2:
            raise ValueError(f"CEAP {task} is one-class under the registered threshold")

    args.output_root.mkdir(parents=True, exist_ok=False)
    output = args.output_root / "ceap_trial_features.csv"
    frame.to_csv(output, index=False)
    manifest = {
        "status": "built_under_registered_ceap_protocol",
        "dataset": "CEAP-360VR",
        "rows": len(frame),
        "subjects": int(frame.subject_id.nunique()),
        "physical_stimuli": int(frame.trial_id.nunique()),
        "features": len(SENSORS) * len(STATISTICS),
        "reservation_sha256": sha256(RESERVATION),
        "reservation_amendment_sha256": sha256(RESERVATION_AMENDMENT),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "output_sha256": sha256(output),
        "builder_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
