from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


WINDOW_SIZE = 1024
STEP_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover shared MAHNOB-HCI stimulus IDs and aggregate window features by trial."
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path(r"E:\AA发表论文的数据\tcgan\feature\hci"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\AA发表论文的数据\dataset\hci-tagging-database"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs/hci_trial_features/trial_features.csv"),
    )
    parser.add_argument(
        "--validate-source-lengths",
        action="store_true",
        help="Stream all source CSVs and verify their window counts. This can take several minutes.",
    )
    return parser.parse_args()


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    time = np.arange(values.shape[0], dtype=float)
    centered_time = time - time.mean()
    denominator = np.sum(centered_time**2)
    slopes = centered_time @ values / denominator if denominator > 0 else np.zeros(values.shape[1])
    row: dict[str, float] = {}
    for index in range(values.shape[1]):
        feature = values[:, index]
        name = f"{prefix}f{index:02d}"
        row[f"{name}_mean"] = float(np.mean(feature))
        row[f"{name}_std"] = float(np.std(feature))
        row[f"{name}_slope"] = float(slopes[index])
    return row


def load_session_metadata(dataset_root: Path) -> pd.DataFrame:
    rows = []
    session_paths = sorted(
        (dataset_root / "Sessions").glob("*/session.xml"), key=lambda path: int(path.parent.name)
    )
    for path in tqdm(session_paths, desc="Reading MAHNOB-HCI XML", unit="session"):
        root = ET.parse(path).getroot()
        attributes = root.attrib
        subject = root.find("subject")
        if subject is None:
            raise ValueError(f"Missing subject element in {path}")
        required = ("sessionId", "cutNr", "cutLenSec", "mediaFile", "feltVlnc", "feltArsl")
        missing = [name for name in required if name not in attributes]
        if missing:
            raise ValueError(f"Missing {missing} in {path}")
        rows.append(
            {
                "subject_id": int(subject.attrib["id"]),
                "session_id": int(attributes["sessionId"]),
                "cut_number": int(attributes["cutNr"]),
                "cut_length_seconds": float(attributes["cutLenSec"]),
                "stimulus_name": attributes["mediaFile"].strip().lower(),
                "valence_score": float(attributes["feltVlnc"]),
                "arousal_score": float(attributes["feltArsl"]),
            }
        )
    metadata = pd.DataFrame(rows).sort_values(["subject_id", "cut_number"])
    metadata["local_trial_id"] = metadata.groupby("subject_id").cumcount()
    stimulus_names = sorted(metadata.stimulus_name.unique())
    stimulus_map = {name: index for index, name in enumerate(stimulus_names)}
    metadata["trial_id"] = metadata.stimulus_name.map(stimulus_map).astype(int)
    return metadata


def validate_subject_contract(metadata: pd.DataFrame, dataset_root: Path) -> None:
    preprocessed_root = dataset_root / "preproc_data"
    for subject_dir in tqdm(sorted(preprocessed_root.glob("S*")), desc="Checking subject contract", unit="subject"):
        subject_id = int(subject_dir.name[1:])
        trial_files = sorted(
            subject_dir.glob("*_EEG.csv"), key=lambda path: int(path.stem.split("_", maxsplit=1)[0])
        )
        subject_meta = metadata.loc[metadata.subject_id == subject_id].sort_values("local_trial_id")
        valence = np.loadtxt(subject_dir / "labels_feltVlnc.csv", ndmin=1)
        arousal = np.loadtxt(subject_dir / "labels_feltArsl.csv", ndmin=1)
        expected = len(subject_meta)
        if not (len(trial_files) == len(valence) == len(arousal) == expected):
            raise ValueError(
                f"Subject {subject_id} count mismatch: files={len(trial_files)}, valence={len(valence)}, "
                f"arousal={len(arousal)}, XML={expected}"
            )
        file_indices = [int(path.stem.split("_", maxsplit=1)[0]) for path in trial_files]
        if file_indices != list(range(expected)):
            raise ValueError(f"Subject {subject_id} trial files are not a contiguous zero-based sequence")
        if not np.array_equal(valence.astype(int), subject_meta.valence_score.to_numpy(int)):
            raise ValueError(f"Subject {subject_id} valence labels disagree with session.xml")
        if not np.array_equal(arousal.astype(int), subject_meta.arousal_score.to_numpy(int)):
            raise ValueError(f"Subject {subject_id} arousal labels disagree with session.xml")


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    args = parse_args()
    metadata = load_session_metadata(args.dataset_root)
    validate_subject_contract(metadata, args.dataset_root)

    features = np.load(args.feature_root / "eeg_feature_fp12.npy").astype(np.float32)
    subjects = np.load(args.feature_root / "subject_ids.npy").reshape(-1).astype(int)
    archived_trials = np.load(args.feature_root / "trial_ids.npy").reshape(-1).astype(int)
    archived_valence = np.load(args.feature_root / "labels_valence.npy").reshape(-1).astype(int)
    archived_arousal = np.load(args.feature_root / "labels_arousal.npy").reshape(-1).astype(int)
    if not all(len(array) == len(features) for array in (subjects, archived_trials, archived_valence, archived_arousal)):
        raise ValueError("MAHNOB-HCI feature arrays have inconsistent row counts")

    groups = pd.DataFrame(
        {"subject_id": subjects, "archived_trial_id": archived_trials, "row_index": np.arange(len(subjects))}
    ).groupby(["subject_id", "archived_trial_id"], sort=True).row_index.apply(list)
    rows = []
    for (subject_id, archived_trial_id), indices in tqdm(
        groups.items(), total=len(groups), desc="Aggregating MAHNOB-HCI trials", unit="trial"
    ):
        local_trial_id = int(archived_trial_id - subject_id * 1000)
        match = metadata.loc[
            (metadata.subject_id == subject_id) & (metadata.local_trial_id == local_trial_id)
        ]
        if len(match) != 1:
            raise ValueError(f"No unique XML mapping for subject={subject_id}, local_trial={local_trial_id}")
        meta = match.iloc[0]
        indices_array = np.asarray(indices, dtype=int)
        expected_valence = int(meta.valence_score >= 5)
        expected_arousal = int(meta.arousal_score >= 5)
        if not np.all(archived_valence[indices_array] == expected_valence):
            raise ValueError(f"Archived valence mismatch for subject={subject_id}, local_trial={local_trial_id}")
        if not np.all(archived_arousal[indices_array] == expected_arousal):
            raise ValueError(f"Archived arousal mismatch for subject={subject_id}, local_trial={local_trial_id}")
        if args.validate_source_lengths:
            source = args.dataset_root / "preproc_data" / f"S{subject_id:02d}" / f"{local_trial_id}_EEG.csv"
            n_samples = count_lines(source)
            expected_windows = max((n_samples - WINDOW_SIZE) // STEP_SIZE + 1, 0)
            if expected_windows != len(indices_array):
                raise ValueError(
                    f"Window count mismatch for {source}: expected={expected_windows}, archived={len(indices_array)}"
                )
        else:
            n_samples = np.nan
        row = {
            "subject_id": int(subject_id),
            "trial_id": int(meta.trial_id),
            "stimulus_name": str(meta.stimulus_name),
            "local_trial_id": local_trial_id,
            "session_id": int(meta.session_id),
            "cut_length_seconds": float(meta.cut_length_seconds),
            "source_samples": n_samples,
            "n_windows": len(indices_array),
            "valence_score": float(meta.valence_score),
            "arousal_score": float(meta.arousal_score),
        }
        row.update(summarize(features[indices_array], "stimulus__"))
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["subject_id", "trial_id"])
    if frame.duplicated(["subject_id", "trial_id"]).any():
        raise ValueError("Recovered subject-by-stimulus keys are not unique")
    if frame.subject_id.nunique() != 27 or frame.trial_id.nunique() != 20 or len(frame) != 527:
        raise ValueError(
            f"Unexpected recovered design: subjects={frame.subject_id.nunique()}, "
            f"stimuli={frame.trial_id.nunique()}, rows={len(frame)}"
        )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_file, index=False)
    missing_cells = frame.subject_id.nunique() * frame.trial_id.nunique() - len(frame)
    print(
        f"Saved {len(frame)} MAHNOB-HCI trials: {frame.subject_id.nunique()} subjects, "
        f"{frame.trial_id.nunique()} shared stimuli, {missing_cells} missing crossed cells -> "
        f"{args.output_file.resolve()}"
    )


if __name__ == "__main__":
    main()
