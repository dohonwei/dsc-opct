from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DEAP windows into trial-level continuous-label features.")
    parser.add_argument(
        "--feature-root", type=Path, default=Path(r"E:\AA发表论文的数据\tcgan\feature\deap")
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path(r"E:\AA发表论文的数据\dataset\DEAP")
    )
    parser.add_argument(
        "--output-file", type=Path, default=Path("outputs/deap_trial_features/trial_features.csv")
    )
    return parser.parse_args()


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    time = np.arange(values.shape[0], dtype=float)
    centered_time = time - time.mean()
    denominator = np.sum(centered_time**2)
    slopes = centered_time @ values / denominator
    row: dict[str, float] = {}
    for index in range(values.shape[1]):
        feature = values[:, index]
        name = f"{prefix}f{index:02d}"
        row[f"{name}_mean"] = float(np.mean(feature))
        row[f"{name}_std"] = float(np.std(feature))
        row[f"{name}_slope"] = float(slopes[index])
    return row


def load_scores(raw_root: Path) -> np.ndarray:
    score_rows = []
    subject_files = sorted(raw_root.glob("s*.dat"))
    if len(subject_files) != 32:
        raise ValueError(f"Expected 32 DEAP subject files, found {len(subject_files)}")
    for subject_index, path in enumerate(tqdm(subject_files, desc="Loading DEAP scores", unit="subject"), start=1):
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        labels = np.asarray(payload["labels"], dtype=float)
        if labels.shape != (40, 4):
            raise ValueError(f"Unexpected labels in {path}: {labels.shape}")
        for trial_index, scores in enumerate(labels):
            score_rows.append((subject_index, trial_index, scores[0], scores[1]))
    return pd.DataFrame(score_rows, columns=["subject_id", "trial_id", "valence_score", "arousal_score"])


def main() -> None:
    args = parse_args()
    eeg = np.load(args.feature_root / "eeg_feature_fp12.npy").astype(np.float32)
    pps = np.load(args.feature_root / "pps_feature.npy").astype(np.float32)
    subjects = np.load(args.feature_root / "subject_ids.npy").reshape(-1).astype(int)
    global_trials = np.load(args.feature_root / "trial_ids.npy").reshape(-1).astype(int)
    expected_global = (subjects - 1) * 40 + global_trials % 40
    if not np.array_equal(global_trials, expected_global):
        raise ValueError("DEAP trial_ids do not follow the documented subject-major global sequence")
    trials = global_trials % 40
    if eeg.shape != (32 * 40 * 29, 40) or pps.shape != (32 * 40 * 29, 6):
        raise ValueError(f"Unexpected DEAP feature shapes: EEG={eeg.shape}, PPS={pps.shape}")
    values = np.concatenate([eeg, pps], axis=1)
    rows = []
    groups = pd.DataFrame({"subject_id": subjects, "trial_id": trials}).groupby(
        ["subject_id", "trial_id"], sort=True
    ).indices
    for (subject_id, trial_id), indices in tqdm(groups.items(), total=len(groups), desc="Aggregating DEAP trials", unit="trial"):
        indices = np.asarray(sorted(indices))
        if len(indices) != 29:
            raise ValueError(f"Expected 29 windows for subject={subject_id}, trial={trial_id}; found {len(indices)}")
        row = {"subject_id": int(subject_id), "trial_id": int(trial_id)}
        row.update(summarize(values[indices], "stimulus__"))
        rows.append(row)
    frame = pd.DataFrame(rows).merge(
        load_scores(args.raw_root), on=["subject_id", "trial_id"], how="left", validate="one_to_one"
    )
    if frame[["valence_score", "arousal_score"]].isna().any().any():
        raise ValueError("Raw DEAP score merge produced missing values")
    archived_valence = np.load(args.feature_root / "labels_valence.npy").reshape(32, 40, 29)[:, :, 0]
    archived_arousal = np.load(args.feature_root / "labels_arousal.npy").reshape(32, 40, 29)[:, :, 0]
    ordered = frame.sort_values(["subject_id", "trial_id"])
    if not np.array_equal((ordered.valence_score.to_numpy().reshape(32, 40) >= 5).astype(int), archived_valence):
        raise ValueError("Continuous valence scores do not match archived binary labels")
    if not np.array_equal((ordered.arousal_score.to_numpy().reshape(32, 40) >= 5).astype(int), archived_arousal):
        raise ValueError("Continuous arousal scores do not match archived binary labels")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_file, index=False)
    print(f"Saved {len(frame)} DEAP trial rows with {len(frame.columns) - 4} features to {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
