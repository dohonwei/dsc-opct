from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm.auto import tqdm


BANDS = ("delta", "theta", "alpha", "beta", "gamma")
SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}
TASK_CONTRASTS = {
    "valence": {1: 1.0, 3: 9.0},
    "arousal": {0: 1.0, 2: 9.0},
}
CHANNEL_INDEX = {"FP1": 0, "FP2": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build two-frontal-channel SEED-IV trial features for untouched validation."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path(r"E:\AA发表论文的数据\dataset\SEED_IV")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/seediv_trial_features")
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trajectory_summary(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    time = np.linspace(-1.0, 1.0, len(values))
    slope = float(np.polyfit(time, values, 1)[0]) if len(values) > 1 else 0.0
    return float(np.mean(values)), float(np.std(values)), slope


def add_summary(row: dict[str, object], prefix: str, values: np.ndarray) -> None:
    mean, std, slope = trajectory_summary(values)
    row[f"stimulus__{prefix}_mean"] = mean
    row[f"stimulus__{prefix}_std"] = std
    row[f"stimulus__{prefix}_slope"] = slope


def trial_row(
    subject: int,
    session: int,
    trial: int,
    emotion: int,
    de: np.ndarray,
    psd: np.ndarray,
) -> dict[str, object]:
    row: dict[str, object] = {
        "subject_id": str(subject),
        "session_id": session,
        "trial_id": session * 100 + trial,
        "emotion_label": emotion,
        "n_windows": int(de.shape[1]),
    }
    frontal_de = {name: de[index] for name, index in CHANNEL_INDEX.items()}
    frontal_psd = {name: psd[index] for name, index in CHANNEL_INDEX.items()}
    relative = {
        name: values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
        for name, values in frontal_psd.items()
    }
    for band_index, band in enumerate(BANDS):
        for channel in CHANNEL_INDEX:
            add_summary(
                row,
                f"log_power_{channel}_{band}",
                frontal_de[channel][:, band_index],
            )
            add_summary(
                row,
                f"relative_power_{channel}_{band}",
                relative[channel][:, band_index],
            )
        left_de = frontal_de["FP1"][:, band_index]
        right_de = frontal_de["FP2"][:, band_index]
        left_relative = relative["FP1"][:, band_index]
        right_relative = relative["FP2"][:, band_index]
        add_summary(row, f"log_difference_{band}", left_de - right_de)
        add_summary(
            row,
            f"normalized_asymmetry_{band}",
            (left_relative - right_relative)
            / np.maximum(np.abs(left_relative) + np.abs(right_relative), 1e-12),
        )
    return row


def main() -> None:
    args = parse_args()
    channel_path = args.data_root / "Channel Order.xlsx"
    readme_path = args.data_root / "ReadMe.txt"
    feature_root = args.data_root / "eeg_feature_smooth"
    for path in (channel_path, readme_path, feature_root):
        if not path.exists():
            raise FileNotFoundError(path)
    channels = pd.read_excel(channel_path, header=None).iloc[:, 0].astype(str).str.strip().tolist()
    if channels[0] != "FP1" or channels[2] != "FP2":
        raise ValueError("SEED-IV channel-order contract does not place FP1/FP2 at indices 0/2")
    files = sorted(feature_root.glob("*/*.mat"))
    if len(files) != 45:
        raise ValueError(f"Expected 45 subject-session feature files, found {len(files)}")
    all_rows: list[dict[str, object]] = []
    progress = tqdm(files, desc="SEED-IV frontal trial features", unit="file", dynamic_ncols=True)
    for path in progress:
        session = int(path.parent.name)
        match = re.match(r"(\d+)_", path.name)
        if match is None:
            raise ValueError(f"Cannot parse subject ID from {path.name}")
        subject = int(match.group(1))
        content = loadmat(path)
        for trial, emotion in enumerate(SESSION_LABELS[session], start=1):
            de_key = f"de_LDS{trial}"
            psd_key = f"psd_LDS{trial}"
            if de_key not in content or psd_key not in content:
                raise KeyError(f"{path.name} lacks {de_key}/{psd_key}")
            de = np.asarray(content[de_key], dtype=float)
            psd = np.asarray(content[psd_key], dtype=float)
            if de.shape[0] != 62 or de.shape[2] != 5 or de.shape != psd.shape:
                raise ValueError(f"Unexpected feature shape in {path.name}/{trial}: {de.shape}")
            all_rows.append(trial_row(subject, session, trial, emotion, de, psd))
    full = pd.DataFrame(all_rows).sort_values(["subject_id", "session_id", "trial_id"])
    if len(full) != 15 * 3 * 24:
        raise ValueError("SEED-IV trial table is incomplete")
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for task, mapping in TASK_CONTRASTS.items():
        task_frame = full.loc[full.emotion_label.isin(mapping)].copy()
        task_frame[f"{task}_score"] = task_frame.emotion_label.map(mapping)
        output = args.output_root / f"seediv_{task}_extreme.csv"
        task_frame.to_csv(output, index=False)
        outputs[task] = {
            "path": str(output),
            "rows": len(task_frame),
            "subjects": int(task_frame.subject_id.nunique()),
            "physical_stimuli": int(task_frame.trial_id.nunique()),
            "class_counts": task_frame[f"{task}_score"].value_counts().sort_index().to_dict(),
            "sha256": sha256(output),
        }
    manifest = {
        "status": "built",
        "dataset": "SEED-IV",
        "source": str(args.data_root),
        "source_files": len(files),
        "channel_contract": {"FP1": 0, "FP2": 2},
        "stimulus_contract": "trial_id=session*100+within-session film index",
        "label_contract": {
            "valence": "happy (3, score 9) versus sad (1, score 1)",
            "arousal": "fear (2, score 9) versus neutral (0, score 1)",
        },
        "excluded_from_each_task": "the two emotion classes outside the prespecified extreme contrast",
        "readme_sha256": sha256(readme_path),
        "channel_order_sha256": sha256(channel_path),
        "outputs": outputs,
    }
    (args.output_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
