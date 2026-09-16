from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
RESERVATION = ROOT / "docs/avdos_v7_external_confirmation_reservation.json"
RESERVATION_AMENDMENT = ROOT / "docs/avdos_v11_external_confirmation_amendment.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
IMPLEMENTATION_LOCK = ROOT / "docs/avdos_v11_external_confirmation_implementation_lock.json"
EXPLORATORY_LOCK = ROOT / "docs/avdos_v11_post_access_exploratory_implementation_lock_amendment_001.json"
SOURCE = Path(
    "E:/AA发表论文的数据/dataset/AVDOS-VR-main/notebooks/temp/2_affect/"
    "Dataset_AVDOS_ManualFeaturesWithAnnotations.csv"
)
ACTIVE_SEGMENTS = ("Positive", "Neutral", "Negative")
MINIMAL_COLUMNS = (
    "HeartRate/Average_mean",
    "HeartRate/Average_std",
    "HeartRate/Average_rms",
    "Ppg/Raw.ppg_mean",
    "Ppg/Raw.ppg_std",
    "Ppg/Raw.ppg_rms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build post-access exploratory AVDOS PPG trials.")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/avdos_exploratory_ppg_features",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_locks(source: Path) -> dict[str, object]:
    for path in (
        RESERVATION,
        RESERVATION_AMENDMENT,
        FREEZE,
        IMPLEMENTATION_LOCK,
        EXPLORATORY_LOCK,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    reservation = json.loads(RESERVATION.read_text(encoding="utf-8"))
    amendment = json.loads(RESERVATION_AMENDMENT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    lock = json.loads(IMPLEMENTATION_LOCK.read_text(encoding="utf-8"))
    exploratory_lock = json.loads(EXPLORATORY_LOCK.read_text(encoding="utf-8"))
    if sha256(source) != reservation["source"]["sha256"]:
        raise RuntimeError("AVDOS source hash differs from the reservation")
    if amendment["reservation_sha256"] != sha256(RESERVATION):
        raise RuntimeError("AVDOS amendment does not match the reservation")
    if lock["reservation_amendment_sha256"] != sha256(RESERVATION_AMENDMENT):
        raise RuntimeError("AVDOS implementation lock does not match the amendment")
    if lock["freeze_sha256"] != sha256(FREEZE):
        raise RuntimeError("AVDOS implementation lock does not match the v11 freeze")
    for relative, expected in freeze["locked_artifacts"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen v11 artifact changed: {relative}")
    for hash_group in ("analysis_code_sha256", "dependency_code_sha256"):
        for relative, expected in lock[hash_group].items():
            if sha256(ROOT / relative) != expected:
                raise RuntimeError(f"Registered AVDOS code changed: {relative}")
    for relative, expected in exploratory_lock["analysis_code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Exploratory AVDOS code changed: {relative}")
    return lock


def output_name(column: str) -> str:
    cleaned = column.replace("/", "_").replace(".", "_").replace("[", "_")
    cleaned = cleaned.replace("]", "_").replace(" ", "_")
    return "feature__" + cleaned


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite AVDOS features: {args.output_root}")
    verify_locks(args.source)
    frame = pd.read_csv(args.source)
    required = {
        "participant",
        "segment",
        "i_window",
        "Valence_mean",
        "Arousal_mean",
        *MINIMAL_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"AVDOS feature table lacks registered columns: {missing}")
    heart_rate = tuple(
        column
        for column in frame.columns
        if column.startswith("HeartRate/Average_")
    )
    raw_ppg = tuple(column for column in frame.columns if column.startswith("Ppg/Raw.ppg_"))
    heart_ppg = tuple(dict.fromkeys((*heart_rate, *raw_ppg)))
    representations = {
        "raw_ppg": list(raw_ppg),
        "heart_rate_signal": list(heart_rate),
        "minimal_ppg": list(MINIMAL_COLUMNS),
        "heart_ppg": list(heart_ppg),
    }
    registered_features = list(heart_ppg)
    active = frame.loc[frame.segment.isin(ACTIVE_SEGMENTS)].copy()
    if active.participant.nunique() != 37:
        raise ValueError(
            f"AVDOS active cohort has {active.participant.nunique()} participants, expected 37"
        )
    rows = []
    groups = list(active.groupby(["participant", "segment"], sort=True, dropna=False))
    progress = tqdm(groups, desc="AVDOS contiguous macro-blocks", unit="segment")
    for (participant, segment), group in progress:
        group = group.sort_values("i_window", kind="stable").reset_index(drop=True)
        blocks = np.array_split(np.arange(len(group)), 2)
        if any(len(block) == 0 for block in blocks):
            raise ValueError(f"AVDOS {participant}/{segment} cannot form two blocks")
        for block_index, indices in enumerate(blocks, start=1):
            block = group.iloc[indices]
            row = {
                "subject_id": str(participant),
                "trial_id": int(ACTIVE_SEGMENTS.index(segment) * 2 + block_index),
                "segment": segment,
                "macro_block": block_index,
                "n_windows": len(block),
                "valence_score": float(block.Valence_mean.mean()),
                "arousal_score": float(block.Arousal_mean.mean()),
            }
            for column in registered_features:
                values = pd.to_numeric(block[column], errors="coerce").to_numpy(float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite AVDOS feature: {participant}/{segment}/{column}")
                row[output_name(column)] = float(np.mean(values))
            rows.append(row)
    trials = pd.DataFrame(rows).sort_values(["subject_id", "trial_id"]).reset_index(drop=True)
    if len(trials) != 222 or trials.subject_id.nunique() != 37:
        raise ValueError("AVDOS trials violate the exploratory 37 x 6 contract")
    if not trials.groupby("subject_id").size().eq(6).all():
        raise ValueError("Every AVDOS participant must contribute six trials")
    labels = trials.segment.ne("Neutral").astype(int).to_numpy()
    if len(np.unique(labels)) != 2:
        raise ValueError("AVDOS official binary arousal mapping is one-class")

    args.output_root.mkdir(parents=True, exist_ok=False)
    output = args.output_root / "avdos_trial_features.csv"
    trials.to_csv(output, index=False)
    feature_map = {column: output_name(column) for column in registered_features}
    manifest = {
        "status": "built_under_post_access_exploratory_avdos_v11_protocol",
        "dataset": "AVDOS-VR",
        "rows": len(trials),
        "subjects": int(trials.subject_id.nunique()),
        "trials_per_subject": 6,
        "segments": list(ACTIVE_SEGMENTS),
        "tasks": ["arousal"],
        "label_definition": "Neutral=0; Negative=1; Positive=1",
        "representations": {
            name: [feature_map[column] for column in columns]
            for name, columns in representations.items()
        },
        "source_sha256": sha256(args.source),
        "reservation_sha256": sha256(RESERVATION),
        "reservation_amendment_sha256": sha256(RESERVATION_AMENDMENT),
        "freeze_sha256": sha256(FREEZE),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK),
        "exploratory_lock_sha256": sha256(EXPLORATORY_LOCK),
        "output_sha256": sha256(output),
        "builder_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
