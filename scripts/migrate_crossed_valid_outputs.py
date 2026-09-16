from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from stimulus_identity_contract import CROSSED_DATASET_NAMES


SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)
ROOT = Path("outputs")
PROTOCOL_SOURCE = ROOT / "unified_protocol_model_benchmark_multiseed"
PROTOCOL_TARGET = ROOT / "unified_protocol_model_benchmark_multiseed_crossed_valid"
EXPOSURE_RUNS = (
    (
        ROOT / "matched_identity_exposure_benchmark",
        ROOT / "matched_identity_exposure_benchmark_crossed_valid",
    ),
    (
        ROOT / "paired_block_identity_exposure_benchmark",
        ROOT / "paired_block_identity_exposure_benchmark_crossed_valid",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate row-level caches to the verified crossed-dataset analysis roots."
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filter_csv(source: Path, target: Path, chunk_size: int, force: bool) -> dict[str, object]:
    if target.exists() and not force:
        frame = pd.read_csv(target, usecols=["dataset"])
        observed = set(frame.dataset.unique())
        if observed != CROSSED_DATASET_NAMES:
            raise ValueError(f"Existing target has invalid datasets: {target}: {sorted(observed)}")
        return {
            "source": str(source),
            "target": str(target),
            "source_sha256": sha256(source),
            "target_sha256": sha256(target),
            "rows": int(len(frame)),
            "rows_by_dataset": frame.dataset.value_counts().sort_index().to_dict(),
            "reused_existing_target": True,
        }

    target.parent.mkdir(parents=True, exist_ok=True)
    first = True
    rows_by_dataset: dict[str, int] = {}
    rows = 0
    reader = pd.read_csv(source, chunksize=chunk_size, low_memory=False)
    for chunk in tqdm(reader, desc=f"Filter {source.name}", unit="chunk", dynamic_ncols=True):
        filtered = chunk.loc[chunk.dataset.isin(CROSSED_DATASET_NAMES)].copy()
        counts = filtered.dataset.value_counts()
        for dataset, count in counts.items():
            rows_by_dataset[str(dataset)] = rows_by_dataset.get(str(dataset), 0) + int(count)
        rows += len(filtered)
        filtered.to_csv(target, mode="w" if first else "a", header=first, index=False)
        first = False
    if first:
        raise ValueError(f"No admitted rows found in {source}")
    if set(rows_by_dataset) != CROSSED_DATASET_NAMES:
        raise ValueError(f"Migrated dataset mismatch for {source}: {sorted(rows_by_dataset)}")
    return {
        "source": str(source),
        "target": str(target),
        "source_sha256": sha256(source),
        "target_sha256": sha256(target),
        "rows": rows,
        "rows_by_dataset": dict(sorted(rows_by_dataset.items())),
        "reused_existing_target": False,
    }


def protocol_sources() -> dict[int, Path]:
    manifest = json.loads((PROTOCOL_SOURCE / "run_manifest.json").read_text(encoding="utf-8"))
    return {int(seed): Path(path) for seed, path in manifest["prediction_sources"].items()}


def validate_group_counts(path: Path, keys: list[str], expected_per_dataset: dict[str, int]) -> None:
    frame = pd.read_csv(path, low_memory=False)
    if set(frame.dataset.unique()) != CROSSED_DATASET_NAMES:
        raise ValueError(f"Dataset contract failed for {path}")
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate row-level keys in {path}")
    for key, count in frame.groupby(keys[:-1]).size().items():
        dataset = key[0] if isinstance(key, tuple) else key
        if count != expected_per_dataset[dataset]:
            raise ValueError(f"Incomplete group {key} in {path}: {count}")


def main() -> None:
    args = parse_args()
    feature_counts = {
        "DEAP": len(pd.read_csv(ROOT / "unified_frontal_features" / "deap.csv", usecols=["subject_id"])),
        "MAHNOB-HCI": len(
            pd.read_csv(ROOT / "unified_frontal_features" / "mahnob-hci.csv", usecols=["subject_id"])
        ),
    }
    records: list[dict[str, object]] = []

    sources = protocol_sources()
    for seed in tqdm(SEEDS, desc="Protocol seeds", unit="seed", dynamic_ncols=True):
        target = PROTOCOL_TARGET / f"seed_{seed}" / "predictions.csv"
        records.append(filter_csv(sources[seed], target, args.chunk_size, args.force))
        validate_group_counts(
            target,
            ["dataset", "task", "protocol", "model", "row_id"],
            feature_counts,
        )

    for source_root, target_root in EXPOSURE_RUNS:
        for seed in tqdm(SEEDS, desc=f"Migrate {source_root.name}", unit="seed", dynamic_ncols=True):
            prediction_target = target_root / f"seed_{seed}" / "predictions.csv"
            audit_target = target_root / f"seed_{seed}" / "split_audit.csv"
            records.append(
                filter_csv(
                    source_root / f"seed_{seed}" / "predictions.csv",
                    prediction_target,
                    args.chunk_size,
                    args.force,
                )
            )
            records.append(
                filter_csv(
                    source_root / f"seed_{seed}" / "split_audit.csv",
                    audit_target,
                    args.chunk_size,
                    args.force,
                )
            )
            validate_group_counts(
                prediction_target,
                ["dataset", "task", "exposure", "model", "row_id"],
                feature_counts,
            )

    manifest = {
        "admitted_datasets": sorted(CROSSED_DATASET_NAMES),
        "excluded_dataset": "EPPVR",
        "reason": (
            "EPPVR record position cannot be verified as shared physical stimulus identity "
            "across participants."
        ),
        "migration_policy": "filter row-level predictions and split audits only; recompute all summaries",
        "files": records,
    }
    manifest_path = ROOT / "crossed_valid_migration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Migrated {len(records)} row-level files; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
