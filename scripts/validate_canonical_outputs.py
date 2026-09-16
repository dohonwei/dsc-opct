from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from stimulus_identity_contract import CROSSED_DATASET_NAMES


ROOTS = (
    "unified_protocol_model_benchmark_multiseed_crossed_valid",
    "matched_identity_exposure_benchmark_crossed_valid",
    "paired_block_identity_exposure_benchmark_crossed_valid",
    "crossed_identity_probes_crossed_valid",
    "representation_identity_audit_crossed_valid",
    "representation_emotion_gate_crossed_valid",
    "representation_exposure_gate_crossed_valid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the canonical crossed-dataset result chain.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def datasets_in_csv(path: Path, chunk_size: int) -> tuple[set[str], int]:
    datasets: set[str] = set()
    rows = 0
    for chunk in pd.read_csv(path, usecols=["dataset"], chunksize=chunk_size):
        datasets.update(chunk.dataset.astype(str).unique())
        rows += len(chunk)
    return datasets, rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "status": "running",
        "admitted_datasets": sorted(CROSSED_DATASET_NAMES),
        "checks": [],
    }
    checks: list[dict[str, object]] = report["checks"]  # type: ignore[assignment]

    csv_files: list[Path] = []
    for name in ROOTS:
        root = args.outputs_root / name
        require(root.is_dir(), f"Missing canonical root: {root}")
        csv_files.extend(path for path in root.rglob("*.csv") if path.is_file())
    for path in tqdm(csv_files, desc="Canonical CSV contracts", unit="file", dynamic_ncols=True):
        columns = pd.read_csv(path, nrows=0).columns
        if "dataset" not in columns:
            continue
        datasets, rows = datasets_in_csv(path, args.chunk_size)
        require(
            datasets == CROSSED_DATASET_NAMES,
            f"Dataset contract failed for {path}: {sorted(datasets)}",
        )
        checks.append({"check": "dataset_set", "path": str(path), "rows": rows, "status": "passed"})

    paired_root = args.outputs_root / "paired_block_identity_exposure_benchmark_crossed_valid"
    audit = pd.read_csv(paired_root / "split_audit.csv")
    require(len(audit) == 2000, f"Expected 2000 paired split-audit rows, observed {len(audit)}")
    require(audit.matches_reference_size.astype(bool).all(), "Training-size matching failed")
    require(audit.matches_reference_class_counts.astype(bool).all(), "Class-count matching failed")
    require((audit.row_overlap_with_test == 0).all(), "Train-test row overlap detected")
    require((audit.shared_core_fraction > 0).all(), "Empty shared training core detected")
    expected_coverage = {
        "unseen_both": (0.0, 0.0),
        "seen_subject": (1.0, 0.0),
        "seen_stimulus": (0.0, 1.0),
        "seen_both": (1.0, 1.0),
    }
    for exposure, (subject_coverage, stimulus_coverage) in expected_coverage.items():
        subset = audit.loc[audit.exposure == exposure]
        require(np.allclose(subset.test_subject_coverage, subject_coverage), f"Bad {exposure} subject coverage")
        require(np.allclose(subset.test_stimulus_coverage, stimulus_coverage), f"Bad {exposure} stimulus coverage")
    checks.append({"check": "paired_split_contract", "rows": len(audit), "status": "passed"})

    expected_tables = {
        paired_root / "factorial_effects_crossed_valid.csv": 12,
        paired_root / "factorial_effects_per_seed_crossed_valid.csv": 60,
        paired_root / "factorial_effects_per_classifier_seed.csv": 240,
        paired_root / "factorial_effects_classifier_sensitivity.csv": 48,
        paired_root / "factorial_effects_multiplicity_sensitivity.csv": 12,
        paired_root / "factorial_effect_equivalence_crossed_valid.csv": 36,
        args.outputs_root / "paired_block_threshold_sensitivity" / "factorial_effects_all_thresholds.csv": 36,
        args.outputs_root / "crossed_identity_probes_crossed_valid" / "identity_class_bootstrap.csv": 8,
        args.outputs_root / "representation_exposure_gate_crossed_valid" / "gate_summary.csv": 3,
    }
    for path, expected_rows in expected_tables.items():
        require(path.is_file(), f"Missing canonical table: {path}")
        rows = len(pd.read_csv(path))
        require(rows == expected_rows, f"{path} has {rows} rows; expected {expected_rows}")
        checks.append({"check": "table_rows", "path": str(path), "rows": rows, "status": "passed"})

    migration_path = args.outputs_root / "crossed_valid_migration_manifest.json"
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    for record in tqdm(migration["files"], desc="Migration hashes", unit="file", dynamic_ncols=True):
        source = Path(record["source"])
        target = Path(record["target"])
        require(source.is_file() and target.is_file(), f"Missing migration artifact: {source} or {target}")
        require(sha256(source) == record["source_sha256"], f"Source hash changed: {source}")
        require(sha256(target) == record["target_sha256"], f"Target hash changed: {target}")
    checks.append({"check": "migration_hashes", "files": len(migration["files"]), "status": "passed"})

    report["status"] = "passed"
    report["n_checks"] = len(checks)
    output_path = args.outputs_root / "canonical_validation_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Canonical validation passed ({len(checks)} checks): {output_path}")


if __name__ == "__main__":
    main()
