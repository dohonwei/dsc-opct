from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import acquire_faced_v11_dataset as acquire  # noqa: E402
import apply_frozen_dcs_opct_v11_to_faced as application  # noqa: E402
import build_faced_trial_features as features  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    cluster_geometry,
    select_distribution_covered_audit,
)
from faced_v11_contract import (  # noqa: E402
    ARCHIVED_MANIFEST,
    BILATERAL_PAIRS,
    DEFAULT_DATA_ROOT,
    FREEZE,
    IMPLEMENTATION_LOCK,
    NON_NEUTRAL_VIDEOS,
    REQUIRED_CHANNELS,
    RESERVATION,
    VIDEO_EMOTION,
    expected_polarity,
    sha256,
    verify_frozen_artifacts,
    verify_implementation_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate FACED code without reading protected values."
    )
    parser.add_argument("--require-lock", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def synthetic_event_table() -> pd.DataFrame:
    rows = []
    onset = 0.0
    for video_index in range(1, 29):
        rows.append(
            {
                "onset": onset,
                "duration": 40.0,
                "trial_type": f"video-{video_index}",
                "emotion_label": VIDEO_EMOTION[video_index],
                "binary_label": expected_polarity(video_index),
                "video_index": video_index,
            }
        )
        onset += 70.0
    return pd.DataFrame(rows)


def synthetic_dose_tables(root: Path) -> None:
    summary_rows = []
    encoding_rows = []
    representations = ("all", "relative_power", "normalized_asymmetry")
    models = ("linear_logistic", "gpu_mlp")
    seeds = (20260813, 20260829, 20260911, 20260923, 20261007)
    for representation_index, representation in enumerate(representations):
        for seed_index, split_seed in enumerate(seeds):
            encoding_rows.append(
                {
                    "dataset": "FACED",
                    "representation": representation,
                    "split_seed": split_seed,
                    "n_features": 10 + representation_index,
                    "encoding_margin": 0.1 + 0.01 * seed_index,
                }
            )
            for model_index, model in enumerate(models):
                for dose in (0.0, 0.25, 0.5, 0.75, 1.0):
                    summary_rows.append(
                        {
                            "dataset": "FACED",
                            "task": "polarity",
                            "axis": "subject",
                            "representation": representation,
                            "model": model,
                            "split_seed": split_seed,
                            "nominal_dose": dose,
                            "achieved_opportunity": 0.02
                            + 0.1 * dose
                            + 0.001 * seed_index,
                            "metadata_prior_opportunity": 0.01 + 0.05 * dose,
                            "exposure_effect": 0.005 * model_index + 0.03 * dose,
                        }
                    )
    pd.DataFrame(summary_rows).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(encoding_rows).to_csv(
        root / "identity_encoding_margin.csv", index=False
    )


def main() -> None:
    args = parse_args()
    checks: dict[str, bool] = {}
    checks["reservation_exists"] = RESERVATION.is_file()
    checks["freeze_hash"] = sha256(FREEZE) == (
        "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
    )
    verify_frozen_artifacts()
    checks["frozen_artifacts"] = True

    manifest = json.loads(ARCHIVED_MANIFEST.read_text(encoding="utf-8"))
    checks["manifest_file_count"] = len(manifest) == 745
    checks["manifest_bdf_count"] = (
        sum(str(item["path"]).endswith(".bdf") for item in manifest) == 123
    )
    checks["manifest_event_count"] = (
        sum(str(item["path"]).endswith("_events.tsv") for item in manifest) == 123
    )
    checks["manifest_channel_count"] = (
        sum(str(item["path"]).endswith("_channels.tsv") for item in manifest) == 123
    )
    checks["manifest_unique_paths"] = len({item["path"] for item in manifest}) == len(
        manifest
    )

    readme = ROOT / "outputs/faced_v11_preaccess_metadata/official_README.md"
    readme_item = next(item for item in manifest if item["path"] == "README.md")
    readme_data = readme.read_bytes()
    checks["git_blob_checksum_support"] = (
        git_blob_sha1(readme_data) == readme_item["checksum"]
    )
    checks["sha256_checksum_support"] = any(
        len(str(item["checksum"])) == 64 for item in manifest
    )
    checks["manifest_core_stable"] = acquire.manifest_core(
        manifest
    ) == acquire.manifest_core(list(reversed(manifest)))

    checks["video_map"] = set(VIDEO_EMOTION) == set(range(1, 29))
    checks["non_neutral_video_count"] = len(NON_NEUTRAL_VIDEOS) == 24
    checks["channel_count"] = (
        len(REQUIRED_CHANNELS) == 32 and len(set(REQUIRED_CHANNELS)) == 32
    )
    checks["bilateral_pairs"] = len(BILATERAL_PAIRS) == 14
    with tempfile.TemporaryDirectory() as directory:
        event_path = Path(directory) / "synthetic_events.tsv"
        synthetic_event_table().to_csv(event_path, sep="\t", index=False)
        parsed = features.event_table(event_path)
        checks["synthetic_event_contract"] = len(parsed) == 28

    rng = np.random.default_rng(20260909)
    sampling_rate = 250.0
    samples = int(12 * sampling_rate)
    time = np.arange(samples) / sampling_rate
    synthetic = np.stack(
        [
            10.0 * np.sin(2 * np.pi * (6.0 + index % 12) * time)
            + rng.normal(0.0, 0.5, samples)
            for index in range(32)
        ]
    )
    row = features.trial_features(synthetic, sampling_rate, "synthetic", 1)
    columns = [key for key in row if key.startswith("stimulus__")]
    checks["synthetic_all_feature_count"] = len(columns) == 1104
    checks["synthetic_relative_feature_count"] = (
        sum(key.startswith("stimulus__relative_power_") for key in columns) == 384
    )
    checks["synthetic_asymmetry_feature_count"] = (
        sum(key.startswith("stimulus__normalized_asymmetry_") for key in columns) == 168
    )
    checks["synthetic_features_finite"] = np.isfinite(
        [row[key] for key in columns]
    ).all()
    checks["cuda_available"] = torch.cuda.is_available()

    with tempfile.TemporaryDirectory() as directory:
        synthetic_root = Path(directory)
        synthetic_dose_tables(synthetic_root)
        target = application.load_unlabeled_target(synthetic_root)
        checks["synthetic_risk_configuration_count"] = len(target) == 120
        target["probability_identity"] = np.linspace(0.1, 0.9, len(target))
        target["probability_wg_opct"] = np.clip(
            0.02 + 0.96 * target.probability_identity, 1e-6, 1 - 1e-6
        )
        geometry = cluster_geometry(target)
        assignment = select_distribution_covered_audit(geometry)
        checks["synthetic_cluster_count"] = len(geometry) == 30
        checks["synthetic_half_audit"] = (
            assignment.partition.value_counts().to_dict()
            == {
                "audit": 15,
                "heldout": 15,
            }
        )
        evaluation = application.attach_outcomes(target, synthetic_root / "summary.csv")
        checks["synthetic_outcome_count"] = len(evaluation) == 120
        checks["synthetic_material_threshold"] = (
            evaluation.material_optimism_event
            == (evaluation.dose_induced_amplification >= 0.02).astype(int)
        ).all()

    protected = (
        list(args.data_root.glob("participants.tsv")) if args.data_root.exists() else []
    )
    if args.data_root.exists():
        protected += list(args.data_root.glob("sub-*/eeg/*_events.tsv"))
        protected += list(args.data_root.glob("sub-*/eeg/*.bdf"))
    checks["no_protected_values_before_lock"] = (
        not protected or IMPLEMENTATION_LOCK.is_file()
    )
    if args.require_lock:
        verify_implementation_lock()
        checks["implementation_lock"] = True

    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not failed else "failed",
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "failed_checks": failed,
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
