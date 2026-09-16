from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_matched_exposure_benchmark as matched
from stimulus_identity_contract import CROSSED_DATASET_NAMES, require_crossed_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct contrasts for paired-block identity exposures.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_json(args.root / "run_manifest.json", typ="series")
    admitted = require_crossed_datasets(
        list(manifest.datasets),
        context="Paired block exposure analysis",
    )
    frames = []
    for split_seed in manifest.split_seeds:
        frame = pd.read_csv(args.root / f"seed_{split_seed}" / "predictions.csv", low_memory=False)
        frame["subject_id"] = frame.subject_id.astype(str)
        frame["trial_id"] = frame.trial_id.astype(int)
        frame["split_seed"] = int(split_seed)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.loc[predictions.dataset.isin(CROSSED_DATASET_NAMES)].copy()
    if set(predictions.dataset.unique()) != set(admitted):
        raise ValueError("Paired prediction cache does not match the admitted dataset manifest")
    models = list(manifest.models)
    contrasts = matched.triple_cluster_bootstrap(
        predictions,
        models,
        args.bootstrap_repetitions,
        args.seed,
    )
    contrasts.to_csv(args.root / "matched_exposure_contrasts_crossed_valid.csv", index=False)
    ranks = matched.rank_sensitivity(pd.read_csv(args.root / "per_seed_summary.csv"), models)
    ranks.to_csv(args.root / "rank_sensitivity.csv", index=False)
    matched.write_report(args.root / "matched_exposure_summary.md", contrasts, ranks)
    inference_manifest = {
        "input_root": str(args.root),
        "admitted_datasets": admitted,
        "models": models,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.seed,
        "bootstrap_seed_scheme": "sha256(base_seed|dataset|task|left_exposure|right_exposure)",
        "bootstrap_clusters": ["subject_id", "trial_id", "split_seed"],
    }
    (args.root / "matched_exposure_contrasts_crossed_valid_manifest.json").write_text(
        json.dumps(inference_manifest, indent=2), encoding="utf-8"
    )
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
