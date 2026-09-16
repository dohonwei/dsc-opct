from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare independent matched sampling with paired block replacement."
    )
    parser.add_argument(
        "--independent-root",
        type=Path,
        default=Path("outputs/matched_identity_exposure_benchmark_crossed_valid"),
    )
    parser.add_argument(
        "--paired-root",
        type=Path,
        default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filename = "factorial_effects_crossed_valid.csv"
    independent = pd.read_csv(args.independent_root / filename)[
        ["dataset", "task", "contrast", "balanced_accuracy_effect"]
    ].rename(columns={"balanced_accuracy_effect": "independent_effect"})
    paired = pd.read_csv(args.paired_root / filename)[
        ["dataset", "task", "contrast", "balanced_accuracy_effect"]
    ].rename(columns={"balanced_accuracy_effect": "paired_block_effect"})
    comparison = independent.merge(
        paired,
        on=["dataset", "task", "contrast"],
        validate="one_to_one",
    )
    comparison["absolute_difference"] = np.abs(
        comparison.independent_effect - comparison.paired_block_effect
    )
    comparison["direction_agrees"] = (
        np.sign(comparison.independent_effect) == np.sign(comparison.paired_block_effect)
    )
    summary_rows = []
    for contrast, group in comparison.groupby("contrast", sort=True):
        summary_rows.append(
            {
                "contrast": contrast,
                "n_dataset_tasks": len(group),
                "direction_agreement": group.direction_agrees.mean(),
                "mean_absolute_difference": group.absolute_difference.mean(),
                "pearson_correlation": group.independent_effect.corr(group.paired_block_effect),
            }
        )
    summary = pd.DataFrame(summary_rows)
    comparison.to_csv(args.paired_root / "design_sensitivity.csv", index=False)
    summary.to_csv(args.paired_root / "design_sensitivity_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
