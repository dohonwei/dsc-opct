from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virecon.statistics import ablation_comparisons, downstream_comparisons, feature_group_comparisons, reconstruction_comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Create subject-level paired statistical comparisons.")
    parser.add_argument("--dataset", choices=["deap", "hci"], required=True)
    parser.add_argument("--run-name", default="loso")
    parser.add_argument("--output-root", default="outputs/virtual_reconstruction")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    root = ROOT / args.output_root
    prefix = f"{args.dataset}_{args.run_name}"
    reconstruction = pd.read_csv(root / f"{prefix}_reconstruction_per_subject.csv")
    feature_groups = pd.read_csv(root / f"{prefix}_feature_groups_per_subject.csv")
    downstream = pd.read_csv(root / f"{prefix}_downstream_per_subject.csv")
    comparisons = pd.concat(
        [
            reconstruction_comparisons(reconstruction, args.seed),
            ablation_comparisons(reconstruction, args.seed),
            feature_group_comparisons(feature_groups, args.seed),
            downstream_comparisons(downstream, args.seed),
        ],
        ignore_index=True,
    )
    path = root / f"{prefix}_paired_statistics.csv"
    comparisons.to_csv(path, index=False)
    print(comparisons.to_string(index=False))
    print(f"Statistics: {path}")


if __name__ == "__main__":
    main()
