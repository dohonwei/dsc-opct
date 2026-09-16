from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MODALITIES = ("FP1", "FP2", "EDA", "PPG", "SKT", "EOG8_candidate", "EOG9_candidate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relate baseline quality to subject-level normalization utility.")
    parser.add_argument("--root", type=Path, default=Path("outputs/temporal_baseline_gate"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 20, 30])
    return parser.parse_args()


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def utility_table(root: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        frame = pd.read_csv(root / f"seed{seed}_per_subject_results.csv")
        wide = frame.pivot(index="held_out_subject", columns="mode", values="balanced_accuracy")
        for subject, values in wide.iterrows():
            rows.append(
                {
                    "held_out_subject": int(subject),
                    "seed": seed,
                    "baseline_utility": values.baseline_referenced - values.stimulus_only,
                }
            )
    utility = pd.DataFrame(rows)
    summary = utility.groupby("held_out_subject", as_index=False).agg(
        mean_baseline_utility=("baseline_utility", "mean"),
        std_baseline_utility=("baseline_utility", "std"),
        positive_seed_fraction=("baseline_utility", lambda x: float(np.mean(x > 0))),
    )
    return utility, summary


def quality_table(cache_path: Path) -> pd.DataFrame:
    with np.load(cache_path) as payload:
        descriptors = payload["descriptors"]
        referenced = payload["referenced"]
        subjects = payload["subjects"]
        labels = payload["labels"]
    rows = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        row: dict[str, float | int] = {
            "held_out_subject": int(subject) + 1,
            "valence_high_fraction": float(labels[mask, 1].mean()),
        }
        for index, modality in enumerate(MODALITIES):
            centers = descriptors[mask, index * 2]
            log_scales = descriptors[mask, index * 2 + 1]
            row[f"quality__{modality}__center_trial_sd"] = float(np.std(centers))
            row[f"quality__{modality}__log_scale_trial_sd"] = float(np.std(log_scales))
            row[f"quality__{modality}__clipped_fraction"] = float(
                np.mean(np.abs(referenced[mask, :, index]) >= 9.999)
            )
            row[f"quality__{modality}__response_abs_median"] = float(
                np.median(np.abs(referenced[mask, :, index]))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        if not column.startswith("quality__"):
            continue
        coefficient, p_value = stats.spearmanr(frame[column], frame.mean_baseline_utility)
        rows.append({"quality_feature": column, "spearman_r": coefficient, "p_raw": p_value})
    output = pd.DataFrame(rows)
    output["p_holm"] = holm(output.p_raw.to_numpy())
    return output.sort_values("p_raw")


def main() -> None:
    args = parse_args()
    utility, utility_summary = utility_table(args.root, args.seeds)
    quality = quality_table(args.root / "temporal_cache.npz")
    merged = utility_summary.merge(quality, on="held_out_subject", validate="one_to_one")
    association = associations(merged)
    utility.to_csv(args.root / "baseline_utility_by_seed.csv", index=False)
    merged.to_csv(args.root / "baseline_utility_heterogeneity.csv", index=False)
    association.to_csv(args.root / "baseline_quality_associations.csv", index=False)
    print(association.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
