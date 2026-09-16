from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_matched_factorial_effects import CONTRASTS, MODELS
from run_multiseed_protocol_sensitivity import holm_adjust


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classifier and multiplicity sensitivity for the paired factorial audit."
    )
    parser.add_argument(
        "--matched-root",
        type=Path,
        default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid"),
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    return parser.parse_args()


def classifier_effects(summary: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    required = {
        "dataset",
        "task",
        "exposure",
        "model",
        "pooled_balanced_accuracy",
        "split_seed",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing per-seed summary columns: {missing}")

    frame = summary.loc[summary.model.isin(models), list(required)].copy()
    keys = ["dataset", "task", "model", "split_seed"]
    if frame.duplicated(keys + ["exposure"]).any():
        raise ValueError("Duplicate exposure rows in per-seed summary")
    wide = frame.pivot(index=keys, columns="exposure", values="pooled_balanced_accuracy")
    expected_exposures = set(next(iter(CONTRASTS.values())))
    if set(wide.columns) != expected_exposures or wide.isna().any().any():
        raise ValueError(
            f"Expected complete exposures {sorted(expected_exposures)}, observed {sorted(wide.columns)}"
        )

    per_seed_rows = []
    for index, row in wide.iterrows():
        dataset, task, model, split_seed = index
        for contrast, coefficients in CONTRASTS.items():
            effect = sum(coefficients[exposure] * row[exposure] for exposure in coefficients)
            per_seed_rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "contrast": contrast,
                    "model": model,
                    "split_seed": int(split_seed),
                    "balanced_accuracy_effect": float(effect),
                }
            )
    per_seed = pd.DataFrame(per_seed_rows)

    rows = []
    for group_keys, group in per_seed.groupby(
        ["dataset", "task", "contrast", "model"], sort=True
    ):
        dataset, task, contrast, model = group_keys
        values = group.balanced_accuracy_effect.to_numpy(float)
        rows.append(
            {
                "dataset": dataset,
                "task": task,
                "contrast": contrast,
                "model": model,
                "n_split_seeds": len(values),
                "balanced_accuracy_effect": np.mean(values),
                "split_seed_sd": np.std(values, ddof=1),
                "split_seed_min": np.min(values),
                "split_seed_max": np.max(values),
                "positive_seed_fraction": np.mean(values > 0),
            }
        )
    return per_seed, pd.DataFrame(rows)


def verify_panel_average(
    per_seed: pd.DataFrame,
    canonical_per_seed: pd.DataFrame,
    models: list[str],
) -> None:
    observed = (
        per_seed.groupby(["dataset", "task", "contrast", "split_seed"], as_index=False)
        .balanced_accuracy_effect.mean()
        .rename(columns={"balanced_accuracy_effect": "reconstructed"})
    )
    expected = canonical_per_seed.rename(
        columns={"balanced_accuracy_effect": "canonical"}
    )
    merged = observed.merge(
        expected,
        on=["dataset", "task", "contrast", "split_seed"],
        how="outer",
        validate="one_to_one",
    )
    if len(models) != len(MODELS):
        return
    if merged[["reconstructed", "canonical"]].isna().any().any() or not np.allclose(
        merged.reconstructed, merged.canonical, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Classifier effects do not reconstruct the canonical panel effects")


def multiplicity_sensitivity(effects: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "task",
        "contrast",
        "balanced_accuracy_effect",
        "p_bootstrap",
        "p_holm_within_contrast",
    }
    missing = sorted(required - set(effects.columns))
    if missing:
        raise ValueError(f"Missing factorial-effect columns: {missing}")
    output = effects[list(required)].copy()
    output["p_holm_global_all_contrasts"] = holm_adjust(
        output.p_bootstrap.to_numpy(float)
    )
    return output.sort_values(["dataset", "task", "contrast"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.matched_root / "per_seed_summary.csv")
    canonical_per_seed = pd.read_csv(
        args.matched_root / "factorial_effects_per_seed_crossed_valid.csv"
    )
    canonical_effects = pd.read_csv(args.matched_root / "factorial_effects_crossed_valid.csv")

    per_seed, classifier_summary = classifier_effects(summary, args.models)
    verify_panel_average(per_seed, canonical_per_seed, args.models)
    multiplicity = multiplicity_sensitivity(canonical_effects)

    per_seed_path = args.matched_root / "factorial_effects_per_classifier_seed.csv"
    summary_path = args.matched_root / "factorial_effects_classifier_sensitivity.csv"
    multiplicity_path = args.matched_root / "factorial_effects_multiplicity_sensitivity.csv"
    per_seed.to_csv(per_seed_path, index=False)
    classifier_summary.to_csv(summary_path, index=False)
    multiplicity.to_csv(multiplicity_path, index=False)

    manifest = {
        "input": str(args.matched_root / "per_seed_summary.csv"),
        "models": args.models,
        "classifier_inference": "descriptive across the five prespecified split seeds",
        "panel_reconstruction_tolerance": 1e-12,
        "primary_multiplicity_family": "four dataset-task comparisons within each factorial contrast",
        "sensitivity_multiplicity_family": "all 12 dataset-task-contrast tests",
        "outputs": [str(per_seed_path), str(summary_path), str(multiplicity_path)],
    }
    manifest_path = args.matched_root / "factorial_robustness_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Classifier sensitivity")
    print(classifier_summary.to_string(index=False))
    print("\nMultiplicity sensitivity")
    print(multiplicity.to_string(index=False))


if __name__ == "__main__":
    main()
