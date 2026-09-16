from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from tqdm.auto import tqdm

PRIMARY_INPUT = Path("outputs/dcs_opct_v11_strong_calibration_baselines/decision_curve_replicates.csv")
CROSSFIT_INPUT = Path("outputs/dcs_opct_v11_crossfit_certified_baselines/decision_curve_replicates.csv")
FREEZE = Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json")
EXPECTED_FREEZE_SHA256 = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
COMPARATORS = (
    "target_platt",
    "target_beta",
    "split_certified_platt",
    "split_certified_beta",
    "crossfit_certified_platt",
    "crossfit_certified_beta",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Paired audit-order inference for frozen DCS-OPCT v11 outputs.")
    parser.add_argument("--primary-input", type=Path, default=PRIMARY_INPUT)
    parser.add_argument("--crossfit-input", type=Path, default=CROSSFIT_INPUT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dcs_opct_v11_paired_inference_crossfit"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--permutation-repetitions", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_ci(values, repetitions, rng):
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    return np.quantile(values[indices].mean(axis=1), (0.025, 0.975))


def randomization_p(values, repetitions, rng):
    observed = abs(float(values.mean()))
    exceed = completed = 0
    while completed < repetitions:
        current = min(5000, repetitions - completed)
        signs = rng.choice((-1.0, 1.0), size=(current, len(values)))
        exceed += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
        completed += current
    return (exceed + 1.0) / (repetitions + 1.0)


def paired_tables(frame, args):
    wide = frame.pivot(index=["dataset", "budget_fraction", "audit_repetition"], columns="method")
    groups = list(frame[["dataset", "budget_fraction"]].drop_duplicates().itertuples(index=False))
    gain_rows, safety_rows = [], []
    progress = tqdm(total=len(groups) * len(COMPARATORS), desc="Paired audit-order inference", unit="comparison", dynamic_ncols=True)
    for group_index, group in enumerate(groups):
        indexed = wide.loc[(group.dataset, group.budget_fraction)]
        dcs_gain = indexed["brier_gain"]["dcs_selective"].to_numpy(float)
        dcs_harm = indexed["material_negative_transfer"]["dcs_selective"].to_numpy(bool)
        configurations = int(indexed["audit_configurations"]["dcs_selective"].iloc[0])
        for comparator_index, comparator in enumerate(COMPARATORS):
            rng = np.random.default_rng(args.seed + group_index * 1009 + comparator_index * 7919)
            comparator_gain = indexed["brier_gain"][comparator].to_numpy(float)
            comparator_harm = indexed["material_negative_transfer"][comparator].to_numpy(bool)
            difference = dcs_gain - comparator_gain
            ci_low, ci_high = bootstrap_ci(difference, args.bootstrap_repetitions, rng)
            gain_rows.append({
                "dataset": group.dataset, "budget_fraction": group.budget_fraction,
                "audit_configurations": configurations, "comparator_method": comparator,
                "paired_repetitions": len(difference),
                "mean_gain_difference_dcs_minus_comparator": float(difference.mean()),
                "mean_gain_difference_ci_low": float(ci_low), "mean_gain_difference_ci_high": float(ci_high),
                "median_gain_difference": float(np.median(difference)),
                "dcs_higher_gain_frequency": float(np.mean(difference > 0)),
                "paired_randomization_p": randomization_p(difference, args.permutation_repetitions, rng),
            })
            dcs_only = int(np.sum(dcs_harm & ~comparator_harm))
            comparator_only = int(np.sum(~dcs_harm & comparator_harm))
            discordant = dcs_only + comparator_only
            safety_rows.append({
                "dataset": group.dataset, "budget_fraction": group.budget_fraction,
                "audit_configurations": configurations, "comparator_method": comparator,
                "dcs_material_negative_count": int(dcs_harm.sum()),
                "comparator_material_negative_count": int(comparator_harm.sum()),
                "dcs_only_harm_count": dcs_only, "comparator_only_harm_count": comparator_only,
                "exact_mcnemar_p": 1.0 if discordant == 0 else float(binomtest(dcs_only, discordant, 0.5).pvalue),
                "material_negative_frequency_difference": float(dcs_harm.mean() - comparator_harm.mean()),
            })
            progress.update(1)
    progress.close()
    gains, safety = pd.DataFrame(gain_rows), pd.DataFrame(safety_rows)
    gains["paired_randomization_p_holm"] = holm(gains.paired_randomization_p)
    safety["exact_mcnemar_p_holm"] = holm(safety.exact_mcnemar_p)
    scope = "retrospective_audit_order_sensitivity_on_fixed_heldout_half"
    gains["inference_scope"], safety["inference_scope"] = scope, scope
    return gains, safety


def upper_bound(events, trials, alpha=0.05):
    if trials == 0:
        return float("nan")
    if events == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, events + 1, trials - events))


def load_combined_replicates(primary_path, crossfit_path):
    primary = pd.read_csv(primary_path)
    crossfit = pd.read_csv(crossfit_path)
    required = {
        "dataset", "budget_fraction", "audit_configurations",
        "audit_repetition", "method", "released", "brier_gain",
        "material_negative_transfer",
    }
    for name, frame in (("primary", primary), ("crossfit", crossfit)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} replicate input is missing columns: {missing}")
        keys = ["dataset", "budget_fraction", "audit_repetition", "method"]
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} replicate input contains duplicate method keys")

    expected_crossfit = {"crossfit_certified_platt", "crossfit_certified_beta"}
    actual_crossfit = set(crossfit.method.unique())
    if actual_crossfit != expected_crossfit:
        raise ValueError(
            f"Cross-fit method set mismatch: expected {sorted(expected_crossfit)}, "
            f"found {sorted(actual_crossfit)}"
        )
    primary_keys = primary.loc[primary.method.eq("dcs_selective"), [
        "dataset", "budget_fraction", "audit_repetition"
    ]].sort_values(["dataset", "budget_fraction", "audit_repetition"]).reset_index(drop=True)
    for method in sorted(expected_crossfit):
        method_keys = crossfit.loc[crossfit.method.eq(method), [
            "dataset", "budget_fraction", "audit_repetition"
        ]].sort_values(["dataset", "budget_fraction", "audit_repetition"]).reset_index(drop=True)
        if not primary_keys.equals(method_keys):
            raise ValueError(f"{method} does not exactly match DCS audit-order keys")
    combined = pd.concat([primary, crossfit], ignore_index=True)
    return combined.sort_values(
        ["dataset", "budget_fraction", "audit_repetition", "method"]
    ).reset_index(drop=True)


def safety_bounds(frame):
    rows = []
    for keys, group in frame.groupby(["dataset", "budget_fraction", "method"], sort=False):
        released = group.released.to_numpy(bool)
        harm = group.material_negative_transfer.to_numpy(bool)
        release_count = int(released.sum())
        harm_count = int(harm.sum())
        released_harm_count = int(np.sum(released & harm))
        rows.append({
            "dataset": keys[0],
            "budget_fraction": keys[1],
            "method": keys[2],
            "audit_repetitions": len(group),
            "release_count": release_count,
            "material_negative_count": harm_count,
            "material_negative_frequency": float(harm.mean()),
            "all_repetition_95pct_upper_bound": upper_bound(harm_count, len(group)),
            "released_action_95pct_upper_bound": upper_bound(released_harm_count, release_count),
            "bound_scope": "descriptive_audit_order_bound_not_future_domain_risk",
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    if sha256(FREEZE) != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("Frozen v11 manifest hash mismatch")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    frame = load_combined_replicates(args.primary_input, args.crossfit_input)
    gains, safety = paired_tables(frame, args)
    bounds = safety_bounds(frame)
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_root / "combined_decision_curve_replicates.csv", index=False)
    gains.to_csv(args.output_root / "paired_gain_inference.csv", index=False)
    safety.to_csv(args.output_root / "paired_safety_inference.csv", index=False)
    bounds.to_csv(args.output_root / "domain_budget_safety_upper_bounds.csv", index=False)
    keys = ["dataset", "budget_fraction", "audit_configurations", "comparator_method", "inference_scope"]
    low_gain = gains.loc[gains.budget_fraction.isin((0.2, 0.4))]
    low_safety = safety.loc[safety.budget_fraction.isin((0.2, 0.4))]
    low_gain.merge(low_safety, on=keys, validate="one_to_one").to_csv(
        args.output_root / "low_label_paired_inference.csv", index=False
    )
    manifest = {
        "status": "retrospective_paired_inference_complete",
        "date": "2026-09-08",
        "frozen_v11_unchanged": True,
        "frozen_v11_sha256": EXPECTED_FREEZE_SHA256,
        "inputs": {
            "primary": str(args.primary_input),
            "crossfit": str(args.crossfit_input),
        },
        "input_sha256": {
            "primary": sha256(args.primary_input),
            "crossfit": sha256(args.crossfit_input),
        },
        "comparators": list(COMPARATORS),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "permutation_repetitions": args.permutation_repetitions,
        "multiple_testing": "Holm across all 150 dataset-budget-comparator cells",
        "claim_boundary": "Audit-order paired sensitivity on reused frozen held-out halves; not independent external repetition or future-domain risk.",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote paired inference to {args.output_root}")


if __name__ == "__main__":
    main()
