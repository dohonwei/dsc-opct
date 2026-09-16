from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
PRIMARY = 0.020
THRESHOLDS = [0.010, 0.015, 0.020, 0.025, 0.030]
BUDGETS = [0.10, 0.20, 0.30, 0.40, 0.50]
COST_RATIOS = np.linspace(0.0, 1.0, 21)
PRED_CLUSTER = ["dataset", "task", "representation", "model", "split_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anchor the frozen material endpoint to engineering decisions.")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/material_event_threshold_robustness_final/leave_dataset_out_predictions.csv"))
    parser.add_argument("--ranking", type=Path, default=Path("outputs/paired_block_identity_exposure_benchmark_crossed_valid/per_seed_summary.csv"))
    parser.add_argument("--freeze", type=Path, default=Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/material_threshold_engineering_anchor"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260908)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> None:
    for path in (args.predictions, args.ranking, args.freeze):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(args.freeze) != FREEZE_HASH:
        raise ValueError("Frozen v11 hash mismatch")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_root.resolve()}")
    if args.bootstrap_repetitions < 1000:
        raise ValueError("At least 1,000 bootstrap repetitions are required")


def ranking_margins(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = frame.loc[frame.exposure.eq("unseen_both")]
    unit_rows, pair_rows = [], []
    keys = ["dataset", "task", "split_seed"]
    for key, group in reference.groupby(keys, sort=True):
        values = group.set_index("model").pooled_balanced_accuracy.astype(float)
        if len(values) != 4:
            raise ValueError(f"Expected four candidates in ranking unit {key}")
        ordered = values.sort_values(ascending=False)
        base = dict(zip(keys, key, strict=True))
        unit_rows.append({**base, "top2_margin": float(ordered.iloc[0] - ordered.iloc[1])})
        for left, right in itertools.combinations(values.index, 2):
            pair_rows.append({**base, "left_model": left, "right_model": right, "absolute_pair_margin": float(abs(values[left] - values[right]))})
    return pd.DataFrame(unit_rows), pd.DataFrame(pair_rows)


def margin_summary(units: pd.DataFrame, pairs: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    top = units.top2_margin.to_numpy(float)
    pair = pairs.absolute_pair_margin.to_numpy(float).reshape(len(units), 6)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(units), size=(repetitions, len(units)))
    rows = []
    for threshold in tqdm(THRESHOLDS, desc="Margin bootstrap", unit="threshold", dynamic_ncols=True):
        top_samples = (top[sampled] <= threshold).mean(axis=1)
        pair_samples = (pair[sampled] <= threshold).mean(axis=(1, 2))
        rows.append({
            "distortion_threshold": threshold,
            "top2_margin_at_or_below": float((top <= threshold).mean()),
            "top2_margin_ci_low": float(np.quantile(top_samples, 0.025)),
            "top2_margin_ci_high": float(np.quantile(top_samples, 0.975)),
            "pair_margin_at_or_below": float((pair <= threshold).mean()),
            "pair_margin_ci_low": float(np.quantile(pair_samples, 0.025)),
            "pair_margin_ci_high": float(np.quantile(pair_samples, 0.975)),
        })
    return pd.DataFrame(rows)


def cluster_indices(group: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    result = {}
    for dataset, dataset_group in group.groupby("dataset", sort=True):
        result[str(dataset)] = [part.index.to_numpy(int) for _, part in dataset_group.groupby(PRED_CLUSTER, sort=True)]
    if set(len(indices) for groups in result.values() for indices in groups) != {4}:
        raise ValueError("Prediction clusters must contain all four nonzero doses")
    return result


def budget_metrics(dataset: np.ndarray, score: np.ndarray, event: np.ndarray, amplitude: np.ndarray, budget: float) -> np.ndarray:
    selected = np.zeros(len(score), dtype=bool)
    for name in np.unique(dataset):
        indices = np.flatnonzero(dataset == name)
        count = int(np.ceil(len(indices) * budget))
        selected[indices[np.argsort(-score[indices], kind="stable")[:count]]] = True
    sensitivity = np.sum(selected & event) / np.sum(event)
    precision = np.sum(selected & event) / np.sum(selected)
    return np.array([sensitivity, precision / event.mean(), amplitude[selected & event].sum() / amplitude[event].sum()])


def audit_budget_summary(predictions: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rows = []
    progress = tqdm(total=len(THRESHOLDS) * len(BUDGETS), desc="Audit-utility bootstrap", unit="cell", dynamic_ncols=True)
    for threshold in THRESHOLDS:
        group = predictions.loc[np.isclose(predictions.material_effect_threshold, threshold)].reset_index(drop=True)
        clusters = cluster_indices(group)
        dataset = group.dataset.to_numpy(str)
        score = group.predicted_probability.to_numpy(float)
        event = group.material_optimism_event.to_numpy(int).astype(bool)
        amplitude = np.maximum(group.dose_induced_amplification.to_numpy(float), 0.0)
        rng = np.random.default_rng(seed + int(threshold * 100000))
        sampled_indices = []
        for _ in range(repetitions):
            parts = []
            for groups in clusters.values():
                chosen = rng.integers(0, len(groups), len(groups))
                parts.extend(groups[index] for index in chosen)
            sampled_indices.append(np.concatenate(parts))
        for budget in BUDGETS:
            observed = budget_metrics(dataset, score, event, amplitude, budget)
            samples = np.empty((repetitions, 3))
            for repetition, indices in enumerate(sampled_indices):
                samples[repetition] = budget_metrics(dataset[indices], score[indices], event[indices], amplitude[indices], budget)
            row = {"material_effect_threshold": threshold, "audit_budget_fraction": budget}
            names = ["event_sensitivity", "event_enrichment", "material_amplification_captured"]
            for column, value, values in zip(names, observed, samples.T, strict=True):
                row[column] = float(value)
                row[f"{column}_ci_low"] = float(np.quantile(values, 0.025))
                row[f"{column}_ci_high"] = float(np.quantile(values, 0.975))
            rows.append(row)
            progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def decision_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    group = predictions.loc[np.isclose(predictions.material_effect_threshold, PRIMARY)]
    event = group.material_optimism_event.to_numpy(int).astype(bool)
    audit = group.predicted_event.to_numpy(int).astype(bool)
    rows = []
    for cost in COST_RATIOS:
        for policy, selected in {"risk_screen": audit, "audit_all": np.ones(len(group), bool), "audit_none": np.zeros(len(group), bool)}.items():
            tp = np.sum(selected & event) / len(group)
            fp = np.sum(selected & ~event) / len(group)
            rows.append({"false_audit_cost_ratio": cost, "policy": policy, "net_benefit": float(tp - cost * fp)})
    result = pd.DataFrame(rows)
    defaults = result[result.policy.isin(["audit_all", "audit_none"])].groupby("false_audit_cost_ratio").net_benefit.max()
    result["net_benefit_vs_best_default"] = [row.net_benefit - defaults.loc[row.false_audit_cost_ratio] for row in result.itertuples(index=False)]
    return result


def make_figure(margins: pd.DataFrame, audit: pd.DataFrame, curve: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 8.2, "axes.titlesize": 9.3, "axes.labelsize": 8.7})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    axes[0, 0].plot(margins.distortion_threshold, margins.top2_margin_at_or_below, "o-", label="Top-two margin", color="#1B6CA8")
    axes[0, 0].plot(margins.distortion_threshold, margins.pair_margin_at_or_below, "s-", label="All pair margins", color="#D1495B")
    axes[0, 0].set(title="A  Candidate margins vulnerable to distortion", ylabel="Fraction at or below threshold", xlabel="Balanced-accuracy distortion")
    axes[0, 0].legend(frameon=False)
    primary = audit[np.isclose(audit.material_effect_threshold, PRIMARY)]
    axes[0, 1].plot(primary.audit_budget_fraction, primary.event_sensitivity, "o-", color="#00876C", label="Events captured")
    axes[0, 1].plot(primary.audit_budget_fraction, primary.material_amplification_captured, "s-", color="#7A5195", label="Amplification captured")
    axes[0, 1].plot(primary.audit_budget_fraction, primary.audit_budget_fraction, "--", color="#6B7280", label="Random expectation")
    axes[0, 1].set(title="B  Risk-ranked audit allocation at 0.02", ylabel="Captured fraction", xlabel="Audit budget fraction")
    axes[0, 1].legend(frameon=False)
    budget20 = audit[np.isclose(audit.audit_budget_fraction, 0.20)]
    axes[1, 0].plot(budget20.material_effect_threshold, budget20.event_enrichment, "o-", color="#E17C05")
    axes[1, 0].axhline(1, color="#6B7280", linestyle="--")
    axes[1, 0].set(title="C  Event enrichment at 20% audit budget", ylabel="Enrichment over prevalence", xlabel="Material-event threshold")
    for policy, group in curve.groupby("policy", sort=False):
        axes[1, 1].plot(group.false_audit_cost_ratio, group.net_benefit, linewidth=1.6, label=policy.replace("_", " "))
    axes[1, 1].set(title="D  Configuration-level decision curve at 0.02", ylabel="Net benefit per configuration", xlabel="False-audit / missed-event cost ratio")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    axes[0, 0].axvline(PRIMARY, color="#111827", linestyle=":")
    axes[1, 0].axvline(PRIMARY, color="#111827", linestyle=":")
    fig.savefig(output / "fig_material_threshold_engineering_anchor.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_material_threshold_engineering_anchor.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate(args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions.held_out.eq(predictions.dataset)].copy()
    ranking = pd.read_csv(args.ranking)
    units, pairs = ranking_margins(ranking)
    margins = margin_summary(units, pairs, args.bootstrap_repetitions, args.seed)
    audit = audit_budget_summary(predictions, args.bootstrap_repetitions, args.seed)
    curve = decision_curve(predictions)
    units.to_csv(args.output_root / "ranking_unit_margins.csv", index=False)
    pairs.to_csv(args.output_root / "candidate_pair_margins.csv", index=False)
    margins.to_csv(args.output_root / "margin_vulnerability_summary.csv", index=False)
    audit.to_csv(args.output_root / "audit_budget_utility.csv", index=False)
    curve.to_csv(args.output_root / "configuration_decision_curve.csv", index=False)
    make_figure(margins, audit, curve, args.output_root)
    names = ["ranking_unit_margins.csv", "candidate_pair_margins.csv", "margin_vulnerability_summary.csv", "audit_budget_utility.csv", "configuration_decision_curve.csv", "fig_material_threshold_engineering_anchor.pdf", "fig_material_threshold_engineering_anchor.png"]
    manifest = {
        "status": "completed_post_hoc_engineering_anchor",
        "analysis_date": "2026-09-08",
        "frozen_v11_unchanged": True,
        "analysis_role": "Post-hoc operational anchoring, not endpoint optimization or physiological validation.",
        "primary_threshold": PRIMARY,
        "ranking_question": "How often is a candidate-model margin small enough for a distortion of this magnitude to change engineering selection?",
        "audit_question": "How much material-event burden is captured when configurations are audited in frozen-risk-score order?",
        "decision_curve_boundary": "A missed material event has unit cost and a false audit the displayed relative cost; this is dimensionless scenario analysis, not measured monetary utility.",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "inputs": {str(args.predictions): sha256(args.predictions), str(args.ranking): sha256(args.ranking), str(args.freeze): sha256(args.freeze)},
        "script": {str(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
        "outputs": {name: sha256(args.output_root / name) for name in names},
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(margins.to_string(index=False))
    print("\nPrimary-threshold audit utility")
    print(audit[np.isclose(audit.material_effect_threshold, PRIMARY)].to_string(index=False))


if __name__ == "__main__":
    main()
