from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EXPECTED_MANUSCRIPT = "dcab80f4727be5bc67586989353c36164b66e2aedf4e364ffa0da32f22c1ed79"
EXPECTED_SUPPLEMENT = "eea5ea34e2a767cfde70fc62e831b2a12429b9aa178a44a3e73f6dccb5820b3f"
PRIMARY_THRESHOLD = 0.02
PRIMARY_BUDGET = 0.20
BLOCK_COLUMNS = ["dataset", "task", "split_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Higher-level dataset-task-seed sensitivity for frozen DCS-OPCT v11."
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("outputs/material_event_threshold_robustness_final/leave_dataset_out_predictions.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(args: argparse.Namespace) -> None:
    locked = {
        Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json"): EXPECTED_FREEZE,
        Path("docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex"): EXPECTED_MANUSCRIPT,
        Path("docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex"): EXPECTED_SUPPLEMENT,
    }
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    for path, expected in locked.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Locked artifact changed: {path}")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    if args.bootstrap_repetitions < 5000:
        raise ValueError("At least 5,000 bootstrap repetitions are required")


def load_primary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame.held_out.eq(frame.dataset)
        & np.isclose(frame.material_effect_threshold, PRIMARY_THRESHOLD)
    ].copy()
    required = set(BLOCK_COLUMNS) | {
        "representation", "model", "nominal_dose", "dose_induced_amplification",
        "material_optimism_event", "predicted_probability", "risk_row_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(frame) != 480 or frame.risk_row_id.nunique() != 480:
        raise ValueError("Expected 480 unique primary prediction rows")
    sizes = frame.groupby(BLOCK_COLUMNS).size()
    if len(sizes) != 20 or set(sizes.unique()) != {24}:
        raise ValueError("Expected twenty dataset-task-seed blocks of 24 rows")
    return frame.sort_values(BLOCK_COLUMNS + ["representation", "model", "nominal_dose"])


def aggregate_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["material_excess"] = np.maximum(work.dose_induced_amplification - PRIMARY_THRESHOLD, 0)
    return (
        work.groupby(BLOCK_COLUMNS, sort=True)
        .agg(
            rows=("risk_row_id", "size"),
            event_rows=("material_optimism_event", "sum"),
            event_prevalence=("material_optimism_event", "mean"),
            mean_amplification=("dose_induced_amplification", "mean"),
            material_excess=("material_excess", "sum"),
            max_risk=("predicted_probability", "max"),
            mean_risk=("predicted_probability", "mean"),
            median_risk=("predicted_probability", "median"),
        )
        .reset_index()
    )


def select_blocks(blocks: pd.DataFrame, score: str) -> pd.DataFrame:
    selected = []
    for _, group in blocks.groupby("dataset", sort=True):
        count = int(np.ceil(len(group) * PRIMARY_BUDGET))
        selected.append(
            group.sort_values([score, "task", "split_seed"], ascending=[False, True, True]).head(count)
        )
    return pd.concat(selected, ignore_index=True)


def selection_metrics(selected: pd.DataFrame, blocks: pd.DataFrame) -> dict[str, float]:
    return {
        "selected_blocks": int(len(selected)),
        "labeled_rows": int(selected.rows.sum()),
        "event_row_sensitivity": float(selected.event_rows.sum() / blocks.event_rows.sum()),
        "material_excess_captured": float(selected.material_excess.sum() / blocks.material_excess.sum()),
        "selected_event_prevalence": float(selected.event_rows.sum() / selected.rows.sum()),
        "event_enrichment": float(
            (selected.event_rows.sum() / selected.rows.sum())
            / (blocks.event_rows.sum() / blocks.rows.sum())
        ),
    }


def enumerate_null(blocks: pd.DataFrame) -> pd.DataFrame:
    dataset_choices = []
    for dataset, group in blocks.groupby("dataset", sort=True):
        group = group.reset_index(drop=True)
        count = int(np.ceil(len(group) * PRIMARY_BUDGET))
        dataset_choices.append((dataset, group, list(itertools.combinations(range(len(group)), count))))
    rows = []
    combinations = itertools.product(*(entry[2] for entry in dataset_choices))
    for index, choice_tuple in enumerate(combinations):
        selected = []
        for (_, group, _), choice in zip(dataset_choices, choice_tuple, strict=True):
            selected.append(group.iloc[list(choice)])
        rows.append({"allocation_id": index, **selection_metrics(pd.concat(selected), blocks)})
    return pd.DataFrame(rows)


def stratified_block_bootstrap(
    frame: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    datasets: dict[str, list[pd.DataFrame]] = {}
    for dataset, dataset_frame in frame.groupby("dataset", sort=True):
        datasets[dataset] = [group for _, group in dataset_frame.groupby(["task", "split_seed"], sort=True)]
    rows = []
    for repetition in tqdm(range(repetitions), desc="Dataset-task-seed bootstrap", unit="rep"):
        samples = []
        for groups in datasets.values():
            draws = rng.integers(0, len(groups), size=len(groups))
            samples.extend(groups[index] for index in draws)
        sample = pd.concat(samples, ignore_index=True)
        rows.append({
            "bootstrap_repetition": repetition,
            "event_prevalence": float(sample.material_optimism_event.mean()),
            "mean_amplification": float(sample.dose_induced_amplification.mean()),
        })
    return pd.DataFrame(rows)


def interval_summary(frame: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, source in [
        ("event_prevalence", frame.material_optimism_event.to_numpy(float)),
        ("mean_amplification", frame.dose_induced_amplification.to_numpy(float)),
    ]:
        values = bootstrap[endpoint].to_numpy(float)
        rows.append({
            "endpoint": endpoint,
            "estimate": float(np.mean(source)),
            "block_bootstrap_q025": float(np.quantile(values, 0.025)),
            "block_bootstrap_q975": float(np.quantile(values, 0.975)),
            "block_bootstrap_width": float(np.quantile(values, 0.975) - np.quantile(values, 0.025)),
            "independent_units": 20,
        })
    return pd.DataFrame(rows)


def make_figure(blocks: pd.DataFrame, summary: pd.DataFrame, null: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    colors = {"DEAP": "#2F6B9A", "MAHNOB-HCI": "#15806F"}
    for dataset, group in blocks.groupby("dataset", sort=True):
        axes[0].scatter(group.max_risk, group.event_prevalence, s=38, alpha=0.85,
                        color=colors[dataset], label=dataset)
    axes[0].set_xlabel("Maximum frozen risk in block")
    axes[0].set_ylabel("Material-event prevalence")
    axes[0].set_title("A  Twenty higher-level blocks", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)

    metrics = ["event_row_sensitivity", "material_excess_captured", "event_enrichment"]
    labels = ["Event rows\ncaptured", "Material excess\ncaptured", "Event\nenrichment"]
    observed = summary.loc[summary.score_aggregation.eq("max_risk")].iloc[0]
    means = [null[metric].mean() for metric in metrics]
    obs = [observed[metric] for metric in metrics]
    x = np.arange(len(metrics))
    axes[1].bar(x - 0.18, means, width=0.36, color="#AAB4C2", label="Exact random mean")
    axes[1].bar(x + 0.18, obs, width=0.36, color="#8A5A2B", label="Frozen ranking")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("B  Whole-block audit yield", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].hist(null.event_row_sensitivity, bins=18, color="#AAB4C2", edgecolor="white")
    axes[2].axvline(observed.event_row_sensitivity, color="#A63D40", linewidth=2,
                    label="Frozen ranking")
    axes[2].set_xlabel("Event-row sensitivity")
    axes[2].set_ylabel("Exact allocations")
    axes[2].set_title("C  Exact allocation null", loc="left", fontweight="bold")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Higher-level dataset-task-seed sensitivity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig_higher_level_cluster_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_higher_level_cluster_sensitivity.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate(args)
    args.output_root.mkdir(parents=True)
    frame = load_primary(args.predictions)
    blocks = aggregate_blocks(frame)
    null = enumerate_null(blocks)
    summary_rows = []
    for score in ["max_risk", "mean_risk", "median_risk"]:
        metrics = selection_metrics(select_blocks(blocks, score), blocks)
        summary_rows.append({
            "score_aggregation": score,
            **metrics,
            "exact_random_p_event_row_sensitivity": float(
                np.mean(null.event_row_sensitivity >= metrics["event_row_sensitivity"] - 1e-15)
            ),
            "exact_random_p_material_excess": float(
                np.mean(null.material_excess_captured >= metrics["material_excess_captured"] - 1e-15)
            ),
        })
    summary = pd.DataFrame(summary_rows)
    bootstrap = stratified_block_bootstrap(frame, args.bootstrap_repetitions, args.seed)
    intervals = interval_summary(frame, bootstrap)
    blocks.to_csv(args.output_root / "dataset_task_seed_blocks.csv", index=False)
    summary.to_csv(args.output_root / "higher_level_audit_yield.csv", index=False)
    null.to_csv(args.output_root / "exact_random_block_allocations.csv", index=False)
    intervals.to_csv(args.output_root / "higher_level_bootstrap_intervals.csv", index=False)
    make_figure(blocks, summary, null, args.output_root)
    conclusion = {
        "status": "posthoc_higher_level_sensitivity_completed",
        "independent_source_domains": 2,
        "conditional_dataset_task_seed_units": 20,
        "claim_boundary": (
            "Resampling and allocation at the dataset-task-seed level address shared trial, split, "
            "representation, model, and dose structure within the two fixed development datasets. "
            "They remain conditional post-hoc sensitivity analyses and do not create independent "
            "future-domain replication, prospective effectiveness, or universal non-harm evidence."
        ),
    }
    (args.output_root / "bounded_conclusion.json").write_text(
        json.dumps(conclusion, indent=2), encoding="utf-8"
    )
    outputs = [path for path in args.output_root.iterdir() if path.is_file()]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": EXPECTED_FREEZE,
        "primary_threshold": PRIMARY_THRESHOLD,
        "primary_budget": PRIMARY_BUDGET,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "input_sha256": sha256(args.predictions),
        "outputs": {path.name: sha256(path) for path in sorted(outputs)},
        "claim_boundary": conclusion["claim_boundary"],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(intervals.to_string(index=False))


if __name__ == "__main__":
    main()
