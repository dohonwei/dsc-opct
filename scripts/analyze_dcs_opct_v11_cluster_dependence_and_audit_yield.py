from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from tqdm.auto import tqdm


EXPECTED_FREEZE_SHA256 = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EXPECTED_MANUSCRIPT_SHA256 = "dcab80f4727be5bc67586989353c36164b66e2aedf4e364ffa0da32f22c1ed79"
EXPECTED_SUPPLEMENTARY_SHA256 = "eea5ea34e2a767cfde70fc62e831b2a12429b9aa178a44a3e73f6dccb5820b3f"
PRIMARY_THRESHOLD = 0.020
PRIMARY_BUDGET = 0.20
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
CLUSTER_COLUMNS = ["dataset", "task", "representation", "model", "split_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc complete-cluster dependence sensitivity and fixed-budget "
            "audit-yield analysis for frozen DCS-OPCT v11."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "outputs/material_event_threshold_robustness_final/"
            "leave_dataset_out_predictions.csv"
        ),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json"),
    )
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=Path("docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex"),
    )
    parser.add_argument(
        "--supplementary",
        type=Path,
        default=Path("docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910"
        ),
    )
    parser.add_argument("--random-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260910)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.predictions, args.freeze, args.manuscript, args.supplementary):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = {
        args.freeze: EXPECTED_FREEZE_SHA256,
        args.manuscript: EXPECTED_MANUSCRIPT_SHA256,
        args.supplementary: EXPECTED_SUPPLEMENTARY_SHA256,
    }
    for path, expected_hash in expected.items():
        observed = sha256(path)
        if observed != expected_hash:
            raise ValueError(
                f"Locked artifact hash mismatch for {path}: "
                f"expected {expected_hash}, observed {observed}"
            )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_root.resolve()}"
        )
    if args.random_repetitions < 1000 or args.bootstrap_repetitions < 1000:
        raise ValueError("At least 1,000 repetitions are required")


def load_predictions(path: Path) -> tuple[pd.DataFrame, list[float]]:
    frame = pd.read_csv(path)
    required = set(CLUSTER_COLUMNS) | {
        "held_out",
        "risk_row_id",
        "nominal_dose",
        "material_effect_threshold",
        "dose_induced_amplification",
        "material_optimism_event",
        "predicted_probability",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    frame = frame.loc[frame.held_out.eq(frame.dataset)].copy()
    thresholds = sorted(frame.material_effect_threshold.astype(float).unique())
    if not any(np.isclose(value, PRIMARY_THRESHOLD) for value in thresholds):
        raise ValueError("Frozen primary material-event threshold is absent")
    keys = ["material_effect_threshold", "risk_row_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Prediction rows are not unique by threshold and risk_row_id")
    cluster_sizes = frame.groupby(["material_effect_threshold", *CLUSTER_COLUMNS]).size()
    if set(cluster_sizes.unique()) != {4}:
        raise ValueError("Every prediction cluster must contain four dose configurations")
    dose_counts = frame.groupby(["material_effect_threshold", *CLUSTER_COLUMNS]).nominal_dose.nunique()
    if not (dose_counts == 4).all():
        raise ValueError("Every prediction cluster must contain four distinct doses")
    return frame, thresholds


def aggregate_clusters(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["material_excess"] = np.maximum(
        work.dose_induced_amplification.to_numpy(float)
        - work.material_effect_threshold.to_numpy(float),
        0.0,
    )
    work["positive_amplification"] = np.maximum(
        work.dose_induced_amplification.to_numpy(float), 0.0
    )
    return (
        work.groupby(["material_effect_threshold", *CLUSTER_COLUMNS], sort=True)
        .agg(
            frozen_cluster_risk=("predicted_probability", "max"),
            mean_frozen_risk=("predicted_probability", "mean"),
            event_configurations=("material_optimism_event", "sum"),
            any_material_event=("material_optimism_event", "max"),
            material_excess=("material_excess", "sum"),
            positive_amplification=("positive_amplification", "sum"),
            configurations=("risk_row_id", "size"),
        )
        .reset_index()
    )


def selection_metrics(selected: pd.DataFrame, all_clusters: pd.DataFrame) -> dict[str, float]:
    total_cluster_events = float(all_clusters.any_material_event.sum())
    total_row_events = float(all_clusters.event_configurations.sum())
    total_excess = float(all_clusters.material_excess.sum())
    precision = float(selected.any_material_event.mean())
    prevalence = float(all_clusters.any_material_event.mean())
    return {
        "selected_clusters": int(len(selected)),
        "labeled_configurations": int(selected.configurations.sum()),
        "cluster_event_sensitivity": float(selected.any_material_event.sum())
        / total_cluster_events,
        "row_event_sensitivity": float(selected.event_configurations.sum())
        / total_row_events,
        "material_excess_captured": (
            float(selected.material_excess.sum()) / total_excess
            if total_excess > 0
            else np.nan
        ),
        "event_cluster_precision": precision,
        "event_enrichment": precision / prevalence,
    }


def select_by_score(group: pd.DataFrame, budget: float, oracle: bool = False) -> pd.DataFrame:
    selected = []
    for _, part in group.groupby("dataset", sort=True):
        count = int(np.ceil(len(part) * budget))
        if oracle:
            columns = [
                "any_material_event",
                "event_configurations",
                "material_excess",
                "frozen_cluster_risk",
                "task",
                "representation",
                "model",
                "split_seed",
            ]
            ascending = [False, False, False, False, True, True, True, True]
        else:
            columns = [
                "frozen_cluster_risk",
                "task",
                "representation",
                "model",
                "split_seed",
            ]
            ascending = [False, True, True, True, True]
        selected.append(part.sort_values(columns, ascending=ascending).head(count))
    return pd.concat(selected, ignore_index=True)


def exact_stratified_event_pvalue(
    group: pd.DataFrame, selected: pd.DataFrame, budget: float
) -> float:
    distribution = np.array([1.0])
    for _, part in group.groupby("dataset", sort=True):
        draws = int(np.ceil(len(part) * budget))
        population = len(part)
        events = int(part.any_material_event.sum())
        lower = max(0, draws - (population - events))
        upper = min(draws, events)
        support = np.arange(lower, upper + 1)
        probability = np.zeros(upper + 1)
        probability[support] = hypergeom.pmf(support, population, events, draws)
        distribution = np.convolve(distribution, probability)
    observed = int(selected.any_material_event.sum())
    return float(distribution[observed:].sum())


def random_selection_metrics(
    group: pd.DataFrame,
    budget: float,
    repetitions: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    dataset_parts = []
    for _, part in group.groupby("dataset", sort=True):
        dataset_parts.append(
            (part.reset_index(drop=True), int(np.ceil(len(part) * budget)))
        )
    rows = []
    for repetition in range(repetitions):
        selected = []
        for part, count in dataset_parts:
            selected.append(part.iloc[rng.choice(len(part), size=count, replace=False)])
        metrics = selection_metrics(pd.concat(selected, ignore_index=True), group)
        rows.append({"random_repetition": repetition, **metrics})
    return pd.DataFrame(rows)


def audit_yield_analysis(
    clusters: pd.DataFrame,
    thresholds: list[float],
    budgets: tuple[float, ...],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    allocation_rows = []
    primary_random = []
    cells = [(threshold, budget) for threshold in thresholds for budget in budgets]
    for cell_index, (threshold, budget) in enumerate(
        tqdm(cells, desc="Complete-cluster audit yield", unit="cell", dynamic_ncols=True)
    ):
        group = clusters.loc[
            np.isclose(clusters.material_effect_threshold, threshold)
        ].reset_index(drop=True)
        selected = select_by_score(group, budget, oracle=False)
        oracle = select_by_score(group, budget, oracle=True)
        risk_metrics = selection_metrics(selected, group)
        oracle_metrics = selection_metrics(oracle, group)
        random = random_selection_metrics(
            group,
            budget,
            repetitions,
            np.random.default_rng(seed + cell_index * 1009),
        )
        row = {
            "material_effect_threshold": float(threshold),
            "audit_budget_fraction": float(budget),
            **{f"risk_{key}": value for key, value in risk_metrics.items()},
            **{f"oracle_{key}": value for key, value in oracle_metrics.items()},
            "cluster_event_exact_random_p_one_sided": exact_stratified_event_pvalue(
                group, selected, budget
            ),
        }
        metric_names = [
            "cluster_event_sensitivity",
            "row_event_sensitivity",
            "material_excess_captured",
            "event_cluster_precision",
            "event_enrichment",
        ]
        for metric in metric_names:
            values = random[metric].to_numpy(float)
            observed = float(risk_metrics[metric])
            row[f"random_{metric}_mean"] = float(np.mean(values))
            row[f"random_{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"random_{metric}_q975"] = float(np.quantile(values, 0.975))
            row[f"risk_minus_random_{metric}"] = observed - float(np.mean(values))
            row[f"randomization_p_{metric}_one_sided"] = float(
                (1 + np.sum(values >= observed - 1e-15)) / (repetitions + 1)
            )
        summary_rows.append(row)

        selected_keys = selected.loc[:, CLUSTER_COLUMNS].copy()
        selected_keys["material_effect_threshold"] = threshold
        selected_keys["audit_budget_fraction"] = budget
        selected_keys["allocation"] = "frozen_risk_ranked"
        selected_keys["allocation_rank"] = np.arange(1, len(selected_keys) + 1)
        allocation_rows.append(selected_keys)

        if np.isclose(threshold, PRIMARY_THRESHOLD):
            saved = random.copy()
            saved.insert(0, "audit_budget_fraction", budget)
            saved.insert(0, "material_effect_threshold", threshold)
            primary_random.append(saved)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(allocation_rows, ignore_index=True),
        pd.concat(primary_random, ignore_index=True),
    )


def anova_icc(values: np.ndarray, cluster_ids: np.ndarray) -> tuple[float, float, float]:
    work = pd.DataFrame({"value": values, "cluster": cluster_ids})
    groups = list(work.groupby("cluster", sort=False).value)
    sizes = {len(group) for _, group in groups}
    if sizes != {4}:
        raise ValueError("ICC calculation requires equal four-row clusters")
    cluster_means = np.asarray([group.mean() for _, group in groups], dtype=float)
    grand_mean = float(work.value.mean())
    n_clusters = len(groups)
    cluster_size = 4
    between = cluster_size * float(np.sum((cluster_means - grand_mean) ** 2))
    ms_between = between / (n_clusters - 1)
    within = float(
        sum(np.sum((group.to_numpy(float) - group.mean()) ** 2) for _, group in groups)
    )
    ms_within = within / (n_clusters * (cluster_size - 1))
    denominator = ms_between + (cluster_size - 1) * ms_within
    icc = (ms_between - ms_within) / denominator if denominator > 0 else np.nan
    design_effect = 1.0 + (cluster_size - 1) * max(float(icc), 0.0)
    effective_n = len(work) / design_effect
    return float(icc), float(design_effect), float(effective_n)


def dependence_summary(frame: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    endpoints = {
        "material_event": "material_optimism_event",
        "amplification": "dose_induced_amplification",
        "frozen_risk_probability": "predicted_probability",
    }
    for threshold in thresholds:
        threshold_frame = frame.loc[
            np.isclose(frame.material_effect_threshold, threshold)
        ].copy()
        scopes = [("overall", threshold_frame)] + [
            (str(dataset), part) for dataset, part in threshold_frame.groupby("dataset")
        ]
        for scope, part in scopes:
            ids = part[CLUSTER_COLUMNS].astype(str).agg("|".join, axis=1).to_numpy()
            for endpoint, column in endpoints.items():
                icc, design_effect, effective_n = anova_icc(
                    part[column].to_numpy(float), ids
                )
                rows.append(
                    {
                        "material_effect_threshold": float(threshold),
                        "scope": scope,
                        "endpoint": endpoint,
                        "configuration_rows": len(part),
                        "complete_clusters": part[CLUSTER_COLUMNS]
                        .drop_duplicates()
                        .shape[0],
                        "cluster_size": 4,
                        "icc_1_1": icc,
                        "design_effect": design_effect,
                        "effective_configuration_rows": effective_n,
                        "effective_fraction": effective_n / len(part),
                    }
                )
    return pd.DataFrame(rows)


def interval_sensitivity(
    frame: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    primary = frame.loc[
        np.isclose(frame.material_effect_threshold, PRIMARY_THRESHOLD)
    ].copy()
    rows = []
    endpoint_functions = {
        "material_event_prevalence": lambda part: float(
            part.material_optimism_event.mean()
        ),
        "mean_amplification": lambda part: float(
            part.dose_induced_amplification.mean()
        ),
    }
    scopes = [("overall", primary)] + [
        (str(dataset), part.copy()) for dataset, part in primary.groupby("dataset")
    ]
    for scope_index, (scope, part) in enumerate(
        tqdm(scopes, desc="Dependence-aware intervals", unit="scope", dynamic_ncols=True)
    ):
        part = part.reset_index(drop=True)
        cluster_frames_by_dataset: dict[str, list[pd.DataFrame]] = {}
        for dataset, dataset_part in part.groupby("dataset", sort=True):
            cluster_frames_by_dataset[str(dataset)] = [
                cluster.reset_index(drop=True)
                for _, cluster in dataset_part.groupby(CLUSTER_COLUMNS, sort=True)
            ]
        rng = np.random.default_rng(seed + scope_index * 4099)
        row_draws = {name: np.empty(repetitions) for name in endpoint_functions}
        cluster_draws = {name: np.empty(repetitions) for name in endpoint_functions}
        for repetition in range(repetitions):
            row_sample = part.iloc[rng.integers(0, len(part), len(part))]
            cluster_parts = []
            for clusters in cluster_frames_by_dataset.values():
                chosen = rng.integers(0, len(clusters), len(clusters))
                cluster_parts.extend(clusters[index] for index in chosen)
            cluster_sample = pd.concat(cluster_parts, ignore_index=True)
            for endpoint, function in endpoint_functions.items():
                row_draws[endpoint][repetition] = function(row_sample)
                cluster_draws[endpoint][repetition] = function(cluster_sample)
        for endpoint, function in endpoint_functions.items():
            row_low, row_high = np.quantile(row_draws[endpoint], [0.025, 0.975])
            cluster_low, cluster_high = np.quantile(
                cluster_draws[endpoint], [0.025, 0.975]
            )
            row_width = float(row_high - row_low)
            cluster_width = float(cluster_high - cluster_low)
            rows.append(
                {
                    "scope": scope,
                    "endpoint": endpoint,
                    "point_estimate": function(part),
                    "naive_row_bootstrap_ci_low": float(row_low),
                    "naive_row_bootstrap_ci_high": float(row_high),
                    "complete_cluster_bootstrap_ci_low": float(cluster_low),
                    "complete_cluster_bootstrap_ci_high": float(cluster_high),
                    "naive_ci_width": row_width,
                    "complete_cluster_ci_width": cluster_width,
                    "cluster_to_naive_width_ratio": cluster_width / row_width,
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    audit: pd.DataFrame, dependence: pd.DataFrame, output_root: Path
) -> None:
    primary = audit.loc[
        np.isclose(audit.material_effect_threshold, PRIMARY_THRESHOLD)
    ].sort_values("audit_budget_fraction")
    x = primary.audit_budget_fraction.to_numpy(float)
    plt.rcParams.update(
        {"font.size": 8.4, "axes.titlesize": 9.5, "axes.labelsize": 8.8}
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.7), constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(
        x,
        primary.risk_cluster_event_sensitivity,
        "o-",
        color="#087F8C",
        label="Frozen risk ranking",
    )
    axis.plot(
        x,
        primary.random_cluster_event_sensitivity_mean,
        "--",
        color="#6B7280",
        label="Random mean",
    )
    axis.plot(
        x,
        primary.oracle_cluster_event_sensitivity,
        "s-",
        color="#C17C24",
        label="Descriptive oracle",
    )
    axis.fill_between(
        x,
        primary.random_cluster_event_sensitivity_q025,
        primary.random_cluster_event_sensitivity_q975,
        color="#9CA3AF",
        alpha=0.18,
    )
    axis.set_title("A  Event-cluster yield at fixed label budgets")
    axis.set_ylabel("Event clusters captured")
    axis.legend(frameon=False, fontsize=7.3)

    axis = axes[0, 1]
    axis.plot(
        x,
        primary.risk_row_event_sensitivity,
        "o-",
        color="#3266A8",
        label="Event configurations",
    )
    axis.plot(
        x,
        primary.risk_material_excess_captured,
        "s-",
        color="#8F4C8A",
        label="Material excess",
    )
    axis.plot(x, x, "--", color="#6B7280", label="Random expectation")
    axis.set_title("B  Burden captured by frozen ranking")
    axis.set_ylabel("Captured fraction")
    axis.legend(frameon=False, fontsize=7.3)

    axis = axes[1, 0]
    axis.plot(x, primary.risk_event_enrichment, "o-", color="#D1495B")
    axis.fill_between(
        x,
        primary.random_event_enrichment_q025,
        primary.random_event_enrichment_q975,
        color="#D1495B",
        alpha=0.12,
    )
    axis.axhline(1.0, color="#6B7280", linestyle="--")
    axis.set_title("C  Event enrichment over prevalence")
    axis.set_ylabel("Enrichment ratio")

    axis = axes[1, 1]
    dep = dependence.loc[
        np.isclose(dependence.material_effect_threshold, PRIMARY_THRESHOLD)
        & dependence.scope.eq("overall")
    ].copy()
    labels = ["Event", "Amplification", "Risk score"]
    colors = ["#087F8C", "#3266A8", "#C17C24"]
    axis.bar(labels, dep.icc_1_1.to_numpy(float), color=colors, width=0.62)
    for index, row in enumerate(dep.itertuples(index=False)):
        axis.text(
            index,
            row.icc_1_1 + 0.025,
            f"n_eff={row.effective_configuration_rows:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    axis.set_ylim(0, 0.9)
    axis.set_title("D  Within-configuration dependence")
    axis.set_ylabel("ICC(1,1)")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        axis.set_axisbelow(True)
    for axis in axes.flat[:3]:
        axis.set_xlabel("Audit budget fraction")
    fig.savefig(
        output_root / "fig_cluster_dependence_audit_yield.pdf", bbox_inches="tight"
    )
    fig.savefig(
        output_root / "fig_cluster_dependence_audit_yield.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    locked_before = {
        "manuscript": sha256(args.manuscript),
        "supplementary": sha256(args.supplementary),
    }
    frame, thresholds = load_predictions(args.predictions)
    clusters = aggregate_clusters(frame)
    args.output_root.mkdir(parents=True, exist_ok=False)

    audit, allocations, primary_random = audit_yield_analysis(
        clusters,
        thresholds,
        BUDGETS,
        args.random_repetitions,
        args.seed,
    )
    dependence = dependence_summary(frame, thresholds)
    intervals = interval_sensitivity(
        frame, args.bootstrap_repetitions, args.seed + 700001
    )

    clusters.to_csv(args.output_root / "complete_cluster_table.csv", index=False)
    audit.to_csv(args.output_root / "audit_yield_summary.csv", index=False)
    allocations.to_csv(args.output_root / "frozen_ranked_allocations.csv", index=False)
    primary_random.to_csv(
        args.output_root / "primary_random_null_replicates.csv", index=False
    )
    dependence.to_csv(args.output_root / "cluster_dependence_summary.csv", index=False)
    intervals.to_csv(
        args.output_root / "interval_dependence_sensitivity.csv", index=False
    )
    make_figure(audit, dependence, args.output_root)

    primary = audit.loc[
        np.isclose(audit.material_effect_threshold, PRIMARY_THRESHOLD)
        & np.isclose(audit.audit_budget_fraction, PRIMARY_BUDGET)
    ].iloc[0]
    conclusion = {
        "analysis_role": "post-hoc dependence and audit-allocation sensitivity",
        "primary_question": (
            "At a frozen 0.02 material-event threshold and 20% complete-cluster "
            "label budget, does the frozen Stage II risk ranking capture more "
            "event-bearing clusters than random allocation?"
        ),
        "primary_result": {
            "labeled_configurations": int(primary.risk_labeled_configurations),
            "cluster_event_sensitivity": float(
                primary.risk_cluster_event_sensitivity
            ),
            "random_mean": float(primary.random_cluster_event_sensitivity_mean),
            "random_97_5_percentile": float(
                primary.random_cluster_event_sensitivity_q975
            ),
            "exact_stratified_one_sided_p": float(
                primary.cluster_event_exact_random_p_one_sided
            ),
            "row_event_sensitivity": float(primary.risk_row_event_sensitivity),
            "material_excess_captured": float(
                primary.risk_material_excess_captured
            ),
        },
        "bounded_interpretation": (
            "On the two public held-out domains used to train and cross-validate the "
            "risk model, complete-cluster frozen ranking retrospectively concentrated "
            "material-event audit yield relative to random allocation. This does not "
            "establish prospective workflow utility, causal necessity, external "
            "effectiveness, or a transferable operating-point guarantee."
        ),
    }
    (args.output_root / "bounded_conclusion.json").write_text(
        json.dumps(conclusion, indent=2), encoding="utf-8"
    )

    output_names = [
        "complete_cluster_table.csv",
        "audit_yield_summary.csv",
        "frozen_ranked_allocations.csv",
        "primary_random_null_replicates.csv",
        "cluster_dependence_summary.csv",
        "interval_dependence_sensitivity.csv",
        "fig_cluster_dependence_audit_yield.pdf",
        "fig_cluster_dependence_audit_yield.png",
        "bounded_conclusion.json",
    ]
    locked_after = {
        "manuscript": sha256(args.manuscript),
        "supplementary": sha256(args.supplementary),
    }
    if locked_after != locked_before:
        raise RuntimeError("Formal manuscript files changed during read-only analysis")
    manifest = {
        "status": "completed_post_hoc_complete_cluster_analysis",
        "analysis_date": "2026-09-10",
        "frozen_v11_unchanged": True,
        "freeze_sha256": sha256(args.freeze),
        "primary_material_event_threshold": PRIMARY_THRESHOLD,
        "primary_audit_budget_fraction": PRIMARY_BUDGET,
        "audit_unit": CLUSTER_COLUMNS,
        "configurations_per_cluster": 4,
        "frozen_cluster_score": (
            "maximum frozen per-configuration predicted probability"
        ),
        "random_comparator": "same per-dataset complete-cluster budget",
        "oracle_role": "outcome-informed descriptive upper bound only",
        "random_repetitions": args.random_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "claim_boundary": conclusion["bounded_interpretation"],
        "locked_tex_sha256_before": locked_before,
        "locked_tex_sha256_after": locked_after,
        "inputs": {
            str(args.predictions): sha256(args.predictions),
            str(args.freeze): sha256(args.freeze),
        },
        "script": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve())
        },
        "outputs": {
            name: sha256(args.output_root / name) for name in output_names
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    display = audit.loc[
        np.isclose(audit.material_effect_threshold, PRIMARY_THRESHOLD),
        [
            "audit_budget_fraction",
            "risk_labeled_configurations",
            "risk_cluster_event_sensitivity",
            "random_cluster_event_sensitivity_mean",
            "risk_row_event_sensitivity",
            "risk_material_excess_captured",
            "cluster_event_exact_random_p_one_sided",
        ],
    ]
    print(display.to_string(index=False))
    print("\nPrimary-threshold overall dependence")
    print(
        dependence.loc[
            np.isclose(dependence.material_effect_threshold, PRIMARY_THRESHOLD)
            & dependence.scope.eq("overall")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
