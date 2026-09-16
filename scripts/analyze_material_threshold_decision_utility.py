from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


EXPECTED_FREEZE_SHA256 = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
PRIMARY_EVENT_THRESHOLD = 0.020
MATERIAL_REGRET_THRESHOLD = 0.020
DEFAULT_COST_RATIOS = [0.0, 0.10, 0.25, 0.50, 1.0, 2.0]
UNIT_COLUMNS = ["dataset", "task", "representation", "split_seed", "nominal_dose"]
CLUSTER_COLUMNS = ["dataset", "task", "representation", "split_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc engineering anchoring of the frozen material-event threshold "
            "to counterfactual model-selection regret and audit utility."
        )
    )
    parser.add_argument(
        "--risk-table",
        type=Path,
        default=Path("outputs/identity_shortcut_risk_classifier/risk_learning_table.csv"),
    )
    parser.add_argument(
        "--threshold-predictions",
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
        "--output-root",
        type=Path,
        default=Path("outputs/material_threshold_decision_utility"),
    )
    parser.add_argument("--cost-ratios", nargs="+", type=float, default=DEFAULT_COST_RATIOS)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(args: argparse.Namespace) -> None:
    required = (args.risk_table, args.threshold_predictions, args.freeze)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")
    freeze_hash = sha256(args.freeze)
    if freeze_hash != EXPECTED_FREEZE_SHA256:
        raise ValueError(
            "Frozen v11 boundary hash mismatch: "
            f"expected {EXPECTED_FREEZE_SHA256}, observed {freeze_hash}"
        )
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_root.resolve()}"
        )
    if args.bootstrap_repetitions < 1000:
        raise ValueError("At least 1,000 bootstrap repetitions are required")
    if any(value < 0 for value in args.cost_ratios):
        raise ValueError("Cost ratios must be nonnegative")


def load_and_validate(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    risk = pd.read_csv(args.risk_table)
    predictions = pd.read_csv(args.threshold_predictions)
    required_risk = set(UNIT_COLUMNS) | {
        "risk_row_id",
        "model",
        "unseen_balanced_accuracy",
        "dose_balanced_accuracy",
        "dose_induced_amplification",
    }
    required_predictions = set(UNIT_COLUMNS) | {
        "held_out",
        "material_effect_threshold",
        "risk_row_id",
        "model",
        "dose_induced_amplification",
        "material_optimism_event",
        "predicted_probability",
        "decision_threshold",
        "predicted_event",
    }
    if missing := required_risk - set(risk.columns):
        raise ValueError(f"Risk table is missing columns: {sorted(missing)}")
    if missing := required_predictions - set(predictions.columns):
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    predictions = predictions.loc[predictions.held_out.eq(predictions.dataset)].copy()
    thresholds = sorted(predictions.material_effect_threshold.astype(float).unique())
    if not any(np.isclose(value, PRIMARY_EVENT_THRESHOLD) for value in thresholds):
        raise ValueError("Threshold predictions do not include the frozen 0.02 endpoint")
    expected_rows = len(risk) * len(thresholds)
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} leave-dataset-out prediction rows, found {len(predictions)}"
        )
    keys = ["material_effect_threshold", "risk_row_id"]
    if predictions.groupby(keys).size().ne(1).any():
        raise ValueError("Threshold predictions are not unique by threshold and risk_row_id")

    merged = predictions.merge(
        risk[["risk_row_id", "unseen_balanced_accuracy", "dose_balanced_accuracy"]],
        on="risk_row_id",
        how="left",
        validate="many_to_one",
    )
    accuracy_columns = ["unseen_balanced_accuracy", "dose_balanced_accuracy"]
    if merged[accuracy_columns].isna().any().any():
        raise ValueError("Prediction rows failed to match the risk table")
    source_amplification = merged.risk_row_id.map(
        risk.set_index("risk_row_id").dose_induced_amplification
    ).to_numpy(float)
    if not np.allclose(
        merged.dose_induced_amplification.to_numpy(float), source_amplification, atol=1e-12
    ):
        raise ValueError("Dose-induced amplification differs between source and predictions")
    if not np.isfinite(merged[accuracy_columns].to_numpy(float)).all():
        raise ValueError("Non-finite balanced-accuracy values found")
    return risk, predictions, thresholds


def build_decision_units(
    risk: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    consequence_rows: list[dict] = []
    groups = list(risk.groupby(UNIT_COLUMNS, sort=True))
    progress = tqdm(
        groups, desc="Reconstructing model choices", unit="unit", dynamic_ncols=True
    )
    for keys, group in progress:
        if group.model.nunique() < 2:
            raise ValueError(f"Decision unit {keys} has fewer than two candidate models")
        reference = group.set_index("model").unseen_balanced_accuracy.astype(float)
        exposed = group.set_index("model").dose_balanced_accuracy.astype(float)
        exposed_winner = str(exposed.idxmax())
        reference_winner = str(reference.idxmax())
        regret = float(reference.max() - reference.loc[exposed_winner])
        consequence_rows.append(
            {
                **dict(zip(UNIT_COLUMNS, keys, strict=True)),
                "n_models": int(group.model.nunique()),
                "exposed_winner": exposed_winner,
                "reference_winner": reference_winner,
                "winner_changed": int(exposed_winner != reference_winner),
                "deployment_regret": regret,
                "material_winner_reversal": int(
                    exposed_winner != reference_winner
                    and regret >= MATERIAL_REGRET_THRESHOLD
                ),
            }
        )
    consequences = pd.DataFrame(consequence_rows)

    aggregate = (
        predictions.groupby(["material_effect_threshold", *UNIT_COLUMNS], sort=True)
        .agg(
            any_material_event=("material_optimism_event", "max"),
            predicted_audit=("predicted_event", "max"),
            maximum_predicted_probability=("predicted_probability", "max"),
            maximum_amplification=("dose_induced_amplification", "max"),
            minimum_amplification=("dose_induced_amplification", "min"),
            model_rows=("model", "nunique"),
        )
        .reset_index()
    )
    units = aggregate.merge(consequences, on=UNIT_COLUMNS, validate="many_to_one")
    if units.model_rows.ne(units.n_models).any():
        raise ValueError("Candidate-model counts disagree after aggregation")
    if len(units) != len(consequences) * len(thresholds):
        raise ValueError("Decision-unit construction produced an unexpected row count")
    return units


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize_policy(group: pd.DataFrame) -> dict:
    event = group.material_winner_reversal.to_numpy(int).astype(bool)
    oracle = group.any_material_event.to_numpy(int).astype(bool)
    audit = group.predicted_audit.to_numpy(int).astype(bool)
    regret = group.deployment_regret.to_numpy(float)
    material_regret = np.where(event, regret, 0.0)

    def diagnostic(flag: np.ndarray, prefix: str) -> dict:
        tp = int(np.sum(flag & event))
        fp = int(np.sum(flag & ~event))
        tn = int(np.sum(~flag & ~event))
        fn = int(np.sum(~flag & event))
        return {
            f"{prefix}_sensitivity": safe_divide(tp, tp + fn),
            f"{prefix}_specificity": safe_divide(tn, tn + fp),
            f"{prefix}_positive_predictive_value": safe_divide(tp, tp + fp),
        }

    values = {
        "n_decision_units": len(group),
        "n_material_reversals": int(event.sum()),
        "material_reversal_prevalence": float(event.mean()),
        "audit_fraction": float(audit.mean()),
        "oracle_event_fraction": float(oracle.mean()),
        "captured_material_reversal_fraction": safe_divide(
            float(np.sum(audit & event)), float(event.sum())
        ),
        "captured_material_regret_fraction": safe_divide(
            float(np.sum(material_regret[audit])), float(np.sum(material_regret))
        ),
        "mean_missed_material_regret_per_unit": float(
            np.sum(material_regret[~audit]) / len(group)
        ),
        "mean_missed_total_regret_per_unit": float(np.sum(regret[~audit]) / len(group)),
        "mean_regret_audited": float(regret[audit].mean()) if audit.any() else float("nan"),
        "mean_regret_not_audited": (
            float(regret[~audit].mean()) if (~audit).any() else float("nan")
        ),
    }
    values.update(diagnostic(oracle, "oracle_event"))
    values.update(diagnostic(audit, "risk_screen"))
    return values


def bootstrap_summary(
    units: pd.DataFrame,
    thresholds: list[float],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    metrics = [
        "audit_fraction",
        "oracle_event_fraction",
        "oracle_event_sensitivity",
        "oracle_event_specificity",
        "risk_screen_sensitivity",
        "risk_screen_specificity",
        "risk_screen_positive_predictive_value",
        "captured_material_reversal_fraction",
        "captured_material_regret_fraction",
        "mean_missed_material_regret_per_unit",
        "mean_missed_total_regret_per_unit",
    ]
    rows: list[dict] = []
    progress = tqdm(thresholds, desc="Cluster bootstrap", unit="threshold", dynamic_ncols=True)
    for threshold in progress:
        group = units.loc[np.isclose(units.material_effect_threshold, threshold)].copy()
        observed = summarize_policy(group)
        cluster_frames = [frame for _, frame in group.groupby(CLUSTER_COLUMNS, sort=True)]
        if len(cluster_frames) != 60 or any(len(frame) != 4 for frame in cluster_frames):
            raise ValueError("Expected 60 complete four-dose decision clusters")
        rng = np.random.default_rng(seed + int(round(threshold * 100000)))
        samples = {metric: np.empty(repetitions) for metric in metrics}
        by_dataset: dict[str, list[pd.DataFrame]] = {}
        for dataset, dataset_group in group.groupby("dataset", sort=True):
            by_dataset[str(dataset)] = [
                frame for _, frame in dataset_group.groupby(CLUSTER_COLUMNS, sort=True)
            ]
        for repetition in range(repetitions):
            sampled_parts = []
            for frames in by_dataset.values():
                indices = rng.integers(0, len(frames), len(frames))
                sampled_parts.extend(frames[index] for index in indices)
            sampled = pd.concat(sampled_parts, ignore_index=True)
            estimate = summarize_policy(sampled)
            for metric in metrics:
                samples[metric][repetition] = estimate[metric]
        row = {"material_effect_threshold": threshold, **observed}
        for metric in metrics:
            finite = samples[metric][np.isfinite(samples[metric])]
            row[f"{metric}_ci_low"] = float(np.quantile(finite, 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(finite, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def net_benefit_table(units: pd.DataFrame, cost_ratios: list[float]) -> pd.DataFrame:
    rows: list[dict] = []
    for threshold, group in units.groupby("material_effect_threshold", sort=True):
        event = group.material_winner_reversal.to_numpy(int).astype(bool)
        audit = group.predicted_audit.to_numpy(int).astype(bool)
        oracle = group.any_material_event.to_numpy(int).astype(bool)
        regret_weight = np.where(
            event,
            group.deployment_regret.to_numpy(float) / MATERIAL_REGRET_THRESHOLD,
            0.0,
        )
        policies = {
            "risk_screen": audit,
            "oracle_material_event": oracle,
            "audit_all": np.ones(len(group), dtype=bool),
            "audit_none": np.zeros(len(group), dtype=bool),
        }
        for cost_ratio in cost_ratios:
            for policy, flag in policies.items():
                captured = float(np.sum(regret_weight[flag & event]) / len(group))
                false_audit = float(np.sum(flag & ~event) / len(group))
                rows.append(
                    {
                        "material_effect_threshold": float(threshold),
                        "false_audit_cost_ratio": float(cost_ratio),
                        "policy": policy,
                        "captured_regret_units_per_decision": captured,
                        "false_audit_fraction": false_audit,
                        "regret_weighted_net_benefit": captured - cost_ratio * false_audit,
                    }
                )
    table = pd.DataFrame(rows)
    defaults = table.loc[table.policy.isin(["audit_all", "audit_none"])].groupby(
        ["material_effect_threshold", "false_audit_cost_ratio"]
    ).regret_weighted_net_benefit.max()
    table["net_benefit_vs_best_default"] = [
        row.regret_weighted_net_benefit
        - defaults.loc[(row.material_effect_threshold, row.false_audit_cost_ratio)]
        for row in table.itertuples(index=False)
    ]
    return table


def pareto_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "material_effect_threshold",
        "audit_fraction",
        "captured_material_reversal_fraction",
        "captured_material_regret_fraction",
        "mean_missed_material_regret_per_unit",
    ]
    work = summary[columns].copy()
    frontier = []
    for row in work.itertuples(index=False):
        dominated = work.loc[
            work.material_effect_threshold.ne(row.material_effect_threshold)
            & work.audit_fraction.le(row.audit_fraction)
            & work.captured_material_regret_fraction.ge(row.captured_material_regret_fraction)
            & (
                work.audit_fraction.lt(row.audit_fraction)
                | work.captured_material_regret_fraction.gt(row.captured_material_regret_fraction)
            )
        ]
        frontier.append(dominated.empty)
    work["pareto_efficient_workload_vs_regret_capture"] = frontier
    work["is_frozen_primary_threshold"] = np.isclose(
        work.material_effect_threshold, PRIMARY_EVENT_THRESHOLD
    )
    return work


def make_figure(summary: pd.DataFrame, net_benefit: pd.DataFrame, output_root: Path) -> None:
    plt.rcParams.update({"font.size": 8.3, "axes.titlesize": 9.4, "axes.labelsize": 8.8})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    x = summary.material_effect_threshold.to_numpy(float)

    axis = axes[0, 0]
    axis.plot(x, summary.oracle_event_sensitivity, "o-", color="#1B6CA8", label="Sensitivity")
    axis.plot(x, summary.oracle_event_specificity, "s-", color="#D1495B", label="Specificity")
    axis.set_title("A  Material event vs material winner reversal")
    axis.set_ylabel("Diagnostic fraction")
    axis.legend(frameon=False, ncol=2)

    axis = axes[0, 1]
    axis.plot(x, summary.audit_fraction, "o-", color="#6B7280", label="Audit workload")
    axis.plot(
        x,
        summary.captured_material_regret_fraction,
        "s-",
        color="#00876C",
        label="Material regret captured",
    )
    axis.set_title("B  Frozen-risk-screen operating consequences")
    axis.set_ylabel("Fraction")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    risk = net_benefit.loc[net_benefit.policy.eq("risk_screen")]
    for ratio, group in risk.groupby("false_audit_cost_ratio", sort=True):
        axis.plot(
            group.material_effect_threshold,
            group.net_benefit_vs_best_default,
            marker="o",
            linewidth=1.4,
            label=f"Cost ratio {ratio:g}",
        )
    axis.axhline(0, color="#6B7280", linestyle="--", linewidth=1.0)
    axis.set_title("C  Regret-weighted net benefit vs best default")
    axis.set_ylabel("0.02-regret units per decision")
    axis.legend(frameon=False, ncol=2, fontsize=7.1)

    axis = axes[1, 1]
    axis.plot(
        summary.audit_fraction,
        summary.captured_material_regret_fraction,
        "o-",
        color="#7A5195",
    )
    for row in summary.itertuples(index=False):
        axis.annotate(
            f"{row.material_effect_threshold:.3f}",
            (row.audit_fraction, row.captured_material_regret_fraction),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_title("D  Workload--regret-capture frontier")
    axis.set_xlabel("Audit workload fraction")
    axis.set_ylabel("Material regret captured")

    for axis in axes.flat:
        if axis is not axes[1, 1]:
            axis.set_xlabel("Material-event threshold")
            axis.axvline(
                PRIMARY_EVENT_THRESHOLD, color="#111827", linestyle=":", linewidth=1.1
            )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.6)
    fig.savefig(output_root / "fig_material_threshold_decision_utility.pdf", bbox_inches="tight")
    fig.savefig(
        output_root / "fig_material_threshold_decision_utility.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    risk, predictions, thresholds = load_and_validate(args)
    args.output_root.mkdir(parents=True, exist_ok=False)

    units = build_decision_units(risk, predictions, thresholds)
    summary = bootstrap_summary(
        units, thresholds, args.bootstrap_repetitions, args.bootstrap_seed
    )
    net_benefit = net_benefit_table(units, sorted(set(args.cost_ratios)))
    pareto = pareto_table(summary)

    units.to_csv(args.output_root / "decision_units.csv", index=False)
    summary.to_csv(args.output_root / "threshold_operational_summary.csv", index=False)
    net_benefit.to_csv(args.output_root / "regret_weighted_net_benefit.csv", index=False)
    pareto.to_csv(args.output_root / "workload_regret_pareto.csv", index=False)
    make_figure(summary, net_benefit, args.output_root)

    output_names = [
        "decision_units.csv",
        "threshold_operational_summary.csv",
        "regret_weighted_net_benefit.csv",
        "workload_regret_pareto.csv",
        "fig_material_threshold_decision_utility.pdf",
        "fig_material_threshold_decision_utility.png",
    ]
    manifest = {
        "status": "completed_post_hoc_engineering_anchoring",
        "analysis_date": "2026-09-08",
        "analysis_role": "post-hoc decision-utility analysis; not threshold optimization",
        "frozen_v11_unchanged": True,
        "primary_material_event_threshold": PRIMARY_EVENT_THRESHOLD,
        "material_winner_reversal_regret_threshold": MATERIAL_REGRET_THRESHOLD,
        "decision_unit": UNIT_COLUMNS,
        "bootstrap_unit": CLUSTER_COLUMNS,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "false_audit_cost_ratios": sorted(set(args.cost_ratios)),
        "net_benefit_definition": (
            "Captured material deployment regret divided by 0.02, per decision unit, "
            "minus false-audit cost ratio times the false-audit fraction."
        ),
        "audit_assumption": (
            "Diagnostic upper-bound assumption that auditing identifies and prevents the "
            "counterfactual model-selection error; no workflow time or monetary cost is claimed."
        ),
        "claim_boundary": (
            "The 0.02 endpoint is an operational model-selection threshold, not a "
            "physiological or clinical materiality threshold. The threshold grid is a "
            "retrospective sensitivity analysis and must not redefine the frozen endpoint."
        ),
        "inputs": {
            str(args.risk_table): sha256(args.risk_table),
            str(args.threshold_predictions): sha256(args.threshold_predictions),
            str(args.freeze): sha256(args.freeze),
        },
        "script": {str(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
        "outputs": {name: sha256(args.output_root / name) for name in output_names},
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    display_columns = [
        "material_effect_threshold",
        "oracle_event_sensitivity",
        "oracle_event_specificity",
        "audit_fraction",
        "risk_screen_sensitivity",
        "risk_screen_specificity",
        "captured_material_regret_fraction",
        "mean_missed_material_regret_per_unit",
    ]
    print(summary[display_columns].to_string(index=False))
    print("\nPareto analysis")
    print(pareto.to_string(index=False))


if __name__ == "__main__":
    main()
