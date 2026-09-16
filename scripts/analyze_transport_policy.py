from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


COSTS = [0.0, 0.00025, 0.0005, 0.001, 0.002, 0.005]
FAILURE_PENALTIES = [0.0, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify uncertainty and decision utility for the frozen transport policy."
    )
    parser.add_argument(
        "--policy-root", type=Path, default=Path("outputs/transport_policy_public_development")
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    return parser.parse_args()


def summarize_sample(sample: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_policy_brier_gain": float(sample.brier_gain.mean()),
        "mean_fixed_brier_gain": float(sample.fixed_brier_gain.mean()),
        "policy_minus_fixed_gain": float(
            (sample.brier_gain - sample.fixed_brier_gain).mean()
        ),
        "mean_oracle_brier_gain": float(sample.oracle_brier_gain.mean()),
        "adaptation_rate": float(np.mean(sample.selected_method != "identity")),
        "negative_transfer_rate": float(np.mean(sample.brier_gain < 0)),
        "auc_noninferiority_rate": float(np.mean(sample.auc_delta >= -0.02)),
    }


def cluster_bootstrap(
    decisions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    cluster_labels = decisions[["seed", "outer_held_family"]].astype(str).agg("|".join, axis=1)
    clusters = {
        label: np.flatnonzero(cluster_labels.to_numpy() == label)
        for label in cluster_labels.unique()
    }
    names = np.asarray(list(clusters))
    rng = np.random.default_rng(seed)
    rows = []
    progress = tqdm(total=repetitions, desc="Transport-policy cluster bootstrap", unit="rep", dynamic_ncols=True)
    for repetition in range(repetitions):
        sampled_names = rng.choice(names, size=len(names), replace=True)
        indices = np.concatenate([clusters[name] for name in sampled_names])
        rows.append({"repetition": repetition, **summarize_sample(decisions.iloc[indices])})
        progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def interval_table(point: dict[str, float], bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, estimate in point.items():
        values = bootstrap[metric].to_numpy(float)
        rows.append({
            "metric": metric,
            "estimate": estimate,
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "bootstrap_valid_repetitions": int(np.isfinite(values).sum()),
            "bootstrap_cluster": "seed x held-out shift family",
        })
    return pd.DataFrame(rows)


def decision_utility(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policy_adapted = (decisions.selected_method != "identity").to_numpy(float)
    fixed_adapted = (decisions.fixed_method != "identity").to_numpy(float)
    oracle_adapted = (decisions.oracle_method != "identity").to_numpy(float)
    strategies = {
        "policy": (decisions.brier_gain.to_numpy(float), policy_adapted),
        "fixed": (decisions.fixed_brier_gain.to_numpy(float), fixed_adapted),
        "identity": (np.zeros(len(decisions)), np.zeros(len(decisions))),
        "oracle": (decisions.oracle_brier_gain.to_numpy(float), oracle_adapted),
    }
    for cost in COSTS:
        for penalty in FAILURE_PENALTIES:
            for strategy, (gain, adapted) in strategies.items():
                utility = gain - cost * adapted - penalty * np.maximum(-gain, 0.0)
                rows.append({
                    "strategy": strategy,
                    "adaptation_cost": cost,
                    "negative_transfer_penalty": penalty,
                    "mean_utility": float(np.mean(utility)),
                    "positive_utility_rate": float(np.mean(utility > 0)),
                    "mean_brier_gain": float(np.mean(gain)),
                    "adaptation_rate": float(np.mean(adapted)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    decisions = pd.read_csv(args.policy_root / "losfo_policy_decisions.csv")
    if len(decisions) != 120 or decisions.scenario_id.nunique() != 120:
        raise ValueError("Expected one decision for each of 120 LOSFO scenarios")
    point = summarize_sample(decisions)
    bootstrap = cluster_bootstrap(decisions, args.bootstrap_repetitions, args.bootstrap_seed)
    intervals = interval_table(point, bootstrap)
    utility = decision_utility(decisions)
    counts = (
        decisions.groupby(["outer_held_family", "selected_method"])
        .size()
        .rename("n_selected")
        .reset_index()
    )
    intervals.to_csv(args.policy_root / "policy_cluster_bootstrap_intervals.csv", index=False)
    utility.to_csv(args.policy_root / "decision_utility_curve.csv", index=False)
    counts.to_csv(args.policy_root / "method_selection_counts.csv", index=False)
    print(intervals.to_string(index=False))
    print(utility.loc[
        (utility.adaptation_cost == 0.001)
        & (utility.negative_transfer_penalty == 1.0)
    ].to_string(index=False))


if __name__ == "__main__":
    main()
