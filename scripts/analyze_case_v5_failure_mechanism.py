from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm.auto import tqdm


GROUPINGS = ["task", "representation", "model", "split_seed", "nominal_dose"]
CLUSTER_COLUMNS = ["task", "representation", "model", "split_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc diagnosis of the CASE v5 certificate failure.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/case_v5_external_confirmation"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/case_v5_posthoc_failure_analysis"),
    )
    return parser.parse_args()


def safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(probability, method="average")
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    labels = frame.material_optimism_event.to_numpy(int)
    identity = frame.probability_identity.to_numpy(float)
    candidate = frame.probability_quantile_mapping.to_numpy(float)
    identity_auc = safe_auc(labels, identity)
    candidate_auc = safe_auc(labels, candidate)
    rank_correlation = pd.Series(identity).corr(pd.Series(candidate), method="spearman")
    return {
        "n": int(len(frame)),
        "events": int(labels.sum()),
        "event_rate": float(labels.mean()),
        "brier_gain": float(np.mean((labels - identity) ** 2 - (labels - candidate) ** 2)),
        "identity_auc": identity_auc,
        "candidate_auc": candidate_auc,
        "auc_delta": (
            float(candidate_auc - identity_auc)
            if np.isfinite(identity_auc) and np.isfinite(candidate_auc)
            else float("nan")
        ),
        "probability_mean_shift": float(np.mean(candidate - identity)),
        "probability_rank_correlation": float(rank_correlation),
        "decision_flip_rate": float(np.mean((identity >= 0.5) != (candidate >= 0.5))),
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite post-hoc analysis: {args.output_root}")
    gate = json.loads((args.input_root / "external_confirmation_gate.json").read_text("utf-8"))
    if gate["candidate_method"] != "quantile_mapping":
        raise ValueError("This diagnostic is registered only for the observed quantile-mapping candidate")

    partitions = {
        "audit": pd.read_csv(args.input_root / "primary_audit_labeled.csv"),
        "heldout": pd.read_csv(args.input_root / "primary_heldout_results.csv"),
    }
    rows = []
    work = []
    for partition, frame in partitions.items():
        work.append((partition, "overall", "all", frame))
        for grouping in GROUPINGS:
            for level, group in frame.groupby(grouping, sort=True, dropna=False):
                work.append((partition, grouping, str(level), group))

    for partition, grouping, level, frame in tqdm(work, desc="CASE post-hoc subgroup metrics", unit="group"):
        rows.append(
            {
                "partition": partition,
                "grouping": grouping,
                "level": level,
                **metrics(frame),
            }
        )
    subgroup = pd.DataFrame(rows)

    cluster_rows = []
    for partition, frame in partitions.items():
        for key, cluster in frame.groupby(CLUSTER_COLUMNS, sort=True, dropna=False):
            result = metrics(cluster)
            cluster_rows.append(
                {
                    "partition": partition,
                    **dict(zip(CLUSTER_COLUMNS, key, strict=True)),
                    **result,
                }
            )
    clusters = pd.DataFrame(cluster_rows)

    finite_subgroups = subgroup.loc[
        subgroup.grouping.ne("overall") & subgroup.auc_delta.notna()
    ]
    summary = {
        "status": "post_hoc_diagnostic_complete",
        "primary_claim_changed": False,
        "primary_claim_supported": bool(gate["claim_supported"]),
        "candidate_method": gate["candidate_method"],
        "selected_method": gate["selected_method"],
        "audit_overall": metrics(partitions["audit"]),
        "heldout_counterfactual_candidate_overall": metrics(partitions["heldout"]),
        "subgroups_with_finite_auc": int(len(finite_subgroups)),
        "subgroups_below_auc_noninferiority_margin": int((finite_subgroups.auc_delta < -0.02).sum()),
        "clusters_with_negative_brier_gain": int((clusters.brier_gain < 0).sum()),
        "clusters_total": int(len(clusters)),
        "interpretation_boundary": (
            "Exploratory diagnosis only. It cannot rescue, repeat, or redefine the registered CASE gate."
        ),
    }

    args.output_root.mkdir(parents=True, exist_ok=False)
    subgroup.to_csv(args.output_root / "subgroup_metrics.csv", index=False)
    clusters.to_csv(args.output_root / "cluster_metrics.csv", index=False)
    (args.output_root / "posthoc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
