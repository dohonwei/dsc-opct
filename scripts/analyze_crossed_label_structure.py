from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from statsmodels.formula.api import mixedlm
from tqdm.auto import tqdm

from stimulus_identity_contract import CROSSED_TRIAL_DATASETS, require_crossed_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crossed subject-by-video label variance audit.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(CROSSED_TRIAL_DATASETS),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/stimulus_shortcut_audit_crossed_valid"),
    )
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def variance_components(frame: pd.DataFrame, column: str) -> dict[str, float]:
    if frame.duplicated(["subject_id", "trial_id"]).any():
        raise ValueError("Variance decomposition requires one row per observed subject-by-stimulus cell")
    model_frame = frame[["subject_id", "trial_id", column]].copy()
    model_frame["all_observations"] = 1
    model = mixedlm(
        f"{column} ~ 1",
        model_frame,
        groups="all_observations",
        re_formula="0",
        vc_formula={"subject": "0 + C(subject_id)", "stimulus": "0 + C(trial_id)"},
    )
    result = None
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidate = model.fit(reml=True, method=method, maxiter=2000, disp=False)
            if candidate.converged:
                result = candidate
                break
        except (np.linalg.LinAlgError, ValueError):
            continue
    if result is None:
        raise RuntimeError(f"Crossed random-effects REML failed for {column}")
    component_map = dict(zip(result.model.exog_vc.names, result.vcomp, strict=True))
    subject_variance = max(float(component_map["subject"]), 0.0)
    trial_variance = max(float(component_map["stimulus"]), 0.0)
    residual_variance = max(float(result.scale), 0.0)
    total = subject_variance + trial_variance + residual_variance
    n_subjects = frame.subject_id.nunique()
    n_trials = frame.trial_id.nunique()
    complete_cells = n_subjects * n_trials
    return {
        "n_subjects": n_subjects,
        "n_trials": n_trials,
        "n_observed_cells": len(frame),
        "crossed_cell_coverage": len(frame) / complete_cells,
        "reml_converged": bool(result.converged),
        "subject_variance": subject_variance,
        "trial_variance": trial_variance,
        "residual_variance": residual_variance,
        "subject_variance_fraction": subject_variance / total,
        "trial_variance_fraction": trial_variance / total,
        "residual_variance_fraction": residual_variance / total,
    }


def entropy_bits(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    probability = counts / counts.sum()
    return float(-np.sum(probability * np.log2(probability)))


def stratified_permuted_mi(
    frame: pd.DataFrame,
    labels: np.ndarray,
    identity_column: str,
    strata_column: str,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    identity = frame[identity_column].to_numpy()
    observed = mutual_info_score(identity, labels) / np.log(2)
    null = np.empty(permutations, dtype=float)
    strata = frame.groupby(strata_column, sort=False).indices
    for permutation in range(permutations):
        shuffled = labels.copy()
        for indices in strata.values():
            shuffled[indices] = rng.permutation(shuffled[indices])
        null[permutation] = mutual_info_score(identity, shuffled) / np.log(2)
    entropy = entropy_bits(labels)
    null_mean = float(null.mean())
    denominator = entropy - null_mean
    corrected = (observed - null_mean) / denominator if denominator > 0 else np.nan
    return {
        "observed_bits": observed,
        "null_mean_bits": null_mean,
        "corrected_normalized": corrected,
        "permutation_p": (1 + np.count_nonzero(null >= observed)) / (permutations + 1),
    }


def main() -> None:
    args = parse_args()
    require_crossed_datasets(args.datasets, context="Crossed label-structure audit")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    trial_rows = []
    progress = tqdm(total=len(args.datasets) * 2, desc="Crossed label audit", unit="task")
    for specification in args.datasets:
        dataset, path = specification.split("=", 1)
        frame = pd.read_csv(path)
        for task in ("arousal", "valence"):
            score_column = f"{task}_score"
            binary = (frame[score_column].to_numpy() >= 5).astype(int)
            label_entropy = entropy_bits(binary)
            components = variance_components(frame, score_column)
            trial_mi = stratified_permuted_mi(
                frame, binary, "trial_id", "subject_id", args.permutations, rng
            )
            subject_mi = stratified_permuted_mi(
                frame, binary, "subject_id", "trial_id", args.permutations, rng
            )
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    **components,
                    "binary_label_entropy_bits": label_entropy,
                    "trial_label_mutual_information_bits": trial_mi["observed_bits"],
                    "subject_label_mutual_information_bits": subject_mi["observed_bits"],
                    "trial_normalized_mutual_information": trial_mi["observed_bits"] / label_entropy,
                    "subject_normalized_mutual_information": subject_mi["observed_bits"] / label_entropy,
                    "trial_stratified_null_mi_bits": trial_mi["null_mean_bits"],
                    "subject_stratified_null_mi_bits": subject_mi["null_mean_bits"],
                    "trial_bias_corrected_normalized_mi": trial_mi["corrected_normalized"],
                    "subject_bias_corrected_normalized_mi": subject_mi["corrected_normalized"],
                    "trial_mi_permutation_p": trial_mi["permutation_p"],
                    "subject_mi_permutation_p": subject_mi["permutation_p"],
                }
            )
            for trial_id, group in frame.groupby("trial_id"):
                trial_rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "trial_id": trial_id,
                        "score_mean": group[score_column].mean(),
                        "score_std": group[score_column].std(),
                        "high_fraction": np.mean(group[score_column] >= 5),
                    }
                )
            progress.update(1)
    progress.close()
    output = pd.DataFrame(rows)
    output.to_csv(args.output_root / "crossed_label_variance.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(args.output_root / "trial_label_priors.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
