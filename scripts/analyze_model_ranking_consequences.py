from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm


DEFAULT_INPUT = Path(
    "outputs/paired_block_identity_exposure_benchmark_crossed_valid/per_seed_summary.csv"
)
DEFAULT_OUTPUT = Path("outputs/model_ranking_consequences_crossed_valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify whether identity exposure changes model selection and deployment regret."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric", default="pooled_balanced_accuracy")
    parser.add_argument("--reference", default="unseen_both")
    parser.add_argument("--minimum-practical-difference", type=float, default=0.02)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def inversion_fraction(left: pd.Series, right: pd.Series, minimum_difference: float) -> tuple[float, int]:
    inversions = 0
    eligible = 0
    for first, second in itertools.combinations(left.index, 2):
        left_delta = left[first] - left[second]
        right_delta = right[first] - right[second]
        if max(abs(left_delta), abs(right_delta)) < minimum_difference:
            continue
        eligible += 1
        inversions += int(np.sign(left_delta) != np.sign(right_delta))
    return (inversions / eligible if eligible else 0.0), eligible


def per_seed_analysis(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    models = sorted(frame.model.unique())
    exposures = sorted(set(frame.exposure.unique()) - {args.reference})
    rows = []
    groups = frame.groupby(["dataset", "task", "split_seed"], sort=True)
    progress = tqdm(
        total=len(groups) * len(exposures),
        desc="Model-ranking consequences",
        unit="contrast",
        dynamic_ncols=True,
    )
    for (dataset, task, split_seed), group in groups:
        table = group.pivot(index="model", columns="exposure", values=args.metric).loc[models]
        reference = table[args.reference]
        reference_winner = reference.idxmax()
        for exposure in exposures:
            candidate = table[exposure]
            candidate_winner = candidate.idxmax()
            tau = stats.kendalltau(candidate.rank(), reference.rank()).statistic
            rho = stats.spearmanr(candidate.rank(), reference.rank()).statistic
            inversion, eligible_pairs = inversion_fraction(
                candidate, reference, args.minimum_practical_difference
            )
            regret = float(reference.max() - reference[candidate_winner])
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "split_seed": split_seed,
                    "exposure": exposure,
                    "reference": args.reference,
                    "n_models": len(models),
                    "exposure_winner": candidate_winner,
                    "reference_winner": reference_winner,
                    "winner_changed": candidate_winner != reference_winner,
                    "deployment_regret": regret,
                    "material_winner_reversal": (
                        candidate_winner != reference_winner
                        and regret >= args.minimum_practical_difference
                    ),
                    "kendall_tau": float(tau),
                    "spearman_rho": float(rho),
                    "pairwise_inversion_fraction": inversion,
                    "eligible_model_pairs": eligible_pairs,
                }
            )
            progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def bootstrap_summary(per_seed: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    groups = list(per_seed.groupby(["dataset", "task", "exposure"], sort=True))
    progress = tqdm(groups, desc="Ranking bootstrap", unit="setting", dynamic_ncols=True)
    for (dataset, task, exposure), group in progress:
        rng = np.random.default_rng(
            args.seed + sum(ord(character) for character in f"{dataset}|{task}|{exposure}")
        )
        metrics = [
            "winner_changed",
            "material_winner_reversal",
            "deployment_regret",
            "kendall_tau",
            "pairwise_inversion_fraction",
        ]
        sampled = {metric: np.empty(args.bootstrap_repetitions) for metric in metrics}
        values = group[metrics].astype(float).to_numpy()
        for repetition in range(args.bootstrap_repetitions):
            indices = rng.integers(0, len(group), len(group))
            means = values[indices].mean(axis=0)
            for metric, value in zip(metrics, means, strict=True):
                sampled[metric][repetition] = value
        row = {
            "dataset": dataset,
            "task": task,
            "exposure": exposure,
            "n_split_seeds": group.split_seed.nunique(),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(group[metric].astype(float).mean())
            row[f"{metric}_ci_low"] = float(np.quantile(sampled[metric], 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(sampled[metric], 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {"dataset", "task", "model", "split_seed", "exposure", args.metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    if args.reference not in set(frame.exposure):
        raise ValueError(f"Reference exposure {args.reference!r} is absent")
    args.output_root.mkdir(parents=True, exist_ok=True)
    per_seed = per_seed_analysis(frame, args)
    summary = bootstrap_summary(per_seed, args)
    per_seed.to_csv(args.output_root / "per_seed_ranking_consequences.csv", index=False)
    summary.to_csv(args.output_root / "ranking_consequences_summary.csv", index=False)
    manifest = {
        "input": str(args.input),
        "metric": args.metric,
        "reference": args.reference,
        "minimum_practical_difference": args.minimum_practical_difference,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "claim_rule": "A material reversal requires a changed winner and reference-condition regret at or above the prespecified minimum difference.",
    }
    (args.output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
