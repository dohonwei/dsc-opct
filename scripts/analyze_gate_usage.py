from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize validation-selected ViRecon gate usage.")
    parser.add_argument("--dataset", choices=["deap", "hci"], required=True)
    parser.add_argument("--run-name", default="loso_bspc")
    parser.add_argument("--output-root", default="outputs/virtual_reconstruction")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / args.output_root
    path = root / f"{args.dataset}_{args.run_name}_reconstruction_per_subject.csv"
    frame = pd.read_csv(path)
    proposed = frame[frame["method"] == "regional"].copy()
    rows = []
    for budget, group in proposed.groupby("budget"):
        weight = group["base_mlp_weight"].to_numpy(float)
        residual = group["residual_scale"].to_numpy(float)
        rows.append({
            "dataset": args.dataset,
            "budget": budget,
            "n_subjects": len(group),
            "mean_mlp_weight": float(weight.mean()),
            "geometry_only_fraction": float(np.mean(np.isclose(weight, 0.0))),
            "mlp_only_fraction": float(np.mean(np.isclose(weight, 1.0))),
            "hybrid_base_fraction": float(np.mean((weight > 0.0) & (weight < 1.0))),
            "residual_accepted_fraction": float(np.mean(residual > 0.0)),
            "mean_residual_scale": float(residual.mean()),
        })
    summary = pd.DataFrame(rows)
    output = root / f"{args.dataset}_{args.run_name}_gate_usage_summary.csv"
    summary.to_csv(output, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"Gate usage: {output}")


if __name__ == "__main__":
    main()
