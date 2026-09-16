from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virecon.data import load_reconstruction_data
from virecon.pipeline import ModelConfig, run_loso, write_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run subject-independent virtual FP1/FP2 reconstruction.")
    parser.add_argument("--config", default="configs/virtual_reconstruction.yaml")
    parser.add_argument("--dataset", choices=["deap", "hci"], default="deap")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "interpolation",
            "ridge",
            "mlp",
            "mc_dropout",
            "regional",
            "regional_no_geo",
            "regional_no_gate",
            "regional_no_uncertainty",
        ],
        default=["interpolation", "ridge", "mlp", "mc_dropout", "regional"],
    )
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    raw = dict(cfg["model"])
    if args.epochs is not None: raw["epochs"] = args.epochs
    if args.quick: raw.update({"epochs": min(raw["epochs"], 2), "patience": 2, "hidden": min(raw["hidden"], 32)})
    data = load_reconstruction_data(cfg["paths"]["feature_root"], args.dataset)
    run_name = args.run_name or ("smoke" if args.quick else "loso")
    output_root = ROOT / cfg["output_root"]
    results = run_loso(
        data,
        cfg["montages"],
        args.methods,
        ModelConfig(**raw),
        args.device,
        args.max_subjects,
        cfg["seed"],
        output_root / "predictions" / f"{args.dataset}_{run_name}",
    )
    paths = write_results(results, output_root, args.dataset, run_name)
    print(results.reconstruction.groupby(["budget", "method"])[["log_mae", "log_rmse", "log_r2", "log_feature_correlation"]].mean().round(4))
    print("Results:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
