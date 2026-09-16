from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiofuse.data import load_eppvr
from physiofuse.pipeline import TrainConfig, run_eppvr_loso, write_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run subject-independent EPPVR PhysioFuse-DG experiments.")
    parser.add_argument("--config", default="configs/physiofuse.yaml")
    parser.add_argument("--methods", nargs="+", default=["eeg", "pps", "early", "late", "physiofuse"], choices=["eeg", "pps", "early", "late", "physiofuse"])
    parser.add_argument("--tasks", nargs="+", default=["arousal", "valence"], choices=["arousal", "valence"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-subjects", type=int, default=None, help="Use only the first N subjects for a smoke test.")
    parser.add_argument("--run-name", default=None, help="Output filename prefix; defaults to eppvr_loso or smoke.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    raw = dict(cfg["train"])
    if args.epochs is not None: raw["epochs"] = args.epochs
    if args.quick:
        raw.update({"epochs": min(raw["epochs"], 2), "patience": 2, "hidden": min(raw["hidden"], 16)})
    data = load_eppvr(cfg["paths"]["eppvr_root"], cfg["paths"]["eppvr_metadata_root"])
    results = run_eppvr_loso(data, args.methods, args.tasks, TrainConfig(**raw), args.seeds or cfg["seeds"], args.device, args.max_subjects)
    run_name = args.run_name or ("smoke" if args.quick else "eppvr_loso")
    paths = write_results(results, ROOT / cfg["output_root"], run_name)
    print(results.groupby(["task", "method"])[["balanced_accuracy", "macro_f1", "mcc", "auroc"]].mean().round(4))
    print(f"Results: {paths[0]} | {paths[1]}")


if __name__ == "__main__":
    main()
