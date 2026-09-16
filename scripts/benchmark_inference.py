from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virecon.models import MLPReconstructor, RegionalUncertaintyReconstructor


def _parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _latency(models: list[torch.nn.Module], values: torch.Tensor, iterations: int) -> tuple[float, float]:
    for _ in range(20):
        output = values
        for model in models:
            model(values)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        for _ in range(iterations):
            for model in models:
                model(values)
    end.record()
    torch.cuda.synchronize()
    latency_ms = start.elapsed_time(end) / iterations
    activation_peak = torch.cuda.max_memory_allocated() - baseline_memory
    parameter_memory = sum(_parameters(model) * 4 for model in models)
    peak_mb = (activation_peak + parameter_memory) / (1024**2)
    return float(latency_ms), float(peak_mb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sparse reconstruction inference on CUDA.")
    parser.add_argument("--config", default="configs/virtual_reconstruction.yaml")
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the deployment benchmark.")
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    rows = []
    for budget, regions in config["montages"].items():
        n_regions = len(regions)
        mlp = MLPReconstructor(n_regions, hidden=config["model"]["hidden"], dropout=0.0).cuda().eval()
        residual = RegionalUncertaintyReconstructor(
            n_regions, hidden=config["model"]["hidden"], dropout=0.0
        ).cuda().eval()
        for batch_size in (1, 256):
            values = torch.randn(batch_size, n_regions, 40, device="cuda")
            mlp_ms, mlp_memory = _latency([mlp], values, args.iterations)
            proposed_ms, proposed_memory = _latency([mlp, residual], values, args.iterations)
            rows.extend(
                [
                    {
                        "budget": budget,
                        "channels": 2 * n_regions,
                        "model": "mlp",
                        "batch_size": batch_size,
                        "parameters": _parameters(mlp),
                        "latency_ms": mlp_ms,
                        "peak_cuda_mb": mlp_memory,
                    },
                    {
                        "budget": budget,
                        "channels": 2 * n_regions,
                        "model": "regional",
                        "batch_size": batch_size,
                        "parameters": _parameters(mlp) + _parameters(residual),
                        "latency_ms": proposed_ms,
                        "peak_cuda_mb": proposed_memory,
                    },
                ]
            )
    frame = pd.DataFrame(rows)
    output = ROOT / config["output_root"] / "inference_benchmark.csv"
    frame.to_csv(output, index=False)
    print(frame.round(4).to_string(index=False))
    print(f"Benchmark: {output}")


if __name__ == "__main__":
    main()
