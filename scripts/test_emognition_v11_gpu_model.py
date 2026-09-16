from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.base import clone
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emognition_v11_contract import GPU_EPOCHS, SEEDS
from identity_shortcut.models import make_shortcut_models


def synthetic_binary_problem(seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(192, 24)).astype(np.float32)
    labels = (
        features[:, 0]
        - 0.7 * features[:, 1]
        + 0.4 * features[:, 2]
        + 0.15 * generator.normal(size=len(features))
        > 0
    ).astype(int)
    return features, labels


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the registered Emognition neural branch")
    progress = tqdm(SEEDS[:2], desc="Emognition synthetic CUDA fits", unit="fit", dynamic_ncols=True)
    for seed in progress:
        features, labels = synthetic_binary_problem(seed)
        template = make_shortcut_models(
            seed,
            n_jobs=1,
            names=["gpu_mlp"],
            device="cuda",
            gpu_epochs=GPU_EPOCHS,
        )["gpu_mlp"]
        model = clone(template)
        model.fit(features, labels)
        fitted = model.named_steps["model"]
        assert fitted.epochs == GPU_EPOCHS
        assert fitted.device_.type == "cuda"
    progress.close()
    print(
        "Emognition synthetic GPU test passed on "
        f"{torch.cuda.get_device_name(0)} with {GPU_EPOCHS} epochs per fit"
    )


if __name__ == "__main__":
    main()
