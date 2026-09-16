from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.base import clone
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amigos_v11_contract import GPU_EPOCHS, MODELS, SEEDS
from identity_shortcut.models import make_shortcut_models


def synthetic_binary_problem(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(192, 84)).astype(np.float32)
    labels = (
        features[:, 0]
        - 0.7 * features[:, 1]
        + 0.4 * features[:, 2]
        + 0.15 * generator.normal(size=len(features))
        > 0
    ).astype(int)
    return features[:160], labels[:160], features[160:]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("AMIGOS registered gpu_mlp test requires CUDA")
    if MODELS != ("linear_logistic", "gpu_mlp") or GPU_EPOCHS != 40:
        raise RuntimeError("AMIGOS model contract changed")

    progress = tqdm(SEEDS[:2], desc="AMIGOS synthetic CUDA fits", unit="fit", dynamic_ncols=True)
    for seed in progress:
        train_x, train_y, test_x = synthetic_binary_problem(seed)
        template = make_shortcut_models(
            seed,
            n_jobs=1,
            names=["gpu_mlp"],
            device="cuda",
            gpu_epochs=GPU_EPOCHS,
        )["gpu_mlp"]
        fitted = clone(template).fit(train_x, train_y)
        probabilities = fitted.predict_proba(test_x)
        assert probabilities.shape == (32, 2)
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        model = fitted.named_steps["model"]
        assert model.epochs == 40
        assert model.device_.type == "cuda"
    progress.close()
    print(
        "AMIGOS synthetic GPU test passed on "
        f"{torch.cuda.get_device_name(0)} with {GPU_EPOCHS} epochs per fit"
    )


if __name__ == "__main__":
    main()
