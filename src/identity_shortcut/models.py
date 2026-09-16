from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.svm import SVC


def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -10.0, 10.0)


class TorchBinaryMLP(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden: int = 64,
        dropout: float = 0.2,
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        random_state: int = 0,
        device: str = "cuda",
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchBinaryMLP":
        import torch
        from torch import nn

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("gpu_mlp requires CUDA, but CUDA is unavailable")
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        device = torch.device(self.device)
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.long)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_tensor, y_tensor),
            batch_size=min(self.batch_size, len(y_tensor)),
            shuffle=True,
            generator=generator,
        )
        second_hidden = max(16, self.hidden // 2)
        model = nn.Sequential(
            nn.Linear(x.shape[1], self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(second_hidden, 2),
        ).to(device)
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2).astype(float)
        weights = counts.sum() / np.maximum(2.0 * counts, 1.0)
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_x.to(device)), batch_y.to(device))
                loss.backward()
                optimizer.step()
        self.model_ = model.eval()
        self.device_ = device
        self.classes_ = np.asarray([0, 1])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode():
            logits = self.model_(torch.as_tensor(x, dtype=torch.float32, device=self.device_))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


def make_shortcut_models(
    seed: int,
    n_jobs: int,
    names: list[str],
    *,
    device: str,
    gpu_epochs: int,
) -> dict[str, Pipeline]:
    scaled = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clip", FunctionTransformer(_clip)),
    ]
    models = {
        "linear_logistic": Pipeline(
            scaled
            + [
                (
                    "model",
                    LogisticRegression(
                        C=0.1,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=seed,
                    ),
                )
            ]
        ),
        "rbf_svm": Pipeline(
            scaled
            + [
                (
                    "model",
                    SVC(C=1.0, gamma="scale", class_weight="balanced", random_state=seed),
                )
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=10,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "gpu_mlp": Pipeline(
            scaled
            + [
                (
                    "model",
                    TorchBinaryMLP(random_state=seed, device=device, epochs=gpu_epochs),
                )
            ]
        ),
    }
    unknown = set(names) - set(models)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return {name: models[name] for name in names}
