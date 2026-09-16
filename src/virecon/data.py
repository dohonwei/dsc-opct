from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReconstructionData:
    dataset: str
    target: np.ndarray
    regions: dict[str, np.ndarray]
    subjects: np.ndarray
    trials: np.ndarray
    labels: dict[str, np.ndarray]
    feature_names: tuple[str, ...]


class FeatureScaler:
    @staticmethod
    def _stabilize(values: np.ndarray) -> np.ndarray:
        return np.sign(values) * np.log1p(np.abs(values))

    @staticmethod
    def _restore(values: np.ndarray) -> np.ndarray:
        return np.sign(values) * np.expm1(np.abs(values))

    def fit(self, values: np.ndarray) -> "FeatureScaler":
        flat = self._stabilize(values).reshape(-1, values.shape[-1])
        self.mean_ = flat.mean(axis=0, keepdims=True).astype(np.float32)
        self.std_ = np.maximum(flat.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((self._stabilize(values) - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        # Predictions outside the observed log-feature domain are numerical
        # failures, not meaningful physical feature magnitudes.
        stabilized = np.clip(values * self.std_ + self.mean_, -30.0, 30.0)
        return self._restore(stabilized).astype(np.float64)


def load_reconstruction_data(feature_root: str | Path, dataset: str) -> ReconstructionData:
    root = Path(feature_root) / dataset.lower()
    if not root.exists():
        raise FileNotFoundError(root)
    regions = {
        path.stem.removeprefix("eeg_feature_"): np.load(path).astype(np.float32)
        for path in root.glob("eeg_feature_*.npy")
    }
    target = regions.pop("fp12")
    expected = {"af34", "f34", "fc12", "t78"}
    if not expected.issubset(regions) or target.shape[1] != 40:
        raise ValueError(f"{dataset} does not satisfy the paper40 sparse-montage contract.")
    subjects, trials = np.load(root / "subject_ids.npy"), np.load(root / "trial_ids.npy")
    labels = {"arousal": np.load(root / "labels_arousal.npy"), "valence": np.load(root / "labels_valence.npy")}
    manifest = json.loads((root / "feature_manifest.json").read_text(encoding="utf-8"))
    feature_names = tuple(manifest.get("feature_names", ()))
    if len(feature_names) != target.shape[1]:
        raise ValueError(f"{dataset} manifest does not describe all target features.")
    if not all(len(values) == len(target) for values in [subjects, trials, *regions.values(), *labels.values()]):
        raise ValueError(f"{dataset} feature arrays have inconsistent sample counts.")
    trial_keys = np.stack([subjects, trials], axis=1)
    _, trial_inverse = np.unique(trial_keys, axis=0, return_inverse=True)
    for label_name, values in labels.items():
        for trial_index in range(int(trial_inverse.max()) + 1):
            if np.unique(values[trial_inverse == trial_index]).size != 1:
                raise ValueError(f"{dataset} {label_name} labels vary within a subject/trial unit.")
    return ReconstructionData(dataset.lower(), target, regions, subjects, trials, labels, feature_names)


def source_tensor(data: ReconstructionData, region_names: list[str], indices: np.ndarray) -> np.ndarray:
    return np.stack([data.regions[name][indices] for name in region_names], axis=1)
