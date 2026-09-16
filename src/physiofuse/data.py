from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrialSet:
    name: str
    eeg: np.ndarray
    pps: np.ndarray
    arousal: np.ndarray
    valence: np.ndarray

    @property
    def n_subjects(self) -> int:
        return self.eeg.shape[0]

    @property
    def n_trials(self) -> int:
        return self.eeg.shape[1]

    def select_subjects(self, indices: np.ndarray) -> "TrialSet":
        return TrialSet(self.name, self.eeg[indices], self.pps[indices], self.arousal[indices], self.valence[indices])


class SequenceScaler:
    def fit(self, values: np.ndarray) -> "SequenceScaler":
        flat = values.reshape(-1, values.shape[-1])
        self.mean = flat.mean(0, keepdims=True).astype(np.float32)
        self.std = np.maximum(flat.std(0, keepdims=True), 1e-6).astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)


def _trial_labels(labels: np.ndarray, subjects: int, trials: int, windows: int, name: str) -> np.ndarray:
    arranged = labels.reshape(subjects, trials, windows)
    if not np.all(arranged == arranged[:, :, :1]):
        raise ValueError(f"{name} changes within a trial; metadata contract is violated.")
    return arranged[:, :, 0].astype(np.int64)


def load_eppvr(eppvr_root: str | Path, metadata_root: str | Path) -> TrialSet:
    root, meta = Path(eppvr_root), Path(metadata_root)
    eeg = np.load(root / "eeg_feature.npy").astype(np.float32)
    pps = np.concatenate([np.load(root / "ppg_feature.npy"), np.load(root / "gsr_feature.npy"), np.load(root / "skt_feature.npy")], axis=1).astype(np.float32)
    arousal, valence = np.load(root / "labels_arousal.npy"), np.load(root / "labels_valence.npy")
    subject_files = sorted(meta.glob("s*.npy"))
    if len(subject_files) != 30 or eeg.shape != (12180, 40) or pps.shape != (12180, 6):
        raise ValueError("Unexpected EPPVR contract: expected 30 subjects, 14 trials, 29 windows, EEG=40 and PPS=6.")
    archived = np.concatenate([np.load(path, allow_pickle=True)[:, 1:].astype(np.int64) for path in subject_files])
    if not (np.array_equal(archived[:, 0], valence) and np.array_equal(archived[:, 1], arousal)):
        raise ValueError("EPPVR aggregate labels do not match s01..s30 metadata ordering.")
    return TrialSet("eppvr", eeg.reshape(30, 14, 29, 40), pps.reshape(30, 14, 29, 6),
                    _trial_labels(arousal, 30, 14, 29, "arousal"), _trial_labels(valence, 30, 14, 29, "valence"))


def load_deap(feature_root: str | Path) -> TrialSet:
    root = Path(feature_root)
    eeg, pps = np.load(root / "eeg_feature_fp12.npy"), np.load(root / "pps_feature.npy")
    arousal, valence = np.load(root / "labels_arousal.npy"), np.load(root / "labels_valence.npy")
    if eeg.shape[0] != 32 * 40 * 29 or eeg.shape[1] != 40 or pps.shape[1] != 6:
        raise ValueError("DEAP feature contract does not match 32 subjects x 40 trials x 29 windows.")
    return TrialSet("deap", eeg.reshape(32, 40, 29, 40).astype(np.float32), pps.reshape(32, 40, 29, 6).astype(np.float32),
                    _trial_labels(arousal, 32, 40, 29, "arousal"), _trial_labels(valence, 32, 40, 29, "valence"))


def flatten_trials(data: TrialSet, task: str, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    part = data.select_subjects(subjects)
    labels = part.arousal if task == "arousal" else part.valence
    return part.eeg.reshape(-1, 29, 40), part.pps.reshape(-1, 29, 6), labels.reshape(-1)
