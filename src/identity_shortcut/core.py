from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


REPRESENTATIONS = {
    "all": ("stimulus__",),
    "absolute_power": ("stimulus__log_power_",),
    "relative_power": ("stimulus__relative_power_",),
    "log_difference": ("stimulus__log_difference_",),
    "normalized_asymmetry": ("stimulus__normalized_asymmetry_",),
    "scale_robust": ("stimulus__relative_power_", "stimulus__normalized_asymmetry_"),
    "baseline_delta": ("delta__",),
}


def stable_seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def representation_columns(frame: pd.DataFrame, representation: str) -> list[str]:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    prefixes = REPRESENTATIONS[representation]
    columns = [column for column in frame if any(column.startswith(prefix) for prefix in prefixes)]
    if not columns:
        raise ValueError(f"No columns found for representation {representation!r}")
    return columns


def conditional_entropy(labels: np.ndarray, identities: np.ndarray) -> float:
    total = len(labels)
    entropy = 0.0
    for identity in np.unique(identities):
        subset = labels[identities == identity]
        probabilities = np.bincount(subset, minlength=2).astype(float)
        probabilities /= probabilities.sum()
        nonzero = probabilities[probabilities > 0]
        entropy -= len(subset) / total * float(np.sum(nonzero * np.log2(nonzero)))
    return entropy


def identity_opportunity(labels: np.ndarray, identities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    identities = np.asarray(identities).astype(str)
    rates = pd.Series(labels).groupby(identities).mean().to_numpy(float)
    return {
        "normalized_mutual_information": float(
            normalized_mutual_info_score(identities, labels, average_method="arithmetic")
        ),
        "conditional_label_entropy": conditional_entropy(labels, identities),
        "identity_rate_std": float(np.std(rates)),
        "identity_rate_range": float(np.ptp(rates)),
    }


def binary_balanced_accuracy(labels: np.ndarray, probability: np.ndarray) -> float:
    prediction = np.asarray(probability) >= 0.5
    recalls = []
    for label in (0, 1):
        mask = np.asarray(labels) == label
        if not np.any(mask):
            return np.nan
        recalls.append(np.mean(prediction[mask] == label))
    return float(np.mean(recalls))


def metadata_prior_opportunity(
    train_labels: np.ndarray,
    train_identities: np.ndarray,
    test_labels: np.ndarray,
    test_identities: np.ndarray,
) -> float:
    global_probability = float(np.mean(train_labels))
    prior = pd.Series(train_labels).groupby(np.asarray(train_identities).astype(str)).mean()
    probability = pd.Series(np.asarray(test_identities).astype(str)).map(prior).to_numpy(float)
    probability[~np.isfinite(probability)] = global_probability
    prior_score = binary_balanced_accuracy(test_labels, probability)
    global_score = binary_balanced_accuracy(
        test_labels, np.full(len(test_labels), global_probability, dtype=float)
    )
    return float(prior_score - global_score)


def class_matched_control_split(
    control: np.ndarray,
    labels: np.ndarray,
    class_counts: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a deterministic class-matched block from an unseen control set."""
    rng = np.random.default_rng(seed)
    removed: list[int] = []
    for label in (0, 1):
        pool = np.asarray(control)[labels[control] == label]
        required = int(class_counts[label])
        if required > len(pool):
            raise ValueError(
                f"Control set cannot supply {required} rows for class {label}; only {len(pool)} available"
            )
        if required:
            removed.extend(rng.choice(pool, size=required, replace=False).tolist())
    removed_array = np.asarray(removed, dtype=int)
    fixed = np.setdiff1d(control, removed_array, assume_unique=False)
    return fixed, removed_array


def _sample_identity_complete_block(
    candidates: np.ndarray,
    labels: np.ndarray,
    class_counts: np.ndarray,
    identities: np.ndarray,
    required_identities: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    required = np.unique(required_identities.astype(str))
    if class_counts.sum() < len(required):
        raise ValueError("Block is too small to cover every required identity")
    by_identity = {identity: candidates[identities[candidates] == identity] for identity in required}
    for _ in range(3000):
        remaining = class_counts.astype(int).copy()
        selected: list[int] = []
        order = required.copy()
        rng.shuffle(order)
        for identity in order:
            available = by_identity[identity]
            available = available[remaining[labels[available]] > 0]
            if not len(available):
                break
            index = int(rng.choice(available))
            selected.append(index)
            remaining[labels[index]] -= 1
        else:
            selected_set = set(selected)
            selected_array = np.fromiter(selected_set, dtype=int)
            valid = True
            for label in (0, 1):
                pool = candidates[
                    (labels[candidates] == label) & ~np.isin(candidates, selected_array)
                ]
                if len(pool) < remaining[label]:
                    valid = False
                    break
                if remaining[label]:
                    selected.extend(rng.choice(pool, size=remaining[label], replace=False).tolist())
            if valid:
                result = np.asarray(selected, dtype=int)
                rng.shuffle(result)
                return result
    raise RuntimeError("Could not draw a class-matched identity-complete block")


@dataclass(frozen=True)
class DoseBlock:
    nominal_dose: float
    achieved_dose: float
    indices: np.ndarray
    opportunity: dict[str, float]
    candidate_min: float
    candidate_max: float


def build_dose_blocks(
    candidates: np.ndarray,
    labels: np.ndarray,
    identities: np.ndarray,
    required_identities: np.ndarray,
    class_counts: np.ndarray,
    doses: list[float],
    draws: int,
    seed: int,
) -> list[DoseBlock]:
    if draws < max(32, len(doses) * 4):
        raise ValueError("At least max(32, four times the dose count) candidate draws are required")
    identities = np.asarray(identities).astype(str)
    unique: dict[tuple[int, ...], tuple[np.ndarray, dict[str, float]]] = {}
    for draw in range(draws):
        block = _sample_identity_complete_block(
            candidates,
            labels,
            class_counts,
            identities,
            required_identities,
            stable_seed(seed, draw),
        )
        key = tuple(sorted(block.tolist()))
        if key not in unique:
            unique[key] = (block, identity_opportunity(labels[block], identities[block]))
    if len(unique) < 2:
        raise RuntimeError("Dose construction produced fewer than two unique blocks")
    pool = list(unique.values())
    values = np.asarray([item[1]["normalized_mutual_information"] for item in pool])
    minimum, maximum = float(values.min()), float(values.max())
    span = maximum - minimum
    output = []
    for dose in doses:
        if not 0.0 <= dose <= 1.0:
            raise ValueError(f"Dose must lie in [0, 1], observed {dose}")
        target = minimum + dose * span
        selected = int(np.argmin(np.abs(values - target)))
        block, opportunity = pool[selected]
        achieved = 0.0 if span <= 1e-12 else (values[selected] - minimum) / span
        output.append(DoseBlock(dose, float(achieved), block, opportunity, minimum, maximum))
    return output
