from __future__ import annotations

from collections.abc import Iterable
import hashlib


CROSSED_DATASET_NAMES = frozenset({"DEAP", "DREAMER", "MAHNOB-HCI", "SEED-IV"})
CROSSED_UNIFIED_DATASETS = (
    "DEAP=outputs/unified_frontal_features/deap.csv",
    "MAHNOB-HCI=outputs/unified_frontal_features/mahnob-hci.csv",
)
CROSSED_TRIAL_DATASETS = (
    "DEAP=outputs/deap_trial_features/trial_features.csv",
    "MAHNOB-HCI=outputs/hci_trial_features/trial_features.csv",
)
STIMULUS_AXES = frozenset({"stimulus", "video", "dual"})


def stable_analysis_seed(base_seed: int, *labels: object) -> int:
    payload = "|".join([str(base_seed), *(str(label) for label in labels)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def dataset_name(specification: str) -> str:
    name, separator, path = specification.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"Dataset specification must be NAME=PATH: {specification!r}")
    return name


def require_crossed_datasets(
    specifications: Iterable[str],
    *,
    context: str,
) -> list[str]:
    names = [dataset_name(item) for item in specifications]
    if len(names) != len(set(names)):
        raise ValueError(f"{context}: duplicate dataset names are not allowed: {names}")
    invalid = sorted(set(names) - CROSSED_DATASET_NAMES)
    if invalid:
        raise ValueError(
            f"{context} requires verified cross-participant physical stimulus identity; "
            f"not admitted: {invalid}. EPPVR may be used only by analyses that do not "
            "treat record position as stimulus identity."
        )
    return names


def require_axis_contract(
    specifications: Iterable[str],
    axes: Iterable[str],
    *,
    context: str,
) -> list[str]:
    axis_set = set(axes)
    if axis_set & STIMULUS_AXES:
        return require_crossed_datasets(specifications, context=context)
    return [dataset_name(item) for item in specifications]
