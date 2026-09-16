from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESERVATION = ROOT / "docs/faced_v11_external_confirmation_reservation.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
IMPLEMENTATION_LOCK = (
    ROOT / "docs/faced_v11_external_confirmation_implementation_lock.json"
)
IMPLEMENTATION_LOCK_AMENDMENTS = (
    ROOT / "docs/faced_v11_external_confirmation_implementation_lock_amendment_001.json",
)
ARCHIVED_MANIFEST = ROOT / "outputs/faced_v11_preaccess_metadata/nemar_manifest.json"

DATASET_ID = "nm000112"
DATASET_VERSION = "v1.1.3"
MANIFEST_URL = "https://data.nemar.org/nm000112/v1.1.3/manifest.json"
DEFAULT_DATA_ROOT = Path(r"E:\AA发表论文的数据\dataset\FACED_NEMAR_v1.1.3")
DEFAULT_FEATURE_ROOT = ROOT / "outputs/faced_v11_trial_features"
DEFAULT_DOSE_ROOT = ROOT / "outputs/faced_v11_counterfactual_identity_dose"
DEFAULT_CONFIRMATION_ROOT = ROOT / "outputs/faced_v11_external_confirmation"

MINIMUM_PARTICIPANTS = 100
SUBJECT_FOLDS = 5
STIMULUS_FOLDS = 5
SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
REPRESENTATIONS = ("all", "relative_power", "normalized_asymmetry")
MODELS = ("linear_logistic", "gpu_mlp")
GPU_EPOCHS = 40
MATERIAL_THRESHOLD = 0.02

BANDS_HZ = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

REQUIRED_CHANNELS = (
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "CZ",
    "C4",
    "T8",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "PZ",
    "P4",
    "P8",
    "PO7",
    "PO3",
    "PO4",
    "PO8",
    "O1",
    "OZ",
    "O2",
)

BILATERAL_PAIRS = (
    ("FP1", "FP2"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("FC5", "FC6"),
    ("FC1", "FC2"),
    ("T7", "T8"),
    ("C3", "C4"),
    ("CP5", "CP6"),
    ("CP1", "CP2"),
    ("P7", "P8"),
    ("P3", "P4"),
    ("PO7", "PO8"),
    ("PO3", "PO4"),
    ("O1", "O2"),
)

VIDEO_EMOTION = {
    **{index: "anger" for index in range(1, 4)},
    **{index: "disgust" for index in range(4, 7)},
    **{index: "fear" for index in range(7, 10)},
    **{index: "sadness" for index in range(10, 13)},
    **{index: "neutral" for index in range(13, 17)},
    **{index: "amusement" for index in range(17, 20)},
    **{index: "inspiration" for index in range(20, 23)},
    **{index: "joy" for index in range(23, 26)},
    **{index: "tenderness" for index in range(26, 29)},
}
NEGATIVE_EMOTIONS = frozenset(("anger", "disgust", "fear", "sadness"))
POSITIVE_EMOTIONS = frozenset(("amusement", "inspiration", "joy", "tenderness"))
NON_NEUTRAL_VIDEOS = tuple(
    index for index, emotion in VIDEO_EMOTION.items() if emotion != "neutral"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_artifacts() -> dict[str, object]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    for relative, expected in freeze["locked_artifacts"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Frozen v11 artifact changed or is missing: {relative}")
    return freeze


def verify_implementation_lock() -> dict[str, object]:
    if not IMPLEMENTATION_LOCK.is_file():
        raise RuntimeError(
            "FACED implementation lock is absent; participant/event/signal access is forbidden"
        )
    lock = json.loads(IMPLEMENTATION_LOCK.read_text(encoding="utf-8"))
    if (
        lock.get("status")
        != "locked_before_faced_participant_event_or_signal_value_access"
    ):
        raise RuntimeError("FACED implementation lock status is invalid")
    if lock.get("reservation_sha256") != sha256(RESERVATION):
        raise RuntimeError("FACED implementation lock does not match the reservation")
    if lock.get("freeze_sha256") != sha256(FREEZE):
        raise RuntimeError("FACED implementation lock does not match the v11 freeze")
    verify_frozen_artifacts()
    effective_hashes = {
        group: dict(lock[group])
        for group in (
            "analysis_code_sha256",
            "dependency_code_sha256",
            "source_metadata_sha256",
        )
    }
    effective_status = lock["status"]
    effective_amendment_id = None
    base_lock_sha256 = sha256(IMPLEMENTATION_LOCK)
    for amendment_path in IMPLEMENTATION_LOCK_AMENDMENTS:
        if not amendment_path.is_file():
            continue
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        if amendment.get("superseded_lock_sha256") != base_lock_sha256:
            raise RuntimeError("FACED implementation amendment does not match base lock")
        failure_path = ROOT / str(amendment.get("failure_record_path", ""))
        if (
            not failure_path.is_file()
            or sha256(failure_path) != amendment.get("failure_record_sha256")
        ):
            raise RuntimeError("FACED implementation amendment failure record changed")
        if amendment.get("scientific_changes") != "none":
            raise RuntimeError("FACED implementation amendment changes scientific rules")
        for group, replacements in amendment["artifact_sha256"].items():
            if group not in effective_hashes:
                raise RuntimeError(f"Unknown FACED amendment artifact group: {group}")
            effective_hashes[group].update(replacements)
        effective_status = amendment["status"]
        effective_amendment_id = amendment["amendment_id"]
    for group in (
        "analysis_code_sha256",
        "dependency_code_sha256",
        "source_metadata_sha256",
    ):
        for relative, expected in effective_hashes[group].items():
            path = ROOT / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"Registered FACED artifact changed: {relative}")
    effective_lock = dict(lock)
    effective_lock["status"] = effective_status
    if effective_amendment_id is not None:
        effective_lock["effective_amendment_id"] = effective_amendment_id
    return effective_lock


def normalized_channel(name: str) -> str:
    return "".join(character for character in str(name).upper() if character.isalnum())


def expected_polarity(video_index: int) -> str:
    emotion = VIDEO_EMOTION[int(video_index)]
    if emotion in NEGATIVE_EMOTIONS:
        return "negative"
    if emotion in POSITIVE_EMOTIONS:
        return "positive"
    return "neutral"
