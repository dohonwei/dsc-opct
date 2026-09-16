from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESERVATION = ROOT / "docs/amigos_v11_external_confirmation_reservation.json"
HISTORY_AUDIT = ROOT / "docs/amigos_v11_preaccess_history_audit.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
RISK_MODEL = ROOT / "outputs/identity_shortcut_risk_classifier/frozen_risk_model.joblib"
DEFAULT_DATA_ROOT = Path(r"E:\AA发表论文的数据\dataset\AMIGOS")

MINIMUM_PARTICIPANTS = 30
SHORT_VIDEO_COUNT = 16
SUBJECT_FOLDS = 5
STIMULUS_FOLDS = 4
SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
REPRESENTATIONS = ("all", "relative_power", "normalized_asymmetry")
MODELS = ("linear_logistic", "gpu_mlp")
GPU_EPOCHS = 40
MATERIAL_THRESHOLD = 0.02

REQUIRED_CHANNELS = (
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
)
BILATERAL_PAIRS = (
    ("AF3", "AF4"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("FC5", "FC6"),
    ("T7", "T8"),
    ("P7", "P8"),
    ("O1", "O2"),
)
BANDS_HZ = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_v11() -> dict:
    freeze = read_json(FREEZE)
    for relative, expected in freeze["locked_artifacts"].items():
        artifact = ROOT / relative
        if not artifact.is_file() or sha256(artifact) != expected:
            raise RuntimeError(f"Frozen v11 artifact changed or is missing: {relative}")
    return freeze


def verify_preaccess_contract() -> tuple[dict, dict, dict]:
    reservation = read_json(RESERVATION)
    history = read_json(HISTORY_AUDIT)
    freeze = verify_frozen_v11()
    if reservation.get("status") != "reserved_before_participant_value_access":
        raise RuntimeError("AMIGOS reservation status is invalid")
    if history.get("status") != "no_prior_amigos_participant_value_use_found":
        raise RuntimeError("AMIGOS pre-access history audit is invalid")
    if reservation["dcs_opct_v11_contract"]["freeze_sha256"] != sha256(FREEZE):
        raise RuntimeError("AMIGOS reservation does not match the v11 freeze")
    if reservation["frozen_risk_contract"]["model_sha256"] != sha256(RISK_MODEL):
        raise RuntimeError("AMIGOS reservation does not match the frozen risk model")
    return reservation, history, freeze


def normalized_channel(name: str) -> str:
    return "".join(character for character in str(name).upper() if character.isalnum())
