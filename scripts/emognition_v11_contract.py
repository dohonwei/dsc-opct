from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESERVATION = ROOT / "docs/emognition_v11_external_confirmation_reservation.json"
HISTORY_AUDIT = ROOT / "docs/emognition_v11_preaccess_history_audit.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
RISK_MODEL = ROOT / "outputs/identity_shortcut_risk_classifier/frozen_risk_model.joblib"
DEFAULT_DATA_ROOT = Path(r"E:\AA发表论文的数据\dataset\Emognition")
DEFAULT_ARCHIVE = DEFAULT_DATA_ROOT / "study_data.zip"

MINIMUM_PARTICIPANTS = 30
STIMULUS_COUNT = 10
MINIMUM_TRIALS_PER_PARTICIPANT = 6
SUBJECT_FOLDS = 5
STIMULUS_FOLDS = 5
SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
REPRESENTATIONS = ("all", "relative_power", "normalized_asymmetry")
MODELS = ("linear_logistic", "gpu_mlp")
GPU_EPOCHS = 40
MATERIAL_THRESHOLD = 0.02
SAM_MIDPOINT = 5
SAM_HIGH_MINIMUM = 6
SAM_LOW_MAXIMUM = 4
EXPECTED_ARCHIVE_MD5 = "28422d18399dc5befab8a6c3d3eddb4a"

REQUIRED_CHANNELS = ("TP9", "AF7", "AF8", "TP10")
BILATERAL_PAIRS = (("AF7", "AF8"), ("TP9", "TP10"))
STIMULUS_NAMES = (
    "AMUSEMENT",
    "ANGER",
    "AWE",
    "DISGUST",
    "ENTHUSIASM",
    "FEAR",
    "LIKING",
    "NEUTRAL",
    "SADNESS",
    "SURPRISE",
)
RAW_CHANNEL_KEYS = tuple(f"RAW_{channel}" for channel in REQUIRED_CHANNELS)
QUALITY_KEYS = (
    "HeadBandOn",
    "HSI_TP9",
    "HSI_AF7",
    "HSI_AF8",
    "HSI_TP10",
)
ALLOWED_HEADBAND_VALUES = (0, 1)
ALLOWED_HSI_VALUES = (1, 2, 4)
USABLE_HSI_VALUES = (1, 2)
RAW_VALUE_POLICY = "native_muse_raw_values_no_unit_conversion"
BANDS_HZ = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
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
        if not artifact.is_file() or hash_file(artifact) != expected:
            raise RuntimeError(f"Frozen v11 artifact changed or is missing: {relative}")
    return freeze


def verify_preaccess_contract() -> tuple[dict, dict, dict]:
    reservation = read_json(RESERVATION)
    history = read_json(HISTORY_AUDIT)
    freeze = verify_frozen_v11()
    if reservation.get("status") != "reserved_before_participant_value_access":
        raise RuntimeError("Emognition reservation status is invalid")
    if history.get("status") != "no_prior_emognition_participant_value_use_found":
        raise RuntimeError("Emognition pre-access history audit is invalid")
    if reservation["dcs_opct_v11_contract"]["freeze_sha256"] != hash_file(FREEZE):
        raise RuntimeError("Emognition reservation does not match the v11 freeze")
    if reservation["frozen_risk_contract"]["model_sha256"] != hash_file(RISK_MODEL):
        raise RuntimeError("Emognition reservation does not match the frozen risk model")
    return reservation, history, freeze


def sam_binary_label(value: float) -> int | None:
    value = float(value)
    if not value.is_integer() or not 1 <= value <= 9:
        raise ValueError(f"SAM rating must be an integer from 1 through 9, found {value}")
    if value >= SAM_HIGH_MINIMUM:
        return 1
    if value <= SAM_LOW_MAXIMUM:
        return 0
    if value == SAM_MIDPOINT:
        return None
    raise AssertionError("Validated SAM rating did not match the locked label rule")
