from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_ID = "ekmed-dcs-opct-v11-one-shot-external-confirmation-20260916"
RESERVATION = ROOT / "docs/ekmed_v11_external_confirmation_reservation.json"
RESERVATION_AMENDMENT = (
    ROOT / "docs/ekmed_v11_external_confirmation_reservation_amendment_001.json"
)
HISTORY_AUDIT = ROOT / "docs/ekmed_v11_preaccess_history_audit.json"
METADATA_RECEIPT = ROOT / "docs/ekmed_v11_zenodo_metadata_receipt_20260916.json"
KEY_MOMENT_RECEIPT = (
    ROOT / "docs/ekmed_v11_public_key_moment_receipt_20260916.json"
)
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
RISK_MODEL = ROOT / "outputs/identity_shortcut_risk_classifier/frozen_risk_model.joblib"
DEFAULT_DATA_ROOT = Path(r"E:\AA发表论文的数据\dataset\EKM-ED")
DEFAULT_ARCHIVE = DEFAULT_DATA_ROOT / "EmoKey Moments EEG Dataset (EKM-ED).zip"

MINIMUM_PARTICIPANTS = 30
STIMULUS_COUNT = 4
MINIMUM_TRIALS_PER_PARTICIPANT = 3
SUBJECT_FOLDS = 5
STIMULUS_FOLDS = 4
SEEDS = (20260813, 20260829, 20260911, 20260923, 20261007)
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0)
REPRESENTATIONS = ("all", "relative_power", "normalized_asymmetry")
MODELS = ("linear_logistic", "gpu_mlp")
GPU_EPOCHS = 40
MATERIAL_THRESHOLD = 0.02
SAM_MIDPOINT = 5
SAM_HIGH_MINIMUM = 6
SAM_LOW_MAXIMUM = 4
EXPECTED_ARCHIVE_BYTES = 4_345_584_031
EXPECTED_ARCHIVE_MD5 = "521774df7dc91c31f2ba09fdefc299c6"

REQUIRED_CHANNELS = ("TP9", "AF7", "AF8", "TP10")
BILATERAL_PAIRS = (("AF7", "AF8"), ("TP9", "TP10"))
STIMULUS_NAMES = ("ANGER", "SADNESS", "HAPPINESS", "FEAR")
KEY_MOMENTS_SECONDS = {
    "ANGER": (83.0, 158.0, 216.0),
    "SADNESS": (133.0, 180.0, 228.0),
    "HAPPINESS": (102.0, 179.0, 230.0),
    "FEAR": (105.0, 180.0, 239.0),
}
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


def verify_preaccess_contract(
    require_archive_absent: bool = True,
) -> tuple[dict, dict, dict]:
    reservation = read_json(RESERVATION)
    amendment = read_json(RESERVATION_AMENDMENT)
    history = read_json(HISTORY_AUDIT)
    receipt = read_json(METADATA_RECEIPT)
    key_moments = read_json(KEY_MOMENT_RECEIPT)
    freeze = verify_frozen_v11()
    if reservation.get("status") != "reserved_before_participant_value_access":
        raise RuntimeError("EKM-ED reservation status is invalid")
    if reservation.get("registration_id") != REGISTRATION_ID:
        raise RuntimeError("EKM-ED reservation identifier is invalid")
    if amendment.get("status") != "reservation_amended_before_archive_member_access":
        raise RuntimeError("EKM-ED reservation amendment status is invalid")
    if amendment.get("reservation_sha256") != hash_file(RESERVATION):
        raise RuntimeError("EKM-ED reservation amendment does not match the reservation")
    if history.get("status") != "no_prior_ekmed_participant_value_use_found":
        raise RuntimeError("EKM-ED pre-access history audit is invalid")
    if receipt.get("record_id") != 8431451:
        raise RuntimeError("EKM-ED metadata receipt does not identify the frozen record")
    registered_moments = {
        key: tuple(float(value) for value in values)
        for key, values in key_moments.get("registered_key_moments_seconds", {}).items()
    }
    if registered_moments != KEY_MOMENTS_SECONDS:
        raise RuntimeError("EKM-ED public key-moment receipt is invalid")
    if reservation["dcs_opct_v11_contract"]["freeze_sha256"] != hash_file(FREEZE):
        raise RuntimeError("EKM-ED reservation does not match the v11 freeze")
    if reservation["frozen_risk_contract"]["model_sha256"] != hash_file(RISK_MODEL):
        raise RuntimeError("EKM-ED reservation does not match the frozen risk model")
    if require_archive_absent and DEFAULT_ARCHIVE.exists():
        raise RuntimeError("EKM-ED archive is already present during pre-access validation")
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
