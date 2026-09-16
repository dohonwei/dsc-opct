from __future__ import annotations

import json
from pathlib import Path

from amigos_v11_contract import (
    BILATERAL_PAIRS,
    DEFAULT_DATA_ROOT,
    DOSES,
    GPU_EPOCHS,
    HISTORY_AUDIT,
    MATERIAL_THRESHOLD,
    MINIMUM_PARTICIPANTS,
    MODELS,
    REPRESENTATIONS,
    REQUIRED_CHANNELS,
    RESERVATION,
    SEEDS,
    SHORT_VIDEO_COUNT,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
    verify_preaccess_contract,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/amigos_v11_preaccess/validation_report.json"


def main() -> None:
    reservation, history, _ = verify_preaccess_contract()
    contract = reservation["counterfactual_contract"]
    feature = reservation["feature_contract"]
    checks = {
        "reservation_exists": RESERVATION.is_file(),
        "history_audit_exists": HISTORY_AUDIT.is_file(),
        "data_absent_at_registration": not history["local_data_root_present_before_registration"],
        "current_local_archive_absent": not DEFAULT_DATA_ROOT.exists(),
        "minimum_participants_locked": MINIMUM_PARTICIPANTS == 30,
        "complete_short_video_map_locked": SHORT_VIDEO_COUNT == 16,
        "registered_montage_locked": list(REQUIRED_CHANNELS) == feature["required_channels"],
        "bilateral_pairs_locked": [list(pair) for pair in BILATERAL_PAIRS]
        == feature["bilateral_pairs"],
        "representations_locked": list(REPRESENTATIONS) == contract["representations"]
        if "representations" in contract
        else list(REPRESENTATIONS) == feature["representations"],
        "models_locked": list(MODELS) == contract["models"],
        "seeds_locked": list(SEEDS) == contract["split_seeds"],
        "doses_locked": list(DOSES) == contract["doses"],
        "folds_locked": SUBJECT_FOLDS == contract["subject_folds"]
        and STIMULUS_FOLDS == contract["stimulus_folds"],
        "cuda_epochs_locked": GPU_EPOCHS == contract["gpu_epochs"],
        "material_threshold_unchanged": MATERIAL_THRESHOLD
        == contract["material_event_threshold"],
        "identity_is_not_safety_success": "not evidence of effectiveness or safety success"
        in reservation["acceptance_gate"]["identity_fallback"],
        "event_variation_required": "event and one non-event"
        in reservation["acceptance_gate"]["endpoint_eligibility"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not failed else "failed",
        "date": "2026-09-09",
        "phase": "pre-access only",
        "dataset": "AMIGOS",
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "authorized_next_action": (
            "obtain the official EULA-authorized AMIGOS archive and place it at "
            "E:/AA发表论文的数据/dataset/AMIGOS; do not parse participant values "
            "until schema manifest, analysis code, synthetic tests, and implementation lock exist"
        ),
        "claim_boundary": reservation["claim_boundary"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if failed:
        raise RuntimeError(f"AMIGOS pre-access validation failed: {failed}")
    print(f"AMIGOS pre-access validation passed: {report['checks_passed']}/{report['checks_total']}")


if __name__ == "__main__":
    main()
