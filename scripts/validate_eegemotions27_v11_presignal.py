from __future__ import annotations

import json
from pathlib import Path

from eegemotions27_v11_contract import (
    BILATERAL_PAIRS,
    DEFAULT_DATA_ROOT,
    DOSES,
    GPU_EPOCHS,
    HISTORY_AUDIT,
    LOCKED_COMMIT,
    MATERIAL_THRESHOLD,
    MINIMUM_PARTICIPANTS,
    MINIMUM_TRIALS_PER_PARTICIPANT,
    MODELS,
    NEGATIVE_EMOTION_IDS,
    POSITIVE_EMOTION_IDS,
    REPRESENTATIONS,
    REQUIRED_CHANNELS,
    RESERVATION,
    SEEDS,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
    verify_presignal_contract,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/eegemotions27_v11_presignal/validation_report.json"


def main() -> None:
    reservation, history, _ = verify_presignal_contract()
    task = reservation["task_contract"]
    feature = reservation["feature_contract"]
    counterfactual = reservation["counterfactual_contract"]
    checks = {
        "reservation_exists": RESERVATION.is_file(),
        "history_disclosure_exists": HISTORY_AUDIT.is_file(),
        "metadata_access_disclosed": "the recursive tree therefore reveals participant-by-emotion file availability"
        in history["information_revealed_by_prior_metadata_access"],
        "signal_values_not_accessed_at_registration": "any raw EEG sample value"
        in history["not_accessed_before_reservation"],
        "separate_from_pristine_family": "not a member"
        in reservation["external_family_membership"],
        "commit_locked": reservation["source"]["locked_commit"] == LOCKED_COMMIT,
        "category_mapping_locked": task["positive_emotion_ids"]
        == list(POSITIVE_EMOTION_IDS)
        and task["negative_emotion_ids"] == list(NEGATIVE_EMOTION_IDS),
        "cohort_minima_locked": MINIMUM_PARTICIPANTS == 30
        and MINIMUM_TRIALS_PER_PARTICIPANT == 6,
        "montage_locked": list(REQUIRED_CHANNELS) == feature["required_channels"],
        "bilateral_pairs_locked": [list(pair) for pair in BILATERAL_PAIRS]
        == feature["bilateral_pairs"],
        "representations_locked": list(REPRESENTATIONS) == feature["representations"],
        "models_locked": list(MODELS) == counterfactual["models"],
        "seeds_and_doses_locked": list(SEEDS) == counterfactual["split_seeds"]
        and list(DOSES) == counterfactual["doses"],
        "folds_locked": SUBJECT_FOLDS == counterfactual["subject_folds"]
        and STIMULUS_FOLDS == counterfactual["stimulus_folds"],
        "cuda_epochs_locked": GPU_EPOCHS == counterfactual["gpu_epochs"],
        "material_threshold_unchanged": MATERIAL_THRESHOLD
        == counterfactual["material_event_threshold"],
        "identity_not_success": "not evidence of effectiveness"
        in reservation["acceptance_gate"]["identity_fallback"],
        "real_repository_not_yet_present": not DEFAULT_DATA_ROOT.exists(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not failed else "failed",
        "date": "2026-09-09",
        "phase": "pre-signal only",
        "dataset": "EEGEmotions-27",
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "authorized_next_action": (
            "run synthetic adapter and metadata-inspector tests, then acquire only the "
            "locked repository commit and create the metadata-only manifest before any "
            "raw EEG text file is opened"
        ),
        "claim_boundary": reservation["claim_boundary"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if failed:
        raise RuntimeError(f"EEGEmotions-27 pre-signal validation failed: {failed}")
    print(
        "EEGEmotions-27 pre-signal validation passed: "
        f"{report['checks_passed']}/{report['checks_total']}"
    )


if __name__ == "__main__":
    main()
