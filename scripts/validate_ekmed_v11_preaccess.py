from __future__ import annotations

import json

from ekmed_v11_contract import (
    BILATERAL_PAIRS,
    DEFAULT_ARCHIVE,
    DOSES,
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_MD5,
    GPU_EPOCHS,
    HISTORY_AUDIT,
    MATERIAL_THRESHOLD,
    METADATA_RECEIPT,
    MINIMUM_PARTICIPANTS,
    MINIMUM_TRIALS_PER_PARTICIPANT,
    MODELS,
    REPRESENTATIONS,
    REQUIRED_CHANNELS,
    RESERVATION,
    ROOT,
    SAM_HIGH_MINIMUM,
    SAM_LOW_MAXIMUM,
    SAM_MIDPOINT,
    SEEDS,
    STIMULUS_COUNT,
    STIMULUS_FOLDS,
    SUBJECT_FOLDS,
    verify_preaccess_contract,
)


OUTPUT = ROOT / "outputs/ekmed_v11_preaccess/validation_report.json"


def main() -> None:
    reservation, history, _ = verify_preaccess_contract(require_archive_absent=True)
    receipt = json.loads(METADATA_RECEIPT.read_text(encoding="utf-8"))
    cohort = reservation["cohort_contract"]
    task = reservation["task_contract"]
    feature = reservation["feature_contract"]
    counterfactual = reservation["counterfactual_contract"]
    relationship = reservation["relationship_to_existing_reservations"]
    archive = receipt["archive"]
    checks = {
        "reservation_exists": RESERVATION.is_file(),
        "history_audit_exists": HISTORY_AUDIT.is_file(),
        "metadata_receipt_exists": METADATA_RECEIPT.is_file(),
        "data_absent_at_registration": not history["local_data_root_present_before_registration"],
        "current_local_archive_absent": not DEFAULT_ARCHIVE.exists(),
        "official_record_open": receipt["access_right"] == "open",
        "official_license_recorded": receipt["license"] == "cc-by-4.0",
        "archive_identity_locked": archive["size_bytes"] == EXPECTED_ARCHIVE_BYTES
        and archive["checksum"] == f"md5:{EXPECTED_ARCHIVE_MD5}",
        "minimum_participants_locked": MINIMUM_PARTICIPANTS
        == cohort["minimum_eligible_participants_per_task"],
        "minimum_trials_locked": MINIMUM_TRIALS_PER_PARTICIPANT
        == cohort["minimum_usable_labeled_trials_per_participant_per_task"],
        "stimulus_count_locked": STIMULUS_COUNT == 4,
        "channels_locked": list(REQUIRED_CHANNELS) == feature["required_channels"],
        "pairs_locked": [list(pair) for pair in BILATERAL_PAIRS]
        == feature["bilateral_pairs"],
        "representations_locked": list(REPRESENTATIONS) == feature["representations"],
        "models_locked": list(MODELS) == counterfactual["models"],
        "seeds_locked": list(SEEDS) == counterfactual["split_seeds"],
        "doses_locked": list(DOSES) == counterfactual["doses"],
        "folds_locked": SUBJECT_FOLDS == counterfactual["subject_folds"]
        and STIMULUS_FOLDS == counterfactual["stimulus_folds"],
        "cuda_epochs_locked": GPU_EPOCHS == counterfactual["gpu_epochs"],
        "material_threshold_unchanged": MATERIAL_THRESHOLD
        == counterfactual["material_event_threshold"],
        "sam_midpoint_excluded": SAM_MIDPOINT == 5
        and SAM_HIGH_MINIMUM == 6
        and SAM_LOW_MAXIMUM == 4
        and "rating 5 excluded" in task["label_rule"],
        "key_moments_primary": "publicly declared" in feature["key_moment_policy"],
        "baseline_not_primary": "not used in the primary" in feature["baseline_policy"],
        "identity_not_safety_success": "not evidence of effectiveness or safety success"
        in reservation["acceptance_gate"]["identity_fallback"],
        "event_variation_required": "event and one non-event"
        in reservation["acceptance_gate"]["endpoint_eligibility"],
        "no_dataset_cherry_picking": relationship["no_outcome_based_dataset_selection"]
        is True,
        "dataset_specific_claim_only": "cannot support replicated"
        in relationship["replication_claim"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not failed else "failed",
        "date": "2026-09-16",
        "phase": "pre-access only",
        "dataset": "EmoKey Moments Muse EEG Dataset (EKM-ED)",
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "authorized_next_action": (
            "download the official Zenodo archive to "
            "E:/AA发表论文的数据/dataset/EKM-ED without opening members; verify the "
            "registered byte size and MD5; then record only the ZIP central directory"
        ),
        "claim_boundary": reservation["claim_boundary"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if failed:
        raise RuntimeError(f"EKM-ED pre-access validation failed: {failed}")
    print(
        f"EKM-ED pre-access validation passed: "
        f"{report['checks_passed']}/{report['checks_total']}"
    )


if __name__ == "__main__":
    main()
