from __future__ import annotations

import json

from emognition_v11_contract import (
    BILATERAL_PAIRS,
    DEFAULT_DATA_ROOT,
    DOSES,
    GPU_EPOCHS,
    HISTORY_AUDIT,
    MATERIAL_THRESHOLD,
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


OUTPUT = ROOT / "outputs/emognition_v11_preaccess/validation_report.json"


def main() -> None:
    reservation, history, _ = verify_preaccess_contract()
    cohort = reservation["cohort_contract"]
    task = reservation["task_contract"]
    feature = reservation["feature_contract"]
    counterfactual = reservation["counterfactual_contract"]
    relationship = reservation["relationship_to_amigos"]
    checks = {
        "reservation_exists": RESERVATION.is_file(),
        "history_audit_exists": HISTORY_AUDIT.is_file(),
        "data_absent_at_registration": not history["local_data_root_present_before_registration"],
        "current_local_archive_absent": not DEFAULT_DATA_ROOT.exists(),
        "minimum_participants_locked": MINIMUM_PARTICIPANTS
        == cohort["minimum_eligible_participants_per_task"],
        "minimum_trials_locked": MINIMUM_TRIALS_PER_PARTICIPANT
        == cohort["minimum_usable_labeled_trials_per_participant_per_task"],
        "stimulus_count_locked": STIMULUS_COUNT == 10,
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
        "baseline_not_primary": "not used in the primary" in feature["baseline_policy"],
        "identity_not_safety_success": "not evidence of effectiveness or safety success"
        in reservation["acceptance_gate"]["identity_fallback"],
        "event_variation_required": "event and one non-event"
        in reservation["acceptance_gate"]["endpoint_eligibility"],
        "no_dataset_cherry_picking": relationship["no_outcome_based_dataset_selection"]
        is True,
        "replication_requires_both": "both AMIGOS and Emognition"
        in relationship["replication_claim"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not failed else "failed",
        "date": "2026-09-09",
        "phase": "pre-access only",
        "dataset": "Emognition Wearable Dataset 2020",
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "authorized_next_action": (
            "obtain written EULA authorization and place the official study_data.zip at "
            "E:/AA发表论文的数据/dataset/Emognition; run only the ZIP central-directory "
            "schema inspector before adapter implementation and lock"
        ),
        "claim_boundary": reservation["claim_boundary"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if failed:
        raise RuntimeError(f"Emognition pre-access validation failed: {failed}")
    print(
        f"Emognition pre-access validation passed: {report['checks_passed']}/{report['checks_total']}"
    )


if __name__ == "__main__":
    main()
