from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/dcs_opct_v11_reproducibility_inventory"
INVENTORY = OUT_DIR / "inventory.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"

FIXED_EVIDENCE = {
    "reproducibility_driver": [
        "scripts/validate_dcs_opct_v11_release_readiness.py",
        "scripts/build_dcs_opct_v11_reproducibility_inventory.py",
        "scripts/validate_dcs_opct_v11_reproducibility_inventory.py",
        "scripts/audit_dcs_opct_v11_q1_submission_readiness.py",
        "scripts/detect_dcs_opct_v11_external_arrival.py",
        "scripts/validate_dcs_opct_v11_external_arrival.py",
        "scripts/validate_dcs_opct_v11_reviewer_closure.py",
    ],
    "access_and_readiness": [
        "docs/dcs_opct_v11_dataset_availability_matrix.md",
        "outputs/dcs_opct_v11_release_readiness/report.json",
        "outputs/amigos_v11_preaccess/stack_validation_report.json",
        "outputs/emognition_v11_preaccess/stack_validation_report.json",
        "outputs/dcs_opct_v11_q1_submission_readiness/readiness_report.json",
        "outputs/dcs_opct_v11_q1_submission_readiness/readiness_report.md",
        "outputs/dcs_opct_v11_external_arrival_readiness/report.json",
        "outputs/dcs_opct_v11_external_arrival_readiness/report.md",
    ],
    "prospective_external_software": [
        "scripts/external_wearable_implementation_lock.py",
        "scripts/external_wearable_one_shot_core.py",
        "scripts/test_external_wearable_implementation_lock.py",
        "scripts/test_external_wearable_one_shot_core.py",
        "scripts/test_external_wearable_one_shot_end_to_end.py",
        "scripts/emognition_v11_schema_adapter.py",
        "scripts/run_emognition_v11_external_confirmation.py",
        "scripts/test_emognition_v11_schema_adapter.py",
        "scripts/test_emognition_v11_archive_adapter.py",
        "docs/emognition_v11_schema_adapter_preaccess_specification.md",
    ],
    "external_access_sidecar": [
        "outputs/dcs_opct_v11_external_access_evidence_v5/manifest.json",
        "outputs/dcs_opct_v11_external_access_evidence_v5/independent_validation_report.json",
        "scripts/build_dcs_opct_v11_external_access_evidence.py",
        "scripts/validate_dcs_opct_v11_external_access_evidence.py",
    ],
    "submission_package": [
        "outputs/dcs_opct_v11_submission_artifacts_crossfit_v11/submission_artifact_manifest.json",
        "outputs/dcs_opct_v11_submission_artifacts_crossfit_v11/independent_validation_report.json",
    ],
    "latest_submission_package": [
        "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/submission_artifact_manifest.json",
        "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/independent_validation_report.json",
        "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/v12_analysis_retrospective_robustness_gate.json",
    ],
    "cluster_dependence_audit_yield": [
        "scripts/analyze_dcs_opct_v11_cluster_dependence_and_audit_yield.py",
        "scripts/validate_dcs_opct_v11_cluster_dependence_and_audit_yield.py",
        "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910/manifest.json",
        "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910/independent_validation_report.json",
        "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910/bounded_conclusion.json",
    ],
    "cluster_score_aggregation_sensitivity": [
        "scripts/analyze_dcs_opct_v11_cluster_score_aggregation_sensitivity.py",
        "scripts/validate_dcs_opct_v11_cluster_score_aggregation_sensitivity.py",
        "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910/manifest.json",
        "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910/independent_validation_report.json",
    ],
    "evidence_integration_preview_v2": [
        "scripts/build_dcs_opct_v11_evidence_integration_preview.py",
        "scripts/validate_dcs_opct_v11_evidence_integration_preview.py",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/preview_manifest.json",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/independent_validation_report.json",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/manuscript_preview.tex",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/manuscript_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/manuscript_preview.log",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/supplementary_preview.tex",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/supplementary_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2/supplementary_preview.log",
    ],
    "nature_review_and_preview_v3": [
        "scripts/build_dcs_opct_v11_nature_review.py",
        "scripts/build_dcs_opct_v11_evidence_integration_preview_v3.py",
        "scripts/validate_dcs_opct_v11_evidence_integration_preview_v3.py",
        "outputs/dcs_opct_v11_nature_review_20260910/review_dcs_opct_v11_20260910.docx",
        "outputs/dcs_opct_v11_nature_review_20260910/review_dcs_opct_v11_20260910.md",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/preview_manifest.json",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/independent_validation_report.json",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/manuscript_preview.tex",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/manuscript_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/manuscript_preview.log",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/supplementary_preview.tex",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/supplementary_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/supplementary_preview.log",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/figures/fig_v11_three_stage_framework.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3/figures/fig_v11_three_stage_framework.png",
    ],
    "higher_level_dependence_and_preview_v4": [
        "scripts/analyze_dcs_opct_v11_higher_level_cluster_sensitivity.py",
        "scripts/validate_dcs_opct_v11_higher_level_cluster_sensitivity.py",
        "scripts/build_dcs_opct_v11_evidence_integration_preview_v4.py",
        "scripts/validate_dcs_opct_v11_evidence_integration_preview_v4.py",
        "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/manifest.json",
        "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/independent_validation_report.json",
        "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/bounded_conclusion.json",
        "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/higher_level_audit_yield.csv",
        "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/higher_level_bootstrap_intervals.csv",
        "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/dataset_task_seed_blocks.csv",
        "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/exact_random_block_allocations.csv",
        "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/fig_higher_level_cluster_sensitivity.pdf",
        "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910/fig_higher_level_cluster_sensitivity.png",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/preview_manifest.json",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/independent_validation_report.json",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/manuscript_preview.tex",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/manuscript_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/manuscript_preview.log",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/supplementary_preview.tex",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/supplementary_preview.pdf",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/supplementary_preview.log",
        "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4/tables/table_higher_level_cluster_sensitivity.tex",
    ],
    "component_diagnostic": [
        "scripts/analyze_dcs_opct_v11_leave_one_component.py",
        "scripts/validate_dcs_opct_v11_leave_one_component.py",
        "outputs/dcs_opct_v11_leave_one_component/same_assignment_leave_one_component.csv",
        "outputs/dcs_opct_v11_leave_one_component/random_half_allocation_repetitions.csv",
        "outputs/dcs_opct_v11_leave_one_component/distribution_coverage_summary.csv",
        "outputs/dcs_opct_v11_leave_one_component/table_leave_one_component.tex",
        "outputs/dcs_opct_v11_leave_one_component/table_distribution_coverage_diagnostic.tex",
        "outputs/dcs_opct_v11_leave_one_component/manifest.json",
        "outputs/dcs_opct_v11_leave_one_component/independent_validation_report.json",
    ],
    "reviewer_closure_20260914": [
        "scripts/analyze_dcs_opct_v11_inductive_target_sensitivity.py",
        "scripts/validate_dcs_opct_v11_inductive_target_sensitivity.py",
        "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/independent_validation_report.json",
        "scripts/analyze_dcs_opct_v11_raw_group_independence_sensitivity.py",
        "scripts/validate_dcs_opct_v11_raw_group_independence_sensitivity.py",
        "outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914/independent_validation_report.json",
        "scripts/analyze_dcs_opct_v11_crossed_cluster_endpoint_uncertainty.py",
        "scripts/analyze_dcs_opct_v11_target_action_endpoint_uncertainty.py",
        "scripts/validate_dcs_opct_v11_endpoint_uncertainty.py",
        "scripts/validate_dcs_opct_v11_target_action_endpoint_uncertainty.py",
        "outputs/dcs_opct_v11_probabilistic_event_risk_20260914/independent_validation_report.json",
        "outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914/independent_validation_report.json",
        "scripts/analyze_dcs_opct_v11_nearest_neighbor_baselines.py",
        "scripts/validate_dcs_opct_v11_nearest_neighbor_baselines.py",
        "outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/independent_validation_report.json",
        "scripts/analyze_dcs_opct_v11_eppvr_raw_partition_first.py",
        "scripts/validate_dcs_opct_v11_eppvr_raw_partition_first.py",
        "scripts/build_dcs_opct_v11_independence_ladder.py",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/analysis_manifest.json",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/participant_population_assignment.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/population_predictions.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/population_split_audit.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/population_identity_probe_summary.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/population_summary.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/population_risk_tables.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/fit_provenance.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/component_fit_diagnostics.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/witness_diagnostics.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/audit_population_geometry.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/audit_population_certificate_assignment.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/certificate.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/evaluation_population_action_predictions.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/evaluation_outcome.csv",
        "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/independent_validation_report.json",
        "docs/elsarticle/figures/fig_v11_independence_ladder.pdf",
        "docs/elsarticle/figures/fig_v11_independence_ladder.png",
        "docs/elsarticle/tables/table_v11_independence_ladder.tex",
    ],
    "manuscript": [
        "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
        "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.pdf",
        "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex",
        "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.pdf",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(relative: str) -> str:
    return Path(relative.replace("\\", "/")).as_posix()


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    groups = {name: [normalized(path) for path in paths] for name, paths in FIXED_EVIDENCE.items()}
    groups["frozen_core_evidence"] = [normalized(path) for path in freeze["locked_artifacts"]]
    preview_root = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2"
    groups["evidence_integration_preview_v2"].extend(
        normalized(str(path.relative_to(ROOT)))
        for path in sorted(preview_root.rglob("*"))
        if path.is_file()
    )
    groups["evidence_integration_preview_v2"] = sorted(
        set(groups["evidence_integration_preview_v2"])
    )

    reverse_groups: dict[str, list[str]] = {}
    for group, paths in groups.items():
        for relative in paths:
            reverse_groups.setdefault(relative, []).append(group)

    missing = [relative for relative in reverse_groups if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot build reproducibility inventory; missing files: {missing}")

    entries = []
    paths = sorted(reverse_groups)
    progress = tqdm(paths, desc="Hash reproducibility evidence", unit="file", dynamic_ncols=True)
    for relative in progress:
        path = ROOT / relative
        entries.append(
            {
                "path": relative,
                "groups": sorted(reverse_groups[relative]),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    progress.close()

    freeze_hash = sha256(FREEZE)
    inventory = {
        "schema_version": "dcs-opct-v11-reproducibility-inventory-1.0",
        "status": "inventory_built_pending_independent_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Distribution-Covered Stratified Order-Preserving Calibration Transport (DCS-OPCT v11)",
        "scope": "Immutable evidence graph for software readiness, frozen development evidence, submission artifacts, and compiled manuscript files.",
        "claim_boundary": (
            "Inventory integrity establishes traceability and reproducibility of recorded artifacts only. "
            "It is not prospective external effectiveness, non-harm, or universal safety evidence."
        ),
        "anti_cycle_rule": (
            "Neither inventory.json nor its validation_report.json is an inventory member; "
            "the readiness report does not depend on this inventory."
        ),
        "freeze_sha256": freeze_hash,
        "expected_freeze_sha256": EXPECTED_FREEZE_HASH,
        "group_counts": {name: len(paths) for name, paths in groups.items()},
        "unique_file_count": len(entries),
        "entries": entries,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INVENTORY.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reproducibility inventory built: {len(entries)} unique files")
    print(f"Inventory: {INVENTORY}")


if __name__ == "__main__":
    main()
