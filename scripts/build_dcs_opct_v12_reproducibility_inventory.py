from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/dcs_opct_v12_reproducibility_inventory"
INVENTORY = OUT_DIR / "inventory.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"

GROUPS = {
    "release_and_inventory_drivers": [
        "scripts/validate_dcs_opct_v12_release_readiness.py",
        "scripts/build_dcs_opct_v12_reproducibility_inventory.py",
        "scripts/validate_dcs_opct_v12_reproducibility_inventory.py",
        "outputs/dcs_opct_v12_release_readiness/report.json",
    ],
    "submission_package": [
        "scripts/make_dcs_opct_v15_submission_artifacts.py",
        "scripts/validate_dcs_opct_v15_submission_artifacts.py",
        "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/submission_artifact_manifest.json",
        "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/independent_validation_report.json",
    ],
    "v12_multidomain_analysis": [
        "docs/dcs_opct_v12_multidomain_risk_robustness_protocol.md",
        "scripts/analyze_dcs_opct_v12_multidomain_risk_robustness.py",
        "scripts/validate_dcs_opct_v12_multidomain_risk_robustness.py",
        "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/domain_metrics.csv",
        "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/fold_coefficients.csv",
        "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/method_summary.csv",
        "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/nested_lodo_predictions.csv",
        "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/retrospective_robustness_gate.json",
        "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/seven_dataset_contract.csv",
        "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/manifest.json",
        "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/independent_validation_report.json",
    ],
    "v12_artifacts": [
        "scripts/make_dcs_opct_v12_multidomain_artifacts_v3.py",
        "scripts/validate_dcs_opct_v12_multidomain_artifacts_v3.py",
        "outputs/dcs_opct_v12_multidomain_artifacts_v3/fig_v12_multidomain_robustness.pdf",
        "outputs/dcs_opct_v12_multidomain_artifacts_v3/fig_v12_multidomain_robustness.png",
        "outputs/dcs_opct_v12_multidomain_artifacts_v3/table_v12_multidomain_robustness.tex",
        "outputs/dcs_opct_v12_multidomain_artifacts_v3/summary.json",
        "outputs/dcs_opct_v12_multidomain_artifacts_v3/manifest.json",
        "outputs/dcs_opct_v12_multidomain_artifacts_v3/independent_validation_report.json",
    ],
    "preserved_v12_failures": [
        "docs/dcs_opct_v12_execution_failure_001.json",
        "docs/dcs_opct_v12_execution_failure_002.json",
        "docs/dcs_opct_v12_execution_failure_003.json",
        "docs/dcs_opct_v12_figure_failure_001.json",
        "docs/dcs_opct_v12_release_readiness_failure_001.json",
        "docs/dcs_opct_v12_release_readiness_failure_002.json",
        "docs/dcs_opct_v12_release_readiness_failure_003.json",
        "docs/dcs_opct_v12_reproducibility_idempotence_failure_001.json",
        "docs/dcs_opct_v14_submission_package_superseded.json",
    ],
    "prospective_external_readiness": [
        "outputs/amigos_v11_preaccess/stack_validation_report.json",
        "outputs/emognition_v11_preaccess/stack_validation_report.json",
        "outputs/dcs_opct_v11_external_access_evidence_v5/manifest.json",
        "outputs/dcs_opct_v11_external_access_evidence_v5/independent_validation_report.json",
        "scripts/emognition_v11_schema_adapter.py",
        "scripts/run_emognition_v11_external_confirmation.py",
        "scripts/test_emognition_v11_schema_adapter.py",
        "scripts/test_emognition_v11_archive_adapter.py",
    ],
    "manuscript": [
        "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
        "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.pdf",
        "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex",
        "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.pdf",
        "docs/elsarticle/dcs_opct_v11_revision_log.md",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    groups = {name: list(paths) for name, paths in GROUPS.items()}
    groups["frozen_v11_core"] = [Path(path.replace("\\", "/")).as_posix() for path in freeze["locked_artifacts"]]
    memberships: dict[str, list[str]] = {}
    for group, paths in groups.items():
        for relative in paths:
            memberships.setdefault(relative, []).append(group)
    missing = [relative for relative in memberships if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot build v12 inventory; missing files: {missing}")

    entries = []
    progress = tqdm(sorted(memberships), desc="Hash v12 reproducibility evidence", unit="file", dynamic_ncols=True)
    for relative in progress:
        path = ROOT / relative
        entries.append({
            "path": relative,
            "groups": sorted(memberships[relative]),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    progress.close()
    inventory = {
        "schema_version": "dcs-opct-v12-reproducibility-inventory-1.0",
        "status": "inventory_built_pending_independent_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "DCS-OPCT v11 with v12 retrospective multi-domain robustness evidence",
        "scope": "Non-self-referential evidence graph for the frozen core, v12 analysis, package, manuscript, GPU readiness, and preserved failures.",
        "claim_boundary": "Inventory integrity is not prospective external effectiveness, universal safety, or successful seven-domain transfer evidence.",
        "anti_cycle_rule": "The inventory and its validation report are not inventory members.",
        "freeze_sha256": sha256(FREEZE),
        "expected_freeze_sha256": EXPECTED_FREEZE_HASH,
        "group_counts": {name: len(paths) for name, paths in groups.items()},
        "unique_file_count": len(entries),
        "entries": entries,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INVENTORY.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DCS-OPCT v12 reproducibility inventory built: {len(entries)} unique files")
    print(f"Inventory: {INVENTORY}")


if __name__ == "__main__":
    main()
