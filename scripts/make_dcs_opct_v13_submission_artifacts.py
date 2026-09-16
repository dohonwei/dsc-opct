from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v11"
OUT = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v13"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"

EXTRA_SOURCES = {
    "v12_revision_log.md": "docs/elsarticle/dcs_opct_v11_revision_log.md",
    "v12_protocol.md": "docs/dcs_opct_v12_multidomain_risk_robustness_protocol.md",
    "v12_analysis.py": "scripts/analyze_dcs_opct_v12_multidomain_risk_robustness.py",
    "v12_analysis_validator.py": "scripts/validate_dcs_opct_v12_multidomain_risk_robustness.py",
    "v12_artifact_builder.py": "scripts/make_dcs_opct_v12_multidomain_artifacts_v3.py",
    "v12_artifact_validator.py": "scripts/validate_dcs_opct_v12_multidomain_artifacts_v3.py",
    "v13_package_builder.py": "scripts/make_dcs_opct_v13_submission_artifacts.py",
    "v13_package_validator.py": "scripts/validate_dcs_opct_v13_submission_artifacts.py",
    "v12_analysis_domain_metrics.csv": "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/domain_metrics.csv",
    "v12_analysis_fold_coefficients.csv": "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/fold_coefficients.csv",
    "v12_analysis_method_summary.csv": "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/method_summary.csv",
    "v12_analysis_nested_lodo_predictions.csv": "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/nested_lodo_predictions.csv",
    "v12_analysis_seven_dataset_contract.csv": "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/seven_dataset_contract.csv",
    "v12_analysis_retrospective_robustness_gate.json": "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/retrospective_robustness_gate.json",
    "v12_analysis_manifest.json": "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/manifest.json",
    "v12_analysis_independent_validation_report.json": "outputs/dcs_opct_v12_multidomain_risk_robustness_v3/independent_validation_report.json",
    "v12_fig_multidomain_robustness.pdf": "outputs/dcs_opct_v12_multidomain_artifacts_v3/fig_v12_multidomain_robustness.pdf",
    "v12_fig_multidomain_robustness.png": "outputs/dcs_opct_v12_multidomain_artifacts_v3/fig_v12_multidomain_robustness.png",
    "v12_table_multidomain_robustness.tex": "outputs/dcs_opct_v12_multidomain_artifacts_v3/table_v12_multidomain_robustness.tex",
    "v12_artifact_summary.json": "outputs/dcs_opct_v12_multidomain_artifacts_v3/summary.json",
    "v12_artifact_manifest.json": "outputs/dcs_opct_v12_multidomain_artifacts_v3/manifest.json",
    "v12_artifact_independent_validation_report.json": "outputs/dcs_opct_v12_multidomain_artifacts_v3/independent_validation_report.json",
    "v12_failure_execution_001.json": "docs/dcs_opct_v12_execution_failure_001.json",
    "v12_failure_execution_002.json": "docs/dcs_opct_v12_execution_failure_002.json",
    "v12_failure_execution_003.json": "docs/dcs_opct_v12_execution_failure_003.json",
    "v12_failure_figure_001.json": "docs/dcs_opct_v12_figure_failure_001.json",
    "v12_failure_release_001.json": "docs/dcs_opct_v12_release_readiness_failure_001.json",
    "v12_failure_release_002.json": "docs/dcs_opct_v12_release_readiness_failure_002.json",
    "manuscript.tex": "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex",
    "manuscript.pdf": "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.pdf",
    "supplementary.tex": "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex",
    "supplementary.pdf": "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.pdf",
    "v11_final_freeze.json": "docs/distribution_covered_stratified_opct_v11_final_freeze.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Immutable v13 package already exists: {OUT}")
    if sha256(FREEZE) != EXPECTED_FREEZE_HASH:
        raise RuntimeError("Frozen v11 manifest changed; refusing to package v12 evidence")

    sources: dict[str, Path] = {}
    for path in sorted(BASE.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name == "submission_artifact_manifest.json":
            name = "v11_base_submission_artifact_manifest.json"
        elif name == "independent_validation_report.json":
            name = "v11_base_independent_validation_report.json"
        sources[name] = path
    for output_name, relative in EXTRA_SOURCES.items():
        if output_name in sources:
            raise RuntimeError(f"Package filename collision: {output_name}")
        sources[output_name] = ROOT / relative

    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot build v12 package; missing sources: {missing}")

    OUT.mkdir(parents=True)
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    progress = tqdm(sorted(sources.items()), desc="Package DCS-OPCT v13 evidence", unit="file", dynamic_ncols=True)
    for output_name, source in progress:
        target = OUT / output_name
        shutil.copy2(source, target)
        source_hashes[source.relative_to(ROOT).as_posix()] = sha256(source)
        output_hashes[target.relative_to(ROOT).as_posix()] = sha256(target)
    progress.close()

    gate = json.loads((OUT / "v12_analysis_retrospective_robustness_gate.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "submission_artifacts_generated_pending_independent_validation",
        "date": datetime.now().astimezone().date().isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "DCS-OPCT v11 with preserved v12 retrospective robustness evidence",
        "frozen_v11_sha256": EXPECTED_FREEZE_HASH,
        "v12_gate_status": gate.get("status"),
        "claim_boundary": (
            "The v12 seven-domain robustness gate failed and is retained as boundary evidence. "
            "This package does not establish prospective external effectiveness, universal safety, "
            "or transferable operating points."
        ),
        "base_package": BASE.relative_to(ROOT).as_posix(),
        "file_count_excluding_manifest_and_validation_report": len(output_hashes),
        "source_sha256": source_hashes,
        "output_sha256": output_hashes,
    }
    (OUT / "submission_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"DCS-OPCT v13 submission package built: {len(output_hashes)} evidence files")
    print(f"Package: {OUT}")


if __name__ == "__main__":
    main()
