from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/dcs_opct_v12_multidomain_risk_robustness_v3"
OUT = ROOT / "outputs/dcs_opct_v12_multidomain_artifacts_v2"
SCRIPT = ROOT / "scripts/make_dcs_opct_v12_multidomain_artifacts.py"
REPORT = OUT / "independent_validation_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, rtol=0, atol=tolerance))


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(SOURCE / "domain_metrics.csv")
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record("manifest_status", manifest["status"] == "complete_preserve_failed_gate", manifest["status"])
    record("artifact_script_hash", manifest["script_sha256"] == sha256(SCRIPT), sha256(SCRIPT))
    record(
        "source_bindings_current",
        manifest["source_manifest_sha256"] == sha256(SOURCE / "manifest.json")
        and manifest["source_validation_sha256"] == sha256(SOURCE / "independent_validation_report.json"),
        {
            "manifest": sha256(SOURCE / "manifest.json"),
            "validation": sha256(SOURCE / "independent_validation_report.json"),
        },
    )
    mismatches = {
        relative: {"expected": expected, "actual": sha256(ROOT / relative)}
        for relative, expected in manifest["output_sha256"].items()
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected
    }
    record("four_artifact_hashes_current", len(manifest["output_sha256"]) == 4 and not mismatches, mismatches)
    selected = metrics.loc[metrics.method.eq("domain_balanced_gpu")]
    frozen = metrics.loc[metrics.method.eq("frozen_two_source_v11")]
    record(
        "summary_metrics_reproduced",
        close(summary["median_auroc"], selected.auroc.median())
        and close(summary["median_brier_skill"], selected.brier_skill.median())
        and close(summary["median_balanced_accuracy"], selected.balanced_accuracy.median()),
        {key: summary[key] for key in ["median_auroc", "median_brier_skill", "median_balanced_accuracy"]},
    )
    weighted_selected = float(np.average(selected.brier, weights=selected.n))
    weighted_frozen = float(np.average(frozen.brier, weights=frozen.n))
    record(
        "weighted_brier_improvement_reproduced",
        close(summary["domain_balanced_weighted_brier"], weighted_selected)
        and close(summary["frozen_v11_weighted_brier"], weighted_frozen)
        and close(summary["weighted_brier_improvement"], weighted_frozen - weighted_selected),
        summary["weighted_brier_improvement"],
    )
    record(
        "failed_gate_visible_in_summary",
        summary["status"] == "prespecified_retrospective_gate_failed"
        and len(summary["failed_gate_checks"]) == 3,
        summary["failed_gate_checks"],
    )
    table = (OUT / "table_v12_multidomain_robustness.tex").read_text(encoding="utf-8")
    record(
        "latex_table_complete",
        all(dataset in table for dataset in ["DEAP", "MAHNOB-HCI", "EPPVR", "CASE", "CEAP", "SEED-IV", "DREAMER"])
        and table.count(r"\\") == 8,
        {"row_terminators": table.count(r"\\")},
    )
    image = mpimg.imread(OUT / "fig_v12_multidomain_robustness.png")
    record(
        "png_resolution_and_content",
        image.shape[0] >= 3000 and image.shape[1] >= 4000 and float(np.std(image[..., :3])) > 0.05,
        {"shape": list(image.shape), "rgb_sd": float(np.std(image[..., :3]))},
    )
    pdf = (OUT / "fig_v12_multidomain_robustness.pdf").read_bytes()
    record("vector_pdf_signature", pdf.startswith(b"%PDF") and len(pdf) > 30_000, len(pdf))
    record(
        "claim_boundary_preserved",
        "no prospective safety or effectiveness claim" in summary["claim_boundary"],
        summary["claim_boundary"],
    )

    passed = sum(check["passed"] for check in checks)
    content = {
        "status": "passed" if passed == len(checks) else "failed",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    if REPORT.is_file():
        previous = json.loads(REPORT.read_text(encoding="utf-8"))
        previous_content = {key: value for key, value in previous.items() if key != "generated_at_utc"}
        if json.dumps(previous_content, sort_keys=True, allow_nan=True) == json.dumps(
            content, sort_keys=True, allow_nan=True
        ):
            generated_at = previous.get("generated_at_utc", generated_at)
    report = {"generated_at_utc": generated_at, **content}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        raise RuntimeError(f"v12 artifact validation failed: {failed}")
    print(f"v12 artifact validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
