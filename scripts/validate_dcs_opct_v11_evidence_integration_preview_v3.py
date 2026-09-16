from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v3"
EXPECTED_HASHES = {
    "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex": "dcab80f4727be5bc67586989353c36164b66e2aedf4e364ffa0da32f22c1ed79",
    "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex": "eea5ea34e2a767cfde70fc62e831b2a12429b9aa178a44a3e73f6dccb5820b3f",
    "docs/distribution_covered_stratified_opct_v11_final_freeze.json": "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee",
    "docs/eegemotions27_v11_external_robustness_implementation_lock.json": "e5f68887071352eead192c3352969c92c5ef7f155227e5f70f89d4cdb326699a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(checks: list[dict[str, Any]], name: str, passed: bool, observed: Any, required: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "observed": observed, "required": required})


def write_idempotent(path: Path, report: dict[str, Any]) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        old = {key: value for key, value in previous.items() if key != "generated_at_utc"}
        new = {key: value for key, value in report.items() if key != "generated_at_utc"}
        if old == new:
            report["generated_at_utc"] = previous.get("generated_at_utc")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def clean_log(path: Path) -> tuple[bool, dict[str, bool]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    findings = {
        "latex_error": bool(re.search(r"LaTeX Error", text, re.I)),
        "fatal_error": bool(re.search(r"Fatal error|Emergency stop", text, re.I)),
        "undefined_control": bool(re.search(r"Undefined control sequence", text, re.I)),
        "undefined_references": "There were undefined references" in text,
        "undefined_citations": "There were undefined citations" in text,
        "overfull_box": bool(re.search(r"Overfull \\[hv]box", text)),
    }
    return not any(findings.values()), findings


def main() -> None:
    checks: list[dict[str, Any]] = []
    observed_hashes = {name: sha256(ROOT / name) for name in EXPECTED_HASHES}
    record(checks, "authoritative TEX and protocol locks unchanged",
           observed_hashes == EXPECTED_HASHES, observed_hashes, EXPECTED_HASHES)

    manifest = json.loads((OUT / "preview_manifest.json").read_text(encoding="utf-8"))
    current = {name: sha256(ROOT / name) for name in manifest["files"]}
    record(checks, "v3 source manifest matches bytes",
           current == manifest["files"], current, manifest["files"])
    record(checks, "preview remains staged",
           manifest.get("status") == "staged_preview_not_merged_into_authoritative_manuscript",
           manifest.get("status"), "staged_preview_not_merged_into_authoritative_manuscript")

    pdf_info: dict[str, Any] = {}
    pdf_ok = True
    for name, expected_pages in {"manuscript_preview.pdf": 36, "supplementary_preview.pdf": 30}.items():
        path = OUT / name
        pages = len(PdfReader(str(path)).pages)
        pdf_info[name] = {"pages": pages, "bytes": path.stat().st_size, "sha256": sha256(path)}
        pdf_ok &= path.read_bytes()[:5] == b"%PDF-" and pages == expected_pages
    record(checks, "compiled PDFs valid with stable page counts", pdf_ok, pdf_info,
           {"manuscript_preview.pdf": 36, "supplementary_preview.pdf": 30})

    log_results = {}
    for name in ["manuscript_preview.log", "supplementary_preview.log"]:
        passed, findings = clean_log(OUT / name)
        log_results[name] = findings
        pdf_ok &= passed
    record(checks, "final TeX logs have no errors, unresolved references, citations, or overfull boxes",
           all(not any(v.values()) for v in log_results.values()), log_results, "all false")

    manuscript = (OUT / "manuscript_preview.tex").read_text(encoding="utf-8")
    supplement = (OUT / "supplementary_preview.tex").read_text(encoding="utf-8")
    combined = manuscript + "\n" + supplement
    forbidden = [
        "minimum-budget condition; budget.",
        "two independent unlabeled transformations",
        "independent witness",
        "independent unlabeled transformations",
        "mechanism--risk--safety loop",
        "controlled causal contrast",
        "universal safe transfer",
    ]
    forbidden_found = {phrase: phrase in combined for phrase in forbidden}
    record(checks, "Nature-review overclaims and fragment removed",
           not any(forbidden_found.values()), forbidden_found, "all false")

    required = [
        "two separately constructed unlabeled transformations",
        "complementary witness",
        "mechanism--risk--release-control loop",
        "controlled protocol contrast",
        "$n_{domain}=2$",
        "conditional allocation-sensitivity replicates",
        "not independent dataset or participant replications",
        "No target outcome entered this optimization",
        "future-domain non-harm",
    ]
    required_found = {phrase: phrase in combined for phrase in required}
    record(checks, "moderated claims and dependence boundaries present",
           all(required_found.values()), required_found, "all true")

    objective_parts = [
        r"\mathcal{L}_k(a,b)",
        r"T_{a,b}(p_i^{(I)})-\widetilde p_i^{(k)}",
        r"\lambda\left[(a-1)^2+b^2\right]",
        r"0.1\leq a\leq5",
        r"|b|\leq5",
    ]
    objective_found = {part: part in manuscript for part in objective_parts}
    record(checks, "OPCT objective matches implementation structure",
           all(objective_found.values()), objective_found, "all true")

    inventory = json.loads((ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/inventory.json").read_text(encoding="utf-8"))
    inventory_validation = json.loads((ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json").read_text(encoding="utf-8"))
    expected_count = int(inventory["unique_file_count"])
    expected_passed = int(inventory_validation["checks_passed"])
    expected_total = int(inventory_validation["checks_total"])
    count_parts = [
        f"{expected_count} files",
        f"{expected_passed}/{expected_total} checks",
        "6/6 gates",
        "passed 7/7",
    ]
    count_found = {part: part in manuscript and part in supplement for part in count_parts}
    record(checks, "current reproducibility counts synchronized in both documents",
           all(count_found.values()), count_found, "all true")

    record(checks, "reported inventory counts match authoritative reports",
           f"{expected_count} files" in combined
           and f"{expected_passed}/{expected_total} checks" in combined,
           [inventory["unique_file_count"], inventory_validation["checks_passed"], inventory_validation["checks_total"]],
           [expected_count, expected_passed, expected_total])

    framework = OUT / "figures/fig_v11_three_stage_framework.png"
    with Image.open(framework) as image:
        dimensions = image.size
    record(checks, "revised framework figure is readable raster companion",
           dimensions[0] >= 3000 and dimensions[1] >= 1200, dimensions, ">=3000 x 1200")

    external = json.loads((ROOT / "outputs/dcs_opct_v11_external_arrival_readiness/report.json").read_text(encoding="utf-8"))
    statuses = [row["status"] for row in external["datasets"]]
    record(checks, "prospective external family remains blocked rather than promoted",
           statuses == ["BLOCKED_EXTERNAL_ACCESS", "BLOCKED_EXTERNAL_ACCESS"], statuses,
           ["BLOCKED_EXTERNAL_ACCESS", "BLOCKED_EXTERNAL_ACCESS"])

    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "claim_boundary": manifest["claim_boundary"],
    }
    write_idempotent(OUT / "independent_validation_report.json", report)
    if report["status"] != "passed":
        failed = [item["check"] for item in checks if not item["passed"]]
        raise RuntimeError(f"v3 preview validation failed: {failed}")
    print(f"DCS-OPCT v3 preview validation passed: {report['checks_passed']}/{report['checks_total']}")


if __name__ == "__main__":
    main()
