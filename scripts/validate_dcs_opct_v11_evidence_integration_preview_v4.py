from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v4"
HIGHER = ROOT / "outputs/dcs_opct_v11_higher_level_cluster_sensitivity_20260910"
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


def main() -> None:
    checks: list[dict[str, Any]] = []
    observed_hashes = {path: sha256(ROOT / path) for path in EXPECTED_HASHES}
    record(checks, "authoritative TEX and locks unchanged", observed_hashes == EXPECTED_HASHES,
           observed_hashes, EXPECTED_HASHES)
    manifest = json.loads((OUT / "preview_manifest.json").read_text(encoding="utf-8"))
    current = {path: sha256(ROOT / path) for path in manifest["files"]}
    record(checks, "v4 manifest hashes", current == manifest["files"], current, manifest["files"])
    record(checks, "v4 remains staged", manifest.get("status") == "staged_preview_not_merged_into_authoritative_manuscript",
           manifest.get("status"), "staged")

    pdf_info = {}
    pdf_ok = True
    for name, pages_expected in {"manuscript_preview.pdf": 37, "supplementary_preview.pdf": 31}.items():
        path = OUT / name
        pages = len(PdfReader(str(path)).pages)
        pdf_info[name] = [pages, path.stat().st_size]
        pdf_ok &= path.read_bytes()[:5] == b"%PDF-" and pages == pages_expected
    record(checks, "compiled PDF page counts", pdf_ok, pdf_info, [37, 31])

    log_findings = {}
    for name in ["manuscript_preview.log", "supplementary_preview.log"]:
        text = (OUT / name).read_text(encoding="utf-8", errors="replace")
        log_findings[name] = {
            "error": bool(re.search(r"LaTeX Error|Fatal error|Emergency stop|Undefined control sequence", text, re.I)),
            "undefined": "There were undefined references" in text or "There were undefined citations" in text,
            "overfull": bool(re.search(r"Overfull \\[hv]box", text)),
        }
    record(checks, "clean final TeX logs", all(not any(v.values()) for v in log_findings.values()),
           log_findings, "all false")

    manuscript = (OUT / "manuscript_preview.tex").read_text(encoding="utf-8")
    supplement = (OUT / "supplementary_preview.tex").read_text(encoding="utf-8")
    combined = manuscript + "\n" + supplement
    required = [
        "20 conditional higher-level blocks", "All 2,025 valid same-budget allocations",
        "$p=0.207$", "$p=0.260$", "does not persist as statistically persuasive workflow-level yield",
        "not a general audit-allocation benefit across tasks, seeds, or future domains",
        "conditional post-hoc sensitivity, not independent-domain validation",
    ]
    presence = {phrase: phrase in combined for phrase in required}
    record(checks, "non-significant higher-level result preserved", all(presence.values()), presence, "all true")

    higher_report = json.loads((HIGHER / "independent_validation_report.json").read_text(encoding="utf-8"))
    record(checks, "higher-level source validated",
           higher_report.get("status") == "passed"
           and higher_report.get("checks_passed") == higher_report.get("checks_total") == 13,
           [higher_report.get("checks_passed"), higher_report.get("checks_total")], [13, 13])
    record(checks, "higher-level manifest tracked",
           manifest.get("higher_level_manifest_sha256") == sha256(HIGHER / "manifest.json"),
           manifest.get("higher_level_manifest_sha256"), sha256(HIGHER / "manifest.json"))

    table = (OUT / "tables/table_higher_level_cluster_sensitivity.tex").read_text(encoding="utf-8")
    table_required = ["0.263", "0.259", "1.316", "0.207", "0.260", "20 dataset--task--seed blocks"]
    record(checks, "higher-level table values", all(value in table for value in table_required),
           {value: value in table for value in table_required}, "all true")

    forbidden = ["minimum-budget condition; budget.", "independent witness", "controlled causal contrast"]
    record(checks, "earlier wording repairs retained", not any(phrase in combined for phrase in forbidden),
           {phrase: phrase in combined for phrase in forbidden}, "all false")

    inventory = json.loads((ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/inventory.json").read_text(encoding="utf-8"))
    inventory_validation = json.loads((ROOT / "outputs/dcs_opct_v11_reproducibility_inventory/validation_report.json").read_text(encoding="utf-8"))
    count_phrases = [
        f"{inventory['unique_file_count']} files",
        f"{inventory_validation['checks_passed']}/{inventory_validation['checks_total']} checks",
    ]
    record(checks, "inventory counts synchronized", all(p in manuscript and p in supplement for p in count_phrases),
           {p: p in manuscript and p in supplement for p in count_phrases}, "all true")

    report = {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "claim_boundary": manifest["claim_boundary"],
    }
    write_idempotent(OUT / "independent_validation_report.json", report)
    if report["status"] != "passed":
        raise RuntimeError([check["check"] for check in checks if not check["passed"]])
    print(f"DCS-OPCT v4 preview validation passed: {report['checks_passed']}/{report['checks_total']}")


if __name__ == "__main__":
    main()
