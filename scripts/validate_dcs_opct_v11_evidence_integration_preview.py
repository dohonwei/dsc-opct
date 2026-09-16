from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_evidence_integration_preview_20260910_v2"

EXPECTED_HASHES = {
    "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex": (
        "dcab80f4727be5bc67586989353c36164b66e2aedf4e364ffa0da32f22c1ed79"
    ),
    "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex": (
        "eea5ea34e2a767cfde70fc62e831b2a12429b9aa178a44a3e73f6dccb5820b3f"
    ),
    "docs/distribution_covered_stratified_opct_v11_final_freeze.json": (
        "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
    ),
    "docs/eegemotions27_v11_external_robustness_implementation_lock.json": (
        "e5f68887071352eead192c3352969c92c5ef7f155227e5f70f89d4cdb326699a"
    ),
}

EXPECTED_PDFS = {
    "manuscript_preview.pdf": 36,
    "supplementary_preview.pdf": 30,
}

READABLE_FIGURES = [
    "fig_direct_calibration_gain_readable.png",
    "fig_direct_calibration_risk_readable.png",
    "fig_split_certified_release_readable.png",
    "fig_split_certified_gain_readable.png",
    "fig_split_certified_risk_readable.png",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    required: Any,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "required": required,
        }
    )


def validate_log(path: Path, pdf_name: str, pages: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    forbidden = {
        "latex_error": r"LaTeX Error",
        "undefined_control_sequence": r"Undefined control sequence",
        "fatal_error": r"Fatal error",
        "emergency_stop": r"Emergency stop",
        "undefined_references": r"There were undefined references",
        "undefined_citations": r"There were undefined citations",
        "undefined_reference_entry": r"Reference .* undefined",
        "undefined_citation_entry": r"Citation .* undefined",
        "overfull_box": r"Overfull \\[hv]box",
    }
    findings = {
        name: bool(re.search(pattern, text, flags=re.IGNORECASE))
        for name, pattern in forbidden.items()
    }
    output_marker = f"Output written on {pdf_name} ({pages} pages"
    return {
        "output_marker": output_marker in text,
        "forbidden_findings": findings,
        "passed": output_marker in text and not any(findings.values()),
    }


def main() -> None:
    checks: list[dict[str, Any]] = []
    progress = tqdm(total=13, desc="DCS-OPCT evidence preview", unit="check")

    observed_hashes = {
        relative: sha256(ROOT / relative) for relative in EXPECTED_HASHES
    }
    record(
        checks,
        "authoritative manuscript sources and protocol locks remain unchanged",
        observed_hashes == EXPECTED_HASHES,
        observed_hashes,
        EXPECTED_HASHES,
    )
    progress.update()

    manifest = json.loads((OUT / "preview_manifest.json").read_text(encoding="utf-8"))
    tracked_hashes = manifest.get("files", {})
    current_hashes = {
        relative: sha256(ROOT / relative) for relative in tracked_hashes
    }
    record(
        checks,
        "all staged TeX, figure, and table files match the preview manifest",
        tracked_hashes == current_hashes
        and manifest.get("status")
        == "staged_preview_not_merged_into_authoritative_manuscript",
        {
            "tracked_files": len(tracked_hashes),
            "hashes_match": tracked_hashes == current_hashes,
            "status": manifest.get("status"),
        },
        "all hashes match and status remains staged preview",
    )
    progress.update()

    pdf_results: dict[str, Any] = {}
    pdfs_ok = True
    for name, expected_pages in EXPECTED_PDFS.items():
        path = OUT / name
        pages = len(PdfReader(str(path)).pages)
        signature = path.read_bytes()[:5]
        pdf_results[name] = {
            "signature": signature.decode("ascii", errors="replace"),
            "pages": pages,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        pdfs_ok = pdfs_ok and signature == b"%PDF-" and pages == expected_pages
    record(
        checks,
        "compiled PDFs have valid signatures and expected page counts",
        pdfs_ok,
        pdf_results,
        EXPECTED_PDFS,
    )
    progress.update()

    log_results = {
        name: validate_log(OUT / name.replace(".pdf", ".log"), name, pages)
        for name, pages in EXPECTED_PDFS.items()
    }
    record(
        checks,
        "final LaTeX logs contain no errors, unresolved references, or overfull boxes",
        all(result["passed"] for result in log_results.values()),
        log_results,
        "output marker present and every forbidden finding false",
    )
    progress.update()

    manuscript = (OUT / "manuscript_preview.tex").read_text(encoding="utf-8")
    supplement = (OUT / "supplementary_preview.tex").read_text(encoding="utf-8")
    combined = manuscript + "\n" + supplement
    required_boundaries = [
        "same-assignment efficiency diagnostic",
        "not proof that either gate is prospectively necessary",
        "were not used for threshold selection",
        "Identity is non-intervention",
        "not counted as effectiveness, non-harm, or successful external calibration",
        "separate pre-signal external robustness test",
        "not prospective effectiveness, universal safety, causal necessity, or threshold optimality",
        "one indivisible cluster",
        "designated post hoc",
        "all pairwise Jaccard indices 1.0",
        "not prospective workflow utility, external effectiveness, or a transferable operating-point guarantee",
    ]
    boundary_presence = {
        phrase: phrase in combined or phrase in manifest.get("claim_boundary", "")
        for phrase in required_boundaries
    }
    record(
        checks,
        "T2, T3, T5, cluster-yield, identity, and EEGEmotions-27 claims remain explicitly bounded",
        all(boundary_presence.values()),
        boundary_presence,
        "all required boundary phrases present",
    )
    progress.update()

    cluster_table = (OUT / "tables/table_complete_cluster_audit_yield.tex").read_text(
        encoding="utf-8"
    )
    cluster_table_checks = {
        "cluster capture": "Event-bearing clusters captured & 0.339" in cluster_table,
        "random comparator": "Random mean 0.200" in cluster_table,
        "exact test": "$p=0.0003$" in cluster_table,
        "post-hoc designation": "post-hoc designated cell" in cluster_table,
        "effective event n": "Effective $n=212$ of 480" in cluster_table,
        "effective amplification n": "Effective $n=173$ of 480" in cluster_table,
        "no broken row marker": "\\\n+" not in cluster_table,
    }
    record(
        checks,
        "complete-cluster primary table contains the validated estimates",
        all(cluster_table_checks.values()),
        cluster_table_checks,
        "all primary estimates present and every row terminator valid",
    )
    progress.update()

    aggregation_root = (
        ROOT / "outputs/dcs_opct_v11_cluster_score_aggregation_sensitivity_20260910"
    )
    aggregation_report = json.loads(
        (aggregation_root / "independent_validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    aggregation_manifest_hash = sha256(aggregation_root / "manifest.json")
    record(
        checks,
        "cluster-score aggregation sensitivity is validated and tracked",
        aggregation_report.get("status") == "passed"
        and aggregation_report.get("checks_passed") == 8
        and aggregation_report.get("checks_total") == 8
        and manifest.get("cluster_score_aggregation_manifest_sha256")
        == aggregation_manifest_hash,
        {
            "status": aggregation_report.get("status"),
            "checks": [
                aggregation_report.get("checks_passed"),
                aggregation_report.get("checks_total"),
            ],
            "manifest_hash": aggregation_manifest_hash,
        },
        "8 of 8 checks passed and manifest hash tracked",
    )
    progress.update()

    cluster_report = json.loads(
        (
            ROOT
            / "outputs/dcs_opct_v11_cluster_dependence_audit_yield_20260910/independent_validation_report.json"
        ).read_text(encoding="utf-8")
    )
    record(
        checks,
        "underlying complete-cluster analysis passed independent validation",
        cluster_report.get("status") == "passed"
        and cluster_report.get("checks_passed") == 16
        and cluster_report.get("checks_total") == 16,
        {
            "status": cluster_report.get("status"),
            "checks_passed": cluster_report.get("checks_passed"),
            "checks_total": cluster_report.get("checks_total"),
        },
        "16 of 16 checks passed",
    )
    progress.update()

    evidence_table = (OUT / "tables/table_evidence_role_matrix.tex").read_text(
        encoding="utf-8"
    )
    evidence_roles = {
        "mechanism sources": "Mechanism sources" in evidence_table,
        "target development": "Target development" in evidence_table,
        "post-access tests": "Post-access tests" in evidence_table,
        "pre-signal test": "Pre-signal test" in evidence_table,
        "reserved tests": "Reserved tests" in evidence_table,
        "reserved results absent": "No result yet; both independent tests are still required"
        in evidence_table,
    }
    record(
        checks,
        "the five evidence roles distinguish results from procedural readiness",
        all(evidence_roles.values()),
        evidence_roles,
        "all five roles and the no-result boundary are present",
    )
    progress.update()

    legacy_refs = [
        "figures/fig_strong_direct_calibration.pdf",
        "figures/fig_split_certified_calibration.pdf",
    ]
    new_refs = [name.replace(".png", ".pdf") for name in READABLE_FIGURES]
    reference_state = {
        "legacy_references": {ref: ref in supplement for ref in legacy_refs},
        "new_references": {ref: ref in supplement for ref in new_refs},
    }
    record(
        checks,
        "legacy dense comparator figures are replaced by five readable figures",
        not any(reference_state["legacy_references"].values())
        and all(reference_state["new_references"].values()),
        reference_state,
        "no legacy references and every readable figure referenced",
    )
    progress.update()

    figure_results: dict[str, Any] = {}
    figures_ok = True
    for name in READABLE_FIGURES:
        path = OUT / "figures" / name
        with Image.open(path) as image:
            width, height = image.size
            extrema = image.convert("L").getextrema()
        result = {
            "width": width,
            "height": height,
            "portrait": height > width,
            "grayscale_extrema": extrema,
            "bytes": path.stat().st_size,
        }
        figure_results[name] = result
        figures_ok = figures_ok and height > width and min(width, height) >= 2000
        figures_ok = figures_ok and extrema[0] < 245 and path.stat().st_size > 30_000
    record(
        checks,
        "five comparator figures are nonblank high-resolution portrait assets",
        figures_ok,
        figure_results,
        "portrait, at least 2000 px per dimension, nonblank, and larger than 30 kB",
    )
    progress.update()

    table_text = (OUT / "tables/table_evidence_role_matrix.tex").read_text(
        encoding="utf-8"
    )
    table_layout = {
        "array_package": r"\usepackage{array}" in supplement,
        "ragged_columns": table_text.count(r">{\raggedright\arraybackslash}") == 4,
        "array_stretch": r"\renewcommand{\arraystretch}{1.08}" in supplement,
    }
    record(
        checks,
        "the evidence-role matrix uses readable ragged paragraph columns",
        all(table_layout.values()),
        table_layout,
        "array package, four ragged columns, and explicit row spacing",
    )
    progress.update()

    rendered_pages = {
        "main T2": OUT / "rendered/main/page-24.png",
        "supplement cluster table": OUT / "rendered/supp/page-05.png",
        "supplement cluster figure": OUT / "rendered/supp/page-06.png",
        "supplement T2": OUT / "rendered/supp/page-15.png",
        "supplement T3": OUT / "rendered/supp/page-17.png",
        "supplement T5": OUT / "rendered/supp/page-18.png",
        "supplement evidence matrix": OUT / "rendered/supp/page-21.png",
        "supplement first readable comparator": OUT / "rendered/supp/page-26.png",
        "supplement final readable comparator": OUT / "rendered/supp/page-30.png",
    }
    rendered_results = {
        name: {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
        for name, path in rendered_pages.items()
    }
    record(
        checks,
        "key evidence pages have current nonempty visual-QA renders",
        all(item["exists"] and item["bytes"] > 30_000 for item in rendered_results.values()),
        rendered_results,
        "all key pages exist and exceed 30 kB",
    )
    progress.update()
    progress.close()

    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    report_path = OUT / "independent_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['checks_passed']}/{report['checks_total']} checks passed")
    print(report_path)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
