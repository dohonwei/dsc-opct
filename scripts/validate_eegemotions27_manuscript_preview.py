from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops
from pypdf import PdfReader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/eegemotions27_v11_manuscript_integration_preview"

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
    "manuscript_preview.pdf": 34,
    "supplementary_preview.pdf": 21,
}


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
    progress = tqdm(total=9, desc="EEGEmotions-27 manuscript preview", unit="check")

    observed_hashes = {
        relative: sha256(ROOT / relative) for relative in EXPECTED_HASHES
    }
    record(
        checks,
        "authoritative manuscript sources and frozen protocol files are unchanged",
        observed_hashes == EXPECTED_HASHES,
        observed_hashes,
        EXPECTED_HASHES,
    )
    progress.update()

    manifest_path = OUT / "preview_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hashes = manifest.get("files", {})
    current_manifest_hashes = {
        relative: sha256(ROOT / relative) for relative in manifest_hashes
    }
    record(
        checks,
        "every staged TeX, figure, and table retains its preview-manifest hash",
        manifest_hashes == current_manifest_hashes
        and manifest.get("status")
        == "staged_preview_not_merged_into_authoritative_manuscript",
        {
            "tracked_files": len(manifest_hashes),
            "hashes_match": manifest_hashes == current_manifest_hashes,
            "status": manifest.get("status"),
        },
        "all tracked hashes match and preview remains unmerged",
    )
    progress.update()

    pdf_results: dict[str, Any] = {}
    pdfs_ok = True
    for name, expected_pages in EXPECTED_PDFS.items():
        path = OUT / name
        signature = path.read_bytes()[:5]
        pages = len(PdfReader(str(path)).pages)
        pdf_results[name] = {
            "signature": signature.decode("ascii", errors="replace"),
            "pages": pages,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        pdfs_ok = pdfs_ok and signature == b"%PDF-" and pages == expected_pages
    record(
        checks,
        "compiled preview PDFs have valid signatures and expected page counts",
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
        "final TeX logs contain no errors, unresolved references, or overfull boxes",
        all(result["passed"] for result in log_results.values()),
        log_results,
        "output marker present and every forbidden finding false",
    )
    progress.update()

    manuscript = (OUT / "manuscript_preview.tex").read_text(encoding="utf-8")
    supplement = (OUT / "supplementary_preview.tex").read_text(encoding="utf-8")
    combined = manuscript + "\n" + supplement
    required_boundaries = [
        "separate pre-signal external robustness test",
        "not participant-experienced valence or arousal",
        "Identity is non-intervention",
        "not effectiveness or safety success",
        "cannot be used to retune the threshold",
        "No universal prevention of negative transfer is claimed",
    ]
    boundary_presence = {
        phrase: phrase in combined for phrase in required_boundaries
    }
    record(
        checks,
        "claim-boundary language remains explicit in the staged manuscript package",
        all(boundary_presence.values()),
        boundary_presence,
        "all required boundary phrases present",
    )
    progress.update()

    evidence_role_tokens = {
        "retrospective budget qualifier": (
            "Across 100 matched-budget retrospective repetitions" in manuscript
        ),
        "four-domain role retained": (
            "four unsupported development or exploratory domains" in manuscript
        ),
        "separate pre-signal role stated": (
            "A separate pre-signal external robustness test also returned identity before "
            "target labels entered action selection."
            in manuscript
        ),
        "all failed external effectiveness tests bounded": (
            "successful external effectiveness on AVDOS-VR, FACED, or EEGEmotions-27"
            in manuscript
        ),
    }
    record(
        checks,
        "domain counts and evidence roles remain consistent after EEGEmotions integration",
        all(evidence_role_tokens.values()),
        evidence_role_tokens,
        "retrospective qualifier, four-domain category, separate pre-signal role, and complete boundary",
    )
    progress.update()

    commit = "0dfec14d177ed73c37a95f06d942e9b58e2bbf77"
    formatting_ok = f"\\texttt{{{commit}}}" in manuscript and "\texttt" not in manuscript
    record(
        checks,
        "the registered repository commit is intact and correctly typeset",
        formatting_ok,
        {
            "full_commit_present": commit in manuscript,
            "texttt_present": f"\\texttt{{{commit}}}" in manuscript,
            "tab_corruption_present": "\texttt" in manuscript,
        },
        "full commit enclosed by a valid LaTeX texttt command with no tab corruption",
    )
    progress.update()

    split_figure_ok = all(
        token in manuscript
        for token in [
            "\\label{fig:eegemotions-applicability}",
            "\\label{fig:eegemotions-dual-generalization}",
            "width=0.78\\textwidth",
            "width=\\textwidth]{figures/fig_eegemotions27_dual_generalization.pdf}",
        ]
    ) and "width=0.49\\textwidth" not in manuscript
    record(
        checks,
        "applicability and dual-generalization results are separate full-width figures",
        split_figure_ok,
        {
            "separate_labels": all(
                label in manuscript
                for label in [
                    "fig:eegemotions-applicability",
                    "fig:eegemotions-dual-generalization",
                ]
            ),
            "legacy_side_by_side_width_present": "width=0.49\\textwidth"
            in manuscript,
        },
        "two separately labeled figures and no legacy side-by-side layout",
    )
    progress.update()

    render_specs = {
        "main": list(range(22, 30)),
        "supp": list(range(12, 22)),
    }
    rendered: dict[str, Any] = {}
    renders_ok = True
    for group, pages in render_specs.items():
        for page in pages:
            path = OUT / "rendered_r2" / group / f"page-{page}.png"
            with Image.open(path).convert("RGB") as image:
                white = Image.new("RGB", image.size, "white")
                nonwhite_bbox = ImageChops.difference(image, white).getbbox()
                key = f"{group}/page-{page}.png"
                rendered[key] = {
                    "size": list(image.size),
                    "nonblank": nonwhite_bbox is not None,
                }
                renders_ok = renders_ok and image.width >= 1000 and nonwhite_bbox is not None
    record(
        checks,
        "all targeted manuscript and supplement pages were rendered and are nonblank",
        renders_ok,
        rendered,
        "18 PNG pages at least 1000 pixels wide with visible content",
    )
    progress.update()
    progress.close()

    passed = all(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed else "failed",
        "date": "2026-09-09",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_checks": len(checks),
        "n_passed": sum(check["passed"] for check in checks),
        "checks": checks,
        "integration_status": "awaiting_author_preview_approval",
        "claim_boundary": (
            "This validation establishes frozen-file integrity, compile quality, and "
            "preview consistency. It does not establish external effectiveness, non-harm, "
            "safety, replication, or a transferable operating-point guarantee."
        ),
    }
    report_path = OUT / "preview_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
