from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from pypdf import PdfReader
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate staged EEGEmotions-27 post-hoc publication artifacts."
    )
    parser.add_argument(
        "--primary-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_external_robustness"),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_posthoc_boundary_analysis_v2"),
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_posthoc_boundary_analysis_v2/figures"),
    )
    parser.add_argument(
        "--table-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_posthoc_boundary_analysis_v2/latex_tables_v4"),
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    report_path = args.analysis_root / "artifact_validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")
    checks: list[dict[str, Any]] = []
    progress = tqdm(total=6, desc="EEGEmotions-27 artifact validation", unit="check")

    primary = json.loads(
        (args.primary_root / "independent_validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    posthoc = json.loads(
        (args.analysis_root / "posthoc_summary.json").read_text(encoding="utf-8")
    )
    record(
        checks,
        "publication artifacts are downstream of a passed immutable-result validation",
        primary.get("status") == "passed"
        and primary.get("n_passed") == primary.get("n_checks") == 12
        and posthoc.get("primary_gate_changed") is False
        and posthoc.get("primary_claim_supported") is False,
        {
            "primary_validation": f"{primary.get('n_passed')}/{primary.get('n_checks')}",
            "primary_gate_changed": posthoc.get("primary_gate_changed"),
            "primary_claim_supported": posthoc.get("primary_claim_supported"),
        },
        "12/12 primary checks and no gate or claim change",
    )
    progress.update()

    expected_rows = {
        "applicability_thresholds.csv": 2,
        "baseline_dual_generalization.csv": 6,
        "dose_monotonicity_by_configuration.csv": 30,
        "dose_trajectory_summary.csv": 30,
        "identity_encoding_summary.csv": 6,
        "model_ranking_reversals.csv": 60,
        "nonreleased_counterfactual_diagnostics.csv": 4,
        "representation_ranking_reversals.csv": 40,
    }
    observed_rows = {}
    csv_ok = True
    for filename, expected in expected_rows.items():
        frame = pd.read_csv(args.analysis_root / filename)
        observed_rows[filename] = len(frame)
        csv_ok = csv_ok and len(frame) == expected and "index" not in frame.columns
    event_table = pd.read_csv(args.analysis_root / "material_event_summary.csv")
    csv_ok = csv_ok and "index" not in event_table.columns
    record(
        checks,
        "clean post-hoc CSV tables cover the expected grids without artifact indices",
        csv_ok,
        observed_rows,
        expected_rows,
    )
    progress.update()

    figure_manifest = json.loads(
        (args.figure_root / "figure_manifest.json").read_text(encoding="utf-8")
    )
    figure_files = sorted(args.figure_root.glob("fig_eegemotions27_*.*"))
    png_files = [path for path in figure_files if path.suffix == ".png"]
    pdf_files = [path for path in figure_files if path.suffix == ".pdf"]
    png_shapes = {}
    png_ok = True
    for path in png_files:
        with Image.open(path) as image:
            png_shapes[path.name] = list(image.size)
            png_ok = png_ok and image.width >= 1500 and image.height >= 900
    pdf_pages = {path.name: len(PdfReader(str(path)).pages) for path in pdf_files}
    figures_ok = (
        len(png_files) == 5
        and len(pdf_files) == 5
        and png_ok
        and all(count == 1 for count in pdf_pages.values())
        and figure_manifest.get("primary_gate_changed") is False
    )
    record(
        checks,
        "five publication figures have high-resolution PNG and one-page vector PDF forms",
        figures_ok,
        {"png_shapes": png_shapes, "pdf_pages": pdf_pages},
        "five PNGs at least 1500x900 and five one-page PDFs",
    )
    progress.update()

    table_manifest = json.loads(
        (args.table_root / "table_manifest.json").read_text(encoding="utf-8")
    )
    table_hashes_ok = all(
        (args.table_root / filename).is_file()
        and sha256(args.table_root / filename) == metadata["sha256"]
        and (args.table_root / filename).stat().st_size == metadata["bytes"]
        for filename, metadata in table_manifest["tables"].items()
    )
    record(
        checks,
        "staged LaTeX source files retain the generated manifest hashes",
        table_hashes_ok and table_manifest.get("primary_gate_changed") is False,
        {"files": len(table_manifest["tables"]), "hashes_ok": table_hashes_ok},
        "five source files with exact hashes and unchanged primary gate",
    )
    progress.update()

    preview_pdf = args.table_root / "eegemotions27_tables_preview.pdf"
    preview_log = args.table_root / "eegemotions27_tables_preview.log"
    log_text = preview_log.read_text(encoding="utf-8", errors="replace")
    preview_pages = len(PdfReader(str(preview_pdf)).pages)
    compile_ok = (
        preview_pages == 4
        and "Output written on eegemotions27_tables_preview.pdf" in log_text
        and "LaTeX Error" not in log_text
        and "Overfull" not in log_text
        and "Fatal error" not in log_text
    )
    record(
        checks,
        "TeX Live 2026 compiled the four-page preview without errors or overfull boxes",
        compile_ok,
        {
            "pages": preview_pages,
            "latex_error": "LaTeX Error" in log_text,
            "overfull": "Overfull" in log_text,
            "fatal_error": "Fatal error" in log_text,
        },
        {"pages": 4, "latex_error": False, "overfull": False, "fatal_error": False},
    )
    progress.update()

    required_boundaries = [
        "non-intervention",
        "not counted as effectiveness or safety success",
        "Post-hoc",
        "cannot alter the frozen external gate",
        "not an independent-sample confidence interval",
    ]
    table_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(args.table_root.glob("table_eegemotions27_*.tex"))
    )
    boundaries_ok = all(text in table_text for text in required_boundaries)
    record(
        checks,
        "captions preserve the inferential and identity-fallback boundaries",
        boundaries_ok,
        {text: text in table_text for text in required_boundaries},
        "all required boundary phrases present",
    )
    progress.update()
    progress.close()

    passed = all(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed else "failed",
        "date": "2026-09-09",
        "dataset": "EEGEmotions-27",
        "n_checks": len(checks),
        "n_passed": sum(check["passed"] for check in checks),
        "checks": checks,
        "preview_pdf": {
            "path": preview_pdf.as_posix(),
            "sha256": sha256(preview_pdf),
            "bytes": preview_pdf.stat().st_size,
        },
        "integration_status": "awaiting_author_figure_and_table_review",
        "claim_boundary": (
            "Artifact validation establishes internal consistency and layout quality only. "
            "It does not change the frozen external result."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
