from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_reviewer_closure_20260914"
MANUSCRIPT = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_manuscript.tex"
SUPPLEMENT = ROOT / "docs/elsarticle/dcs_opct_v11_bspc_supplementary.tex"
MAIN_LOG = ROOT / "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.log"
SUPP_LOG = ROOT / "docs/elsarticle/build/dcs_opct_v11_bspc_supplementary.log"
FIGURE = ROOT / "docs/elsarticle/figures/fig_v11_three_stage_framework.pdf"
INDEPENDENCE_FIGURE = ROOT / "docs/elsarticle/figures/fig_v11_independence_ladder.pdf"
MAIN_PDF = ROOT / "docs/elsarticle/build/dcs_opct_v11_bspc_manuscript.pdf"

REPORTS = {
    "inductive_target": ROOT
    / "outputs/dcs_opct_v11_inductive_target_sensitivity_20260914/independent_validation_report.json",
    "raw_group_independence": ROOT
    / "outputs/dcs_opct_v11_raw_group_independence_sensitivity_20260914/independent_validation_report.json",
    "raw_partition_first": ROOT
    / "outputs/dcs_opct_v11_eppvr_raw_partition_first_20260914/independent_validation_report.json",
    "nearest_neighbor": ROOT
    / "outputs/dcs_opct_v11_nearest_neighbor_baselines_20260914/independent_validation_report.json",
    "target_action_endpoint": ROOT
    / "outputs/dcs_opct_v11_target_action_endpoint_uncertainty_20260914/independent_validation_report.json",
    "source_endpoint": ROOT
    / "outputs/dcs_opct_v11_probabilistic_event_risk_20260914/independent_validation_report.json",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_latex_log(text: str) -> bool:
    banned = (
        r"Overfull \\[hv]box",
        r"LaTeX Warning: There were undefined references",
        r"LaTeX Warning: Citation .* undefined",
        r"LaTeX Warning: Reference .* undefined",
        r"! LaTeX Error:",
        r"Fatal error occurred",
    )
    return not any(re.search(pattern, text) for pattern in banned)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    report_data = {name: read_json(path) for name, path in REPORTS.items()}

    checks = {
        "all_component_validators_pass": all(
            report.get("status") == "passed"
            and report.get("checks_passed") == report.get("checks_total")
            for report in report_data.values()
        ),
        "expected_component_check_counts": {
            name: report_data[name].get("checks_total") for name in report_data
        }
        == {
            "inductive_target": 12,
            "raw_group_independence": 13,
            "raw_partition_first": 21,
            "nearest_neighbor": 14,
            "target_action_endpoint": 12,
            "source_endpoint": 12,
        },
        "main_latex_log_clean": clean_latex_log(
            MAIN_LOG.read_text(encoding="utf-8", errors="replace")
        ),
        "supplement_latex_log_clean": clean_latex_log(
            SUPP_LOG.read_text(encoding="utf-8", errors="replace")
        ),
        "framework_figure_in_latest_pdf": MAIN_PDF.stat().st_mtime >= FIGURE.stat().st_mtime,
        "independence_figure_in_latest_pdf": (
            MAIN_PDF.stat().st_mtime >= INDEPENDENCE_FIGURE.stat().st_mtime
        ),
        "transductive_boundary_explicit": "transductive" in manuscript
        and "outcome-held-out" in manuscript,
        "raw_partition_boundary_explicit": all(
            term in manuscript
            for term in (
                "No participant, global raw row, training row",
                "no positive fully training-data-independent release",
                "fail-closed",
            )
        ),
        "endpoint_uncertainty_explicit": "measurement-uncertain" in manuscript,
        "nearest_neighbor_boundary_explicit": all(
            term in manuscript for term in ("CPCS", "ATC", "TransCal")
        ),
        "no_universal_safety_claim": "universal transport safety" in manuscript
        and "do not establish" in manuscript,
        "public_repository_link_visible": (
            "https://github.com/dohonwei/dsc-opct" in manuscript
            and "TODO(author): add anonymized repository URL or archival DOI" not in manuscript
        ),
        "funding_statement_verified": (
            "This work was supported by the National Natural Science Foundation of China (No. 62172081)."
            in manuscript
            and "TODO(author): add confirmed Funding statement" not in manuscript
        ),
        "supplement_present": len(supplement) > 1000,
    }
    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "passed" if not failed else "failed",
        "checks_passed": sum(bool(value) for value in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "author_dependent_blockers": [],
        "claim_boundary": (
            "Passing closes the currently implemented reviewer analyses and document checks. "
            "It verifies a fully participant- and training-row-disjoint sensitivity but does not "
            "establish a positive fully training-data-independent release, prospective external "
            "effectiveness, or universal non-harm."
        ),
    }
    (OUT / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
