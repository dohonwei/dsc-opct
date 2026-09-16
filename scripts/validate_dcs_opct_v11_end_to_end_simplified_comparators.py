from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
DATASETS = {"EPPVR", "SEED-IV", "DREAMER", "CASE", "CEAP"}
VARIANTS = {"risk_ranking_only", "certificate_only", "single_unlabeled_gate", "full_dcs_opct"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v11 simplified comparator artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dcs_opct_v11_end_to_end_simplified_comparators_20260910_v2"))
    parser.add_argument("--freeze", type=Path, default=Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.output_root
    required = ["end_to_end_comparator_domains.csv", "end_to_end_comparator_summary.csv", "fig_end_to_end_simplified_comparators.pdf", "fig_end_to_end_simplified_comparators.png", "conclusion.json", "manifest.json"]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    domains = pd.read_csv(root / "end_to_end_comparator_domains.csv")
    summary = pd.read_csv(root / "end_to_end_comparator_summary.csv")
    conclusion = json.loads((root / "conclusion.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: Any, required_value: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "required": required_value})

    freeze_hash = sha256(args.freeze)
    record("freeze_hash", freeze_hash == EXPECTED_FREEZE, freeze_hash, EXPECTED_FREEZE)
    record("manifest_boundary", manifest["frozen_v11_unchanged"] is True and "Post-hoc" in manifest["analysis_role"], manifest["analysis_role"], "post-hoc frozen analysis")
    record("domain_dimensions", len(domains) == 20 and set(domains.dataset) == DATASETS and set(domains.variant) == VARIANTS, [len(domains), sorted(domains.dataset.unique())], [20, sorted(DATASETS)])
    record("summary_dimensions", len(summary) == 4 and set(summary.variant) == VARIANTS, len(summary), 4)
    record("common_ceiling", summary.same_audit_budget_ceiling.nunique() == 1 and int(summary.same_audit_budget_ceiling.iloc[0]) == 720, summary.same_audit_budget_ceiling.tolist(), [720] * 4)
    release_sets = summary.loc[summary.variant.ne("risk_ranking_only"), "released_domain_names"]
    record("release_set_preserved", release_sets.nunique() == 1 and release_sets.iloc[0] == "EPPVR;CASE;CEAP", release_sets.tolist(), "EPPVR;CASE;CEAP")
    full = summary.loc[summary.variant.eq("full_dcs_opct")].iloc[0]
    certificate = summary.loc[summary.variant.eq("certificate_only")].iloc[0]
    record("label_reduction", int(full.action_labels_required) == 480 and int(certificate.action_labels_required) == 720 and np.isclose(conclusion["full_label_reduction_vs_certificate_only"], 1 / 3), [int(full.action_labels_required), int(certificate.action_labels_required), conclusion["full_label_reduction_vs_certificate_only"]], [480, 720, 1 / 3])
    record("no_material_negative_transfer", summary.material_negative_transfer_count.eq(0).all(), summary.material_negative_transfer_count.tolist(), [0] * 4)
    record("risk_only_nonintervention", domains.loc[domains.variant.eq("risk_ranking_only"), "selected_action"].eq("identity").all(), domains.loc[domains.variant.eq("risk_ranking_only"), "selected_action"].tolist(), ["identity"] * 5)
    claim = conclusion["claim_boundary"].lower()
    record("claim_boundary", "not proof" in claim and "external effectiveness" in claim, conclusion["claim_boundary"], "bounded component diagnostic")
    image = mpimg.imread(root / "fig_end_to_end_simplified_comparators.png")
    record("figure_dimensions", image.shape[0] >= 700 and image.shape[1] >= 1800, list(image.shape), ">=700 x >=1800")
    inputs_valid = all(Path(path).is_file() and sha256(Path(path)) == digest for path, digest in manifest["inputs"].items())
    outputs_valid = all((root / name).is_file() and sha256(root / name) == digest for name, digest in manifest["outputs"].items())
    record("input_hashes", inputs_valid, bool(inputs_valid), True)
    record("output_hashes", outputs_valid, bool(outputs_valid), True)
    report = {"status": "passed" if all(item["passed"] for item in checks) else "failed", "analysis_date": "2026-09-10", "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks), "checks": checks}
    (root / "independent_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['checks_passed']}/{report['checks_total']} checks passed")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
