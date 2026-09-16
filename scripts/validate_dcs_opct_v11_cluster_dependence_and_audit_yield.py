from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "dcs_opct_v11_cluster_dependence_audit_yield_20260910"
FREEZE = ROOT / "docs" / "distribution_covered_stratified_opct_v11_final_freeze.json"
MANUSCRIPT = ROOT / "docs" / "elsarticle" / "dcs_opct_v11_bspc_manuscript.tex"
SUPPLEMENTARY = ROOT / "docs" / "elsarticle" / "dcs_opct_v11_bspc_supplementary.tex"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EXPECTED_MANUSCRIPT = "dcab80f4727be5bc67586989353c36164b66e2aedf4e364ffa0da32f22c1ed79"
EXPECTED_SUPPLEMENTARY = "eea5ea34e2a767cfde70fc62e831b2a12429b9aa178a44a3e73f6dccb5820b3f"
CLUSTER_COLUMNS = ["dataset", "task", "representation", "model", "split_seed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    audit = pd.read_csv(OUTPUT / "audit_yield_summary.csv")
    clusters = pd.read_csv(OUTPUT / "complete_cluster_table.csv")
    allocations = pd.read_csv(OUTPUT / "frozen_ranked_allocations.csv")
    random = pd.read_csv(OUTPUT / "primary_random_null_replicates.csv")
    dependence = pd.read_csv(OUTPUT / "cluster_dependence_summary.csv")
    intervals = pd.read_csv(OUTPUT / "interval_dependence_sensitivity.csv")
    conclusion = json.loads(
        (OUTPUT / "bounded_conclusion.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record("freeze_hash", sha256(FREEZE) == EXPECTED_FREEZE, sha256(FREEZE))
    record(
        "formal_tex_unchanged",
        sha256(MANUSCRIPT) == EXPECTED_MANUSCRIPT
        and sha256(SUPPLEMENTARY) == EXPECTED_SUPPLEMENTARY,
        [sha256(MANUSCRIPT), sha256(SUPPLEMENTARY)],
    )
    record(
        "manifest_lock_consistency",
        manifest["locked_tex_sha256_before"]
        == manifest["locked_tex_sha256_after"],
        manifest["locked_tex_sha256_after"],
    )
    record(
        "complete_cluster_dimensions",
        len(clusters) == 600
        and set(clusters.configurations.unique()) == {4}
        and not clusters.duplicated(
            ["material_effect_threshold", *CLUSTER_COLUMNS]
        ).any(),
        [len(clusters), sorted(clusters.configurations.unique().tolist())],
    )
    record(
        "analysis_grid_dimensions",
        len(audit) == 25 and len(allocations) == 900 and len(random) == 50000,
        [len(audit), len(allocations), len(random)],
    )
    primary = audit.loc[
        np.isclose(audit.material_effect_threshold, 0.02)
        & np.isclose(audit.audit_budget_fraction, 0.20)
    ].iloc[0]
    record(
        "primary_budget_is_complete_cluster",
        primary.risk_selected_clusters == 24
        and primary.risk_labeled_configurations == 96,
        [
            int(primary.risk_selected_clusters),
            int(primary.risk_labeled_configurations),
        ],
    )
    record(
        "primary_cluster_event_capture",
        np.isclose(primary.risk_cluster_event_sensitivity, 19 / 56),
        float(primary.risk_cluster_event_sensitivity),
    )
    record(
        "primary_row_event_capture",
        np.isclose(primary.risk_row_event_sensitivity, 43 / 114),
        float(primary.risk_row_event_sensitivity),
    )
    record(
        "primary_ranking_beats_random_tail",
        primary.risk_cluster_event_sensitivity
        > primary.random_cluster_event_sensitivity_q975,
        [
            float(primary.risk_cluster_event_sensitivity),
            float(primary.random_cluster_event_sensitivity_q975),
        ],
    )
    record(
        "primary_exact_inference",
        np.isclose(
            primary.cluster_event_exact_random_p_one_sided,
            0.0003446716618680,
        )
        and primary.cluster_event_exact_random_p_one_sided < 0.001,
        float(primary.cluster_event_exact_random_p_one_sided),
    )
    overall_event = dependence.loc[
        np.isclose(dependence.material_effect_threshold, 0.02)
        & dependence.scope.eq("overall")
        & dependence.endpoint.eq("material_event")
    ].iloc[0]
    record(
        "event_cluster_dependence_reproduced",
        np.isclose(overall_event.icc_1_1, 0.4198659354052407)
        and np.isclose(
            overall_event.effective_configuration_rows, 212.4271844660194
        ),
        [
            float(overall_event.icc_1_1),
            float(overall_event.effective_configuration_rows),
        ],
    )
    record(
        "dependence_sensitivity_dimensions",
        len(dependence) == 45 and len(intervals) == 6,
        [len(dependence), len(intervals)],
    )
    record(
        "cluster_intervals_not_silently_narrower",
        bool((intervals.cluster_to_naive_width_ratio > 0.95).all()),
        intervals[
            ["scope", "endpoint", "cluster_to_naive_width_ratio"]
        ].to_dict("records"),
    )
    record(
        "bounded_claim_language",
        all(
            phrase in conclusion["bounded_interpretation"]
            for phrase in [
                "retrospectively",
                "does not establish prospective workflow utility",
                "external effectiveness",
                "transferable operating-point guarantee",
            ]
        ),
        conclusion["bounded_interpretation"],
    )
    image = mpimg.imread(OUTPUT / "fig_cluster_dependence_audit_yield.png")
    record(
        "figure_dimensions",
        image.shape[0] >= 1600 and image.shape[1] >= 2000,
        list(image.shape),
    )
    record(
        "output_hashes",
        all(
            sha256(OUTPUT / name) == value
            for name, value in manifest["outputs"].items()
        ),
        len(manifest["outputs"]),
    )

    report = {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "analysis_date": "2026-09-10",
        "checks_passed": sum(check["passed"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    (OUTPUT / "independent_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"{report['checks_passed']}/{report['checks_total']} checks passed")
    if report["status"] != "passed":
        failed = [check["name"] for check in checks if not check["passed"]]
        print("Failed checks:", ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
