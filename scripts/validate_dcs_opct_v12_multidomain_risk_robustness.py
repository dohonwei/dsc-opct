from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v12_multidomain_risk_robustness_v3"
V2 = ROOT / "outputs/dcs_opct_v12_multidomain_risk_robustness_v2"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
SCRIPT = ROOT / "scripts/analyze_dcs_opct_v12_multidomain_risk_robustness.py"
PROTOCOL = ROOT / "docs/dcs_opct_v12_multidomain_risk_robustness_protocol.md"
REPORT = OUT / "independent_validation_report.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
DATASET_COUNTS = {
    "DEAP": (240, 41),
    "MAHNOB-HCI": (240, 73),
    "EPPVR": (320, 52),
    "CASE": (320, 67),
    "CEAP": (320, 79),
    "SEED-IV": (240, 14),
    "DREAMER": (240, 70),
}
CLUSTERS = ["dataset", "task", "representation", "model", "split_seed"]
EXPANDED_METHODS = {"pooled_gpu", "domain_balanced_gpu", "smooth_worst_domain_gpu"}


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
    gate = json.loads((OUT / "retrospective_robustness_gate.json").read_text(encoding="utf-8"))
    contract = pd.read_csv(OUT / "seven_dataset_contract.csv")
    predictions = pd.read_csv(OUT / "nested_lodo_predictions.csv")
    metrics = pd.read_csv(OUT / "domain_metrics.csv")
    coefficients = pd.read_csv(OUT / "fold_coefficients.csv")
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    tasks = tqdm(total=17, desc="Validate v12 multi-domain robustness", unit="check", dynamic_ncols=True)
    record(
        "manifest_status_and_boundary",
        manifest.get("status") == "complete_preserve_pass_or_failure"
        and "Retrospective robustness branch" in manifest.get("claim_boundary", ""),
        {"status": manifest.get("status"), "claim_boundary": manifest.get("claim_boundary")},
    )
    tasks.update(1)
    record("freeze_hash_unchanged", sha256(FREEZE) == EXPECTED_FREEZE_HASH == manifest["v11_freeze_sha256"], sha256(FREEZE))
    tasks.update(1)
    record("analysis_script_hash", sha256(SCRIPT) == manifest["script_sha256"], sha256(SCRIPT))
    tasks.update(1)
    record("protocol_hash", sha256(PROTOCOL) == manifest["protocol_sha256"], sha256(PROTOCOL))
    tasks.update(1)

    input_mismatch = {
        relative: {"expected": expected, "actual": sha256(ROOT / relative)}
        for relative, expected in manifest["input_sha256"].items()
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected
    }
    record("eleven_inputs_current", len(manifest["input_sha256"]) == 11 and not input_mismatch, input_mismatch)
    tasks.update(1)
    output_mismatch = {
        relative: {"expected": expected, "actual": sha256(ROOT / relative)}
        for relative, expected in manifest["output_sha256"].items()
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected
    }
    record("six_outputs_current", len(manifest["output_sha256"]) == 6 and not output_mismatch, output_mismatch)
    tasks.update(1)

    observed_counts = {
        dataset: (len(group), int(group.material_optimism_event.sum()))
        for dataset, group in contract.groupby("dataset")
    }
    record("seven_dataset_counts_and_events", observed_counts == DATASET_COUNTS, observed_counts)
    tasks.update(1)
    cluster_sizes = contract.groupby(CLUSTERS).size()
    record(
        "complete_four_dose_clusters",
        len(contract) == 1920 and cluster_sizes.eq(4).all()
        and not contract.duplicated([*CLUSTERS, "nominal_dose"]).any(),
        {"rows": len(contract), "cluster_size_counts": cluster_sizes.value_counts().to_dict()},
    )
    tasks.update(1)
    record(
        "prediction_grid_complete",
        len(predictions) == 1920 * 4
        and set(predictions.method) == EXPANDED_METHODS | {"frozen_two_source_v11"},
        predictions.method.value_counts().to_dict(),
    )
    tasks.update(1)
    record(
        "outer_holdout_rows_only",
        predictions.dataset.eq(predictions.outer_held_out).all(),
        int(predictions.dataset.ne(predictions.outer_held_out).sum()),
    )
    tasks.update(1)
    threshold_counts = (
        predictions.loc[predictions.method.isin(EXPANDED_METHODS)]
        .groupby(["dataset", "method"]).decision_threshold.nunique()
    )
    threshold_detail = {
        f"{dataset}:{method}": int(count)
        for (dataset, method), count in threshold_counts.items()
    }
    record(
        "one_nested_threshold_per_fold",
        len(threshold_counts) == 21 and threshold_counts.eq(1).all(),
        threshold_detail,
    )
    tasks.update(1)

    metric_errors = []
    for row in metrics.itertuples(index=False):
        group = predictions.loc[(predictions.dataset == row.dataset) & (predictions.method == row.method)]
        labels = group.material_optimism_event.to_numpy(int)
        probability = group.probability.to_numpy(float)
        if not close(roc_auc_score(labels, probability), row.auroc):
            metric_errors.append(f"{row.dataset}:{row.method}:auroc")
        if not close(brier_score_loss(labels, probability), row.brier):
            metric_errors.append(f"{row.dataset}:{row.method}:brier")
        if not close(balanced_accuracy_score(labels, probability >= row.threshold), row.balanced_accuracy):
            metric_errors.append(f"{row.dataset}:{row.method}:balanced_accuracy")
    record("domain_metrics_reproduced", not metric_errors, metric_errors)
    tasks.update(1)
    record(
        "gpu_execution_recorded",
        manifest["gpu_fits"] == 147 and "RTX 4070 Ti SUPER" in manifest["gpu"]
        and manifest["peak_gpu_memory_bytes"] > 0,
        {key: manifest[key] for key in ["gpu", "gpu_fits", "peak_gpu_memory_bytes"]},
    )
    tasks.update(1)
    failed_gate_checks = sorted(name for name, passed in gate["checks"].items() if not passed)
    expected_failed = sorted([
        "five_of_seven_auroc_at_least_060",
        "median_auroc_at_least_065",
        "median_balanced_accuracy_at_least_060",
    ])
    record("prespecified_failure_retained", gate["status"] == "failed" and failed_gate_checks == expected_failed, failed_gate_checks)
    tasks.update(1)
    selected = metrics.loc[metrics.method.eq("domain_balanced_gpu")]
    record(
        "reported_selected_summary_reproduced",
        close(selected.auroc.median(), gate["selected_method"]["median_auroc"])
        and close(selected.brier_skill.median(), gate["selected_method"]["median_brier_skill"])
        and close(selected.balanced_accuracy.median(), gate["selected_method"]["median_balanced_accuracy"]),
        gate["selected_method"],
    )
    tasks.update(1)
    selected_coefficients = coefficients.loc[coefficients.method.eq("domain_balanced_gpu")]
    record(
        "domain_balanced_mechanism_signs_positive",
        len(selected_coefficients) == 35 and selected_coefficients.coefficient.gt(0).all(),
        selected_coefficients.groupby("feature").coefficient.min().to_dict(),
    )
    tasks.update(1)
    comparable = sorted(set(manifest["output_sha256"]).intersection(
        str(path.relative_to(ROOT)).replace("v2", "v3") for path in V2.glob("*.csv")
    ))
    v2_v3 = {}
    for v3_relative in comparable:
        v3_path = ROOT / v3_relative
        v2_path = Path(str(v3_path).replace("_v3", "_v2"))
        v2_v3[v3_path.name] = sha256(v3_path) == sha256(v2_path)
    record("v2_v3_scientific_outputs_identical", len(v2_v3) == 5 and all(v2_v3.values()), v2_v3)
    tasks.update(1)
    tasks.close()

    passed = sum(bool(check["passed"]) for check in checks)
    content = {
        "status": "passed" if passed == len(checks) else "failed",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "claim_boundary": "Validation preserves the failed retrospective gate and does not authorize a prospective or universal safety claim.",
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
        raise RuntimeError(f"v12 multi-domain validation failed: {failed}")
    print(f"v12 multi-domain validation passed: {passed}/{len(checks)} checks")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
