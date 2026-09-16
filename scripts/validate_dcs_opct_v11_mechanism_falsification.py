from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_mechanism_falsification_20260910"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
PROTOCOL = ROOT / "docs/dcs_opct_v11_mechanism_falsification_protocol.md"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
DATASETS = {"DEAP", "MAHNOB-HCI", "EPPVR", "CASE", "CEAP", "SEED-IV", "DREAMER"}
MODELS = {"prevalence_only", "encoding_only", "opportunity_only", "additive_mechanism", "full_five_variable"}
CLUSTERS = ["dataset", "task", "representation", "model", "split_seed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_sign_randomization(effects: np.ndarray) -> float:
    observed = float(effects.mean())
    null = [float((effects * np.asarray(signs)).mean()) for signs in itertools.product([-1, 1], repeat=len(effects))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def add(checks: list[dict[str, Any]], name: str, passed: bool, observed: Any, required: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "observed": observed, "required": required})


def write_idempotent(path: Path, report: dict[str, Any]) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k != "generated_at_utc"} == {k: v for k, v in report.items() if k != "generated_at_utc"}:
            report["generated_at_utc"] = previous.get("generated_at_utc")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    checks: list[dict[str, Any]] = []
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((OUT / "mechanism_falsification_report.json").read_text(encoding="utf-8"))

    add(checks, "freeze unchanged", sha256(FREEZE) == EXPECTED_FREEZE, sha256(FREEZE), EXPECTED_FREEZE)
    add(checks, "protocol tracked", manifest["protocol_sha256"] == sha256(PROTOCOL) == report["protocol_sha256"],
        [manifest["protocol_sha256"], report["protocol_sha256"]], sha256(PROTOCOL))
    current_inputs = {path: sha256(ROOT / path) for path in manifest["input_sha256"]}
    add(checks, "input hashes", current_inputs == manifest["input_sha256"], current_inputs, manifest["input_sha256"])
    current_files = {path: sha256(ROOT / path) for path in manifest["files"]}
    add(checks, "artifact hashes", current_files == manifest["files"], current_files, manifest["files"])
    add(checks, "CUDA execution", "NVIDIA GeForce RTX 4070 Ti SUPER" in report["cuda_device"], report["cuda_device"], "RTX 4070 Ti SUPER")

    contract = pd.read_csv(OUT / "seven_dataset_mechanism_contract.csv")
    cluster_sizes = contract.groupby(CLUSTERS).size()
    contract_ok = len(contract) == 1920 and contract.groupby(CLUSTERS).ngroups == 480 and cluster_sizes.eq(4).all() and set(contract.dataset) == DATASETS
    add(checks, "seven-domain complete-cluster contract", contract_ok,
        [len(contract), contract.groupby(CLUSTERS).ngroups, cluster_sizes.value_counts().to_dict(), sorted(contract.dataset.unique())],
        [1920, 480, {4: 480}, sorted(DATASETS)])

    metrics = pd.read_csv(OUT / "nested_lodo_domain_metrics.csv")
    metrics_ok = len(metrics) == 35 and set(metrics.dataset) == DATASETS and set(metrics.model) == MODELS and metrics.groupby("dataset").model.nunique().eq(5).all()
    add(checks, "nested LODO coverage", metrics_ok, [len(metrics), sorted(metrics.dataset.unique()), sorted(metrics.model.unique())], "7 datasets x 5 models")
    predictions = pd.read_csv(OUT / "nested_lodo_predictions.csv")
    add(checks, "prediction coverage", len(predictions) == 9600 and predictions.predicted_probability.between(0, 1).all(),
        [len(predictions), float(predictions.predicted_probability.min()), float(predictions.predicted_probability.max())], [9600, "probabilities in [0,1]"])

    summary = pd.read_csv(OUT / "nested_model_summary.csv").set_index("model")
    recomputed = metrics.groupby("model").agg(median_auroc=("auroc", "median"), mean_auroc=("auroc", "mean"), mean_brier=("brier", "mean"))
    summary_ok = all(np.allclose(summary.loc[recomputed.index, col], recomputed[col], atol=1e-12) for col in recomputed.columns)
    add(checks, "model summaries recompute", summary_ok, recomputed.to_dict(), "saved values within 1e-12")

    paired = pd.read_csv(OUT / "dataset_level_paired_inference.csv").set_index("comparison_model")
    wide_auc = metrics.pivot(index="dataset", columns="model", values="auroc")
    wide_brier = metrics.pivot(index="dataset", columns="model", values="brier")
    paired_ok = True
    paired_observed = {}
    for model in paired.index:
        auc = wide_auc[model].to_numpy(float) - wide_auc.encoding_only.to_numpy(float)
        brier = wide_brier.encoding_only.to_numpy(float) - wide_brier[model].to_numpy(float)
        observed = [exact_sign_randomization(auc), exact_sign_randomization(brier)]
        paired_observed[model] = observed
        paired_ok &= np.isclose(observed[0], paired.loc[model, "exact_p_auc_gain"]) and np.isclose(observed[1], paired.loc[model, "exact_p_brier_improvement"])
    add(checks, "dataset-level exact inference", paired_ok, paired_observed, "all exact 2^7 sign randomizations reproduce")

    contrasts = pd.read_csv(OUT / "cluster_bootstrap_quartile_contrasts.csv").set_index("feature")
    gradient_ok = (
        contrasts.loc["encoding_margin", "bootstrap_ci_low"] < 0 < contrasts.loc["encoding_margin", "bootstrap_ci_high"]
        and contrasts.loc["opportunity_delta", "bootstrap_ci_low"] > 0
        and contrasts.loc["metadata_prior_opportunity", "bootstrap_ci_low"] > 0
        and contrasts.bootstrap_repetitions.eq(5000).all()
    )
    add(checks, "opportunity-specific gradient", gradient_ok, contrasts.to_dict(orient="index"),
        "encoding CI crosses zero; both opportunity CIs are positive; 5000 cluster bootstraps")

    external = pd.read_csv(OUT / "external_encoding_utilization_counterexamples.csv").set_index("dataset")
    external_ok = (
        set(external.index) == {"FACED", "EEGEmotions-27"}
        and external.loc["FACED", "material_positive_events"] == 0
        and external.loc["FACED", "action"] == "identity"
        and external.loc["EEGEmotions-27", "action"] == "identity"
        and external.loc["EEGEmotions-27", "evidence_role"] == "separate pre-signal external robustness test"
    )
    add(checks, "bounded external counterexamples", external_ok, external.reset_index().to_dict(orient="records"), "FACED and EEGEmotions-27 bounded roles preserved")

    image = Image.open(OUT / "fig_mechanism_falsification.png")
    add(checks, "figure nonempty", image.width >= 2000 and image.height >= 800, [image.width, image.height], ">=2000 x 800")
    add(checks, "failed composite flag preserved", report["mechanism_falsification_supported"] is False,
        report["mechanism_falsification_supported"], False)
    boundary_tokens = ["not a sufficient", "post-hoc", "prospective transfer effectiveness", "superiority of the interaction term"]
    add(checks, "claim boundary", all(token in report["claim_boundary"] for token in boundary_tokens),
        {token: token in report["claim_boundary"] for token in boundary_tokens}, "all true")

    validation = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks),
        "checks": checks,
        "interpretation": (
            "Opportunity gradients were supported while the encoding gradient was not, but the prespecified composite "
            "mechanism-falsification flag failed and remains false."
        ),
    }
    write_idempotent(OUT / "independent_validation_report.json", validation)
    if validation["status"] != "passed":
        raise RuntimeError([item["check"] for item in checks if not item["passed"]])
    print(f"Mechanism-falsification validation passed: {validation['checks_passed']}/{validation['checks_total']}")


if __name__ == "__main__":
    main()
