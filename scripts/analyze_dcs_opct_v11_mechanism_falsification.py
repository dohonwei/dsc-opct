from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from develop_witness_gated_covariance_opct_v9 import load_development  # noqa: E402


PROTOCOL = ROOT / "docs/dcs_opct_v11_mechanism_falsification_protocol.md"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
PUBLIC = ROOT / "outputs/identity_shortcut_risk_classifier/risk_learning_table.csv"
FACED_ENCODING = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/faced_identity_encoding_summary.csv"
FACED_OUTCOMES = ROOT / "outputs/dcs_opct_v11_submission_artifacts_crossfit_v15/faced_configuration_outcomes.csv"
EEG_SUMMARY = ROOT / "outputs/eegemotions27_v11_posthoc_boundary_analysis_v2/posthoc_summary.json"
OUT = ROOT / "outputs/dcs_opct_v11_mechanism_falsification_20260910"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
DATASETS = ["DEAP", "MAHNOB-HCI", "EPPVR", "CASE", "CEAP", "SEED-IV", "DREAMER"]
CLUSTERS = ["dataset", "task", "representation", "model", "split_seed"]
MODEL_FEATURES = {
    "prevalence_only": [],
    "encoding_only": ["encoding_margin"],
    "opportunity_only": ["opportunity_delta", "metadata_prior_opportunity"],
    "additive_mechanism": [
        "opportunity_delta", "metadata_prior_opportunity", "encoding_margin", "model_capacity"
    ],
    "full_five_variable": [
        "opportunity_delta", "metadata_prior_opportunity", "encoding_margin",
        "dose_encoding_interaction", "model_capacity",
    ],
}
QUARTILE_FEATURES = ["encoding_margin", "opportunity_delta", "metadata_prior_opportunity"]
L2 = 0.1
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_SEED = 20260910


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> pd.DataFrame:
    columns = [*CLUSTERS, "nominal_dose", *MODEL_FEATURES["full_five_variable"], "material_optimism_event"]
    public = pd.read_csv(PUBLIC)
    public = public.loc[(public.axis == "subject") & (public.nominal_dose > 0), columns].copy()
    frames = [public]
    for dataset, (frame, _) in load_development().items():
        work = frame.copy()
        work["dataset"] = dataset
        frames.append(work[columns])
    contract = pd.concat(frames, ignore_index=True)
    contract = contract.loc[contract.dataset.isin(DATASETS)].copy()
    contract["dataset"] = pd.Categorical(contract.dataset, DATASETS, ordered=True)
    contract = contract.sort_values([*CLUSTERS, "nominal_dose"]).reset_index(drop=True)
    contract["dataset"] = contract.dataset.astype(str)
    if set(contract.dataset) != set(DATASETS):
        raise ValueError("Seven-dataset contract is incomplete")
    if not contract.groupby(CLUSTERS).size().eq(4).all():
        raise ValueError("Every mechanism cluster must contain four nonzero doses")
    if not np.isfinite(contract[MODEL_FEATURES["full_five_variable"]].to_numpy(float)).all():
        raise ValueError("Non-finite mechanism predictor")
    if not set(contract.material_optimism_event.unique()).issubset({0, 1}):
        raise ValueError("Material event must be binary")
    return contract


class GpuLogistic:
    def __init__(self, features: list[str]):
        self.features = features

    def fit(self, frame: pd.DataFrame) -> "GpuLogistic":
        raw = frame[self.features].to_numpy(np.float64)
        self.mean = raw.mean(axis=0)
        self.scale = raw.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        x = torch.as_tensor((raw - self.mean) / self.scale, dtype=torch.float64, device="cuda")
        y = torch.as_tensor(frame.material_optimism_event.to_numpy(float), dtype=torch.float64, device="cuda")
        domains = frame.dataset.astype("category")
        codes = torch.as_tensor(domains.cat.codes.to_numpy(copy=True), dtype=torch.long, device="cuda")
        n_domains = len(domains.cat.categories)
        self.coefficient = torch.nn.Parameter(torch.zeros(len(self.features), dtype=torch.float64, device="cuda"))
        prevalence = float(np.clip(y.mean().item(), 1e-6, 1 - 1e-6))
        self.intercept = torch.nn.Parameter(torch.tensor(np.log(prevalence / (1 - prevalence)), dtype=torch.float64, device="cuda"))
        optimizer = torch.optim.LBFGS(
            [self.coefficient, self.intercept], lr=0.8, max_iter=250,
            tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            logits = x @ self.coefficient + self.intercept
            row_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            domain_losses = torch.stack([row_loss[codes == index].mean() for index in range(n_domains)])
            loss = domain_losses.mean() + 0.5 * L2 * self.coefficient.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        self.coefficient_np = self.coefficient.detach().cpu().numpy()
        self.intercept_value = float(self.intercept.detach().cpu())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = (frame[self.features].to_numpy(float) - self.mean) / self.scale
        logits = x @ self.coefficient_np + self.intercept_value
        return 1 / (1 + np.exp(-np.clip(logits, -35, 35)))


def metric_row(dataset: str, model: str, y: np.ndarray, p: np.ndarray, train_prevalence: float) -> dict[str, object]:
    brier = float(brier_score_loss(y, p))
    null_brier = float(np.mean((y - train_prevalence) ** 2))
    prevalence = float(y.mean())
    ap = float(average_precision_score(y, p))
    return {
        "dataset": dataset, "model": model, "n": len(y), "events": int(y.sum()),
        "prevalence": prevalence, "auroc": float(roc_auc_score(y, p)),
        "average_precision": ap, "average_precision_lift": ap / prevalence,
        "brier": brier, "brier_skill": 1 - brier / null_brier,
        "log_loss": float(log_loss(y, p)),
    }


def exact_sign_randomization(effects: np.ndarray) -> float:
    observed = float(effects.mean())
    null = [float((effects * np.asarray(signs)).mean()) for signs in itertools.product([-1, 1], repeat=len(effects))]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def nested_models(contract: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, predictions = [], []
    progress = tqdm(total=len(DATASETS) * len(MODEL_FEATURES), desc="CUDA mechanism LODO", unit="fit")
    for held_out in DATASETS:
        train = contract.loc[contract.dataset.ne(held_out)]
        test = contract.loc[contract.dataset.eq(held_out)]
        y = test.material_optimism_event.to_numpy(int)
        prevalence = float(train.material_optimism_event.mean())
        for name, features in MODEL_FEATURES.items():
            if features:
                probability = GpuLogistic(features).fit(train).predict(test)
            else:
                probability = np.full(len(test), prevalence)
            metrics.append(metric_row(held_out, name, y, probability, prevalence))
            saved = test[[*CLUSTERS, "nominal_dose", "material_optimism_event"]].copy()
            saved["held_out_dataset"] = held_out
            saved["mechanism_model"] = name
            saved["predicted_probability"] = probability
            predictions.append(saved)
            progress.update(1)
    progress.close()
    metric_frame = pd.DataFrame(metrics)
    summary = metric_frame.groupby("model", sort=False).agg(
        median_auroc=("auroc", "median"), mean_auroc=("auroc", "mean"),
        median_brier=("brier", "median"), mean_brier=("brier", "mean"),
        median_brier_skill=("brier_skill", "median"), mean_log_loss=("log_loss", "mean"),
    ).reset_index()
    wide_auc = metric_frame.pivot(index="dataset", columns="model", values="auroc")
    wide_brier = metric_frame.pivot(index="dataset", columns="model", values="brier")
    comparisons = []
    for name in MODEL_FEATURES:
        if name == "encoding_only":
            continue
        auc_gain = (wide_auc[name] - wide_auc.encoding_only).loc[DATASETS].to_numpy(float)
        brier_gain = (wide_brier.encoding_only - wide_brier[name]).loc[DATASETS].to_numpy(float)
        comparisons.append({
            "comparison_model": name,
            "domains_auc_improved": int((auc_gain > 0).sum()),
            "mean_auc_gain": float(auc_gain.mean()),
            "exact_p_auc_gain": exact_sign_randomization(auc_gain),
            "domains_brier_improved": int((brier_gain > 0).sum()),
            "mean_brier_improvement": float(brier_gain.mean()),
            "exact_p_brier_improvement": exact_sign_randomization(brier_gain),
        })
    return metric_frame, summary, pd.DataFrame(comparisons), pd.concat(predictions, ignore_index=True)


def quartile_analysis(contract: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = contract.copy()
    for feature in QUARTILE_FEATURES:
        work[f"{feature}_quartile"] = work.groupby("dataset")[feature].transform(
            lambda values: pd.qcut(values.rank(method="first"), 4, labels=False)
        ).astype(int)
    rows = []
    for feature in QUARTILE_FEATURES:
        column = f"{feature}_quartile"
        grouped = work.groupby(["dataset", column], sort=True).material_optimism_event.agg(["mean", "sum", "count"]).reset_index()
        grouped["feature"] = feature
        grouped = grouped.rename(columns={column: "quartile", "mean": "event_rate", "sum": "events", "count": "n"})
        rows.append(grouped)
    rates = pd.concat(rows, ignore_index=True)

    cluster_id = work[CLUSTERS].astype(str).agg("|".join, axis=1)
    work["cluster_id"] = cluster_id
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrast_rows = []
    for feature in QUARTILE_FEATURES:
        qcol = f"{feature}_quartile"
        per_dataset_arrays = {}
        observed_domain = []
        for dataset in DATASETS:
            part = work.loc[work.dataset.eq(dataset)]
            clusters = []
            for _, group in part.groupby("cluster_id", sort=True):
                clusters.append(np.array([
                    group.loc[group[qcol].eq(0), "material_optimism_event"].sum(), group[qcol].eq(0).sum(),
                    group.loc[group[qcol].eq(3), "material_optimism_event"].sum(), group[qcol].eq(3).sum(),
                ], dtype=float))
            array = np.stack(clusters)
            per_dataset_arrays[dataset] = array
            total = array.sum(axis=0)
            observed_domain.append(total[2] / total[3] - total[0] / total[1])
        draws = np.empty(BOOTSTRAP_REPETITIONS, dtype=float)
        for rep in tqdm(range(BOOTSTRAP_REPETITIONS), desc=f"Cluster bootstrap {feature}", unit="rep"):
            domain_effects = []
            for dataset in DATASETS:
                array = per_dataset_arrays[dataset]
                sampled = array[rng.integers(0, len(array), len(array))].sum(axis=0)
                domain_effects.append(sampled[2] / sampled[3] - sampled[0] / sampled[1])
            draws[rep] = np.mean(domain_effects)
        observed = float(np.mean(observed_domain))
        contrast_rows.append({
            "feature": feature, "equal_domain_q4_minus_q1": observed,
            "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
            "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
            "domains_positive": int((np.asarray(observed_domain) > 0).sum()),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        })
    return rates, pd.DataFrame(contrast_rows)


def external_counterexamples() -> pd.DataFrame:
    faced_encoding = pd.read_csv(FACED_ENCODING)
    faced_outcomes = pd.read_csv(FACED_OUTCOMES)
    eeg = json.loads(EEG_SUMMARY.read_text(encoding="utf-8"))
    return pd.DataFrame([
        {
            "dataset": "FACED", "evidence_role": "post-access exploratory",
            "encoding_margin_min": float(faced_encoding.encoding_margin_mean.min()),
            "encoding_margin_max": float(faced_encoding.encoding_margin_mean.max()),
            "material_positive_events": int(faced_outcomes.material_optimism_event.sum()),
            "total_event_rows": int(len(faced_outcomes)), "monotonic_configurations": 0,
            "action": "identity", "boundary": "High decodability coexisted with zero frozen material events.",
        },
        {
            "dataset": "EEGEmotions-27", "evidence_role": "separate pre-signal external robustness test",
            "encoding_margin_min": float(eeg["identity_encoding_margin_range"][0]),
            "encoding_margin_max": float(eeg["identity_encoding_margin_range"][1]),
            "material_positive_events": int(eeg["audit_material_positive_events"] + eeg["heldout_material_positive_events"]),
            "total_event_rows": 120, "monotonic_configurations": int(eeg["monotonic_positive_configurations"]),
            "action": eeg["selected_method"],
            "boundary": "High decodability coexisted with sparse heterogeneous events and no globally monotonic dose response.",
        },
    ])


def make_figure(metrics: pd.DataFrame, rates: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    order = ["encoding_only", "opportunity_only", "additive_mechanism", "full_five_variable"]
    labels = ["Encoding only", "Opportunity only", "Additive", "Full five-variable"]
    x = np.arange(len(DATASETS))
    width = 0.19
    for index, (model, label) in enumerate(zip(order, labels)):
        values = metrics.loc[metrics.model.eq(model)].set_index("dataset").loc[DATASETS, "auroc"]
        axes[0].bar(x + (index - 1.5) * width, values, width=width, label=label)
    axes[0].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x, DATASETS, rotation=35, ha="right")
    axes[0].set_ylabel("Leave-one-dataset-out AUROC")
    axes[0].set_title("A. Encoding alone did not identify utilization risk")
    axes[0].legend(frameon=False, fontsize=8)

    colors = {"encoding_margin": "#4C78A8", "opportunity_delta": "#F58518", "metadata_prior_opportunity": "#54A24B"}
    for feature in QUARTILE_FEATURES:
        part = rates.loc[rates.feature.eq(feature)].groupby("quartile").agg(events=("events", "sum"), n=("n", "sum"))
        axes[1].plot(np.arange(1, 5), part.events / part.n, marker="o", linewidth=2, color=colors[feature], label=feature.replace("_", " "))
    axes[1].set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
    axes[1].set_ylabel("Material-event rate")
    axes[1].set_title("B. Opportunity, not decodability, showed a dose gradient")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mechanism_falsification.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_mechanism_falsification.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if sha256(FREEZE) != EXPECTED_FREEZE:
        raise RuntimeError("Frozen v11 hash changed")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(BOOTSTRAP_SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    contract = load_contract()
    contract.to_csv(OUT / "seven_dataset_mechanism_contract.csv", index=False)
    metrics, summary, comparisons, predictions = nested_models(contract)
    metrics.to_csv(OUT / "nested_lodo_domain_metrics.csv", index=False)
    summary.to_csv(OUT / "nested_model_summary.csv", index=False)
    comparisons.to_csv(OUT / "dataset_level_paired_inference.csv", index=False)
    predictions.to_csv(OUT / "nested_lodo_predictions.csv", index=False)
    rates, contrasts = quartile_analysis(contract)
    rates.to_csv(OUT / "within_dataset_quartile_event_rates.csv", index=False)
    contrasts.to_csv(OUT / "cluster_bootstrap_quartile_contrasts.csv", index=False)
    external = external_counterexamples()
    external.to_csv(OUT / "external_encoding_utilization_counterexamples.csv", index=False)
    make_figure(metrics, rates)

    summary_i = summary.set_index("model")
    full_cmp = comparisons.set_index("comparison_model").loc["full_five_variable"]
    contrast_i = contrasts.set_index("feature")
    supported = bool(
        summary_i.loc["encoding_only", "median_auroc"] <= 0.55
        and summary_i.loc["full_five_variable", "median_auroc"] >= 0.60
        and contrast_i.loc["opportunity_delta", "equal_domain_q4_minus_q1"] > 0
        and contrast_i.loc["metadata_prior_opportunity", "equal_domain_q4_minus_q1"] > 0
        and external.loc[external.dataset.eq("FACED"), "material_positive_events"].iloc[0] == 0
    )
    report = {
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "post_hoc_data_informed_mechanism_falsification",
        "freeze_sha256": sha256(FREEZE),
        "protocol_sha256": sha256(PROTOCOL),
        "cuda_device": torch.cuda.get_device_name(0),
        "rows": len(contract), "complete_clusters": int(contract.groupby(CLUSTERS).ngroups),
        "encoding_only_median_auroc": float(summary_i.loc["encoding_only", "median_auroc"]),
        "full_model_median_auroc": float(summary_i.loc["full_five_variable", "median_auroc"]),
        "full_vs_encoding_domains_brier_improved": int(full_cmp.domains_brier_improved),
        "full_vs_encoding_exact_p_brier": float(full_cmp.exact_p_brier_improvement),
        "quartile_contrasts": contrasts.to_dict(orient="records"),
        "mechanism_falsification_supported": supported,
        "claim_boundary": (
            "Identity decodability alone was not a sufficient cross-domain indicator of material utilization. "
            "This post-hoc analysis does not establish prospective transfer effectiveness, causal invariance, "
            "universal safety, or superiority of the interaction term."
        ),
    }
    (OUT / "mechanism_falsification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    tracked = [path for path in OUT.iterdir() if path.is_file() and path.name != "manifest.json"]
    manifest = {
        "status": "built_pending_independent_validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha256(FREEZE), "protocol_sha256": sha256(PROTOCOL),
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in [PUBLIC, FACED_ENCODING, FACED_OUTCOMES, EEG_SUMMARY]},
        "files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in tracked},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
