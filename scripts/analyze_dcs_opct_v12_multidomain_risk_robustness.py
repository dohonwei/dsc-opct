from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from develop_witness_gated_covariance_opct_v9 import load_development  # noqa: E402


PROTOCOL = ROOT / "docs/dcs_opct_v12_multidomain_risk_robustness_protocol.md"
PUBLIC = ROOT / "outputs/identity_shortcut_risk_classifier/risk_learning_table.csv"
PUBLIC_OOF = ROOT / "outputs/identity_shortcut_risk_classifier/grouped_cv_predictions.csv"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
OUT = ROOT / "outputs/dcs_opct_v12_multidomain_risk_robustness_v3"
DATASETS = ["DEAP", "MAHNOB-HCI", "EPPVR", "CASE", "CEAP", "SEED-IV", "DREAMER"]
FEATURES = [
    "opportunity_delta",
    "metadata_prior_opportunity",
    "encoding_margin",
    "dose_encoding_interaction",
    "model_capacity",
]
CLUSTERS = ["dataset", "task", "representation", "model", "split_seed"]
METHODS = ["pooled_gpu", "domain_balanced_gpu", "smooth_worst_domain_gpu"]
MATERIAL_THRESHOLD = 0.02
L2 = 0.1
WORST_DOMAIN_TEMPERATURE = 0.10
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
INPUTS = [
    PUBLIC,
    PUBLIC_OOF,
    ROOT / "outputs/eppvr_transport_policy_retrospective/eppvr_policy_predictions.csv",
    ROOT / "outputs/seediv_transport_policy_external/seediv_policy_predictions.csv",
    ROOT / "outputs/dreamer_v4_external_confirmation/mechanism_probabilities_blinded.csv",
    ROOT / "outputs/dreamer_counterfactual_external/summary.csv",
    ROOT / "outputs/case_v5_external_confirmation/primary_audit_labeled.csv",
    ROOT / "outputs/case_v5_external_confirmation/primary_heldout_results.csv",
    ROOT / "outputs/ceap_v6_external_confirmation/primary_audit_labeled.csv",
    ROOT / "outputs/ceap_v6_external_confirmation/primary_heldout_results.csv",
    FREEZE,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> pd.DataFrame:
    public = pd.read_csv(PUBLIC)
    public = public.loc[(public.axis == "subject") & (public.nominal_dose > 0)].copy()
    public = public[[*CLUSTERS, "nominal_dose", *FEATURES, "material_optimism_event"]]

    target_frames = []
    for dataset, (frame, _) in load_development().items():
        work = frame.copy()
        work["dataset"] = dataset
        target_frames.append(
            work[[*CLUSTERS, "nominal_dose", *FEATURES, "material_optimism_event"]]
        )
    combined = pd.concat([public, *target_frames], ignore_index=True)
    combined["dataset"] = pd.Categorical(combined.dataset, categories=DATASETS, ordered=True)
    combined = combined.sort_values([*CLUSTERS, "nominal_dose"]).reset_index(drop=True)
    combined["dataset"] = combined.dataset.astype(str)

    if set(combined.dataset) != set(DATASETS):
        raise ValueError("The seven-dataset contract is incomplete")
    values = combined[FEATURES].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Mechanism predictors must be finite")
    if not set(combined.material_optimism_event.unique()).issubset({0, 1}):
        raise ValueError("The endpoint must be binary")
    sizes = combined.groupby(CLUSTERS, observed=True).size()
    if not sizes.eq(4).all():
        raise ValueError(f"Every configuration cluster must contain four doses: {sizes.value_counts().to_dict()}")
    if combined.duplicated([*CLUSTERS, "nominal_dose"]).any():
        raise ValueError("Duplicate dataset-cluster-dose rows detected")
    class_counts = combined.groupby("dataset").material_optimism_event.nunique()
    if not class_counts.eq(2).all():
        raise ValueError(f"Every dataset must contain both endpoint classes: {class_counts.to_dict()}")
    return combined


@dataclass
class FittedModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    final_loss: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = (frame[FEATURES].to_numpy(float) - self.mean) / self.scale
        logits = x @ self.coefficient + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))


def fit_gpu(frame: pd.DataFrame, method: str) -> FittedModel:
    if method not in METHODS:
        raise ValueError(method)
    x_raw = frame[FEATURES].to_numpy(np.float64)
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale[scale < 1e-12] = 1.0
    x = torch.as_tensor((x_raw - mean) / scale, dtype=torch.float64, device="cuda")
    y = torch.as_tensor(frame.material_optimism_event.to_numpy(float), dtype=torch.float64, device="cuda")
    domains = frame.dataset.astype("category")
    domain_code = torch.as_tensor(
        domains.cat.codes.to_numpy(copy=True), dtype=torch.long, device="cuda"
    )
    n_domains = len(domains.cat.categories)
    coefficient = torch.nn.Parameter(torch.zeros(len(FEATURES), dtype=torch.float64, device="cuda"))
    prevalence = float(np.clip(y.mean().item(), 1e-6, 1.0 - 1e-6))
    intercept = torch.nn.Parameter(torch.tensor(np.log(prevalence / (1.0 - prevalence)), dtype=torch.float64, device="cuda"))
    optimizer = torch.optim.LBFGS(
        [coefficient, intercept], lr=0.8, max_iter=250, tolerance_grad=1e-10,
        tolerance_change=1e-12, line_search_fn="strong_wolfe"
    )
    final_loss = float("nan")

    def closure() -> torch.Tensor:
        nonlocal final_loss
        optimizer.zero_grad(set_to_none=True)
        logits = x @ coefficient + intercept
        row_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        if method == "pooled_gpu":
            data_loss = row_loss.mean()
        else:
            domain_losses = torch.stack([row_loss[domain_code == index].mean() for index in range(n_domains)])
            if method == "domain_balanced_gpu":
                data_loss = domain_losses.mean()
            else:
                tau = WORST_DOMAIN_TEMPERATURE
                data_loss = tau * torch.logsumexp(domain_losses / tau, dim=0)
        loss = data_loss + 0.5 * L2 * coefficient.square().sum()
        loss.backward()
        final_loss = float(loss.detach().cpu())
        return loss

    optimizer.step(closure)
    return FittedModel(
        mean=mean,
        scale=scale,
        coefficient=coefficient.detach().cpu().numpy(),
        intercept=float(intercept.detach().cpu()),
        final_loss=final_loss,
    )


def select_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    scores = np.array([balanced_accuracy_score(labels, probability >= value) for value in candidates])
    return float(candidates[np.flatnonzero(np.isclose(scores, scores.max()))[-1]])


def metric_row(dataset: str, method: str, labels: np.ndarray, probability: np.ndarray,
               threshold: float, training_prevalence: float) -> dict[str, object]:
    brier = float(brier_score_loss(labels, probability))
    null_brier = float(np.mean((labels - training_prevalence) ** 2))
    prevalence = float(labels.mean())
    ap = float(average_precision_score(labels, probability))
    return {
        "dataset": dataset,
        "method": method,
        "n": len(labels),
        "events": int(labels.sum()),
        "prevalence": prevalence,
        "threshold": threshold,
        "training_prevalence": training_prevalence,
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": ap,
        "average_precision_lift": ap / prevalence,
        "brier": brier,
        "brier_skill": 1.0 - brier / null_brier,
        "balanced_accuracy": float(balanced_accuracy_score(labels, probability >= threshold)),
    }


def frozen_reference(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame.loc[~frame.dataset.isin(["DEAP", "MAHNOB-HCI"])].copy()
    target_probability = []
    for dataset, (work, _) in load_development().items():
        rows = work.copy()
        rows["dataset"] = dataset
        target_probability.append(rows[[*CLUSTERS, "nominal_dose", "probability_identity"]])
    target_probability = pd.concat(target_probability, ignore_index=True)
    target = target.merge(target_probability, on=[*CLUSTERS, "nominal_dose"], validate="one_to_one")

    public_oof = pd.read_csv(PUBLIC_OOF)
    public_oof = public_oof.loc[public_oof.scheme.eq("leave_dataset_out")].copy()
    public_contract = pd.read_csv(PUBLIC)
    public_contract = public_contract.loc[(public_contract.axis == "subject") & (public_contract.nominal_dose > 0)].copy()
    public_contract = public_contract.merge(
        public_oof[["risk_row_id", "predicted_probability"]], on="risk_row_id", validate="one_to_one"
    )
    public_contract = public_contract[[*CLUSTERS, "nominal_dose", "predicted_probability"]]
    public_rows = frame.loc[frame.dataset.isin(["DEAP", "MAHNOB-HCI"])].merge(
        public_contract, on=[*CLUSTERS, "nominal_dose"], validate="one_to_one"
    )
    public_rows = public_rows.rename(columns={"predicted_probability": "probability_identity"})
    return pd.concat([public_rows, target], ignore_index=True)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v12 multi-domain robustness analysis")
    if sha256(FREEZE) != EXPECTED_FREEZE_HASH:
        raise RuntimeError("The frozen v11 file changed")
    torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    OUT.mkdir(parents=True, exist_ok=False)
    frame = load_contract()
    frame.to_csv(OUT / "seven_dataset_contract.csv", index=False)

    total_fits = len(DATASETS) * len(METHODS) * len(DATASETS)
    progress = tqdm(total=total_fits, desc="Nested seven-domain GPU risk analysis", unit="fit", dynamic_ncols=True)
    prediction_frames = []
    coefficient_rows = []
    metric_rows = []
    for held_out in DATASETS:
        train = frame.loc[frame.dataset.ne(held_out)].copy()
        test = frame.loc[frame.dataset.eq(held_out)].copy()
        for method in METHODS:
            inner_predictions = []
            for inner_held_out in train.dataset.drop_duplicates().tolist():
                inner_train = train.loc[train.dataset.ne(inner_held_out)]
                inner_test = train.loc[train.dataset.eq(inner_held_out)]
                model = fit_gpu(inner_train, method)
                inner_predictions.append(
                    pd.DataFrame({
                        "label": inner_test.material_optimism_event.to_numpy(int),
                        "probability": model.predict(inner_test),
                    })
                )
                progress.update(1)
            inner = pd.concat(inner_predictions, ignore_index=True)
            threshold = select_threshold(inner.label.to_numpy(int), inner.probability.to_numpy(float))
            model = fit_gpu(train, method)
            probability = model.predict(test)
            progress.update(1)
            prediction_frames.append(
                test[[*CLUSTERS, "nominal_dose", "material_optimism_event"]].assign(
                    method=method, probability=probability, decision_threshold=threshold,
                    predicted_event=(probability >= threshold).astype(int), outer_held_out=held_out
                )
            )
            metric_rows.append(metric_row(
                held_out, method, test.material_optimism_event.to_numpy(int), probability,
                threshold, float(train.material_optimism_event.mean())
            ))
            for feature, value in zip(FEATURES, model.coefficient, strict=True):
                coefficient_rows.append({"held_out": held_out, "method": method, "feature": feature, "coefficient": value})
    progress.close()

    frozen = frozen_reference(frame)
    freeze_manifest = json.loads((ROOT / "outputs/identity_shortcut_risk_classifier/freeze_manifest.json").read_text(encoding="utf-8"))
    frozen_threshold = float(freeze_manifest["decision_threshold"])
    for dataset, group in frozen.groupby("dataset", sort=False):
        labels = group.material_optimism_event.to_numpy(int)
        probability = group.probability_identity.to_numpy(float)
        training_prevalence = float(freeze_manifest["public_training_event_prevalence"])
        metric_rows.append(metric_row(dataset, "frozen_two_source_v11", labels, probability, frozen_threshold, training_prevalence))
        prediction_frames.append(
            group[[*CLUSTERS, "nominal_dose", "material_optimism_event"]].assign(
                method="frozen_two_source_v11", probability=probability,
                decision_threshold=frozen_threshold, predicted_event=(probability >= frozen_threshold).astype(int),
                outer_held_out=dataset,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    predictions.to_csv(OUT / "nested_lodo_predictions.csv", index=False)
    metrics.to_csv(OUT / "domain_metrics.csv", index=False)
    coefficients.to_csv(OUT / "fold_coefficients.csv", index=False)

    expanded = metrics.loc[metrics.method.isin(METHODS)]
    summary = expanded.groupby("method").agg(
        median_auroc=("auroc", "median"), mean_auroc=("auroc", "mean"),
        median_brier_skill=("brier_skill", "median"), mean_brier=("brier", "mean"),
        median_balanced_accuracy=("balanced_accuracy", "median"),
        domains_auroc_ge_060=("auroc", lambda x: int((x >= 0.60).sum())),
        domains_brier_skill_lt_minus_005=("brier_skill", lambda x: int((x < -0.05).sum())),
    ).reset_index()
    summary.to_csv(OUT / "method_summary.csv", index=False)
    selected = summary.set_index("method").loc["domain_balanced_gpu"]
    pooled = summary.set_index("method").loc["pooled_gpu"]
    gate_checks = {
        "five_of_seven_auroc_at_least_060": bool(selected.domains_auroc_ge_060 >= 5),
        "median_auroc_at_least_065": bool(selected.median_auroc >= 0.65),
        "median_brier_skill_positive": bool(selected.median_brier_skill > 0),
        "at_most_one_domain_brier_skill_below_minus_005": bool(selected.domains_brier_skill_lt_minus_005 <= 1),
        "median_balanced_accuracy_at_least_060": bool(selected.median_balanced_accuracy >= 0.60),
        "mean_brier_within_0002_of_pooled": bool(selected.mean_brier <= pooled.mean_brier + 0.002),
    }
    gate = {
        "status": "passed" if all(gate_checks.values()) else "failed",
        "evidence_role": "post_hoc_retrospective_multidomain_robustness_only",
        "checks": gate_checks,
        "selected_method": selected.to_dict(),
        "pooled_reference": pooled.to_dict(),
        "claim_boundary": "Passing does not establish prospective external effectiveness or universal safety.",
    }
    (OUT / "retrospective_robustness_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    output_paths = sorted(path for path in OUT.iterdir() if path.is_file())
    manifest = {
        "status": "complete_preserve_pass_or_failure",
        "date": "2026-09-09",
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "datasets": DATASETS,
        "features": FEATURES,
        "methods": METHODS,
        "nested_outer_folds": len(DATASETS),
        "gpu_fits": total_fits,
        "material_event_threshold": MATERIAL_THRESHOLD,
        "v11_freeze_sha256": sha256(FREEZE),
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in INPUTS},
        "output_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
        "claim_boundary": "Retrospective robustness branch; frozen v11 is unchanged and no prospective claim is permitted.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
