from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_consensus_order_preserving_transport_v7 import load_development  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    CLUSTER_COLUMNS,
    STRATUM_COLUMNS,
    partition_frame,
)
from develop_order_preserving_transport_v6 import RULE  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import prepare as prepare_v9  # noqa: E402
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402


ASSIGNMENTS = Path(
    "outputs/distribution_covered_stratified_opct_v11_development/"
    "distribution_covered_assignments.csv"
)
CERTIFICATES = Path(
    "outputs/distribution_covered_stratified_opct_v11_development/"
    "primary_certificates.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen DCS-OPCT v11 with direct target Platt and isotonic "
            "calibration under the identical audit-label budget."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dcs_opct_v11_target_calibration_baselines"),
    )
    parser.add_argument("--platt-epochs", type=int, default=1500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_positive_platt(
    probability: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    seed: int,
    progress: tqdm,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    x = torch.as_tensor(logit(clipped), dtype=torch.float32, device="cuda")
    y = torch.as_tensor(labels, dtype=torch.float32, device="cuda")
    raw_scale = torch.nn.Parameter(torch.zeros((), device="cuda"))
    shift = torch.nn.Parameter(torch.zeros((), device="cuda"))
    optimizer = torch.optim.Adam([raw_scale, shift], lr=0.03)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.nn.functional.softplus(raw_scale) + 1e-4
        prediction = torch.sigmoid(scale * x + shift)
        loss = torch.nn.functional.binary_cross_entropy(prediction, y)
        loss.backward()
        optimizer.step()
        progress.update(1)
    return (
        float((torch.nn.functional.softplus(raw_scale) + 1e-4).detach().cpu()),
        float(shift.detach().cpu()),
    )


def apply_platt(probability: np.ndarray, scale: float, shift: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return expit(scale * logit(clipped) + shift)


def safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else np.nan


def cluster_metrics(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    work = frame.copy()
    labels = work.material_optimism_event.to_numpy(int)
    identity = work.probability_identity.to_numpy(float)
    candidate = work[f"probability_{method}"].to_numpy(float)
    work["identity_loss"] = (labels - identity) ** 2
    work["candidate_loss"] = (labels - candidate) ** 2
    return (
        work.groupby(list(CLUSTER_COLUMNS), sort=True, dropna=False)
        .agg(
            identity_loss=("identity_loss", "mean"),
            candidate_loss=("candidate_loss", "mean"),
        )
        .reset_index()
    )


def stratified_bootstrap_gain(
    clusters: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = np.zeros(repetitions, dtype=float)
    total = 0
    for _, stratum in clusters.groupby(list(STRATUM_COLUMNS), sort=False, dropna=False):
        gain = (stratum.identity_loss - stratum.candidate_loss).to_numpy(float)
        sampled = rng.integers(0, len(gain), size=(repetitions, len(gain)))
        draws += gain[sampled].sum(axis=1)
        total += len(gain)
    draws /= total
    point = float(np.mean(clusters.identity_loss - clusters.candidate_loss))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def plot_results(summary: pd.DataFrame, output: Path) -> None:
    datasets = summary.dataset.drop_duplicates().tolist()
    methods = ["dcs_opct", "target_platt", "target_isotonic"]
    labels = ["DCS-OPCT", "Target Platt", "Target isotonic"]
    colors = ["#1b7f79", "#356fa3", "#bd7a18"]
    x = np.arange(len(datasets), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)
    for index, (method, label, color) in enumerate(zip(methods, labels, colors, strict=True)):
        part = summary.set_index(["dataset", "method"]).loc[
            [(dataset, method) for dataset in datasets]
        ]
        offset = (index - 1) * width
        axes[0].bar(x + offset, part.brier_gain, width, label=label, color=color)
        axes[1].bar(x + offset, part.auc_delta, width, label=label, color=color)
    axes[0].axhline(0.0, color="#303840", linewidth=1)
    axes[1].axhline(0.0, color="#303840", linewidth=1)
    axes[0].set_ylabel("Held-out Brier gain vs identity")
    axes[1].set_ylabel("Held-out AUROC change vs identity")
    for axis in axes:
        axis.set_xticks(x, datasets, rotation=20, ha="right")
        axis.grid(axis="y", color="#dce2e8", linewidth=0.8)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.05, 1.18))
    fig.suptitle("Matched target-label-budget calibration baselines", fontsize=15, fontweight="bold")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the matched Platt baseline")
    if args.smoke:
        args.platt_epochs = min(args.platt_epochs, 50)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)

    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )
    prepared, _, _ = prepare_v9(load_development(), config, args.seed)
    assignments = pd.read_csv(ASSIGNMENTS)
    selected = pd.read_csv(CERTIFICATES).set_index("dataset").selected_method.to_dict()
    training_progress = tqdm(
        total=len(prepared) * args.platt_epochs,
        desc="CUDA positive-Platt baselines",
        unit="epoch",
        dynamic_ncols=True,
    )
    prediction_frames = []
    summary_rows = []
    for dataset, frame in prepared.items():
        assignment = assignments.loc[assignments.dataset.eq(dataset)].drop(columns="dataset")
        audit, heldout = partition_frame(frame, assignment)
        audit_y = audit.material_optimism_event.to_numpy(int)
        audit_identity = audit.probability_identity.to_numpy(float)
        heldout_identity = heldout.probability_identity.to_numpy(float)

        scale, shift = fit_positive_platt(
            audit_identity,
            audit_y,
            args.platt_epochs,
            args.seed + len(prediction_frames) * 1009,
            training_progress,
        )
        heldout["probability_target_platt"] = apply_platt(heldout_identity, scale, shift)
        isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        isotonic.fit(audit_identity, audit_y)
        heldout["probability_target_isotonic"] = isotonic.predict(heldout_identity)
        heldout["probability_dcs_opct"] = (
            heldout.probability_wg_opct.to_numpy(float)
            if selected[dataset] == "wg_opct"
            else heldout_identity.copy()
        )
        heldout["dataset"] = dataset
        heldout["platt_scale"] = scale
        heldout["platt_shift"] = shift
        prediction_frames.append(heldout)

        labels = heldout.material_optimism_event.to_numpy(int)
        identity_auc = safe_auc(labels, heldout_identity)
        identity_brier = float(np.mean((labels - heldout_identity) ** 2))
        for method_index, method in enumerate(
            ("dcs_opct", "target_platt", "target_isotonic")
        ):
            probability = heldout[f"probability_{method}"].to_numpy(float)
            clusters = cluster_metrics(heldout, method)
            point, ci_low, ci_high = stratified_bootstrap_gain(
                clusters,
                args.bootstrap_repetitions,
                args.seed + len(summary_rows) * 7919 + method_index,
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "evidence_role": "retrospective_matched_budget_comparator",
                    "n_audit": len(audit),
                    "n_heldout": len(heldout),
                    "identity_brier": identity_brier,
                    "candidate_brier": float(np.mean((labels - probability) ** 2)),
                    "brier_gain": point,
                    "brier_gain_ci_low": ci_low,
                    "brier_gain_ci_high": ci_high,
                    "identity_auc": identity_auc,
                    "candidate_auc": safe_auc(labels, probability),
                    "auc_delta": safe_auc(labels, probability) - identity_auc,
                    "platt_scale": scale if method == "target_platt" else np.nan,
                    "platt_shift": shift if method == "target_platt" else np.nan,
                }
            )
    training_progress.close()

    args.output_root.mkdir(parents=True, exist_ok=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    predictions.to_csv(args.output_root / "heldout_predictions.csv", index=False)
    summary.to_csv(args.output_root / "matched_budget_summary.csv", index=False)
    plot_results(summary, args.output_root / "fig_matched_budget_calibration_baselines")
    manifest = {
        "status": "retrospective_matched_budget_comparison_complete",
        "date": "2026-09-08",
        "claim_boundary": (
            "Comparator analysis only; it does not alter the frozen DCS-OPCT v11 "
            "algorithm, release gates, or prospective-confirmation status."
        ),
        "audit_fraction": 0.5,
        "platt_epochs": args.platt_epochs,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "gpu_name": torch.cuda.get_device_name(0),
        "input_sha256": {
            "assignments": sha256(ASSIGNMENTS),
            "certificates": sha256(CERTIFICATES),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
