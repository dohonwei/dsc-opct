from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_dcs_opct_v11_audit_budget_decision_curves import (  # noqa: E402
    ASSIGNMENTS, BUDGET_FRACTIONS, LOCKS, apply_platt, frame_seed,
    heldout_metrics, stratified_brier_certificate, summarize,
)
from analyze_dcs_opct_v11_strong_calibration_baselines import (  # noqa: E402
    apply_beta, fit_monotone_beta_batch,
)
from analyze_dcs_opct_v11_audit_budget_decision_curves import fit_positive_platt_batch  # noqa: E402
from develop_consensus_order_preserving_transport_v7 import load_development  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    CLUSTER_COLUMNS, STRATUM_COLUMNS, partition_frame,
)
from develop_order_preserving_transport_v6 import RULE  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import prepare as prepare_v9  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig, balanced_cluster_order, rows_for_clusters,
)

FREEZE = Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json")
EXPECTED_FREEZE_SHA256 = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
PARAMETERS = Path("outputs/dcs_opct_v11_strong_calibration_baselines/calibration_parameters.csv")
LEGACY_REPLICATES = Path("outputs/dcs_opct_v11_strong_calibration_baselines/decision_curve_replicates.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-fit certified Platt and beta baselines on frozen v11 audits.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dcs_opct_v11_crossfit_certified_baselines"))
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--calibration-epochs", type=int, default=1000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--seed-step", type=int, default=7919)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_index():
    frame = pd.read_csv(PARAMETERS)
    return frame.set_index(["dataset", "budget_fraction", "audit_repetition", "method"])


def main():
    args = parse_args()
    if sha256(FREEZE) != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("Frozen v11 manifest hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 4)
        args.calibration_epochs = min(args.calibration_epochs, 50)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 100)
    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )
    prepared, _, _ = prepare_v9(load_development(), config, args.seed)
    assignments = pd.read_csv(ASSIGNMENTS)
    locks = pd.read_csv(LOCKS).set_index("dataset")
    parameters = parameter_index()
    total_fits = len(prepared) * len(BUDGET_FRACTIONS) * args.calibration_epochs * 2
    progress = tqdm(total=total_fits, desc="CUDA opposite-fold calibration fits", unit="epoch", dynamic_ncols=True)
    rows, certificates = [], []
    gain_matrices = {}
    heldout_strata = {}
    for dataset_index, (dataset, frame) in enumerate(prepared.items()):
        assignment = assignments.loc[assignments.dataset.eq(dataset)].drop(columns="dataset")
        audit_pool, heldout = partition_frame(frame, assignment)
        heldout_clusters = heldout.loc[:, list(CLUSTER_COLUMNS)].drop_duplicates().sort_values(list(CLUSTER_COLUMNS)).reset_index(drop=True)
        heldout_clusters["cluster_index"] = np.arange(len(heldout_clusters), dtype=int)
        heldout = heldout.merge(heldout_clusters, on=list(CLUSTER_COLUMNS), how="left", validate="many_to_one")
        heldout_strata[dataset] = [
            group.cluster_index.to_numpy(int)
            for _, group in heldout_clusters.groupby(list(STRATUM_COLUMNS), sort=False, dropna=False)
        ]
        orders = [
            balanced_cluster_order(audit_pool, config, args.seed + dataset_index * 1000003 + repetition * args.seed_step)
            for repetition in range(args.audit_repetitions)
        ]
        total_clusters = len(orders[0])
        identity_heldout = heldout.probability_identity.to_numpy(float)
        for fraction in BUDGET_FRACTIONS:
            cluster_budget = max(2, int(round(total_clusters * fraction)))
            audits, fold_a, fold_b = [], [], []
            for order in orders:
                selected = order[:cluster_budget]
                audits.append(audit_pool.loc[rows_for_clusters(audit_pool, CLUSTER_COLUMNS, selected)].reset_index(drop=True))
                fold_a.append(audit_pool.loc[rows_for_clusters(audit_pool, CLUSTER_COLUMNS, selected[::2])].reset_index(drop=True))
                fold_b.append(audit_pool.loc[rows_for_clusters(audit_pool, CLUSTER_COLUMNS, selected[1::2])].reset_index(drop=True))
            probability_b = np.stack([part.probability_identity.to_numpy(float) for part in fold_b])
            labels_b = np.stack([part.material_optimism_event.to_numpy(int) for part in fold_b])
            platt_b_scale, platt_b_shift = fit_positive_platt_batch(
                probability_b, labels_b, args.calibration_epochs,
                args.seed + dataset_index * 6007 + cluster_budget, progress,
            )
            beta_b_a, beta_b_b, beta_b_shift = fit_monotone_beta_batch(
                probability_b, labels_b, args.calibration_epochs,
                args.seed + dataset_index * 7001 + cluster_budget, progress,
            )
            for repetition in range(args.audit_repetitions):
                audit = audits[repetition]
                part_a = fold_a[repetition].copy()
                part_b = fold_b[repetition].copy()
                key = (dataset, fraction, repetition)
                platt_a = parameters.loc[key + ("split_certified_platt",)]
                beta_a = parameters.loc[key + ("split_certified_beta",)]
                platt_final = parameters.loc[key + ("target_platt",)]
                beta_final = parameters.loc[key + ("target_beta",)]

                part_a["probability_crossfit_platt"] = apply_platt(
                    part_a.probability_identity.to_numpy(float),
                    float(platt_b_scale[repetition]), float(platt_b_shift[repetition]),
                )
                part_b["probability_crossfit_platt"] = apply_platt(
                    part_b.probability_identity.to_numpy(float),
                    float(platt_a["parameter_a"]), float(platt_a["shift"]),
                )
                part_a["probability_crossfit_beta"] = apply_beta(
                    part_a.probability_identity.to_numpy(float),
                    float(beta_b_a[repetition]), float(beta_b_b[repetition]),
                    float(beta_b_shift[repetition]),
                )
                part_b["probability_crossfit_beta"] = apply_beta(
                    part_b.probability_identity.to_numpy(float),
                    float(beta_a["parameter_a"]), float(beta_a["parameter_b"]),
                    float(beta_a["shift"]),
                )
                certificate_frame = pd.concat([part_a, part_b], ignore_index=True)
                platt_certificate = stratified_brier_certificate(
                    certificate_frame, "probability_crossfit_platt",
                    args.bootstrap_repetitions,
                    frame_seed(certificate_frame, args.seed + dataset_index * 8009 + cluster_budget * 1009),
                    True,
                )
                beta_certificate = stratified_brier_certificate(
                    certificate_frame, "probability_crossfit_beta",
                    args.bootstrap_repetitions,
                    frame_seed(certificate_frame, args.seed + dataset_index * 9001 + cluster_budget * 1013),
                    True,
                )
                final_platt = apply_platt(
                    identity_heldout, float(platt_final["parameter_a"]),
                    float(platt_final["shift"])
                )
                final_beta = apply_beta(
                    identity_heldout, float(beta_final["parameter_a"]),
                    float(beta_final["parameter_b"]), float(beta_final["shift"]),
                )
                candidates = {
                    "crossfit_certified_platt": (final_platt, platt_certificate),
                    "crossfit_certified_beta": (final_beta, beta_certificate),
                }
                for method, (candidate, certificate) in candidates.items():
                    released = bool(certificate["certified"])
                    probability = candidate if released else identity_heldout
                    metrics, cluster_gains = heldout_metrics(heldout, probability, released)
                    rows.append({
                        "dataset": dataset, "budget_fraction": fraction,
                        "audit_clusters": cluster_budget,
                        "audit_configurations": cluster_budget * config.configurations_per_cluster,
                        "audit_repetition": repetition, "method": method, **metrics,
                    })
                    gain_matrices.setdefault((dataset, fraction, method), []).append(cluster_gains)
                    certificates.append({
                        "dataset": dataset, "budget_fraction": fraction,
                        "audit_repetition": repetition, "method": method,
                        "fit_fold_a_configurations": len(part_a),
                        "fit_fold_b_configurations": len(part_b),
                        "certificate_configurations": len(certificate_frame),
                        **certificate,
                    })
    progress.close()
    replicates = pd.DataFrame(rows)
    summary = summarize(
        replicates, gain_matrices, heldout_strata,
        args.bootstrap_repetitions, args.seed + 11003,
    )
    legacy = pd.read_csv(LEGACY_REPLICATES)
    legacy = legacy.loc[
        legacy.method.eq("dcs_selective")
        & legacy.audit_repetition.lt(args.audit_repetitions)
    ]
    paired_rows = []
    for method in replicates.method.drop_duplicates():
        comparison = legacy.merge(
            replicates.loc[replicates.method.eq(method)],
            on=["dataset", "budget_fraction", "audit_repetition"],
            suffixes=("_dcs", "_comparator"), validate="one_to_one",
        )
        for keys, group in comparison.groupby(["dataset", "budget_fraction"], sort=False):
            difference = group.brier_gain_dcs - group.brier_gain_comparator
            paired_rows.append({
                "dataset": keys[0], "budget_fraction": keys[1],
                "comparator_method": method,
                "mean_gain_difference_dcs_minus_comparator": float(difference.mean()),
                "dcs_higher_gain_frequency": float(np.mean(difference > 0)),
                "dcs_material_negative_count": int(group.material_negative_transfer_dcs.sum()),
                "comparator_material_negative_count": int(group.material_negative_transfer_comparator.sum()),
                "descriptive_only": True,
            })
    args.output_root.mkdir(parents=True, exist_ok=False)
    replicates.to_csv(args.output_root / "decision_curve_replicates.csv", index=False)
    summary.to_csv(args.output_root / "decision_curve_summary.csv", index=False)
    pd.DataFrame(certificates).to_csv(args.output_root / "certification_diagnostics.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(args.output_root / "paired_vs_dcs.csv", index=False)
    manifest = {
        "status": "crossfit_certified_baselines_complete",
        "date": "2026-09-08",
        "frozen_v11_unchanged": True,
        "frozen_v11_sha256": EXPECTED_FREEZE_SHA256,
        "methods": ["crossfit_certified_platt", "crossfit_certified_beta"],
        "audit_repetitions": args.audit_repetitions,
        "calibration_epochs": args.calibration_epochs,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "gpu_name": torch.cuda.get_device_name(0),
        "parameter_input_sha256": sha256(PARAMETERS),
        "legacy_replicates_sha256": sha256(LEGACY_REPLICATES),
        "claim_boundary": "Retrospective same-audit cross-fit certification on frozen v11 development domains and reused held-out halves.",
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote cross-fit certified baselines to {args.output_root}")


if __name__ == "__main__":
    main()
