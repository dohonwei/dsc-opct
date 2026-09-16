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

from develop_order_preserving_transport_v6 import RULE, load_all_datasets  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    balanced_cluster_order,
    certify_checkpoint,
    rows_for_clusters,
)
from run_order_preserving_transport_v6_ablation import (  # noqa: E402
    DEVELOPMENT,
    PROTOCOL,
    VARIANTS,
    full_data_analyses,
    prepare_datasets,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OPCT v6 ablation artifacts.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/order_preserving_transport_v6_ablation"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frames_match(observed: pd.DataFrame, expected: pd.DataFrame, tolerance: float = 1e-10) -> bool:
    if list(observed.columns) != list(expected.columns) or len(observed) != len(expected):
        return False
    for column in observed.columns:
        left = observed[column]
        right = expected[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(
                left.to_numpy(float), right.to_numpy(float),
                atol=tolerance, rtol=0, equal_nan=True,
            ):
                return False
        elif not left.fillna("<NA>").astype(str).equals(
            right.fillna("<NA>").astype(str)
        ):
            return False
    return True


def main() -> None:
    args = parse_args()
    report_path = args.output_root / "independent_validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")
    manifest = json.loads((args.output_root / "ablation_manifest.json").read_text("utf-8"))
    stored_locks = pd.read_csv(args.output_root / "unlabeled_variant_locks.csv")
    stored_certificates = pd.read_csv(args.output_root / "simultaneous_certificates.csv")
    stored_results = pd.read_csv(args.output_root / "heldout_candidate_results.csv")
    stored_summary = pd.read_csv(args.output_root / "ablation_summary.csv")
    stored_paired = pd.read_csv(args.output_root / "paired_audit_contrasts.csv")
    stored_contrasts = pd.read_csv(args.output_root / "full_data_cluster_bootstrap_contrasts.csv")
    stored_heterogeneity = pd.read_csv(args.output_root / "opct_subgroup_heterogeneity.csv")
    checks = []

    def record(name: str, passed: bool, observed: object, required: object) -> None:
        checks.append(
            {"check": name, "passed": bool(passed), "observed": observed, "required": required}
        )

    analysis_script = ROOT / "scripts/run_order_preserving_transport_v6_ablation.py"
    record(
        "ablation script hash is unchanged",
        sha256(analysis_script) == manifest["script_sha256"],
        sha256(analysis_script), manifest["script_sha256"],
    )
    record(
        "ablation protocol hash is unchanged",
        sha256(ROOT / PROTOCOL) == manifest["protocol_sha256"],
        sha256(ROOT / PROTOCOL), manifest["protocol_sha256"],
    )
    record(
        "upstream manifests retain exact hashes",
        sha256(ROOT / DEVELOPMENT / "development_manifest.json")
        == manifest["development_manifest_sha256"]
        and sha256(ROOT / DEVELOPMENT / "independent_validation_report.json")
        == manifest["independent_validation_sha256"],
        True, True,
    )
    record(
        "formal ablation used CUDA",
        bool(manifest["cuda_verified"]) and torch.cuda.is_available(),
        manifest["gpu_name"], "CUDA available and GPU recorded",
    )

    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=int(manifest["bootstrap_repetitions"]),
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, int(manifest["bootstrap_repetitions"]) // 10),
    )
    prepared, rebuilt_locks = prepare_datasets(
        load_all_datasets(), config, int(manifest["seed"])
    )
    record(
        "all unlabeled variant locks reconstruct",
        frames_match(rebuilt_locks, stored_locks, 1e-7),
        rebuilt_locks.to_dict("records"),
        "exact categorical fields and numeric tolerance 1e-7",
    )
    invariant_rows = rebuilt_locks.loc[
        rebuilt_locks.method.isin(["intercept_only", "slope_only", "opct"])
    ]
    record(
        "every monotone variant has zero inversions",
        bool(invariant_rows.order_inversions.eq(0).all()),
        int(invariant_rows.order_inversions.sum()), 0,
    )

    lock_index = rebuilt_locks.set_index(["dataset", "method"])
    certificate_rows = []
    result_rows = []
    progress = tqdm(
        total=len(prepared) * int(manifest["audit_repetitions"]),
        desc="independent v6 ablation reconstruction",
        unit="audit",
        dynamic_ncols=True,
    )
    for dataset, frame in prepared.items():
        for repetition in range(int(manifest["audit_repetitions"])):
            audit_seed = int(manifest["seed"]) + repetition * int(manifest["seed_step"])
            order = balanced_cluster_order(frame, config, audit_seed)
            audit_clusters = order[: config.cluster_budgets[0]]
            audit_mask = rows_for_clusters(frame, config.cluster_columns, audit_clusters)
            audit = frame.loc[audit_mask].reset_index(drop=True)
            heldout = frame.loc[~audit_mask].reset_index(drop=True)
            candidates = [
                method for method in VARIANTS
                if bool(lock_index.loc[(dataset, method), "applicable"])
            ]
            certificates = (
                certify_checkpoint(
                    audit, candidates, config,
                    audit_seed + RULE.configuration_budget * 1009,
                    total_registered_candidates=len(VARIANTS),
                ).set_index("method")
                if candidates else pd.DataFrame()
            )
            for method in VARIANTS:
                applicable = bool(lock_index.loc[(dataset, method), "applicable"])
                has_certificate = bool(not certificates.empty and method in certificates.index)
                certificate = certificates.loc[method] if has_certificate else None
                certified = bool(applicable and has_certificate and certificate.certified)
                certificate_rows.append(
                    {
                        "dataset": dataset,
                        "audit_repetition": repetition,
                        "audit_seed": audit_seed,
                        "method": method,
                        "applicable": applicable,
                        "certified": certified,
                        "brier_gain": np.nan if certificate is None else float(certificate.brier_gain),
                        "brier_gain_lcb": np.nan if certificate is None else float(certificate.brier_gain_lcb),
                        "auc_delta": np.nan if certificate is None else float(certificate.auc_delta),
                        "auc_delta_lcb": np.nan if certificate is None else float(certificate.auc_delta_lcb),
                        "failure_reason": (
                            "not_applicable" if certificate is None
                            else str(certificate.failure_reason)
                        ),
                    }
                )
                metrics = evaluate_action(heldout, method)
                result_rows.append(
                    {
                        "dataset": dataset,
                        "audit_repetition": repetition,
                        "audit_seed": audit_seed,
                        "method": method,
                        "applicable": applicable,
                        "certified": certified,
                        "released": bool(applicable and certified),
                        **{
                            key: value for key, value in metrics.items()
                            if key not in {"selected_method", "adapted"}
                        },
                    }
                )
            progress.update(1)
            progress.set_postfix(dataset=dataset, refresh=False)
    progress.close()

    rebuilt_certificates = pd.DataFrame(certificate_rows)
    rebuilt_results = pd.DataFrame(result_rows)
    rebuilt_summary = summarize(rebuilt_results, rebuilt_locks)
    record(
        "all simultaneous certificates reconstruct",
        frames_match(rebuilt_certificates, stored_certificates),
        len(rebuilt_certificates), len(stored_certificates),
    )
    record(
        "all held-out candidate metrics reconstruct",
        frames_match(rebuilt_results, stored_results),
        len(rebuilt_results), len(stored_results),
    )
    record(
        "ablation summary reconstructs",
        frames_match(rebuilt_summary, stored_summary),
        rebuilt_summary.to_dict("records"), stored_summary.to_dict("records"),
    )

    contrasts, heterogeneity = full_data_analyses(
        prepared, int(manifest["bootstrap_repetitions"]), int(manifest["seed"]) + 424242
    )
    record(
        "cluster-bootstrap contrasts reconstruct",
        frames_match(contrasts, stored_contrasts),
        len(contrasts), len(stored_contrasts),
    )
    record(
        "subgroup heterogeneity table reconstructs",
        frames_match(heterogeneity, stored_heterogeneity),
        len(heterogeneity), len(stored_heterogeneity),
    )
    paired_rows = []
    for dataset, group in rebuilt_results.groupby("dataset", sort=False):
        pivot = group.pivot(index="audit_repetition", columns="method", values="brier_gain")
        for comparator in ("direct_base", "intercept_only", "slope_only"):
            difference = pivot.opct - pivot[comparator]
            paired_rows.append(
                {
                    "dataset": dataset,
                    "contrast": f"opct_minus_{comparator}",
                    "n_paired_audits": len(difference),
                    "mean_difference": float(difference.mean()),
                    "median_difference": float(difference.median()),
                    "opct_win_rate": float((difference > 0).mean()),
                    "tie_rate": float(np.isclose(difference, 0, atol=1e-12).mean()),
                }
            )
    rebuilt_paired = pd.DataFrame(paired_rows)
    record(
        "paired audit contrasts reconstruct",
        frames_match(rebuilt_paired, stored_paired),
        len(rebuilt_paired), len(stored_paired),
    )
    released = rebuilt_results.loc[rebuilt_results.released]
    record(
        "every released ablation action is safe",
        bool(
            released.material_negative_transfer.eq(False).all()
            and released.auc_noninferior.eq(True).all()
        ),
        {
            "released": len(released),
            "material_negative_transfer": int(released.material_negative_transfer.sum()),
            "auc_violations": int((~released.auc_noninferior).sum()),
        },
        {"material_negative_transfer": 0, "auc_violations": 0},
    )

    passed = all(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed else "failed",
        "date": "2026-09-07",
        "n_checks": len(checks),
        "n_passed": sum(check["passed"] for check in checks),
        "checks": checks,
        "prespecified_gate_status": json.loads(
            (args.output_root / "ablation_acceptance_gate.json").read_text("utf-8")
        ),
        "claim_boundary": "Retrospective ablation; CEAP reservation remains untouched.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
