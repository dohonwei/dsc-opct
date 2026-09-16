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

from develop_minimal_intervention_transport_v5 import (  # noqa: E402
    RULE as V5_RULE,
    rank_minimal_interventions,
)
from develop_order_preserving_transport_v6 import (  # noqa: E402
    RULE,
    cluster_assignment_rows,
    fit_opct,
    load_all_datasets,
)
from develop_risk_controlled_transport_v5 import evaluate_action, summarize  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    balanced_cluster_order,
    certify_checkpoint,
    probability_methods,
    rows_for_clusters,
    validate_prediction_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate OPCT v6 development artifacts.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/order_preserving_transport_v6_development"),
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
                left.to_numpy(float), right.to_numpy(float), atol=tolerance, rtol=0, equal_nan=True
            ):
                return False
        else:
            if not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
                return False
    return True


def main() -> None:
    args = parse_args()
    report_path = args.output_root / "independent_validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")
    manifest = json.loads((args.output_root / "development_manifest.json").read_text("utf-8"))
    stored_actions = pd.read_csv(args.output_root / "unlabeled_opct_locks.csv")
    stored_certificates = pd.read_csv(args.output_root / "fixed_horizon_certificates.csv")
    stored_assignments = pd.read_csv(args.output_root / "audit_cluster_assignments.csv")
    stored_results = pd.read_csv(args.output_root / "repeated_audit_results.csv")
    stored_summary = pd.read_csv(args.output_root / "rule_summary.csv")

    checks = []

    def record(name: str, passed: bool, observed: object, required: object) -> None:
        checks.append(
            {"check": name, "passed": bool(passed), "observed": observed, "required": required}
        )

    develop_script = ROOT / "scripts/develop_order_preserving_transport_v6.py"
    record(
        "formal development script hash is unchanged",
        sha256(develop_script) == manifest["script_sha256"],
        sha256(develop_script),
        manifest["script_sha256"],
    )
    input_hashes_ok = all(
        sha256(
            {
                "protocol": ROOT / "docs/order_preserving_transport_v6_development_protocol.md",
                "public_results": ROOT / "outputs/transport_policy_public_development/synthetic_candidate_results.csv",
                "policy": ROOT / "outputs/transport_policy_public_development/frozen_transport_policy.joblib",
                "case_gate": ROOT / "outputs/case_v5_external_confirmation/external_confirmation_gate.json",
                "case_audit": ROOT / "outputs/case_v5_external_confirmation/primary_audit_labeled.csv",
                "case_heldout": ROOT / "outputs/case_v5_external_confirmation/primary_heldout_results.csv",
            }[name]
        )
        == expected
        for name, expected in manifest["input_sha256"].items()
    )
    record("all registered inputs retain exact hashes", input_hashes_ok, input_hashes_ok, True)
    record(
        "formal run used CUDA",
        bool(manifest["cuda_verified"]) and torch.cuda.is_available(),
        manifest["gpu_name"],
        "CUDA available and GPU name recorded",
    )

    config = RiskControlledConfig(**manifest["risk_control_config"])
    datasets = load_all_datasets()
    action_rows = []
    prepared = {}
    for dataset, (raw_frame, signatures) in datasets.items():
        frame = raw_frame.copy()
        validate_prediction_contract(frame, config)
        _, base_method, action_reason = rank_minimal_interventions(
            signatures, probability_methods(frame), config, V5_RULE
        )
        if base_method == "identity":
            projected = frame.probability_identity.to_numpy(float).copy()
            diagnostics = {
                "scale": 1.0,
                "shift": 0.0,
                "final_loss": 0.0,
                "probability_mean_shift": 0.0,
                "probability_rank": 1.0,
                "decision_flip_rate": 0.0,
                "order_inversions": 0,
                "applicable": False,
            }
            projection_reason = "abstain_v5_unlabeled_actionability_gate_failed"
        else:
            projected, diagnostics = fit_opct(
                frame.probability_identity.to_numpy(float),
                frame[f"probability_{base_method}"].to_numpy(float),
                RULE,
                int(manifest["seed"]),
                "cuda",
            )
            projection_reason = (
                "opct_locked_before_audit_outcomes"
                if bool(diagnostics["applicable"])
                else "abstain_opct_unlabeled_invariant_failed"
            )
        frame["probability_opct"] = projected
        prepared[dataset] = (frame, base_method, diagnostics)
        action_rows.append(
            {
                "dataset": dataset,
                "rule": RULE.name,
                "base_method": base_method,
                "base_action_reason": action_reason,
                "projection_reason": projection_reason,
                **diagnostics,
            }
        )
    rebuilt_actions = pd.DataFrame(action_rows)
    record(
        "unlabeled OPCT locks and CUDA parameters reconstruct",
        frames_match(rebuilt_actions, stored_actions, 1e-7),
        rebuilt_actions.to_dict("records"),
        "exact categorical fields and numeric tolerance 1e-7",
    )
    invariant_ok = bool(
        rebuilt_actions.probability_rank.ge(RULE.minimum_rank).all()
        and rebuilt_actions.order_inversions.eq(0).all()
    )
    record(
        "strict order-preservation invariant holds",
        invariant_ok,
        {
            "minimum_rank": float(rebuilt_actions.probability_rank.min()),
            "total_inversions": int(rebuilt_actions.order_inversions.sum()),
        },
        {"minimum_rank": RULE.minimum_rank, "total_inversions": 0},
    )

    certificate_rows = []
    result_rows = []
    assignment_rows = []
    progress = tqdm(
        total=len(prepared) * int(manifest["audit_repetitions"]),
        desc="independent v6 certificate reconstruction",
        unit="audit",
        dynamic_ncols=True,
    )
    for dataset, (frame, base_method, diagnostics) in prepared.items():
        for repetition in range(int(manifest["audit_repetitions"])):
            seed = int(manifest["seed"]) + repetition * int(manifest["seed_step"])
            order = balanced_cluster_order(frame, config, seed)
            audit_clusters = order[: config.cluster_budgets[0]]
            audit_mask = rows_for_clusters(frame, config.cluster_columns, audit_clusters)
            audit = frame.loc[audit_mask].reset_index(drop=True)
            heldout = frame.loc[~audit_mask].reset_index(drop=True)
            assignment_rows.extend(
                cluster_assignment_rows(dataset, repetition, seed, order, config)
            )
            if base_method == "identity" or not bool(diagnostics["applicable"]):
                certificate = None
                selected = "identity"
                reason = "abstain_no_applicable_opct_candidate"
            else:
                certificate = certify_checkpoint(
                    audit,
                    ["opct"],
                    config,
                    seed + RULE.configuration_budget * 1009,
                    total_registered_candidates=1,
                ).iloc[0]
                selected = "opct" if bool(certificate.certified) else "identity"
                reason = (
                    "selected_simultaneously_certified_opct"
                    if selected == "opct"
                    else f"abstain_confirmation_failed:{certificate.failure_reason}"
                )
            certificate_rows.append(
                {
                    "dataset": dataset,
                    "rule": RULE.name,
                    "audit_repetition": repetition,
                    "audit_seed": seed,
                    "base_method": base_method,
                    "selected_method": selected,
                    "selection_reason": reason,
                    "n_audit_configurations": len(audit),
                    "n_heldout_configurations": len(heldout),
                    "brier_gain": None if certificate is None else float(certificate.brier_gain),
                    "brier_gain_lcb": None if certificate is None else float(certificate.brier_gain_lcb),
                    "brier_certified": None if certificate is None else bool(certificate.brier_certified),
                    "auc_delta": None if certificate is None else float(certificate.auc_delta),
                    "auc_delta_lcb": None if certificate is None else float(certificate.auc_delta_lcb),
                    "auc_noninferior": None if certificate is None else bool(certificate.auc_noninferior),
                    "certified": None if certificate is None else bool(certificate.certified),
                    "failure_reason": None if certificate is None else str(certificate.failure_reason),
                }
            )
            result_rows.append(
                {
                    "dataset": dataset,
                    "rule": RULE.name,
                    "audit_repetition": repetition,
                    "audit_seed": seed,
                    "base_method": base_method,
                    "actual_configuration_budget": len(audit),
                    "n_heldout_configurations": len(heldout),
                    **evaluate_action(heldout, selected),
                }
            )
            progress.update(1)
            progress.set_postfix(dataset=dataset, refresh=False)
    progress.close()

    rebuilt_certificates = pd.DataFrame(certificate_rows)
    rebuilt_assignments = pd.DataFrame(assignment_rows)
    rebuilt_results = pd.DataFrame(result_rows)
    rebuilt_summary = summarize(rebuilt_results)
    record(
        "all audit assignments reconstruct",
        frames_match(rebuilt_assignments, stored_assignments),
        len(rebuilt_assignments),
        len(stored_assignments),
    )
    record(
        "all fixed-horizon certificates reconstruct",
        frames_match(rebuilt_certificates, stored_certificates),
        len(rebuilt_certificates),
        len(stored_certificates),
    )
    record(
        "all held-out decisions and metrics reconstruct",
        frames_match(rebuilt_results, stored_results),
        len(rebuilt_results),
        len(stored_results),
    )
    record(
        "dataset summaries reconstruct",
        frames_match(rebuilt_summary, stored_summary),
        rebuilt_summary.to_dict("records"),
        stored_summary.to_dict("records"),
    )
    released = rebuilt_results.loc[rebuilt_results.adapted]
    safety_ok = bool(
        released.material_negative_transfer.eq(False).all()
        and released.auc_noninferior.eq(True).all()
    )
    record(
        "every released action satisfies realized safety endpoints",
        safety_ok,
        {
            "released": int(len(released)),
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
        "claim_boundary": "Retrospective development only; untouched external confirmation remains required.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
