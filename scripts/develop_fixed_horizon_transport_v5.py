from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_label_independent_transport_v5 import (  # noqa: E402
    LabelIndependentRule,
    choose_unlabeled_action,
)
from develop_risk_controlled_transport_v5 import evaluate_action, load_datasets, summarize  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    balanced_cluster_order,
    certify_checkpoint,
    probability_methods,
    rows_for_clusters,
)

RULE = LabelIndependentRule("fixed_horizon_64_m001", 0.001, 0.001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Develop the fixed-horizon v5 certificate.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/fixed_horizon_transport_v5_development"),
    )
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--seed-step", type=int, default=7919)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required. Use --allow-cpu only for debugging.")
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 5)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)

    public_path = Path("outputs/transport_policy_public_development/synthetic_candidate_results.csv")
    policy_path = Path("outputs/transport_policy_public_development/frozen_transport_policy.joblib")
    public = pd.read_csv(public_path)
    policy = joblib.load(policy_path)
    datasets = load_datasets(public, policy)
    config = RiskControlledConfig(
        configuration_budgets=(64,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )

    action_rows: list[dict[str, object]] = []
    certificate_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    progress = tqdm(
        total=len(datasets) * args.audit_repetitions,
        desc="v5 fixed-horizon certification",
        unit="audit",
        dynamic_ncols=True,
    )
    for dataset, (frame, signatures) in datasets.items():
        candidate, reason = choose_unlabeled_action(
            signatures, probability_methods(frame), config, RULE
        )
        action_rows.append(
            {
                "dataset": dataset,
                "rule": RULE.name,
                "candidate_method": candidate,
                "action_lock_reason": reason,
            }
        )
        for repetition in range(args.audit_repetitions):
            seed = args.seed + repetition * args.seed_step
            order = balanced_cluster_order(frame, config, seed)
            audit_clusters = order[: config.cluster_budgets[0]]
            audit = frame.loc[
                rows_for_clusters(frame, config.cluster_columns, audit_clusters)
            ].reset_index(drop=True)
            if candidate == "identity":
                selected = "identity"
                certificate = None
            else:
                certificate = certify_checkpoint(
                    audit, [candidate], config, seed + 64 * 1009, total_registered_candidates=1
                ).iloc[0]
                selected = candidate if bool(certificate.certified) else "identity"
            test = frame.loc[
                ~rows_for_clusters(frame, config.cluster_columns, audit_clusters)
            ].reset_index(drop=True)
            result_rows.append(
                {
                    "dataset": dataset,
                    "rule": RULE.name,
                    "audit_repetition": repetition,
                    "audit_seed": seed,
                    "candidate_method": candidate,
                    "actual_configuration_budget": 64,
                    "n_heldout_configurations": len(test),
                    **evaluate_action(test, selected),
                }
            )
            certificate_rows.append(
                {
                    "dataset": dataset,
                    "rule": RULE.name,
                    "audit_repetition": repetition,
                    "audit_seed": seed,
                    "candidate_method": candidate,
                    "selected_method": selected,
                    "selection_reason": (
                        "abstain_no_unlabeled_candidate"
                        if certificate is None
                        else (
                            "selected_fixed_horizon_certificate"
                            if selected != "identity"
                            else f"abstain_confirmation_failed:{certificate.failure_reason}"
                        )
                    ),
                    "brier_gain": None if certificate is None else float(certificate.brier_gain),
                    "brier_gain_lcb": None if certificate is None else float(certificate.brier_gain_lcb),
                    "auc_delta": None if certificate is None else float(certificate.auc_delta),
                    "auc_delta_lcb": None if certificate is None else float(certificate.auc_delta_lcb),
                    "valid_auc_bootstraps": None if certificate is None else int(certificate.valid_auc_bootstraps),
                }
            )
            progress.update(1)
            progress.set_postfix(dataset=dataset, method=selected, refresh=False)
    progress.close()

    args.output_root.mkdir(parents=True, exist_ok=False)
    results = pd.DataFrame(result_rows)
    summary = summarize(results)
    pd.DataFrame(action_rows).to_csv(args.output_root / "unlabeled_action_locks.csv", index=False)
    pd.DataFrame(certificate_rows).to_csv(args.output_root / "fixed_horizon_certificates.csv", index=False)
    results.to_csv(args.output_root / "repeated_audit_results.csv", index=False)
    summary.to_csv(args.output_root / "rule_summary.csv", index=False)
    manifest = {
        "status": "fixed_horizon_v5_retrospective_development_complete",
        "date": "2026-09-07",
        "claim_status": "development_only_external_confirmation_required",
        "rule": RULE.__dict__,
        "configuration_budget": 64,
        "family_size": 2,
        "target_label_role": "single_fixed_horizon_confirmation_only",
        "audit_repetitions": args.audit_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "cuda_verified": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "statistical_backend": "numpy_cpu_cluster_bootstrap",
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output_root / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(action_rows).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
