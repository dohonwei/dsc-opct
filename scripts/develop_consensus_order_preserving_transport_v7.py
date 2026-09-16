from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_order_preserving_transport_v6 import (  # noqa: E402
    RULE,
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


PROTOCOL = Path("docs/consensus_order_preserving_transport_v7_development_protocol.md")
AVDOS_RESERVATION = Path("docs/avdos_v7_external_confirmation_reservation.json")
COMPONENTS = ("mean_shift", "coral", "quantile_mapping", "support_clipping", "scmt")
QUORUM = 3
RULE_NAME = "copct_v7_consensus_q3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Develop C-OPCT v7 consensus transport.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/consensus_order_preserving_transport_v7_development"),
    )
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--seed-step", type=int, default=7919)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_development() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    datasets = load_all_datasets()
    ceap_root = Path("outputs/ceap_v6_external_confirmation")
    validation = json.loads(
        (ceap_root / "independent_validation_report.json").read_text(encoding="utf-8")
    )
    if validation["status"] != "passed" or validation["external_gate"]["claim_supported"]:
        raise RuntimeError("CEAP v6 failure must be independently locked before v7 development")
    ceap = pd.concat(
        [
            pd.read_csv(ceap_root / "primary_audit_labeled.csv"),
            pd.read_csv(ceap_root / "primary_heldout_results.csv"),
        ],
        ignore_index=True,
    )
    datasets["CEAP"] = (ceap, pd.read_csv(ceap_root / "candidate_signatures.csv"))
    return datasets


def consensus_diagnostics(identity: np.ndarray, probability: np.ndarray) -> dict[str, object]:
    order = np.argsort(identity, kind="stable")
    rank = float(spearmanr(identity, probability).statistic)
    inversions = int(np.sum(np.diff(probability[order]) < -1e-12))
    mean_shift = float(np.mean(probability - identity))
    return {
        "probability_mean_shift": mean_shift,
        "probability_rank": rank,
        "decision_flip_rate": float(
            np.mean((identity >= 0.5) != (probability >= 0.5))
        ),
        "order_inversions": inversions,
        "applicable": bool(
            abs(mean_shift) <= RULE.maximum_probability_mean_shift
            and rank >= RULE.minimum_rank
            and inversions == 0
        ),
    }


def assignment_rows(
    dataset: str,
    repetition: int,
    seed: int,
    order: list[tuple[object, ...]],
    config: RiskControlledConfig,
) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset,
            "rule": RULE_NAME,
            "audit_repetition": repetition,
            "audit_seed": seed,
            "cluster_order_position": position,
            "partition": "audit" if position < config.cluster_budgets[0] else "heldout",
            **dict(zip(config.cluster_columns, cluster, strict=True)),
        }
        for position, cluster in enumerate(order)
    ]


def prepare(
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: RiskControlledConfig,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    component_rows = []
    lock_rows = []
    progress = tqdm(
        total=len(datasets) * len(COMPONENTS) * RULE.epochs,
        desc="v7 CUDA component projections",
        unit="epoch",
        dynamic_ncols=True,
    )
    for dataset, (raw, _) in datasets.items():
        frame = raw.copy().reset_index(drop=True)
        validate_prediction_contract(frame, config)
        available = set(probability_methods(frame))
        identity = frame.probability_identity.to_numpy(float)
        eligible = []
        for index, method in enumerate(COMPONENTS):
            if method not in available:
                progress.update(RULE.epochs)
                component_rows.append(
                    {"dataset": dataset, "method": method, "available": False, "applicable": False}
                )
                continue
            projected, diagnostics = fit_opct(
                identity,
                frame[f"probability_{method}"].to_numpy(float),
                RULE,
                seed + index * 1009,
                "cuda",
                progress,
            )
            frame[f"probability_component_{method}"] = projected
            component_rows.append(
                {"dataset": dataset, "method": method, "available": True, **diagnostics}
            )
            if bool(diagnostics["applicable"]):
                eligible.append(projected)
        quorum_pass = len(eligible) >= QUORUM
        consensus = np.median(np.stack(eligible), axis=0) if quorum_pass else identity.copy()
        diagnostics = consensus_diagnostics(identity, consensus)
        applicable = bool(quorum_pass and diagnostics["applicable"])
        if not applicable:
            consensus = identity.copy()
            diagnostics = consensus_diagnostics(identity, consensus)
        frame["probability_copct"] = consensus
        lock_rows.append(
            {
                "dataset": dataset,
                "method": "copct",
                "n_registered_components": len(COMPONENTS),
                "n_eligible_components": len(eligible),
                "quorum": QUORUM,
                "quorum_pass": quorum_pass,
                "applicable": applicable,
                "selection_reason": (
                    "unlabeled_consensus_locked"
                    if applicable
                    else "abstain_component_quorum_or_invariant_failed"
                ),
                **{key: value for key, value in diagnostics.items() if key != "applicable"},
            }
        )
        prepared[dataset] = frame
        progress.set_postfix(dataset=dataset, eligible=len(eligible), refresh=False)
    progress.close()
    return prepared, pd.DataFrame(component_rows), pd.DataFrame(lock_rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for registered C-OPCT development")
    reservation = json.loads(AVDOS_RESERVATION.read_text(encoding="utf-8"))
    if reservation["status"] != "reserved_before_participant_value_access":
        raise RuntimeError("AVDOS reservation is invalid")
    config = RiskControlledConfig(
        configuration_budgets=(RULE.configuration_budget,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        auc_noninferiority_margin=RULE.auc_noninferiority_margin,
        minimum_brier_gain=RULE.minimum_certified_gain,
        minimum_valid_auc_bootstraps=max(20, args.bootstrap_repetitions // 10),
    )
    prepared, components, locks = prepare(load_development(), config, args.seed)
    lock_index = locks.set_index("dataset")
    result_rows = []
    certificate_rows = []
    assignment_records = []
    progress = tqdm(
        total=len(prepared) * args.audit_repetitions,
        desc="v7 repeated fixed-horizon certification",
        unit="audit",
        dynamic_ncols=True,
    )
    for dataset, frame in prepared.items():
        applicable = bool(lock_index.loc[dataset, "applicable"])
        for repetition in range(args.audit_repetitions):
            audit_seed = args.seed + repetition * args.seed_step
            order = balanced_cluster_order(frame, config, audit_seed)
            audit_clusters = order[: config.cluster_budgets[0]]
            mask = rows_for_clusters(frame, config.cluster_columns, audit_clusters)
            audit = frame.loc[mask].reset_index(drop=True)
            heldout = frame.loc[~mask].reset_index(drop=True)
            assignment_records.extend(
                assignment_rows(dataset, repetition, audit_seed, order, config)
            )
            certificate = None
            if applicable:
                certificate = certify_checkpoint(
                    audit,
                    ["copct"],
                    config,
                    audit_seed + RULE.configuration_budget * 1009,
                    total_registered_candidates=1,
                ).iloc[0]
            selected = (
                "copct"
                if certificate is not None and bool(certificate.certified)
                else "identity"
            )
            certificate_rows.append(
                {
                    "dataset": dataset,
                    "audit_repetition": repetition,
                    "audit_seed": audit_seed,
                    "applicable": applicable,
                    "selected_method": selected,
                    "selection_reason": (
                        "selected_certified_copct"
                        if selected == "copct"
                        else (
                            "abstain_unlabeled_consensus_not_applicable"
                            if certificate is None
                            else f"abstain_confirmation_failed:{certificate.failure_reason}"
                        )
                    ),
                    "brier_gain": np.nan if certificate is None else float(certificate.brier_gain),
                    "brier_gain_lcb": np.nan if certificate is None else float(certificate.brier_gain_lcb),
                    "auc_delta": np.nan if certificate is None else float(certificate.auc_delta),
                    "auc_delta_lcb": np.nan if certificate is None else float(certificate.auc_delta_lcb),
                    "certified": False if certificate is None else bool(certificate.certified),
                }
            )
            result_rows.append(
                {
                    "dataset": dataset,
                    "rule": RULE_NAME,
                    "audit_repetition": repetition,
                    "audit_seed": audit_seed,
                    "actual_configuration_budget": len(audit),
                    "n_heldout_configurations": len(heldout),
                    **evaluate_action(heldout, selected),
                }
            )
            progress.update(1)
            progress.set_postfix(dataset=dataset, selected=selected, refresh=False)
    progress.close()

    results = pd.DataFrame(result_rows)
    certificates = pd.DataFrame(certificate_rows)
    assignments = pd.DataFrame(assignment_records)
    summary = summarize(results)
    args.output_root.mkdir(parents=True, exist_ok=False)
    components.to_csv(args.output_root / "component_projection_diagnostics.csv", index=False)
    locks.to_csv(args.output_root / "unlabeled_consensus_locks.csv", index=False)
    certificates.to_csv(args.output_root / "fixed_horizon_certificates.csv", index=False)
    assignments.to_csv(args.output_root / "audit_cluster_assignments.csv", index=False)
    results.to_csv(args.output_root / "repeated_audit_results.csv", index=False)
    summary.to_csv(args.output_root / "rule_summary.csv", index=False)

    focus = summary.set_index("dataset")
    released = results.loc[results.adapted]
    gate = {
        "all_consensus_order_inversions_zero": bool(locks.order_inversions.eq(0).all()),
        "zero_released_material_negative_transfer": bool(
            released.material_negative_transfer.eq(False).all()
        ),
        "all_released_auc_noninferior": bool(released.auc_noninferior.eq(True).all()),
        "eppvr_case_ceap_release_rate_ge_080": bool(
            focus.loc[["EPPVR", "CASE", "CEAP"], "adaptation_coverage"].ge(0.80).all()
        ),
        "eppvr_case_ceap_mean_gain_gt_0005": bool(
            focus.loc[["EPPVR", "CASE", "CEAP"], "mean_brier_gain_when_adapted"].gt(0.005).all()
        ),
        "eppvr_case_ceap_min_gain_gt_0001": bool(
            focus.loc[["EPPVR", "CASE", "CEAP"], "minimum_brier_gain_when_adapted"].gt(0.001).all()
        ),
        "seediv_unlabeled_quorum_abstention": bool(
            not lock_index.loc["SEED-IV", "applicable"]
            and focus.loc["SEED-IV", "adaptation_coverage"] == 0
        ),
    }
    gate["all_passed"] = bool(all(gate.values()))
    (args.output_root / "development_acceptance_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    inputs = {
        "protocol": PROTOCOL,
        "avdos_reservation": AVDOS_RESERVATION,
        "ceap_validation": Path(
            "outputs/ceap_v6_external_confirmation/independent_validation_report.json"
        ),
    }
    manifest = {
        "status": "copct_v7_retrospective_development_complete",
        "date": "2026-09-08",
        "claim_status": "development_only_avdos_remains_untouched",
        "components": list(COMPONENTS),
        "quorum": QUORUM,
        "opct_component_rule": asdict(RULE),
        "risk_control_config": config.to_dict(),
        "audit_repetitions": args.audit_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "gpu_name": torch.cuda.get_device_name(0),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
    }
    (args.output_root / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(locks.to_string(index=False))
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
