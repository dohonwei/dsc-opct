from __future__ import annotations

import argparse
from dataclasses import asdict
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

from develop_consensus_order_preserving_transport_v7 import (  # noqa: E402
    assignment_rows,
    consensus_diagnostics,
    load_development,
)
from develop_order_preserving_transport_v6 import RULE, fit_opct  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action, summarize  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    balanced_cluster_order,
    certify_checkpoint,
    probability_methods,
    rows_for_clusters,
    validate_prediction_contract,
)


PROTOCOL = Path("docs/complementary_geometry_opct_v8_development_protocol.md")
AVDOS_RESERVATION = Path("docs/avdos_v7_external_confirmation_reservation.json")
V7_GATE = Path(
    "outputs/consensus_order_preserving_transport_v7_development/development_acceptance_gate.json"
)
COMPONENTS = ("coral", "quantile_mapping")
RULE_NAME = "cg_opct_v8_equal_geometry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Develop complementary-geometry OPCT v8.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/complementary_geometry_opct_v8_development"),
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


def prepare(
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: RiskControlledConfig,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    prepared = {}
    component_rows = []
    lock_rows = []
    progress = tqdm(
        total=len(datasets) * len(COMPONENTS) * RULE.epochs,
        desc="v8 CUDA complementary projections",
        unit="epoch",
        dynamic_ncols=True,
    )
    for dataset, (raw, _) in datasets.items():
        frame = raw.copy().reset_index(drop=True)
        validate_prediction_contract(frame, config)
        available = set(probability_methods(frame))
        identity = frame.probability_identity.to_numpy(float)
        projected = []
        for index, method in enumerate(COMPONENTS):
            if method not in available:
                progress.update(RULE.epochs)
                component_rows.append(
                    {"dataset": dataset, "method": method, "available": False, "applicable": False}
                )
                continue
            probability, diagnostics = fit_opct(
                identity,
                frame[f"probability_{method}"].to_numpy(float),
                RULE,
                seed + index * 1009,
                "cuda",
                progress,
            )
            frame[f"probability_component_{method}"] = probability
            component_rows.append(
                {"dataset": dataset, "method": method, "available": True, **diagnostics}
            )
            if bool(diagnostics["applicable"]):
                projected.append(probability)
        both_pass = len(projected) == len(COMPONENTS)
        consensus = np.mean(projected, axis=0) if both_pass else identity.copy()
        diagnostics = consensus_diagnostics(identity, consensus)
        applicable = bool(both_pass and diagnostics["applicable"])
        if not applicable:
            consensus = identity.copy()
            diagnostics = consensus_diagnostics(identity, consensus)
        frame["probability_cg_opct"] = consensus
        lock_rows.append(
            {
                "dataset": dataset,
                "method": "cg_opct",
                "required_components": "+".join(COMPONENTS),
                "n_eligible_components": len(projected),
                "both_components_pass": both_pass,
                "applicable": applicable,
                "selection_reason": (
                    "complementary_geometry_consensus_locked"
                    if applicable
                    else "abstain_complementary_projection_ineligible"
                ),
                **{key: value for key, value in diagnostics.items() if key != "applicable"},
            }
        )
        prepared[dataset] = frame
        progress.set_postfix(dataset=dataset, eligible=len(projected), refresh=False)
    progress.close()
    return prepared, pd.DataFrame(component_rows), pd.DataFrame(lock_rows)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CG-OPCT development")
    if json.loads(V7_GATE.read_text(encoding="utf-8"))["all_passed"]:
        raise RuntimeError("v8 is justified only by the preserved v7 development failure")
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
        desc="v8 repeated fixed-horizon certification",
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
            records = assignment_rows(dataset, repetition, audit_seed, order, config)
            for record in records:
                record["rule"] = RULE_NAME
            assignment_records.extend(records)
            certificate = None
            if applicable:
                certificate = certify_checkpoint(
                    audit,
                    ["cg_opct"],
                    config,
                    audit_seed + RULE.configuration_budget * 1009,
                    total_registered_candidates=1,
                ).iloc[0]
            selected = (
                "cg_opct"
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
                        "selected_certified_cg_opct"
                        if selected == "cg_opct"
                        else (
                            "abstain_unlabeled_geometry_inapplicable"
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
    locks.to_csv(args.output_root / "unlabeled_geometry_locks.csv", index=False)
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
        "seediv_complementary_geometry_abstention": bool(
            not lock_index.loc["SEED-IV", "applicable"]
            and focus.loc["SEED-IV", "adaptation_coverage"] == 0
        ),
    }
    gate["all_passed"] = bool(all(gate.values()))
    (args.output_root / "development_acceptance_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "cg_opct_v8_retrospective_development_complete",
        "date": "2026-09-08",
        "claim_status": "development_only_avdos_remains_untouched",
        "components": list(COMPONENTS),
        "weights": [0.5, 0.5],
        "opct_component_rule": asdict(RULE),
        "risk_control_config": config.to_dict(),
        "audit_repetitions": args.audit_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "gpu_name": torch.cuda.get_device_name(0),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {
            "protocol": sha256(PROTOCOL),
            "avdos_reservation": sha256(AVDOS_RESERVATION),
            "v7_failed_gate": sha256(V7_GATE),
        },
    }
    (args.output_root / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(locks.to_string(index=False))
    print(summary.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
