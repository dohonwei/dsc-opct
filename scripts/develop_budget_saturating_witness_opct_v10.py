from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_complementary_geometry_opct_v8 import prepare as prepare_v8_components  # noqa: E402
from develop_consensus_order_preserving_transport_v7 import (  # noqa: E402
    assignment_rows,
    consensus_diagnostics,
    load_development,
)
from develop_order_preserving_transport_v6 import RULE  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action, summarize  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import (  # noqa: E402
    MAX_NORMALIZED_DISAGREEMENT,
    MIN_DIRECTION_AGREEMENT,
    MIN_DISPLACEMENT_COSINE,
    witness_statistics,
)
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    RiskControlledConfig,
    balanced_cluster_order,
    certify_checkpoint,
    rows_for_clusters,
)


PROTOCOL = Path("docs/budget_saturating_witness_opct_v10_development_protocol.md")
AVDOS_RESERVATION = Path("docs/avdos_v7_external_confirmation_reservation.json")
V9_GATE = Path(
    "outputs/witness_gated_covariance_opct_v9_development/development_acceptance_gate.json"
)
PRIMARY_COMPONENT = "coral"
WITNESS_COMPONENT = "quantile_mapping"
STRENGTH_GRID = (1.00, 1.25, 1.50, 1.75, 2.00)
RULE_NAME = "bsw_opct_v10_budget_saturating_witness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop budget-saturating witness OPCT v10."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/budget_saturating_witness_opct_v10_development"),
    )
    parser.add_argument("--audit-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--seed-step", type=int, default=7919)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strength_path(
    identity: np.ndarray,
    fitted_scale: float,
    fitted_shift: float,
) -> tuple[np.ndarray, float | None, pd.DataFrame]:
    identity = np.clip(np.asarray(identity, dtype=float), 1e-6, 1.0 - 1e-6)
    identity_logit = logit(identity)
    rows = []
    selected_probability = identity.copy()
    selected_strength: float | None = None
    for strength in STRENGTH_GRID:
        scale = 1.0 + strength * (fitted_scale - 1.0)
        shift = strength * fitted_shift
        if scale <= 0.0:
            rows.append(
                {
                    "strength": strength,
                    "scale": scale,
                    "shift": shift,
                    "admissible": False,
                    "failure_reason": "nonpositive_slope",
                }
            )
            continue
        probability = expit(scale * identity_logit + shift)
        diagnostics = consensus_diagnostics(identity, probability)
        admissible = bool(diagnostics["applicable"])
        rows.append(
            {
                "strength": strength,
                "scale": scale,
                "shift": shift,
                "admissible": admissible,
                "failure_reason": "" if admissible else "opct_invariant_failed",
                **{key: value for key, value in diagnostics.items() if key != "applicable"},
            }
        )
        if admissible:
            selected_probability = probability
            selected_strength = strength
    return selected_probability, selected_strength, pd.DataFrame(rows)


def prepare(
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: RiskControlledConfig,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    component_frames, component_diagnostics, _ = prepare_v8_components(
        datasets, config, seed
    )
    diagnostic_index = component_diagnostics.set_index(["dataset", "method"])
    prepared: dict[str, pd.DataFrame] = {}
    lock_rows = []
    path_rows = []
    for dataset, frame in component_frames.items():
        identity = frame.probability_identity.to_numpy(float)
        both_components_pass = all(
            bool(diagnostic_index.loc[(dataset, method), "applicable"])
            for method in (PRIMARY_COMPONENT, WITNESS_COMPONENT)
        )
        if both_components_pass:
            primary = frame[f"probability_component_{PRIMARY_COMPONENT}"].to_numpy(float)
            witness_probability = frame[
                f"probability_component_{WITNESS_COMPONENT}"
            ].to_numpy(float)
            witness = witness_statistics(identity, primary, witness_probability)
        else:
            witness = {
                "directional_agreement": np.nan,
                "displacement_cosine": np.nan,
                "normalized_disagreement": np.nan,
                "witness_pass": False,
            }

        selected_strength = None
        action = identity.copy()
        if both_components_pass and bool(witness["witness_pass"]):
            primary_diagnostics = diagnostic_index.loc[(dataset, PRIMARY_COMPONENT)]
            action, selected_strength, path = strength_path(
                identity,
                float(primary_diagnostics["scale"]),
                float(primary_diagnostics["shift"]),
            )
            path.insert(0, "dataset", dataset)
            path_rows.extend(path.to_dict("records"))
        else:
            for strength in STRENGTH_GRID:
                path_rows.append(
                    {
                        "dataset": dataset,
                        "strength": strength,
                        "admissible": False,
                        "failure_reason": "component_or_witness_ineligible",
                    }
                )
        action_diagnostics = consensus_diagnostics(identity, action)
        applicable = bool(
            both_components_pass
            and witness["witness_pass"]
            and selected_strength is not None
            and action_diagnostics["applicable"]
        )
        if not applicable:
            action = identity.copy()
            action_diagnostics = consensus_diagnostics(identity, action)
        frame["probability_bsw_opct"] = action
        prepared[dataset] = frame
        lock_rows.append(
            {
                "dataset": dataset,
                "method": "bsw_opct",
                "primary_component": PRIMARY_COMPONENT,
                "witness_component": WITNESS_COMPONENT,
                "both_components_pass": both_components_pass,
                **witness,
                "selected_strength": selected_strength,
                "applicable": applicable,
                "selection_reason": (
                    "largest_admissible_strength_locked"
                    if applicable
                    else "abstain_component_witness_or_strength_ineligible"
                ),
                **{
                    key: value
                    for key, value in action_diagnostics.items()
                    if key != "applicable"
                },
            }
        )
    return (
        prepared,
        component_diagnostics,
        pd.DataFrame(path_rows),
        pd.DataFrame(lock_rows),
    )


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BSW-OPCT v10 development")
    if args.smoke:
        args.audit_repetitions = min(args.audit_repetitions, 5)
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)
    if json.loads(V9_GATE.read_text(encoding="utf-8"))["all_passed"]:
        raise RuntimeError("v10 is justified only by the preserved v9 failure")
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
    prepared, components, paths, locks = prepare(load_development(), config, args.seed)
    lock_index = locks.set_index("dataset")
    result_rows = []
    certificate_rows = []
    assignment_records = []
    progress = tqdm(
        total=len(prepared) * args.audit_repetitions,
        desc="v10 repeated fixed-horizon certification",
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
                    ["bsw_opct"],
                    config,
                    audit_seed + RULE.configuration_budget * 1009,
                    total_registered_candidates=1,
                ).iloc[0]
            selected = (
                "bsw_opct"
                if certificate is not None and bool(certificate.certified)
                else "identity"
            )
            certificate_rows.append(
                {
                    "dataset": dataset,
                    "audit_repetition": repetition,
                    "audit_seed": audit_seed,
                    "applicable": applicable,
                    "selected_strength": lock_index.loc[dataset, "selected_strength"],
                    "selected_method": selected,
                    "selection_reason": (
                        "selected_certified_bsw_opct"
                        if selected == "bsw_opct"
                        else (
                            "abstain_unlabeled_action_inapplicable"
                            if certificate is None
                            else f"abstain_confirmation_failed:{certificate.failure_reason}"
                        )
                    ),
                    "brier_gain": np.nan if certificate is None else float(certificate.brier_gain),
                    "brier_gain_lcb": (
                        np.nan if certificate is None else float(certificate.brier_gain_lcb)
                    ),
                    "auc_delta": np.nan if certificate is None else float(certificate.auc_delta),
                    "auc_delta_lcb": (
                        np.nan if certificate is None else float(certificate.auc_delta_lcb)
                    ),
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
    paths.to_csv(args.output_root / "unlabeled_strength_path.csv", index=False)
    locks.to_csv(args.output_root / "unlabeled_action_locks.csv", index=False)
    certificates.to_csv(args.output_root / "fixed_horizon_certificates.csv", index=False)
    assignments.to_csv(args.output_root / "audit_cluster_assignments.csv", index=False)
    results.to_csv(args.output_root / "repeated_audit_results.csv", index=False)
    summary.to_csv(args.output_root / "rule_summary.csv", index=False)

    focus = summary.set_index("dataset")
    released = results.loc[results.adapted]
    gate = {
        "all_returned_order_inversions_zero": bool(locks.order_inversions.eq(0).all()),
        "zero_released_material_negative_transfer": bool(
            released.material_negative_transfer.eq(False).all()
        ),
        "all_released_auc_noninferior": bool(released.auc_noninferior.eq(True).all()),
        "eppvr_case_ceap_release_rate_ge_080": bool(
            focus.loc[["EPPVR", "CASE", "CEAP"], "adaptation_coverage"].ge(0.80).all()
        ),
        "eppvr_case_ceap_mean_gain_gt_0005": bool(
            focus.loc[
                ["EPPVR", "CASE", "CEAP"], "mean_brier_gain_when_adapted"
            ].gt(0.005).all()
        ),
        "eppvr_case_ceap_min_gain_gt_0001": bool(
            focus.loc[
                ["EPPVR", "CASE", "CEAP"], "minimum_brier_gain_when_adapted"
            ].gt(0.001).all()
        ),
        "seediv_and_dreamer_witness_abstention": bool(
            not lock_index.loc["SEED-IV", "applicable"]
            and not lock_index.loc["DREAMER", "applicable"]
            and focus.loc["SEED-IV", "adaptation_coverage"] == 0
            and focus.loc["DREAMER", "adaptation_coverage"] == 0
        ),
    }
    gate["all_passed"] = bool(all(gate.values()))
    (args.output_root / "development_acceptance_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "bsw_opct_v10_retrospective_development_complete",
        "date": "2026-09-08",
        "claim_status": "development_only_avdos_remains_untouched",
        "primary_component": PRIMARY_COMPONENT,
        "witness_component": WITNESS_COMPONENT,
        "strength_grid": list(STRENGTH_GRID),
        "witness_thresholds": {
            "minimum_directional_agreement": MIN_DIRECTION_AGREEMENT,
            "minimum_displacement_cosine": MIN_DISPLACEMENT_COSINE,
            "maximum_normalized_disagreement": MAX_NORMALIZED_DISAGREEMENT,
        },
        "opct_component_rule": asdict(RULE),
        "risk_control_config": config.to_dict(),
        "audit_repetitions": args.audit_repetitions,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "seed_step": args.seed_step,
        "gpu_name": torch.cuda.get_device_name(0),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {
            "protocol": sha256(PROTOCOL),
            "avdos_reservation": sha256(AVDOS_RESERVATION),
            "v9_failed_gate": sha256(V9_GATE),
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
