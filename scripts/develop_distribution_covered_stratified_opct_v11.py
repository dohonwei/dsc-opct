from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_order_preserving_transport_v6 import RULE  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from develop_witness_gated_covariance_opct_v9 import (  # noqa: E402
    MAX_NORMALIZED_DISAGREEMENT,
    MIN_DIRECTION_AGREEMENT,
    MIN_DISPLACEMENT_COSINE,
    load_development,
    prepare as prepare_v9,
)
from identity_shortcut.risk_controlled_transport import RiskControlledConfig  # noqa: E402


PROTOCOL = Path("docs/distribution_covered_stratified_opct_v11_development_protocol.md")
AVDOS_RESERVATION = Path("docs/avdos_v7_external_confirmation_reservation.json")
V9_GATE = Path(
    "outputs/witness_gated_covariance_opct_v9_development/development_acceptance_gate.json"
)
V10_GATE = Path(
    "outputs/budget_saturating_witness_opct_v10_development/development_acceptance_gate.json"
)
CLUSTER_COLUMNS = ("task", "representation", "model", "split_seed")
STRATUM_COLUMNS = ("task", "representation", "model")
GEOMETRY_COLUMNS = (
    "identity_mean",
    "identity_sd",
    "action_mean",
    "action_sd",
    "displacement_mean",
    "displacement_sd",
    "decision_flip_fraction",
)
SD_WEIGHT = 0.05
RULE_NAME = "dcs_opct_v11_distribution_covered_stratified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Develop distribution-covered stratified OPCT v11."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/distribution_covered_stratified_opct_v11_development"),
    )
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


def cluster_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["displacement"] = work.probability_wg_opct - work.probability_identity
    work["decision_flip"] = (
        (work.probability_identity >= 0.5) != (work.probability_wg_opct >= 0.5)
    ).astype(float)
    rows = []
    for key, group in work.groupby(list(CLUSTER_COLUMNS), sort=True, dropna=False):
        rows.append(
            {
                **dict(zip(CLUSTER_COLUMNS, key, strict=True)),
                "identity_mean": float(group.probability_identity.mean()),
                "identity_sd": float(group.probability_identity.std(ddof=0)),
                "action_mean": float(group.probability_wg_opct.mean()),
                "action_sd": float(group.probability_wg_opct.std(ddof=0)),
                "displacement_mean": float(group.displacement.mean()),
                "displacement_sd": float(group.displacement.std(ddof=0)),
                "decision_flip_fraction": float(group.decision_flip.mean()),
            }
        )
    geometry = pd.DataFrame(rows)
    for column in GEOMETRY_COLUMNS:
        values = geometry[column].to_numpy(float)
        scale = float(np.std(values))
        geometry[f"z_{column}"] = (values - float(np.mean(values))) / max(scale, 1e-12)
    return geometry


def select_distribution_covered_audit(geometry: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [f"z_{column}" for column in GEOMETRY_COLUMNS]
    level_indices = {
        column: {
            value: index
            for index, value in enumerate(
                sorted(geometry[column].drop_duplicates().tolist(), key=str)
            )
        }
        for column in STRATUM_COLUMNS
    }
    assignments = []
    for stratum, group in geometry.groupby(
        list(STRATUM_COLUMNS), sort=True, dropna=False
    ):
        if len(group) != 5:
            raise ValueError(
                f"Expected five split-seed clusters per stratum, observed {len(group)}"
            )
        parity = sum(
            level_indices[column][value]
            for column, value in zip(STRATUM_COLUMNS, stratum, strict=True)
        ) % 2
        audit_count = 3 if parity == 0 else 2
        target_mean = group[feature_columns].to_numpy(float).mean(axis=0)
        target_sd = group[feature_columns].to_numpy(float).std(axis=0)
        candidates = []
        for indices in itertools.combinations(group.index.tolist(), audit_count):
            selected = geometry.loc[list(indices), feature_columns].to_numpy(float)
            score = float(
                np.sum((selected.mean(axis=0) - target_mean) ** 2)
                + SD_WEIGHT * np.sum((selected.std(axis=0) - target_sd) ** 2)
            )
            split_seed_tie_break = tuple(
                str(geometry.loc[index, "split_seed"]) for index in indices
            )
            candidates.append((score, split_seed_tie_break, indices))
        score, _, selected_indices = min(candidates, key=lambda item: (item[0], item[1]))
        selected_set = set(selected_indices)
        for index in group.index:
            assignments.append(
                {
                    **{
                        column: geometry.loc[index, column]
                        for column in CLUSTER_COLUMNS
                    },
                    "partition": "audit" if index in selected_set else "heldout",
                    "audit_count_in_stratum": audit_count,
                    "stratum_coreset_score": score,
                }
            )
    assignment = pd.DataFrame(assignments)
    counts = assignment.partition.value_counts().to_dict()
    expected_half = len(assignment) // 2
    if len(assignment) % 2 or counts != {
        "audit": expected_half,
        "heldout": expected_half,
    }:
        raise ValueError("Distribution-covered assignment must split clusters in half")
    return assignment


def partition_frame(
    frame: pd.DataFrame, assignment: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_keys = assignment.loc[assignment.partition.eq("audit"), list(CLUSTER_COLUMNS)]
    audit = frame.merge(audit_keys, on=list(CLUSTER_COLUMNS), how="inner")
    heldout = frame.merge(
        audit_keys, on=list(CLUSTER_COLUMNS), how="left", indicator=True
    )
    heldout = heldout.loc[heldout._merge.eq("left_only")].drop(columns="_merge")
    return audit.reset_index(drop=True), heldout.reset_index(drop=True)


def stratified_certificate(
    audit: pd.DataFrame,
    config: RiskControlledConfig,
    seed: int,
    applicable: bool,
    rank: float,
    inversions: int,
) -> dict[str, object]:
    if not applicable:
        return {
            "brier_gain": np.nan,
            "brier_gain_lcb": np.nan,
            "auc_delta": 0.0,
            "auc_delta_lcb": 0.0,
            "brier_certified": False,
            "auc_noninferior": False,
            "certified": False,
            "failure_reason": "unlabeled_action_inapplicable",
        }
    work = audit.copy()
    labels = work.material_optimism_event.to_numpy(int)
    identity = work.probability_identity.to_numpy(float)
    action = work.probability_wg_opct.to_numpy(float)
    work["paired_brier_gain"] = (labels - identity) ** 2 - (labels - action) ** 2
    cluster_gain = (
        work.groupby(list(CLUSTER_COLUMNS), sort=False, dropna=False)
        .paired_brier_gain.mean()
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    draws = np.zeros(config.bootstrap_repetitions, dtype=float)
    total_clusters = 0
    for _, stratum in cluster_gain.groupby(
        list(STRATUM_COLUMNS), sort=False, dropna=False
    ):
        gains = stratum.paired_brier_gain.to_numpy(float)
        sampled = rng.integers(
            0, len(gains), size=(config.bootstrap_repetitions, len(gains))
        )
        draws += gains[sampled].sum(axis=1)
        total_clusters += len(gains)
    draws /= total_clusters
    tail_alpha = config.familywise_alpha / 2.0
    point_gain = float(cluster_gain.paired_brier_gain.mean())
    gain_lcb = float(np.quantile(draws, tail_alpha))
    brier_pass = gain_lcb > config.minimum_brier_gain
    auc_pass = bool(
        rank >= RULE.minimum_rank
        and inversions == 0
        and 0.0 >= config.auc_noninferiority_margin
    )
    return {
        "brier_gain": point_gain,
        "brier_gain_lcb": gain_lcb,
        "auc_delta": 0.0,
        "auc_delta_lcb": 0.0,
        "tail_alpha": tail_alpha,
        "bootstrap_repetitions": config.bootstrap_repetitions,
        "brier_certified": bool(brier_pass),
        "auc_noninferior": bool(auc_pass),
        "certified": bool(brier_pass and auc_pass),
        "failure_reason": (
            "certified"
            if brier_pass and auc_pass
            else (
                "brier_lower_bound_not_positive"
                if not brier_pass
                else "analytic_auc_invariant_failed"
            )
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DCS-OPCT v11 development")
    if args.smoke:
        args.bootstrap_repetitions = min(args.bootstrap_repetitions, 200)
    if json.loads(V9_GATE.read_text(encoding="utf-8"))["all_passed"]:
        raise RuntimeError("v11 requires the preserved v9 development failure")
    if json.loads(V10_GATE.read_text(encoding="utf-8"))["all_passed"]:
        raise RuntimeError("v11 requires the preserved v10 development failure")
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
    prepared, components, locks = prepare_v9(load_development(), config, args.seed)
    lock_index = locks.set_index("dataset")
    geometry_frames = []
    assignment_frames = []
    certificate_rows = []
    heldout_rows = []
    for dataset, frame in prepared.items():
        geometry = cluster_geometry(frame)
        geometry.insert(0, "dataset", dataset)
        assignment = select_distribution_covered_audit(geometry.drop(columns="dataset"))
        assignment.insert(0, "dataset", dataset)
        audit, heldout = partition_frame(frame, assignment.drop(columns="dataset"))
        lock = lock_index.loc[dataset]
        certificate = stratified_certificate(
            audit,
            config,
            args.seed + RULE.configuration_budget * 1009,
            bool(lock.applicable),
            float(lock.probability_rank),
            int(lock.order_inversions),
        )
        selected = "wg_opct" if bool(certificate["certified"]) else "identity"
        certificate_rows.append(
            {
                "dataset": dataset,
                "rule": RULE_NAME,
                "selected_method": selected,
                "selection_reason": (
                    "selected_distribution_covered_stratified_certificate"
                    if selected == "wg_opct"
                    else f"abstain:{certificate['failure_reason']}"
                ),
                **certificate,
            }
        )
        heldout_rows.append(
            {
                "dataset": dataset,
                "rule": RULE_NAME,
                "n_audit_configurations": len(audit),
                "n_heldout_configurations": len(heldout),
                **evaluate_action(heldout, selected),
            }
        )
        geometry_frames.append(geometry)
        assignment_frames.append(assignment)

    certificates = pd.DataFrame(certificate_rows)
    heldout_results = pd.DataFrame(heldout_rows)
    geometry_table = pd.concat(geometry_frames, ignore_index=True)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    args.output_root.mkdir(parents=True, exist_ok=False)
    components.to_csv(args.output_root / "component_projection_diagnostics.csv", index=False)
    locks.to_csv(args.output_root / "unlabeled_action_locks.csv", index=False)
    geometry_table.to_csv(args.output_root / "unlabeled_cluster_geometry.csv", index=False)
    assignments.to_csv(args.output_root / "distribution_covered_assignments.csv", index=False)
    certificates.to_csv(args.output_root / "primary_certificates.csv", index=False)
    heldout_results.to_csv(args.output_root / "primary_heldout_results.csv", index=False)

    certificate_index = certificates.set_index("dataset")
    heldout_index = heldout_results.set_index("dataset")
    focus = ["EPPVR", "CASE", "CEAP"]
    gate = {
        "eppvr_case_ceap_certified": bool(
            certificate_index.loc[focus, "certified"].eq(True).all()
        ),
        "eppvr_case_ceap_heldout_gain_gt_0001": bool(
            heldout_index.loc[focus, "brier_gain"].gt(0.001).all()
        ),
        "zero_released_material_negative_transfer": bool(
            heldout_results.loc[heldout_results.adapted, "material_negative_transfer"]
            .eq(False)
            .all()
        ),
        "all_released_auc_noninferior": bool(
            heldout_results.loc[heldout_results.adapted, "auc_noninferior"]
            .eq(True)
            .all()
        ),
        "all_action_order_inversions_zero": bool(locks.order_inversions.eq(0).all()),
        "seediv_and_dreamer_abstain": bool(
            heldout_index.loc[["SEED-IV", "DREAMER"], "adapted"].eq(False).all()
        ),
    }
    gate["all_passed"] = bool(all(gate.values()))
    (args.output_root / "development_acceptance_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "dcs_opct_v11_development_complete",
        "date": "2026-09-08",
        "claim_status": "development_only_avdos_remains_untouched",
        "probability_action": "wg_opct_v9_unchanged",
        "geometry_columns": list(GEOMETRY_COLUMNS),
        "stratum_columns": list(STRATUM_COLUMNS),
        "sd_weight": SD_WEIGHT,
        "witness_thresholds": {
            "minimum_directional_agreement": MIN_DIRECTION_AGREEMENT,
            "minimum_displacement_cosine": MIN_DISPLACEMENT_COSINE,
            "maximum_normalized_disagreement": MAX_NORMALIZED_DISAGREEMENT,
        },
        "opct_rule": asdict(RULE),
        "risk_control_config": config.to_dict(),
        "gpu_name": torch.cuda.get_device_name(0),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {
            "protocol": sha256(PROTOCOL),
            "avdos_reservation": sha256(AVDOS_RESERVATION),
            "v9_failed_gate": sha256(V9_GATE),
            "v10_failed_gate": sha256(V10_GATE),
        },
    }
    (args.output_root / "development_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(certificates.to_string(index=False))
    print(heldout_results.to_string(index=False))
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
