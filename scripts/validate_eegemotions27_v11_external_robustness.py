from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMPLEMENTATION_LOCK_SHA256 = (
    "e5f68887071352eead192c3352969c92c5ef7f155227e5f70f89d4cdb326699a"
)
EXPECTED_FREEZE_SHA256 = (
    "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
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
KEYS = ("dataset", "task", "axis", "representation", "model", "split_seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently validate the frozen EEGEmotions-27 v11 result."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/eegemotions27_v11_external_robustness"),
    )
    parser.add_argument(
        "--implementation-lock",
        type=Path,
        default=Path("docs/eegemotions27_v11_external_robustness_implementation_lock.json"),
    )
    parser.add_argument(
        "--reservation",
        type=Path,
        default=Path("docs/eegemotions27_v11_external_robustness_reservation.json"),
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    required: Any,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "required": required,
        }
    )


def frame_equivalent(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    keys: list[str],
) -> bool:
    if set(left.columns) != set(right.columns):
        return False
    columns = sorted(left.columns)
    left = left.sort_values(keys).reset_index(drop=True)[columns]
    right = right.sort_values(keys).reset_index(drop=True)[columns]
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError:
        return False
    return True


def reconstruct_geometry(mechanism: pd.DataFrame) -> pd.DataFrame:
    work = mechanism.copy()
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
        scale = max(float(np.std(values)), 1e-12)
        geometry[f"z_{column}"] = (values - float(np.mean(values))) / scale
    return geometry


def reconstruct_assignment(geometry: pd.DataFrame, sd_weight: float) -> pd.DataFrame:
    feature_columns = [f"z_{column}" for column in GEOMETRY_COLUMNS]
    level_indices = {
        column: {
            level: index
            for index, level in enumerate(
                sorted(geometry[column].drop_duplicates().tolist(), key=str)
            )
        }
        for column in STRATUM_COLUMNS
    }
    rows = []
    for stratum, group in geometry.groupby(
        list(STRATUM_COLUMNS), sort=True, dropna=False
    ):
        if len(group) != 5:
            raise ValueError(f"Expected five seeds per stratum, observed {len(group)}")
        parity = sum(
            level_indices[column][level]
            for column, level in zip(STRATUM_COLUMNS, stratum, strict=True)
        ) % 2
        audit_count = 3 if parity == 0 else 2
        target = group[feature_columns].to_numpy(float)
        candidates = []
        for indices in itertools.combinations(group.index.tolist(), audit_count):
            selected = geometry.loc[list(indices), feature_columns].to_numpy(float)
            score = float(
                np.sum((selected.mean(axis=0) - target.mean(axis=0)) ** 2)
                + sd_weight
                * np.sum((selected.std(axis=0) - target.std(axis=0)) ** 2)
            )
            tie_break = tuple(str(geometry.loc[index, "split_seed"]) for index in indices)
            candidates.append((score, tie_break, indices))
        score, _, selected_indices = min(candidates, key=lambda item: (item[0], item[1]))
        selected_set = set(selected_indices)
        for index in group.index:
            rows.append(
                {
                    **{column: geometry.loc[index, column] for column in CLUSTER_COLUMNS},
                    "partition": "audit" if index in selected_set else "heldout",
                    "audit_count_in_stratum": audit_count,
                    "stratum_coreset_score": score,
                }
            )
    return pd.DataFrame(rows)


def safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(probability, method="average")
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def action_metrics(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    labels = frame.material_optimism_event.to_numpy(int)
    identity = frame.probability_identity.to_numpy(float)
    selected = frame[f"probability_{method}"].to_numpy(float)
    gain = float(np.mean((labels - identity) ** 2 - (labels - selected) ** 2))
    auc_delta = float(safe_auc(labels, selected) - safe_auc(labels, identity))
    return {
        "selected_method": method,
        "adapted": method != "identity",
        "brier_gain": gain,
        "auc_delta": auc_delta,
        "negative_transfer": bool(method != "identity" and gain < 0.0),
        "material_negative_transfer": bool(method != "identity" and gain < -0.001),
        "auc_noninferior": bool(np.isfinite(auc_delta) and auc_delta >= -0.02),
    }


def main() -> None:
    args = parse_args()
    report_path = args.output_root / "independent_validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")

    implementation = json.loads(args.implementation_lock.read_text(encoding="utf-8"))
    reservation = json.loads(args.reservation.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    pre_root = args.output_root / "preoutcome"
    action_root = args.output_root / "action"
    model_root = args.output_root / "model_outcomes"
    outcome_root = args.output_root / "outcome"
    pre_lock_path = pre_root / "preoutcome_design_lock.json"
    action_lock_path = action_root / "primary_action_and_audit_lock.json"
    pre_lock = json.loads(pre_lock_path.read_text(encoding="utf-8"))
    action_lock = json.loads(action_lock_path.read_text(encoding="utf-8"))
    model_manifest = json.loads(
        (model_root / "model_outcome_manifest.json").read_text(encoding="utf-8")
    )
    certificate = json.loads(
        (outcome_root / "primary_candidate_certificate.json").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (outcome_root / "external_confirmation_gate.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (outcome_root / "external_confirmation_manifest.json").read_text(encoding="utf-8")
    )

    summary = pd.read_csv(model_root / "summary.csv")
    split_audit = pd.read_csv(model_root / "split_audit.csv")
    components = pd.read_csv(action_root / "component_projection_diagnostics.csv")
    mechanism = pd.read_csv(action_root / "mechanism_probabilities_blinded.csv")
    stored_geometry = pd.read_csv(action_root / "unlabeled_cluster_geometry.csv")
    assignments = pd.read_csv(action_root / "distribution_covered_assignments.csv")
    audit = pd.read_csv(outcome_root / "primary_audit_labeled.csv")
    heldout = pd.read_csv(outcome_root / "primary_heldout_results.csv")

    checks: list[dict[str, Any]] = []
    steps = tqdm(total=12, desc="EEGEmotions-27 independent validation", unit="check")

    implementation_hash = sha256(args.implementation_lock)
    freeze_hash = sha256(args.freeze)
    locked_inventory_ok = all(
        (ROOT / relative).is_file() and sha256(ROOT / relative) == expected
        for inventory_name in (
            "frozen_artifact_sha256",
            "analysis_code_sha256",
            "dependency_code_sha256",
        )
        for relative, expected in implementation[inventory_name].items()
    )
    record(
        checks,
        "implementation lock, freeze, reservation, and code inventory retain exact hashes",
        implementation_hash == EXPECTED_IMPLEMENTATION_LOCK_SHA256
        and freeze_hash == EXPECTED_FREEZE_SHA256
        and implementation["freeze_sha256"] == freeze_hash
        and reservation["dcs_opct_v11_contract"]["freeze_sha256"] == freeze_hash
        and reservation["source"]["locked_commit"]
        == implementation["registered_execution"]["locked_commit"]
        and locked_inventory_ok,
        {
            "implementation_lock_sha256": implementation_hash,
            "freeze_sha256": freeze_hash,
            "locked_inventory_ok": locked_inventory_ok,
        },
        "exact presignal lock/freeze hashes and unchanged registered artifacts",
    )
    steps.update()

    pre_filenames = {
        "mechanism_summary": "mechanism_summary.csv",
        "identity_encoding": "identity_encoding_margin.csv",
        "split_audit": "split_audit.csv",
        "prepared_payload": "prepared_dose_design.joblib",
        "adapter_audits": "adapter_output_audits.json",
    }
    pre_hashes_ok = all(
        sha256(pre_root / pre_filenames[name]) == expected
        for name, expected in pre_lock["artifact_sha256"].items()
    )
    record(
        checks,
        "pre-outcome design lock and all bound artifacts retain exact hashes",
        sha256(pre_lock_path) == manifest["preoutcome_lock_sha256"]
        and sha256(pre_lock_path) == action_lock["provenance"]["preoutcome_lock_sha256"]
        and pre_hashes_ok,
        {"lock_sha256": sha256(pre_lock_path), "artifacts_ok": pre_hashes_ok},
        manifest["preoutcome_lock_sha256"],
    )
    steps.update()

    action_filenames = {
        "components": "component_projection_diagnostics.csv",
        "geometry": "unlabeled_cluster_geometry.csv",
        "assignment": "distribution_covered_assignments.csv",
        "mechanism": "mechanism_probabilities_blinded.csv",
    }
    action_hashes_ok = all(
        sha256(action_root / action_filenames[name]) == expected
        for name, expected in action_lock["artifact_sha256"].items()
    )
    record(
        checks,
        "unlabeled action lock and all blinded artifacts retain exact hashes",
        sha256(action_lock_path) == manifest["action_lock_sha256"]
        and sha256(action_lock_path) == model_manifest["action_lock_sha256"]
        and action_hashes_ok,
        {"lock_sha256": sha256(action_lock_path), "artifacts_ok": action_hashes_ok},
        manifest["action_lock_sha256"],
    )
    steps.update()

    model_filenames = {
        "predictions": "predictions.csv",
        "split_audit": "split_audit.csv",
        "summary": "summary.csv",
        "identity_encoding": "identity_encoding_margin.csv",
    }
    model_hashes_ok = all(
        sha256(model_root / model_filenames[name]) == expected
        for name, expected in model_manifest["artifact_sha256"].items()
    )
    record(
        checks,
        "post-lock model artifacts and outcome manifest hash chain are intact",
        model_hashes_ok
        and sha256(model_root / "summary.csv") == manifest["full_summary_sha256"]
        and sha256(outcome_root / "external_confirmation_gate.json")
        == manifest["gate_sha256"]
        and model_manifest["preoutcome_lock_sha256"] == manifest["preoutcome_lock_sha256"],
        {
            "model_artifacts_ok": model_hashes_ok,
            "summary_sha256": sha256(model_root / "summary.csv"),
            "gate_sha256": sha256(outcome_root / "external_confirmation_gate.json"),
        },
        "exact model manifest and final manifest hashes",
    )
    steps.update()

    config = pre_lock["config"]
    grid_columns = list(KEYS) + ["nominal_dose"]
    expected_summary_rows = (
        len(config["tasks"])
        * len(config["representations"])
        * len(config["models"])
        * len(config["seeds"])
        * len(config["doses"])
    )
    levels_ok = (
        set(summary.dataset) == {config["dataset"]}
        and set(summary.task) == set(config["tasks"])
        and set(summary.axis) == {config["axis"]}
        and set(summary.representation) == set(config["representations"])
        and set(summary.model) == set(config["models"])
        and set(summary.split_seed) == set(config["seeds"])
        and set(summary.nominal_dose) == set(config["doses"])
    )
    record(
        checks,
        "registered model summary is the exact 150-row configuration-dose grid",
        len(summary) == expected_summary_rows
        and not summary.duplicated(grid_columns).any()
        and levels_ok,
        {"rows": len(summary), "unique_rows": summary[grid_columns].drop_duplicates().shape[0]},
        {"rows": expected_summary_rows, "complete_levels": True},
    )
    steps.update()

    split_keys = ["dataset", "task", "axis", "split_seed", "fold", "nominal_dose"]
    record(
        checks,
        "split audit contains all 5 seeds by 20 crossed folds by 5 doses",
        len(split_audit) == 500
        and not split_audit.duplicated(split_keys).any()
        and set(split_audit.split_seed) == set(config["seeds"])
        and set(split_audit.nominal_dose) == set(config["doses"])
        and split_audit.fold.nunique() == 20
        and split_audit.row_overlap_with_test.eq(0).all(),
        {
            "rows": len(split_audit),
            "folds": int(split_audit.fold.nunique()),
            "maximum_row_overlap": int(split_audit.row_overlap_with_test.max()),
        },
        {"rows": 500, "folds": 20, "maximum_row_overlap": 0},
    )
    steps.update()

    action = freeze["probability_action"]
    component_rebuilt = []
    for row in components.itertuples(index=False):
        applicable = bool(
            abs(float(row.probability_mean_shift))
            <= float(action["maximum_probability_mean_shift"])
            and float(row.probability_rank) >= float(action["minimum_probability_rank"])
            and int(row.order_inversions) <= int(action["maximum_order_inversions"])
        )
        component_rebuilt.append(
            {
                "method": row.method,
                "stored": bool(row.applicable),
                "reconstructed": applicable,
            }
        )
    components_applicable = all(item["reconstructed"] for item in component_rebuilt)
    witness_not_evaluated = all(
        np.isnan(float(action_lock["witness"][key]))
        for key in (
            "directional_agreement",
            "displacement_cosine",
            "normalized_disagreement",
        )
    )
    reconstructed_candidate_applicable = bool(
        components_applicable and action_lock["witness"]["witness_pass"]
    )
    record(
        checks,
        "component applicability and pre-outcome identity fallback reconstruct exactly",
        all(item["stored"] == item["reconstructed"] for item in component_rebuilt)
        and not components_applicable
        and witness_not_evaluated
        and reconstructed_candidate_applicable == action_lock["candidate_applicable"]
        and action_lock["candidate_method"] == "identity"
        and np.allclose(
            mechanism.probability_wg_opct,
            mechanism.probability_identity,
            rtol=0.0,
            atol=0.0,
        ),
        {
            "components": component_rebuilt,
            "witness_not_evaluated": witness_not_evaluated,
            "candidate_method": action_lock["candidate_method"],
        },
        "both components inapplicable; witness skipped; exact identity fallback",
    )
    steps.update()

    rebuilt_geometry = reconstruct_geometry(mechanism)
    rebuilt_assignment = reconstruct_assignment(
        rebuilt_geometry, float(freeze["distribution_covered_audit"]["sd_weight"])
    )
    assignment_counts = assignments.partition.value_counts().to_dict()
    assignment_ok = frame_equivalent(
        assignments,
        rebuilt_assignment,
        keys=list(CLUSTER_COLUMNS),
    )
    record(
        checks,
        "distribution-covered geometry and 15/15 cluster assignment reconstruct exactly",
        frame_equivalent(stored_geometry, rebuilt_geometry, keys=list(CLUSTER_COLUMNS))
        and assignment_ok
        and len(assignments) == 30
        and assignment_counts == {"audit": 15, "heldout": 15},
        {
            "clusters": len(assignments),
            "counts": assignment_counts,
            "assignment_exact": assignment_ok,
        },
        {
            "clusters": 30,
            "counts": {"audit": 15, "heldout": 15},
            "assignment_exact": True,
        },
    )
    steps.update()

    outcome_columns = list(KEYS) + ["nominal_dose", "exposure_effect"]
    outcomes = summary[outcome_columns].copy()
    anchors = outcomes.loc[
        outcomes.nominal_dose.eq(0.0), list(KEYS) + ["exposure_effect"]
    ].rename(columns={"exposure_effect": "dose_zero_exposure_effect"})
    outcomes = outcomes.merge(anchors, on=list(KEYS), validate="many_to_one")
    outcomes = outcomes.loc[outcomes.nominal_dose.gt(0.0)].copy()
    outcomes["dose_induced_amplification"] = (
        outcomes.exposure_effect - outcomes.dose_zero_exposure_effect
    )
    outcomes["material_optimism_event"] = (
        outcomes.dose_induced_amplification >= float(config["material_threshold"])
    ).astype(int)
    evaluation = mechanism.merge(
        outcomes,
        on=list(KEYS) + ["nominal_dose"],
        validate="one_to_one",
    )
    partition_keys = assignments.loc[
        assignments.partition.eq("audit"), list(CLUSTER_COLUMNS)
    ]
    rebuilt_audit = evaluation.merge(partition_keys, on=list(CLUSTER_COLUMNS), how="inner")
    rebuilt_heldout = evaluation.merge(
        partition_keys, on=list(CLUSTER_COLUMNS), how="left", indicator=True
    )
    rebuilt_heldout = rebuilt_heldout.loc[
        rebuilt_heldout._merge.eq("left_only")
    ].drop(columns="_merge")
    outcome_keys = list(KEYS) + ["nominal_dose"]
    record(
        checks,
        "locked outcomes reconstruct into exact 60/60 audit and held-out rows",
        len(evaluation) == 120
        and len(audit) == 60
        and len(heldout) == 60
        and frame_equivalent(audit, rebuilt_audit, keys=outcome_keys)
        and frame_equivalent(heldout, rebuilt_heldout, keys=outcome_keys),
        {
            "evaluation_rows": len(evaluation),
            "audit_rows": len(audit),
            "heldout_rows": len(heldout),
        },
        {"evaluation_rows": 120, "audit_rows": 60, "heldout_rows": 60},
    )
    steps.update()

    expected_certificate = {
        "brier_gain": np.nan,
        "brier_gain_lcb": np.nan,
        "auc_delta": 0.0,
        "auc_delta_lcb": 0.0,
        "brier_certified": False,
        "auc_noninferior": False,
        "certified": False,
        "failure_reason": "unlabeled_action_inapplicable",
    }
    certificate_ok = all(
        (
            np.isnan(float(certificate[key]))
            if isinstance(expected, float) and np.isnan(expected)
            else certificate[key] == expected
        )
        for key, expected in expected_certificate.items()
    )
    record(
        checks,
        "certificate failure is the registered unlabeled-action-inapplicable boundary",
        certificate_ok,
        certificate,
        expected_certificate,
    )
    steps.update()

    rebuilt_metrics = action_metrics(heldout, "identity")
    metrics_ok = all(
        (
            rebuilt_metrics[key] == gate["heldout_metrics"][key]
            if isinstance(rebuilt_metrics[key], (bool, str))
            else abs(
                float(rebuilt_metrics[key]) - float(gate["heldout_metrics"][key])
            )
            < 1e-12
        )
        for key in rebuilt_metrics
    )
    record(
        checks,
        "held-out identity non-intervention metrics reconstruct exactly",
        metrics_ok,
        rebuilt_metrics,
        gate["heldout_metrics"],
    )
    steps.update()

    endpoint_estimable = (
        sorted(audit.material_optimism_event.unique().tolist()) == [0, 1]
        and sorted(heldout.material_optimism_event.unique().tolist()) == [0, 1]
    )
    certificate_pass = bool(certificate["certified"])
    effectiveness_pass = bool(
        gate["selected_method"] == "wg_opct" and rebuilt_metrics["brier_gain"] > 0.001
    )
    released_nonharm = (
        None
        if gate["selected_method"] == "identity"
        else bool(
            rebuilt_metrics["brier_gain"] >= 0.0
            and not rebuilt_metrics["negative_transfer"]
            and rebuilt_metrics["auc_noninferior"]
        )
    )
    claim_supported = bool(effectiveness_pass and released_nonharm is True)
    gate_ok = (
        endpoint_estimable == gate["endpoint_estimable"]
        and certificate_pass == gate["certificate_pass"]
        and effectiveness_pass == gate["effectiveness_pass"]
        and released_nonharm == gate["released_action_nonharm_pass"]
        and claim_supported == gate["claim_supported"]
        and claim_supported == manifest["claim_supported"]
        and gate["candidate_method"] == "identity"
        and gate["selected_method"] == "identity"
        and gate["safety_success"] is False
    )
    record(
        checks,
        "final gate follows the frozen conjunction and preserves the identity boundary",
        gate_ok,
        {
            "endpoint_estimable": endpoint_estimable,
            "certificate_pass": certificate_pass,
            "effectiveness_pass": effectiveness_pass,
            "released_action_nonharm_pass": released_nonharm,
            "claim_supported": claim_supported,
        },
        "exact match to gate and manifest; identity is not a safety success",
    )
    steps.update()
    steps.close()

    passed = all(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed else "failed",
        "date": "2026-09-09",
        "dataset": "EEGEmotions-27",
        "validation_role": "independent artifact and endpoint reconstruction",
        "n_checks": len(checks),
        "n_passed": sum(check["passed"] for check in checks),
        "checks": checks,
        "claim_supported": claim_supported if passed else False,
        "interpretation": (
            "The frozen result is internally consistent: both unlabeled projections exceeded the maximum probability-mean-shift boundary, so v11 abstained to identity before target labels entered the action decision. This is conservative non-release, not evidence of effectiveness or safety success."
            if passed
            else "At least one frozen artifact or endpoint reconstruction check failed."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
