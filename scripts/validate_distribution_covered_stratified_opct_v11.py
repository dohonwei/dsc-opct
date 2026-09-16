from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_consensus_order_preserving_transport_v7 import load_development  # noqa: E402
from develop_distribution_covered_stratified_opct_v11 import (  # noqa: E402
    CLUSTER_COLUMNS,
    GEOMETRY_COLUMNS,
    STRATUM_COLUMNS,
)
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402


OUTPUT = Path("outputs/distribution_covered_stratified_opct_v11_development")
REPORT = OUTPUT / "independent_validation_report.json"
AVDOS_PATH = Path(
    "E:/AA发表论文的数据/dataset/AVDOS-VR-main/notebooks/temp/2_affect/"
    "Dataset_AVDOS_ManualFeaturesWithAnnotations.csv"
)
EXPECTED_AVDOS_SHA256 = "e6639be584e5ac2c3ff0a786fd77f88ae164fee5369f82a89b0f90a7fe97650b"
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, passed: bool, detail: object) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def reconstruct_action(
    dataset: str,
    frame: pd.DataFrame,
    components: pd.DataFrame,
    locks: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    identity = np.clip(output.probability_identity.to_numpy(float), 1e-6, 1.0 - 1e-6)
    lock = locks.loc[locks.dataset.eq(dataset)].iloc[0]
    if bool(lock.applicable):
        coral = components.loc[
            components.dataset.eq(dataset) & components.method.eq("coral")
        ].iloc[0]
        action = expit(
            float(coral["scale"]) * logit(identity) + float(coral["shift"])
        )
    else:
        action = identity.copy()
    output["probability_wg_opct"] = action
    return output


def recompute_stratified_certificate(
    audit: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    work = audit.copy()
    labels = work.material_optimism_event.to_numpy(int)
    identity = work.probability_identity.to_numpy(float)
    action = work.probability_wg_opct.to_numpy(float)
    work["gain"] = (labels - identity) ** 2 - (labels - action) ** 2
    cluster_gain = (
        work.groupby(list(CLUSTER_COLUMNS), sort=False, dropna=False)
        .gain.mean()
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    draws = np.zeros(repetitions, dtype=float)
    total = 0
    for _, stratum in cluster_gain.groupby(
        list(STRATUM_COLUMNS), sort=False, dropna=False
    ):
        gains = stratum.gain.to_numpy(float)
        sampled = rng.integers(0, len(gains), size=(repetitions, len(gains)))
        draws += gains[sampled].sum(axis=1)
        total += len(gains)
    draws /= total
    return float(cluster_gain.gain.mean()), float(np.quantile(draws, 0.025))


def main() -> None:
    manifest = json.loads((OUTPUT / "development_manifest.json").read_text(encoding="utf-8"))
    gate = json.loads((OUTPUT / "development_acceptance_gate.json").read_text(encoding="utf-8"))
    components = pd.read_csv(OUTPUT / "component_projection_diagnostics.csv")
    locks = pd.read_csv(OUTPUT / "unlabeled_action_locks.csv")
    geometry = pd.read_csv(OUTPUT / "unlabeled_cluster_geometry.csv")
    assignments = pd.read_csv(OUTPUT / "distribution_covered_assignments.csv")
    certificates = pd.read_csv(OUTPUT / "primary_certificates.csv")
    heldout_results = pd.read_csv(OUTPUT / "primary_heldout_results.csv")
    datasets = load_development()
    checks = []

    input_paths = {
        "protocol": Path("docs/distribution_covered_stratified_opct_v11_development_protocol.md"),
        "avdos_reservation": Path("docs/avdos_v7_external_confirmation_reservation.json"),
        "v9_failed_gate": Path(
            "outputs/witness_gated_covariance_opct_v9_development/development_acceptance_gate.json"
        ),
        "v10_failed_gate": Path(
            "outputs/budget_saturating_witness_opct_v10_development/development_acceptance_gate.json"
        ),
    }
    observed_input_hashes = {name: sha256(path) for name, path in input_paths.items()}
    checks.append(
        check(
            "manifest_input_hashes",
            observed_input_hashes == manifest["input_sha256"],
            observed_input_hashes,
        )
    )
    script_hash = sha256(Path("scripts/develop_distribution_covered_stratified_opct_v11.py"))
    checks.append(
        check(
            "development_script_hash",
            script_hash == manifest["script_sha256"],
            script_hash,
        )
    )
    avdos_hash = sha256(AVDOS_PATH)
    checks.append(
        check(
            "avdos_still_reserved_hash_only",
            avdos_hash == EXPECTED_AVDOS_SHA256,
            avdos_hash,
        )
    )
    forbidden_geometry = {
        "material_optimism_event",
        "brier_gain",
        "auc_delta",
        "certified",
    }
    checks.append(
        check(
            "audit_design_tables_label_free",
            forbidden_geometry.isdisjoint(geometry.columns)
            and forbidden_geometry.isdisjoint(assignments.columns),
            {
                "geometry_columns": geometry.columns.tolist(),
                "assignment_columns": assignments.columns.tolist(),
            },
        )
    )

    per_dataset = {}
    all_numeric_match = True
    all_half_splits = True
    all_geometry_columns = True
    for dataset, (raw, _) in datasets.items():
        frame = reconstruct_action(dataset, raw, components, locks)
        assignment = assignments.loc[assignments.dataset.eq(dataset)].drop(columns="dataset")
        audit_keys = assignment.loc[assignment.partition.eq("audit"), list(CLUSTER_COLUMNS)]
        audit = frame.merge(audit_keys, on=list(CLUSTER_COLUMNS), how="inner")
        heldout = frame.merge(
            audit_keys, on=list(CLUSTER_COLUMNS), how="left", indicator=True
        )
        heldout = heldout.loc[heldout._merge.eq("left_only")].drop(columns="_merge")
        saved_certificate = certificates.loc[certificates.dataset.eq(dataset)].iloc[0]
        saved_heldout = heldout_results.loc[heldout_results.dataset.eq(dataset)].iloc[0]
        counts = assignment.partition.value_counts().to_dict()
        all_half_splits &= counts.get("audit") == counts.get("heldout")
        all_geometry_columns &= all(
            f"z_{column}" in geometry.columns for column in GEOMETRY_COLUMNS
        )

        if saved_certificate.selected_method == "wg_opct":
            point, lcb = recompute_stratified_certificate(
                audit,
                int(saved_certificate.bootstrap_repetitions),
                20260908 + 160 * 1009,
            )
            certificate_match = (
                abs(point - float(saved_certificate.brier_gain)) <= TOLERANCE
                and abs(lcb - float(saved_certificate.brier_gain_lcb)) <= TOLERANCE
            )
        else:
            point, lcb = np.nan, np.nan
            certificate_match = pd.isna(saved_certificate.brier_gain)
        heldout_point = evaluate_action(heldout, str(saved_heldout.selected_method))
        heldout_match = (
            abs(heldout_point["brier_gain"] - float(saved_heldout.brier_gain)) <= TOLERANCE
            and abs(heldout_point["auc_delta"] - float(saved_heldout.auc_delta)) <= TOLERANCE
            and bool(heldout_point["material_negative_transfer"])
            == bool(saved_heldout.material_negative_transfer)
        )
        all_numeric_match &= certificate_match and heldout_match
        per_dataset[dataset] = {
            "partition_counts": counts,
            "recomputed_certificate_gain": point,
            "recomputed_certificate_lcb": lcb,
            "recomputed_heldout_gain": heldout_point["brier_gain"],
            "certificate_match": certificate_match,
            "heldout_match": heldout_match,
        }

    checks.append(check("all_assignments_are_exact_halves", all_half_splits, per_dataset))
    checks.append(
        check("registered_geometry_schema_complete", all_geometry_columns, list(GEOMETRY_COLUMNS))
    )
    checks.append(check("certificates_and_heldout_metrics_reproduce", all_numeric_match, per_dataset))
    checks.append(
        check(
            "negative_controls_abstain",
            heldout_results.loc[
                heldout_results.dataset.isin(["SEED-IV", "DREAMER"]), "adapted"
            ].eq(False).all(),
            heldout_results.loc[
                heldout_results.dataset.isin(["SEED-IV", "DREAMER"]),
                ["dataset", "selected_method", "adapted"],
            ].to_dict("records"),
        )
    )
    checks.append(check("recorded_gate_passes", bool(gate["all_passed"]), gate))
    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "date": "2026-09-08",
        "validator_sha256": sha256(Path(__file__)),
        "checks_passed": int(sum(item["passed"] for item in checks)),
        "checks_total": len(checks),
        "checks": checks,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
