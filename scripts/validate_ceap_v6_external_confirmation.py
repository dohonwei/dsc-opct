from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from apply_frozen_opct_v6_to_ceap import (  # noqa: E402
    attach_outcomes,
    config_from_freeze,
)
from build_ceap_trial_features import source_tree  # noqa: E402
from develop_order_preserving_transport_v6 import RULE, fit_opct  # noqa: E402
from develop_risk_controlled_transport_v5 import evaluate_action  # noqa: E402
from identity_shortcut.risk_controlled_transport import (  # noqa: E402
    balanced_cluster_order,
    certify_checkpoint,
    rows_for_clusters,
)
import run_counterfactual_identity_dose as dose  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CEAP OPCT v6 confirmation.")
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/ceap_v6_external_confirmation")
    )
    parser.add_argument(
        "--dose-root", type=Path, default=Path("outputs/ceap_counterfactual_identity_dose")
    )
    parser.add_argument(
        "--feature-root", type=Path, default=Path("outputs/ceap_trial_features")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:\AA发表论文的数据\dataset\CEAP-360VR\CEAP-360VR"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frames_match(left: pd.DataFrame, right: pd.DataFrame, tolerance: float = 1e-10) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(
                left[column].to_numpy(float), right[column].to_numpy(float),
                atol=tolerance, rtol=0, equal_nan=True,
            ):
                return False
        elif not left[column].fillna("<NA>").astype(str).equals(
            right[column].fillna("<NA>").astype(str)
        ):
            return False
    return True


def scalar_match(left: object, right: object, tolerance: float = 1e-10) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return bool(left) == bool(right)
    if isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
        right, (int, float, np.integer, np.floating)
    ):
        return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=0))
    return left == right


def json_native(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    report_path = args.output_root / "independent_validation_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")
    reservation_path = ROOT / "docs/ceap_v6_external_confirmation_reservation.json"
    reservation_amendment_path = (
        ROOT / "docs/ceap_v6_external_confirmation_reservation_amendment_001.json"
    )
    implementation_path = (
        ROOT / "docs/ceap_v6_external_confirmation_implementation_lock_amendment_001.json"
    )
    freeze_path = ROOT / "docs/order_preserving_transport_v6_final_freeze.json"
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    amendment = json.loads(reservation_amendment_path.read_text(encoding="utf-8"))
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (args.output_root / "external_confirmation_manifest.json").read_text(encoding="utf-8")
    )
    gate = json.loads((args.output_root / "external_confirmation_gate.json").read_text("utf-8"))
    lock_path = args.output_root / "primary_action_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = []

    def record(name: str, passed: bool, observed: object, required: object) -> None:
        checks.append(
            {"check": name, "passed": bool(passed), "observed": observed, "required": required}
        )

    record(
        "registration chain retains exact hashes",
        implementation["reservation_sha256"] == sha256(reservation_path)
        and implementation["reservation_amendment_sha256"] == sha256(reservation_amendment_path)
        and implementation["freeze_sha256"] == sha256(freeze_path),
        True, True,
    )
    code_ok = all(
        sha256(ROOT / relative) == expected
        for relative, expected in implementation["analysis_code_sha256"].items()
    )
    record("all registered analysis code retains exact hashes", code_ok, code_ok, True)
    count, size, digest = source_tree(
        args.data_root, reservation["source"]["included_directories"]
    )
    source_ok = (
        count == amendment["source_file_count"]
        and size == amendment["source_total_bytes"]
        and digest == amendment["canonical_tree_sha256"]
    )
    record(
        "registered CEAP source tree is unchanged",
        source_ok,
        {"files": count, "bytes": size, "sha256": digest},
        {
            "files": amendment["source_file_count"],
            "bytes": amendment["source_total_bytes"],
            "sha256": amendment["canonical_tree_sha256"],
        },
    )
    feature_manifest = json.loads(
        (args.feature_root / "build_manifest.json").read_text(encoding="utf-8")
    )
    feature_table = pd.read_csv(args.feature_root / "ceap_trial_features.csv")
    feature_ok = bool(
        feature_manifest["rows"] == 256
        and feature_manifest["subjects"] == 32
        and feature_manifest["physical_stimuli"] == 8
        and feature_manifest["output_sha256"]
        == sha256(args.feature_root / "ceap_trial_features.csv")
        and len(feature_table) == 256
    )
    record("CEAP feature contract and hash verify", feature_ok, feature_manifest, "32 x 8")

    predictions = pd.read_csv(args.dose_root / "predictions.csv")
    split_audit = pd.read_csv(args.dose_root / "split_audit.csv")
    stored_summary = pd.read_csv(args.dose_root / "summary.csv")
    rebuilt_summary = dose.summarize(predictions, split_audit)
    record(
        "counterfactual summary reconstructs from row predictions",
        frames_match(rebuilt_summary, stored_summary),
        len(rebuilt_summary), len(stored_summary),
    )
    dose_manifest = json.loads((args.dose_root / "run_manifest.json").read_text("utf-8"))
    record(
        "counterfactual run used registered CUDA settings",
        dose_manifest["gpu_epochs"] == 40 and bool(dose_manifest["gpu_name"]),
        {"gpu": dose_manifest["gpu_name"], "epochs": dose_manifest["gpu_epochs"]},
        {"gpu": "recorded", "epochs": 40},
    )
    record(
        "pre-outcome action lock is immutable",
        sha256(lock_path) == manifest["action_lock_sha256"],
        sha256(lock_path), manifest["action_lock_sha256"],
    )

    mechanism = pd.read_csv(args.output_root / "mechanism_probabilities_blinded.csv")
    if lock["base_method"] == "identity":
        rebuilt_opct = mechanism.probability_identity.to_numpy(float)
        rebuilt_diagnostics = lock["opct_diagnostics"]
    else:
        rebuilt_opct, rebuilt_diagnostics = fit_opct(
            mechanism.probability_identity.to_numpy(float),
            mechanism[f"probability_{lock['base_method']}"].to_numpy(float),
            RULE,
            reservation["confirmation_contract"]["primary_audit_seed"],
            "cuda",
        )
    record(
        "frozen OPCT projection reconstructs",
        np.allclose(
            rebuilt_opct, mechanism.probability_opct.to_numpy(float), atol=1e-7, rtol=0
        ) and all(
            scalar_match(rebuilt_diagnostics[key], lock["opct_diagnostics"][key], 1e-7)
            for key in rebuilt_diagnostics
        ),
        rebuilt_diagnostics, lock["opct_diagnostics"],
    )
    config = config_from_freeze(freeze)
    evaluation = attach_outcomes(
        mechanism,
        args.dose_root / "summary.csv",
        float(
            json.loads(
                (ROOT / "outputs/identity_shortcut_risk_classifier/freeze_manifest.json")
                .read_text(encoding="utf-8")
            )["material_effect_threshold"]
        ),
    )
    seed = int(reservation["confirmation_contract"]["primary_audit_seed"])
    order = balanced_cluster_order(evaluation, config, seed)
    audit_clusters = order[: config.cluster_budgets[0]]
    audit_mask = rows_for_clusters(evaluation, config.cluster_columns, audit_clusters)
    audit = evaluation.loc[audit_mask].reset_index(drop=True)
    heldout = evaluation.loc[~audit_mask].reset_index(drop=True)
    stored_audit = pd.read_csv(args.output_root / "primary_audit_labeled.csv")
    stored_heldout = pd.read_csv(args.output_root / "primary_heldout_results.csv")
    record(
        "primary audit and held-out partitions reconstruct",
        frames_match(audit, stored_audit) and frames_match(heldout, stored_heldout),
        {"audit": len(audit), "heldout": len(heldout)},
        {"audit": 160, "heldout": 160},
    )
    candidate = lock["candidate_method"]
    if candidate == "identity":
        certificate = None
        selected = "identity"
    else:
        certificate = certify_checkpoint(
            audit, ["opct"], config,
            seed + RULE.configuration_budget * 1009,
            total_registered_candidates=1,
        ).iloc[0]
        selected = "opct" if bool(certificate.certified) else "identity"
    metrics = evaluate_action(heldout, selected)
    certificate_payload = json.loads(
        (args.output_root / "primary_candidate_certificate.json").read_text("utf-8")
    )
    certificate_ok = (
        certificate_payload["selected_method"] == selected
        and (
            certificate is None
            or all(
                scalar_match(certificate_payload["certificate"][key], certificate[key])
                for key in certificate.index
            )
        )
    )
    record("primary certificate reconstructs", certificate_ok, selected, certificate_payload)
    metrics_ok = all(
        scalar_match(gate["heldout_metrics"][key], value)
        for key, value in metrics.items()
    )
    record("held-out metrics reconstruct", metrics_ok, metrics, gate["heldout_metrics"])
    record(
        "claim flag equals all frozen gates",
        gate["claim_supported"]
        == bool(gate["certificate_pass"] and gate["effectiveness_pass"] and gate["safety_pass"]),
        gate["claim_supported"], "conjunction of effectiveness and safety gates",
    )

    passed = all(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed else "failed",
        "date": "2026-09-07",
        "n_checks": len(checks),
        "n_passed": sum(check["passed"] for check in checks),
        "checks": checks,
        "external_gate": gate,
    }
    report_path.write_text(json.dumps(json_native(report), indent=2), encoding="utf-8")
    print(json.dumps(json_native(report), indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
