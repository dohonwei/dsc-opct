from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RISK_ROOT = ROOT / "outputs" / "identity_shortcut_risk_classifier"
SCMT_ROOT = ROOT / "outputs" / "scmt_public_development"
EXTERNAL_ROOT = ROOT / "outputs" / "eppvr_frozen_risk_validation"
TRANSPORT_ROOT = ROOT / "outputs" / "eppvr_scmt_transport"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> None:
    checks: list[dict] = []
    risk_manifest = json.loads((RISK_ROOT / "freeze_manifest.json").read_text(encoding="utf-8"))
    scmt_manifest_path = SCMT_ROOT / "scmt_freeze_manifest.json"
    scmt_manifest = json.loads(scmt_manifest_path.read_text(encoding="utf-8"))
    retrospective = json.loads(
        (TRANSPORT_ROOT / "retrospective_evaluation_manifest.json").read_text(encoding="utf-8")
    )
    locked_gate_path = EXTERNAL_ROOT / "external_validation_gate.json"
    module_path = ROOT / "src" / "identity_shortcut" / "domain_transport.py"
    check(risk_manifest["admissible"], "source risk model is frozen and admissible", checks)
    check(scmt_manifest["admissible"], "SCMT public development passed its gate", checks)
    check(
        scmt_manifest["source_risk_model_sha256"] == risk_manifest["model_sha256"],
        "SCMT references the frozen source-risk hash",
        checks,
    )
    check(
        sha256(module_path) == scmt_manifest["domain_transport_code_sha256"],
        "SCMT implementation hash matches its freeze",
        checks,
    )
    check(
        sha256(scmt_manifest_path) == retrospective["scmt_manifest_sha256"],
        "EPPVR application references the current SCMT manifest",
        checks,
    )
    check(
        sha256(locked_gate_path) == retrospective["locked_external_gate_sha256_before"]
        == retrospective["locked_external_gate_sha256_after"],
        "locked zero-shot EPPVR gate remained byte-identical",
        checks,
    )
    synthetic = pd.read_csv(SCMT_ROOT / "synthetic_shift_results.csv")
    check(
        synthetic[["seed", "shift_family", "shift_dose"]].drop_duplicates().shape[0] == 75,
        "synthetic development contains all 75 scenarios",
        checks,
    )
    check(
        set(synthetic.method)
        == {
            "identity",
            "mean_shift",
            "coral",
            "quantile_mapping",
            "support_clipping",
            "scmt_conservative",
            "scmt_balanced",
            "scmt_support_strict",
        },
        "synthetic benchmark contains every frozen method",
        checks,
    )
    predictions = pd.read_csv(TRANSPORT_ROOT / "eppvr_transport_predictions.csv")
    locked = pd.read_csv(EXTERNAL_ROOT / "eppvr_risk_predictions.csv")
    check(len(predictions) == len(locked) == 320, "EPPVR row count remains 320", checks)
    check(
        np.allclose(
            predictions.probability_identity,
            locked.predicted_material_risk,
            atol=5e-8,
            rtol=0,
        ),
        "identity probabilities reproduce the locked external predictions",
        checks,
    )
    intervals = pd.read_csv(TRANSPORT_ROOT / "paired_bootstrap_differences.csv")
    check(
        intervals.bootstrap_valid_repetitions.min() == 5000,
        "all paired intervals use 5000 valid configuration-cluster draws",
        checks,
    )
    diagnostics = pd.read_csv(TRANSPORT_ROOT / "scmt_seed_diagnostics.csv")
    check(set(diagnostics.seed) == {17, 29, 43, 71, 101}, "all five SCMT seeds are present", checks)
    check(diagnostics.applied.all(), "all final SCMT seeds passed the unlabeled gate", checks)
    check(
        (diagnostics.mean_displacement <= 1.25 + 1e-6).all(),
        "all SCMT mappings obey the frozen trust-region radius",
        checks,
    )
    report = {
        "status": "passed",
        "n_checks": len(checks),
        "checks": checks,
        "scientific_claim_status": "post_external_retrospective_method_development",
        "required_next_validation": "an untouched fourth dataset",
    }
    (TRANSPORT_ROOT / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
