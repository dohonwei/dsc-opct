from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CLUSTERS = ["task", "representation", "model", "split_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DREAMER v4 external confirmation artifacts.")
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/dreamer_v4_external_confirmation")
    )
    parser.add_argument(
        "--application-freeze", type=Path, default=Path("docs/dreamer_v4_application_freeze.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/dreamer_v4_external_confirmation/validation_report.json")
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    return float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else np.nan


def check(name: str, passed: bool, observed, required: str) -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "observed": observed, "required": required}


def main() -> None:
    args = parse_args()
    required = [
        "mechanism_probabilities_blinded.csv",
        "candidate_signatures.csv",
        "applicability_gates.csv",
        "primary_checkpoint_decisions.csv",
        "primary_candidate_certificates.csv",
        "primary_audit_cluster_order.csv",
        "primary_action_lock.json",
        "primary_heldout_predictions.csv",
        "repeated_checkpoint_decisions.csv",
        "repeated_audit_results.csv",
        "external_confirmation_gate.json",
        "external_confirmation_manifest.json",
        "scmt_candidate_diagnostics.csv",
    ]
    missing = [name for name in required if not (args.root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing external confirmation artifacts: {missing}")
    freeze = json.loads(args.application_freeze.read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "external_confirmation_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((args.root / "primary_action_lock.json").read_text(encoding="utf-8"))
    gate = json.loads((args.root / "external_confirmation_gate.json").read_text(encoding="utf-8"))
    mechanism = pd.read_csv(args.root / "mechanism_probabilities_blinded.csv")
    decisions = pd.read_csv(args.root / "primary_checkpoint_decisions.csv")
    order = pd.read_csv(args.root / "primary_audit_cluster_order.csv")
    heldout = pd.read_csv(args.root / "primary_heldout_predictions.csv")
    repeated = pd.read_csv(args.root / "repeated_audit_results.csv")
    repeated_decisions = pd.read_csv(args.root / "repeated_checkpoint_decisions.csv")
    checks: list[dict[str, object]] = []
    app_script = ROOT / "scripts" / "apply_frozen_risk_controlled_transport_to_dreamer.py"
    checks.append(check(
        "application script matches the preregistered hash",
        sha256(app_script) == freeze["application_script_sha256"],
        sha256(app_script),
        freeze["application_script_sha256"],
    ))
    probability_columns = [column for column in mechanism if column.startswith("probability_")]
    forbidden = {"material_optimism_event", "exposure_effect", "dose_induced_amplification"}
    checks.append(check(
        "blinded mechanism table has 240 configurations and no outcome columns",
        len(mechanism) == 240 and not forbidden.intersection(mechanism.columns) and len(probability_columns) == 6,
        {"rows": len(mechanism), "probability_columns": probability_columns, "forbidden_present": sorted(forbidden.intersection(mechanism.columns))},
        "240 rows, six candidate probabilities, and zero outcome columns",
    ))
    cluster_sizes = mechanism.groupby(CLUSTERS).size()
    checks.append(check(
        "all physical audit clusters contain four nonzero dose configurations",
        len(cluster_sizes) == 60 and cluster_sizes.eq(4).all(),
        cluster_sizes.value_counts().sort_index().to_dict(),
        "60 clusters of size 4",
    ))
    checks.append(check(
        "primary checkpoint budgets follow the frozen sequence",
        decisions.configuration_budget.tolist() == [16, 32, 64],
        decisions.configuration_budget.tolist(),
        "[16, 32, 64]",
    ))
    final = decisions.iloc[-1]
    checks.append(check(
        "persisted action lock agrees with the final primary decision",
        lock["selected_method"] == final.selected_method and lock["actual_cluster_budget"] == int(final.actual_cluster_budget),
        {"lock_method": lock["selected_method"], "decision_method": final.selected_method, "lock_clusters": lock["actual_cluster_budget"], "decision_clusters": int(final.actual_cluster_budget)},
        "exact agreement",
    ))
    checks.append(check(
        "manifest binds the immutable action lock",
        manifest["primary_action_lock_sha256"] == sha256(args.root / "primary_action_lock.json"),
        manifest["primary_action_lock_sha256"],
        sha256(args.root / "primary_action_lock.json"),
    ))
    audit_keys = set(map(tuple, order.iloc[: int(final.actual_cluster_budget)][CLUSTERS].itertuples(index=False, name=None)))
    heldout_keys = set(map(tuple, heldout[CLUSTERS].drop_duplicates().itertuples(index=False, name=None)))
    checks.append(check(
        "primary audit and held-out clusters are disjoint",
        not audit_keys.intersection(heldout_keys),
        len(audit_keys.intersection(heldout_keys)),
        "0 overlapping clusters",
    ))
    expected_heldout = 240 - int(final.actual_configuration_budget)
    checks.append(check(
        "held-out row count matches the locked audit budget",
        len(heldout) == expected_heldout,
        len(heldout),
        str(expected_heldout),
    ))
    selected = lock["selected_method"]
    labels = heldout.material_optimism_event.to_numpy(int)
    identity = heldout.probability_identity.to_numpy(float)
    chosen = heldout[f"probability_{selected}"].to_numpy(float)
    gain = float(np.mean((labels - identity) ** 2 - (labels - chosen) ** 2))
    auc_delta = safe_auc(labels, chosen) - safe_auc(labels, identity)
    checks.append(check(
        "primary point estimates reproduce from held-out predictions",
        np.isclose(gain, gate["primary"]["brier_gain"], equal_nan=True) and np.isclose(auc_delta, gate["primary"]["auc_delta"], equal_nan=True),
        {"brier_gain": gain, "auc_delta": auc_delta},
        "match external_confirmation_gate.json",
    ))
    expected_seeds = freeze["repeated_audit_seeds"]
    checks.append(check(
        "all 100 prespecified repeated audit seeds were used",
        repeated.audit_seed.tolist() == expected_seeds and repeated.audit_repetition.nunique() == 100,
        {"rows": len(repeated), "first_seed": int(repeated.audit_seed.iloc[0]), "last_seed": int(repeated.audit_seed.iloc[-1])},
        "exact registered seed list",
    ))
    checks.append(check(
        "repeated checkpoint table is complete",
        len(repeated_decisions) == 300 and repeated_decisions.groupby("audit_repetition").size().eq(3).all(),
        len(repeated_decisions),
        "300 rows, three checkpoints per repetition",
    ))
    coverage = float(repeated.adapted.mean())
    negative = float(repeated.negative_transfer.mean())
    checks.append(check(
        "reported robustness frequencies reproduce",
        np.isclose(coverage, gate["repeated_audit_coverage"]) and np.isclose(negative, gate["repeated_audit_negative_transfer_frequency"]),
        {"coverage": coverage, "negative_transfer_frequency": negative},
        "match external_confirmation_gate.json",
    ))
    criteria = gate["criteria"]
    checks.append(check(
        "claim flag equals the conjunction of every frozen external criterion",
        bool(gate["claim_supported"]) == bool(all(criteria.values())),
        {"claim_supported": gate["claim_supported"], "criteria": criteria},
        "claim_supported == all(criteria)",
    ))
    passed = all(item["passed"] for item in checks)
    report = {
        "status": "passed" if passed else "failed",
        "date": "2026-09-07",
        "n_checks": len(checks),
        "checks": checks,
        "claim_supported": bool(gate["claim_supported"]) if passed else False,
        "conclusion": gate["interpretation"] if passed else "Artifact validation failed; no manuscript claim is admissible.",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
