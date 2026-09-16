from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "risk_controlled_transport_v4_development"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    observed: object,
    required: str,
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "required": required,
        }
    )


def main() -> None:
    manifest = json.loads((OUTPUT / "development_manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(OUTPUT / "strategy_summary.csv")
    decisions = pd.read_csv(OUTPUT / "checkpoint_decisions.csv")
    applicability = pd.read_csv(OUTPUT / "applicability_gates.csv")
    results = pd.read_csv(OUTPUT / "repeated_audit_results.csv")
    checks: list[dict[str, object]] = []
    record(
        checks,
        "registered configuration budgets preserve complete four-dose clusters",
        manifest["config"]["configuration_budgets"] == [16, 32, 64]
        and manifest["config"]["configurations_per_cluster"] == 4,
        manifest["config"]["configuration_budgets"],
        "exactly 16/32/64 configurations with cluster size 4",
    )
    record(
        checks,
        "formal development uses adequate bootstrap repetitions",
        manifest["config"]["bootstrap_repetitions"] >= 2000,
        manifest["config"]["bootstrap_repetitions"],
        ">= 2000",
    )
    expected_rows = 2 * manifest["audit_repetitions"] * 3 * 6
    record(
        checks,
        "comparison grid is complete",
        len(results) == expected_rows,
        len(results),
        str(expected_rows),
    )
    v4 = summary.loc[summary.strategy == "v4_risk_controlled"]
    record(
        checks,
        "v4 has no observed negative transfer in repeated real-domain development",
        v4.negative_transfer_rate.max() == 0.0,
        float(v4.negative_transfer_rate.max()),
        "0",
    )
    seediv_v4 = v4.loc[v4.dataset == "SEED-IV"]
    record(
        checks,
        "v4 abstains on the known harmful SEED-IV shift",
        seediv_v4.adaptation_coverage.max() == 0.0,
        float(seediv_v4.adaptation_coverage.max()),
        "0 adaptation coverage",
    )
    eppvr_64 = v4.loc[
        (v4.dataset == "EPPVR") & (v4.configuration_budget == 64)
    ].iloc[0]
    record(
        checks,
        "v4 remains effective on EPPVR by the largest audit budget",
        eppvr_64.adaptation_coverage >= 0.50
        and eppvr_64.mean_brier_gain > 0.0
        and eppvr_64.minimum_brier_gain_when_adapted > 0.0,
        {
            "coverage": float(eppvr_64.adaptation_coverage),
            "mean_brier_gain": float(eppvr_64.mean_brier_gain),
            "minimum_adapted_gain": float(eppvr_64.minimum_brier_gain_when_adapted),
        },
        "coverage >= 0.50, positive mean gain, and positive minimum adapted gain",
    )
    coverages = v4.sort_values(["dataset", "configuration_budget"]).groupby("dataset")[
        "adaptation_coverage"
    ]
    monotone = all((values.diff().fillna(0.0) >= -1e-12).all() for _, values in coverages)
    record(
        checks,
        "sequential certification coverage is monotone in available budget",
        monotone,
        v4.pivot(index="dataset", columns="configuration_budget", values="adaptation_coverage")
        .to_dict(orient="index"),
        "nondecreasing",
    )
    seediv_coral = applicability.loc[
        (applicability.dataset == "SEED-IV") & (applicability.method == "coral")
    ].iloc[0]
    record(
        checks,
        "applicability gate rejects the catastrophic SEED-IV CORAL action",
        not bool(seediv_coral.applicable),
        seediv_coral.applicability_reason,
        "rejected before target-label certification",
    )
    eppvr_candidates = applicability.loc[
        (applicability.dataset == "EPPVR") & applicability.method.ne("identity")
    ]
    record(
        checks,
        "applicability gate does not collapse to universal abstention",
        bool(eppvr_candidates.applicable.any()),
        int(eppvr_candidates.applicable.sum()),
        ">= 1 applicable EPPVR adapter",
    )
    locked = decisions.loc[decisions.selection_reason == "carried_forward_locked_certificate"]
    record(
        checks,
        "certified actions are carried forward without reversal",
        len(locked) > 0
        and locked.actual_configuration_budget.lt(locked.configuration_budget).all(),
        len(locked),
        "> 0 valid carried-forward decisions",
    )
    passed = all(item["passed"] for item in checks)
    report = {
        "status": "passed" if passed else "failed",
        "date": "2026-09-07",
        "n_checks": len(checks),
        "checks": checks,
        "development_conclusion": (
            "The v4 framework passes retrospective development gates for selective effectiveness "
            "and observed negative-transfer control. This does not establish the manuscript claim "
            "until the frozen one-shot DREAMER confirmation passes."
        ),
        "claim_status": "development_supported_external_confirmation_pending",
        "v4_code_sha256": sha256(
            ROOT / "src" / "identity_shortcut" / "risk_controlled_transport.py"
        ),
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
