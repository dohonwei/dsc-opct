from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import pandas as pd

EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
DATASETS = {"EPPVR", "CASE", "CEAP", "DREAMER", "SEED-IV", "AVDOS-VR", "FACED", "EEGEmotions-27"}
RELEASES = {"EPPVR", "CASE", "CEAP"}
PARAMETERS = {
    "maximum_probability_mean_shift", "minimum_probability_rank",
    "maximum_order_inversions", "minimum_directional_agreement",
    "minimum_displacement_cosine", "maximum_normalized_disagreement",
    "minimum_brier_gain_lcb", "auc_noninferiority_margin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen v11 action-stability artifacts.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/dcs_opct_v11_frozen_action_stability_20260910_v2"))
    parser.add_argument("--freeze", type=Path, default=Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.output_root
    required = [
        "threshold_grid.csv", "one_at_a_time_action_stability.csv",
        "domain_gate_margins.csv", "joint_local_pressure_cube.csv",
        "joint_release_patterns.csv", "summary.json",
        "fig_frozen_action_stability.pdf", "fig_frozen_action_stability.png",
        "manifest.json",
    ]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)

    grid = pd.read_csv(root / "threshold_grid.csv")
    actions = pd.read_csv(root / "one_at_a_time_action_stability.csv")
    margins = pd.read_csv(root / "domain_gate_margins.csv")
    joint = pd.read_csv(root / "joint_local_pressure_cube.csv")
    patterns = pd.read_csv(root / "joint_release_patterns.csv")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: Any, required_value: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "required": required_value})

    freeze_hash = sha256(args.freeze)
    record("freeze_hash", freeze_hash == EXPECTED_FREEZE, freeze_hash, EXPECTED_FREEZE)
    record("manifest_freeze_hash", manifest["freeze_sha256"] == EXPECTED_FREEZE, manifest["freeze_sha256"], EXPECTED_FREEZE)
    record("post_hoc_boundary", manifest["frozen_v11_unchanged"] is True and "no threshold optimization" in manifest["analysis_role"], manifest["analysis_role"], "frozen post-hoc analysis")
    record("grid_dimensions", len(grid) == 40 and set(grid.parameter) == PARAMETERS and int(grid.is_frozen.sum()) == 8, [len(grid), int(grid.is_frozen.sum())], [40, 8])
    record("action_dimensions", len(actions) == 320 and set(actions.dataset) == DATASETS and set(actions.parameter) == PARAMETERS, [len(actions), sorted(actions.dataset.unique())], [320, sorted(DATASETS)])
    record("margin_dimensions", len(margins) == 64 and set(margins.dataset) == DATASETS, len(margins), 64)
    record("joint_dimensions", len(joint) == 3 ** 8 and int(patterns.scenario_count.sum()) == 3 ** 8, [len(joint), int(patterns.scenario_count.sum())], [6561, 6561])

    frozen_rows = actions.loc[actions.grid_index.eq(2)]
    released = set(frozen_rows.loc[frozen_rows.certified, "dataset"])
    record("frozen_actions", released == RELEASES, sorted(released), sorted(RELEASES))
    maximum_unique = int(frozen_rows.groupby("dataset").selected_action.nunique().max())
    record("frozen_consistency", maximum_unique == 1, maximum_unique, 1)
    external_identity = frozen_rows.loc[frozen_rows.dataset.isin({"AVDOS-VR", "FACED", "EEGEmotions-27"}), "selected_action"].eq("identity").all()
    record("external_identity", external_identity, bool(external_identity), True)

    claim = summary["claim_boundary"].lower()
    record("claim_boundary", "post-hoc" in claim and "does not" in claim and "external effectiveness" in claim, summary["claim_boundary"], "bounded post-hoc claim")
    image = mpimg.imread(root / "fig_frozen_action_stability.png")
    record("figure_dimensions", image.shape[0] >= 1800 and image.shape[1] >= 2000, list(image.shape), ">=1800 x >=2000")
    inputs_valid = all(Path(path).is_file() and sha256(Path(path)) == digest for path, digest in manifest["input_sha256"].items())
    outputs_valid = all((root / name).is_file() and sha256(root / name) == digest for name, digest in manifest["outputs"].items())
    record("input_hashes", inputs_valid, bool(inputs_valid), True)
    record("output_hashes", outputs_valid, bool(outputs_valid), True)

    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "analysis_date": "2026-09-10",
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    (root / "independent_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['checks_passed']}/{report['checks_total']} checks passed")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
