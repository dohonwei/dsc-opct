from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import pandas as pd

ROOT = Path("outputs/dcs_opct_v11_worked_audit_example_20260910_v4")
FREEZE = Path("docs/distribution_covered_stratified_opct_v11_final_freeze.json")
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = ["worked_audit_trace.csv", "worked_audit_example.md", "fig_worked_audit_trace.pdf", "fig_worked_audit_trace.png", "manifest.json"]
    for name in required:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(ROOT / name)
    frame = pd.read_csv(ROOT / "worked_audit_trace.csv")
    text = (ROOT / "worked_audit_example.md").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: Any, required_value: Any) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "required": required_value})

    record("freeze_hash", sha256(FREEZE) == EXPECTED_FREEZE, sha256(FREEZE), EXPECTED_FREEZE)
    record("row_count", len(frame) == 8 and frame.dataset.nunique() == 8, [len(frame), frame.dataset.nunique()], [8, 8])
    released = frame.loc[frame.selected_action.ne("identity")]
    record("release_set", set(released.dataset) == {"EPPVR", "CASE", "CEAP"}, released.dataset.tolist(), ["EPPVR", "CASE", "CEAP"])
    record("release_label_count", int(released.action_labels_accessed.sum()) == 480, int(released.action_labels_accessed.sum()), 480)
    stopped = frame.loc[frame.selected_action.eq("identity")]
    record("stopped_before_action_labels", stopped.action_labels_accessed.eq(0).all(), stopped.action_labels_accessed.tolist(), [0] * 5)
    component_stops = frame.loc[frame.component_gate.eq("stop")]
    record("downstream_not_reached", component_stops.witness_gate.eq("not reached").all() and component_stops.certificate_gate.eq("not reached").all(), component_stops[["dataset", "witness_gate", "certificate_gate"]].to_dict("records"), "all downstream stages not reached")
    eeg = frame.loc[frame.dataset.eq("EEGEmotions-27")].iloc[0]
    record("eeg_evidence_role", eeg.evidence_role == "separate pre-signal external robustness test", eeg.evidence_role, "separate pre-signal external robustness test")
    record("identity_boundary", stopped.claim_status.str.contains("not effectiveness or safety success", regex=False).all(), stopped.claim_status.tolist(), "identity is not success")
    record("worked_text", "action-stage labels accessed = 0" in text and "cannot be counted as effectiveness" in text and "not a new experiment" in text, text[:250], "bounded worked example")
    image = mpimg.imread(ROOT / "fig_worked_audit_trace.png")
    record("figure_dimensions", image.shape[0] >= 900 and image.shape[1] >= 1900, list(image.shape), ">=900 x >=1900")
    inputs_valid = all(Path(path).is_file() and sha256(Path(path)) == digest for path, digest in manifest["inputs"].items())
    outputs_valid = all((ROOT / name).is_file() and sha256(ROOT / name) == digest for name, digest in manifest["outputs"].items())
    record("input_hashes", inputs_valid, bool(inputs_valid), True)
    record("output_hashes", outputs_valid, bool(outputs_valid), True)
    report = {"status": "passed" if all(item["passed"] for item in checks) else "failed", "analysis_date": "2026-09-10", "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks), "checks": checks}
    (ROOT / "independent_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{report['checks_passed']}/{report['checks_total']} checks passed")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
