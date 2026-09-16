from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_endpoint_harmonization_20260910"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EXPECTED_DATASETS = {
    "DEAP", "MAHNOB-HCI", "EPPVR", "CASE", "CEAP-360VR", "SEED-IV", "DREAMER",
    "AVDOS-VR", "FACED", "EEGEmotions-27", "AMIGOS", "Emognition",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, observed: object, required: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed, "required": required})

    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    structured = json.loads((OUT / "endpoint_harmonization.json").read_text(encoding="utf-8"))
    eppvr = json.loads((OUT / "eppvr_metadata_trace.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(OUT / "endpoint_harmonization.csv")
    freeze_hash = sha256(ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json")
    record("freeze unchanged", freeze_hash == EXPECTED_FREEZE == manifest["freeze_sha256"], freeze_hash, EXPECTED_FREEZE)
    current = {path: sha256(ROOT / path) for path in manifest["files"]}
    record("artifact hashes", current == manifest["files"], current, manifest["files"])
    record("complete dataset family", set(frame.dataset) == EXPECTED_DATASETS,
           sorted(set(frame.dataset)), sorted(EXPECTED_DATASETS))
    record("row count", len(frame) == manifest["row_count"] == len(structured["rows"]) == 21, len(frame), 21)
    record("no missing semantic fields", not frame.isna().any().any(),
           frame.columns[frame.isna().any()].tolist(), [])
    record("EPPVR raw contract",
           eppvr["participants"] == 30 and eppvr["raw_data_shape"] == [[14, 10, 7000]]
           and eppvr["raw_label_shape"] == [[14, 3]],
           [eppvr["participants"], eppvr["raw_data_shape"], eppvr["raw_label_shape"]],
           [30, [[14, 10, 7000]], [[14, 3]]])
    timing = eppvr["trial_timing"]
    record("EPPVR timing", eppvr["sampling_rate_hz"] == 100 and timing["baseline_seconds"] == 10.0
           and timing["post_baseline_seconds"] == 60.0,
           [eppvr["sampling_rate_hz"], timing], "100 Hz; 10-s baseline; 60-s post-baseline")
    record("EPPVR archive identity", eppvr["archive_copy_byte_identity"] is True
           and len(eppvr["source_subject_sha256"]) == 30,
           [eppvr["archive_copy_byte_identity"], len(eppvr["source_subject_sha256"])], [True, 30])
    unknown = ["device_manufacturer_model", "electrode_reference_and_ground",
               "sensor_placement_for_channels_5_to_9", "ethics_committee", "ethics_approval_number",
               "written_informed_consent", "privacy_and_secondary_analysis_authorization"]
    record("governance unknowns explicit",
           all(eppvr[key] == "AUTHOR_CONFIRMATION_REQUIRED" for key in unknown),
           {key: eppvr[key] for key in unknown}, "all AUTHOR_CONFIRMATION_REQUIRED")
    protocol = frame.loc[frame.dataset.isin(["AMIGOS", "Emognition"])]
    record("reserved datasets remain protocol only",
           len(protocol) == 4 and protocol.verification_status.eq("PROTOCOL_ONLY_NO_PARTICIPANT_VALUES").all(),
           protocol[["dataset", "verification_status"]].to_dict("records"), "four protocol-only rows")
    category = frame.loc[frame.dataset.isin(["SEED-IV", "AVDOS-VR", "FACED", "EEGEmotions-27", "AMIGOS"])]
    bounded = category.comparability_boundary.str.contains(
        "not participant-experienced|not a 1-9", case=False, regex=True
    )
    record("category endpoints bounded", bounded.all(),
           category.loc[~bounded, ["dataset", "comparability_boundary"]].to_dict("records"), [])
    record("identity fallback wording",
           "identity fallback gives zero change by construction; it is non-intervention" in
           (ROOT / "docs/dcs_opct_v11_claim_evidence_matrix.md").read_text(encoding="utf-8").lower(),
           "claim matrix inspected", "non-intervention wording")
    source_mismatch = {}
    for path, expected in structured["source_sha256"].items():
        target = ROOT / path
        actual = sha256(target) if target.is_file() else None
        if actual != expected:
            source_mismatch[path] = {"expected": expected, "actual": actual}
    record("source hashes", not source_mismatch, source_mismatch, {})
    tex = (OUT / "table_endpoint_harmonization.tex").read_text(encoding="utf-8")
    semantic_tex = tex.replace(r"\allowbreak{}", "")
    record("publication table coverage", all(name in semantic_tex for name in EXPECTED_DATASETS),
           {name: name in semantic_tex for name in sorted(EXPECTED_DATASETS)}, "all true")
    report = {
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks),
        "checks": checks, "claim_boundary": structured["claim_boundary"],
    }
    (OUT / "independent_validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if report["status"] != "passed":
        raise RuntimeError([item["check"] for item in checks if not item["passed"]])
    print(f"Endpoint harmonization validation passed: {report['checks_passed']}/{report['checks_total']}")


if __name__ == "__main__":
    main()
