from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/dcs_opct_v11_external_access_evidence_v5"
MANIFEST = OUT / "manifest.json"
REPORT = OUT / "independent_validation_report.json"
EXPECTED_FREEZE = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"
EXPECTED_EULA = "cbb89107841e5c1dccaa7c1d7e865c2af5e87874417ce4333b87ccf86fbdc62f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = read_json(MANIFEST)
    checks = []
    progress = tqdm(total=12, desc="Validate external-access package", unit="check", dynamic_ncols=True)

    def record(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        progress.update(1)

    record(
        "manifest_status",
        manifest.get("status") == "external_access_evidence_packaged_pending_validation",
        manifest.get("status"),
    )
    record("freeze_unchanged", manifest.get("frozen_v11_sha256") == EXPECTED_FREEZE, manifest.get("frozen_v11_sha256"))
    source_hashes = manifest.get("source_sha256", {})
    source_to_output = manifest.get("source_to_output", {})
    source_copies_valid = (
        set(source_hashes) == set(source_to_output)
        and set(source_to_output.values()) == set(manifest.get("output_sha256", {}))
        and all(
            manifest["output_sha256"].get(source_to_output[relative]) == digest
            for relative, digest in source_hashes.items()
        )
    )
    record("source_to_package_hashes_valid", source_copies_valid, len(source_hashes))
    output_valid = all(
        (OUT / name).is_file() and sha256(OUT / name) == digest
        for name, digest in manifest.get("output_sha256", {}).items()
    )
    record("package_hashes_valid", output_valid, len(manifest.get("output_sha256", {})))
    expected_files = set(manifest.get("output_sha256", {})) | {"manifest.json"}
    observed_files = {path.name for path in OUT.iterdir() if path.is_file() and path != REPORT}
    record("package_inventory_exact", expected_files == observed_files, sorted(observed_files))

    emognition = read_json(OUT / "emognition_preaccess_stack_report.json")
    amigos = read_json(OUT / "amigos_preaccess_stack_report.json")
    readiness = read_json(OUT / "emognition_access_readiness_report.json")
    record(
        "emognition_cuda_stack_passed",
        emognition.get("status") == "passed"
        and len(emognition.get("tests", [])) == 15
        and emognition.get("checks", {}).get("cuda_available") is True,
        {"status": emognition.get("status"), "tests": len(emognition.get("tests", []))},
    )
    evidence_mismatches = {}
    for relative, expected in emognition.get("evidence_sha256", {}).items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        if actual != expected:
            evidence_mismatches[relative] = {"expected": expected, "actual": actual}
    record(
        "emognition_embedded_evidence_hashes_current",
        not evidence_mismatches,
        evidence_mismatches,
    )
    record(
        "amigos_cuda_stack_passed",
        amigos.get("status") == "passed"
        and len(amigos.get("tests", [])) == 13
        and amigos.get("checks", {}).get("cuda_available") is True,
        {"status": amigos.get("status"), "tests": len(amigos.get("tests", []))},
    )
    record(
        "emognition_access_readiness_passed",
        readiness.get("status") == "passed"
        and readiness.get("checks_passed") == readiness.get("checks_total") == 13,
        {"status": readiness.get("status"), "checks": readiness.get("checks_passed")},
    )
    attempts = [read_json(OUT / f"amigos_access_attempt_{index:03d}.json") for index in range(1, 4)]
    record(
        "amigos_failures_preserved_without_data_access",
        all(
            item.get("data_downloaded") is False
            and item.get("participant_values_accessed") is False
            for item in attempts
        ),
        [item.get("status") for item in attempts],
    )
    record(
        "official_eula_integrity",
        sha256(OUT / "emognition_official_eula.pdf") == EXPECTED_EULA,
        sha256(OUT / "emognition_official_eula.pdf"),
    )
    record(
        "claim_boundary_and_no_restricted_data",
        manifest.get("restricted_participant_data_included") is False
        and "not external validation" in manifest.get("claim_boundary", ""),
        manifest.get("claim_boundary"),
    )
    progress.close()

    passed = sum(item["passed"] for item in checks)
    report = {
        "status": "passed" if passed == len(checks) else "failed",
        "date": "2026-09-09",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "manifest_sha256": sha256(MANIFEST),
        "claim_boundary": "Validation of this sidecar supplies no external empirical result.",
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "passed":
        raise RuntimeError(
            f"External-access evidence validation failed: "
            f"{[item['name'] for item in checks if not item['passed']]}"
        )
    print(f"External-access evidence validation passed: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
