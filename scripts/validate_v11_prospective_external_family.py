from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/v11_prospective_external_family_registry.json"
FREEZE = ROOT / "docs/distribution_covered_stratified_opct_v11_final_freeze.json"
EXPECTED_FREEZE_HASH = "fe0d6b2f3bb7a90da51fd7b030f0f4874d2dee9ab14175a20f9f4e8d0d71b0ee"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    members = {member["dataset"]: member for member in registry["members"]}
    checks = {
        "registry_status": registry["status"]
        == "prospective_external_family_reserved_before_member_data_access",
        "freeze_unchanged": sha256(FREEZE) == EXPECTED_FREEZE_HASH
        == registry["freeze_sha256"],
        "two_locked_members": set(members)
        == {"AMIGOS", "Emognition Wearable Dataset 2020"},
        "reservations_match_hashes": all(
            sha256(ROOT / member["reservation_file"])
            == member["reservation_sha256"]
            for member in members.values()
        ),
        "no_member_values_accessed": all(
            member["participant_values_accessed_before_family_registration"] is False
            for member in members.values()
        ),
        "all_outcomes_reported": registry["reporting_policy"][
            "all_acquired_members_reported"
        ]
        and registry["reporting_policy"]["all_one_shot_outcomes_preserved"],
        "no_at_least_one_success_claim": "at-least-one-success"
        in registry["multiplicity_boundary"],
        "replication_requires_both": "both AMIGOS and Emognition"
        in registry["claim_policy"]["replicated_external_effectiveness"],
        "identity_not_success": "cannot count as effectiveness or safety success"
        in registry["reporting_policy"]["identity_fallback"],
        "universal_safety_forbidden": "Never permitted"
        in registry["claim_policy"]["universal_safety"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Prospective external-family checks failed: {failed}")
    print(f"Prospective external-family registry passed: {len(checks)}/{len(checks)}")


if __name__ == "__main__":
    main()
