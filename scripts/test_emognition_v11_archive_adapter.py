from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

from emognition_v11_adapter_contract import validate_emognition_task_bundle
from emognition_v11_contract import STIMULUS_NAMES
from emognition_v11_schema_adapter import (
    EmognitionTrialExclusion,
    load_emognition_task_bundles,
)


PARTICIPANTS = tuple(str(value) for value in range(22, 52))
PREFIX = "study_data/"


def questionnaire(subject_id: str, *, malformed: bool = False) -> dict:
    if malformed:
        return {"metadata": {"id": int(subject_id), "movie_order": list(STIMULUS_NAMES)}, "questionnaires": {}}
    items = []
    for index, stimulus in enumerate(STIMULUS_NAMES):
        sam = {
            "VALENCE": 4 if index % 2 == 0 else 6,
            "AROUSAL": 6 if index % 2 == 0 else 4,
            "MOTIVATION": 5,
        }
        if subject_id == "22" and index == 0:
            sam.pop("AROUSAL")
        items.append({"movie": stimulus, "emotions": {}, "sam": sam})
    return {
        "metadata": {
            "id": int(subject_id),
            "movie_order": ["BASELINE", *STIMULUS_NAMES],
        },
        "questionnaires": items,
    }


def build_archive(path: Path, *, malformed: bool = False) -> dict:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(f"{PREFIX}", "")
        for subject_id in PARTICIPANTS:
            handle.writestr(f"{PREFIX}{subject_id}/", "")
            handle.writestr(
                f"{PREFIX}{subject_id}/{subject_id}_QUESTIONNAIRES.json",
                json.dumps(questionnaire(subject_id, malformed=malformed and subject_id == "22")),
            )
            for index, stimulus in enumerate(STIMULUS_NAMES):
                if subject_id == "23" and index == 1:
                    continue
                handle.writestr(
                    f"{PREFIX}{subject_id}/{subject_id}_{stimulus}_STIMULUS_MUSE.json",
                    json.dumps({"fixture": "ok"}),
                )
    with zipfile.ZipFile(path) as handle:
        entries = [
            {"path": info.filename, "is_directory": info.is_dir()}
            for info in handle.infolist()
        ]
    return {
        "status": "emognition_schema_only_acquisition_manifest_complete",
        "archive_md5": "synthetic",
        "archive_sha256": "synthetic",
        "entries": entries,
    }


def fake_trial_record(
    payload: dict,
    *,
    subject_id: str,
    stimulus_name: str,
    stimulus_index: int,
    profile,
):
    if payload.get("fixture") != "ok":
        raise EmognitionTrialExclusion("synthetic unusable trial")
    base = float(int(subject_id) + stimulus_index)
    return (
        {
            "subject_id": subject_id,
            "trial_id": stimulus_index,
            "stimulus_index": stimulus_index,
            "stimulus_name": stimulus_name,
            "sampling_rate_hz": 256.0,
            "n_windows": 3,
            "stimulus__log_power_TP9_theta_mean": base,
            "stimulus__relative_power_TP9_theta_mean": base / 100.0,
            "stimulus__normalized_asymmetry_AF7_AF8_theta_mean": base / 200.0,
        },
        {
            "samples": 2048,
            "usable_samples": 2048,
            "usable_fraction": 1.0,
            "raw_value_policy": profile.raw_value_policy,
            "quality_policy": profile.quality_policy,
        },
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "study_data.zip"
        schema = root / "schema.json"
        manifest = build_archive(archive)
        schema.write_text(json.dumps(manifest), encoding="utf-8")
        with (
            patch(
                "emognition_v11_schema_adapter._verify_archive_and_manifest",
                return_value=manifest,
            ) as verifier,
            patch(
                "emognition_v11_schema_adapter.build_trial_record",
                side_effect=fake_trial_record,
            ),
            patch("emognition_v11_schema_adapter.hash_file", return_value="synthetic"),
        ):
            bundles, provenance = load_emognition_task_bundles(archive, schema)
        verifier.assert_called_once_with(archive, schema)

        assert len(bundles["valence"].frame) == 299
        assert len(bundles["arousal"].frame) == 298
        for task, bundle in bundles.items():
            audit = validate_emognition_task_bundle(
                bundle.frame, bundle.labels, bundle.feature_sets, task
            )
            assert audit["n_subjects"] == 30
            assert audit["n_stimuli"] == 10
        assert provenance["eligibility"]["arousal"]["missing_rating_trials_excluded"] == 1
        assert provenance["eligibility"]["arousal"]["midpoint_trials_excluded"] == 0
        assert provenance["exclusion_counts"]["missing_registered_signal_file"] == 1

        malformed_archive = root / "malformed.zip"
        malformed_schema = root / "malformed_schema.json"
        malformed_manifest = build_archive(malformed_archive, malformed=True)
        malformed_schema.write_text(json.dumps(malformed_manifest), encoding="utf-8")
        with (
            patch(
                "emognition_v11_schema_adapter._verify_archive_and_manifest",
                return_value=malformed_manifest,
            ),
            patch(
                "emognition_v11_schema_adapter.build_trial_record",
                side_effect=fake_trial_record,
            ),
            patch("emognition_v11_schema_adapter.hash_file", return_value="synthetic"),
        ):
            try:
                load_emognition_task_bundles(malformed_archive, malformed_schema)
            except ValueError:
                pass
            else:
                raise AssertionError("Malformed questionnaire nesting was not rejected")

    print(
        "Emognition archive-adapter integration test passed: ZIP prefix, missing files, "
        "task-specific missing ratings, eligibility, and schema refusal"
    )


if __name__ == "__main__":
    main()
