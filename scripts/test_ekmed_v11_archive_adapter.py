from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from ekmed_v11_adapter_contract import validate_ekmed_task_bundle
from ekmed_v11_contract import STIMULUS_NAMES
from ekmed_v11_schema_adapter import (
    QUESTIONNAIRE_COLUMNS,
    SIGNAL_COLUMNS,
    load_ekmed_task_bundles,
)


PARTICIPANTS = tuple(str(value) for value in range(1, 31))
ROOT_PREFIX = (
    "EmoKey Moments EEG Dataset (EKM-ED)/muse_wearable_data/"
    "preprocessed/clean-signals/0.0078125S/"
)
QUESTIONNAIRE = (
    "EmoKey Moments EEG Dataset (EKM-ED)/questionnaires/preprocessed/"
    "Ficha_Evaluacion_Participante_SAM_Refactored.csv"
)


def questionnaire_csv() -> str:
    rows = [",ID,VALENCE,AROUSAL,EMOTION"]
    index = 0
    for subject_id in PARTICIPANTS:
        for stimulus_index, stimulus in enumerate(STIMULUS_NAMES):
            valence = 4 if stimulus_index % 2 == 0 else 6
            arousal = 6 if stimulus_index % 2 == 0 else 4
            if subject_id == "1" and stimulus == "ANGER":
                arousal = 5
            rows.append(f"{index},{subject_id},{valence},{arousal},{stimulus}")
            index += 1
    return "\n".join(rows) + "\n"


def build_fixture(root: Path) -> tuple[Path, Path, dict]:
    archive = root / "ekmed_fixture.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(QUESTIONNAIRE, questionnaire_csv())
        for subject_id in PARTICIPANTS:
            for stimulus in STIMULUS_NAMES:
                if subject_id == "2" and stimulus == "SADNESS":
                    continue
                handle.writestr(
                    f"{ROOT_PREFIX}{subject_id}/{stimulus}.csv",
                    ",".join(SIGNAL_COLUMNS) + "\nfixture\n",
                )
    with zipfile.ZipFile(archive) as handle:
        members = [
            {
                "name": info.filename,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "compression_type": info.compress_type,
            }
            for info in handle.infolist()
            if not info.is_dir()
        ]
    central = {
        "status": "central_directory_recorded_without_member_reads",
        "central_directory": {"members": members},
    }
    central_path = root / "central_directory_manifest.json"
    central_path.write_text(json.dumps(central), encoding="utf-8")
    receipt = {
        "status": "header_schema_recorded_without_data_rows",
        "archive_sha256": "synthetic",
        "central_manifest_sha256": "synthetic-central",
        "questionnaire": {
            "member": QUESTIONNAIRE,
            "delimiter": ",",
            "encoding": "utf-8-sig",
            "columns": list(QUESTIONNAIRE_COLUMNS),
        },
        "signal": {
            "member": f"{ROOT_PREFIX}1/ANGER.csv",
            "delimiter": ",",
            "encoding": "utf-8-sig",
            "columns": list(SIGNAL_COLUMNS),
        },
    }
    receipt_path = root / "header_schema_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return archive, receipt_path, central


def fake_parse_signal_frame(
    frame: pd.DataFrame,
    *,
    subject_id: str,
    stimulus_name: str,
    stimulus_index: int,
    profile,
):
    base = float(int(subject_id) + stimulus_index)
    return (
        {
            "subject_id": subject_id,
            "trial_id": stimulus_index,
            "stimulus_index": stimulus_index,
            "stimulus_name": stimulus_name,
            "sampling_rate_hz": 128.0,
            "n_windows": 3,
            "key_moment_count": 3,
            "stimulus__log_power_TP9_theta_mean": base,
            "stimulus__relative_power_TP9_theta_mean": base / 100.0,
            "stimulus__normalized_asymmetry_AF7_AF8_theta_mean": base / 200.0,
        },
        {
            "samples": 31_000,
            "key_moment_windows": 3,
            "time_alignment": profile.time_alignment,
            "amplitude_policy": profile.amplitude_policy,
            "feature_source": profile.feature_source,
        },
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        archive, receipt, central = build_fixture(Path(temporary))
        with (
            patch(
                "ekmed_v11_schema_adapter._verify_archive_and_receipt",
                return_value=(json.loads(receipt.read_text()), central),
            ),
            patch(
                "ekmed_v11_schema_adapter.parse_signal_frame",
                side_effect=fake_parse_signal_frame,
            ),
            patch("ekmed_v11_schema_adapter.hash_file", return_value="synthetic"),
        ):
            bundles, provenance = load_ekmed_task_bundles(archive, receipt)

        assert len(bundles["valence"].frame) == 119
        assert len(bundles["arousal"].frame) == 118
        for task, bundle in bundles.items():
            audit = validate_ekmed_task_bundle(
                bundle.frame, bundle.labels, bundle.feature_sets, task
            )
            assert audit["n_subjects"] == 30
            assert audit["n_stimuli"] == 4
        assert provenance["eligibility"]["arousal"]["midpoint_trials_excluded"] == 1
        assert provenance["exclusion_counts"]["missing_registered_clean_signal_file"] == 1
        assert provenance["questionnaire_participants_attempted"] == 30

    print(
        "EKM-ED archive-adapter synthetic test passed: inventory lock, missing files, "
        "midpoint exclusion, task eligibility, and structural validation"
    )


if __name__ == "__main__":
    main()
