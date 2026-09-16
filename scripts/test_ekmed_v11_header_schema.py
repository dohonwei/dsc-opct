from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from inspect_ekmed_v11_header_schema import build_header_receipt, select_members


class EKMHeaderSchemaTest(unittest.TestCase):
    def build_fixture(self, root: Path) -> tuple[Path, Path]:
        archive_path = root / "fixture.zip"
        questionnaire = (
            "EKM-ED/questionnaires/preprocessed/"
            "Ficha_Evaluacion_Participante_SAM_Refactored.csv"
        )
        signal = (
            "EmoKey Moments EEG Dataset (EKM-ED)/muse_wearable_data/"
            "preprocessed/clean-signals/0.0078125S/1/ANGER.csv"
        )
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                questionnaire,
                ",ID,VALENCE,AROUSAL,EMOTION\n0,1,2,8,ANGER\n",
            )
            archive.writestr(signal, "Time,TP9,AF7,AF8,TP10\n0,1,2,3,4\n")
            archive.writestr(
                "EmoKey Moments EEG Dataset (EKM-ED)/muse_wearable_data/"
                "preprocessed/unclean-signals/muse/0.0078125/1/ANGER.csv",
                "Wrong,Branch\n99,99\n",
            )
        archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = [
                {
                    "name": info.filename,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                    "compression_type": info.compress_type,
                }
                for info in archive.infolist()
                if not info.is_dir()
            ]
        manifest_path = root / "central.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "central_directory_recorded_without_member_reads",
                    "archive_sha256": archive_sha,
                    "central_directory": {"members": members},
                }
            ),
            encoding="utf-8",
        )
        return archive_path, manifest_path

    def test_reads_only_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path, manifest_path = self.build_fixture(Path(temporary))
            receipt = build_header_receipt(archive_path, manifest_path)
            self.assertEqual(receipt["status"], "header_schema_recorded_without_data_rows")
            self.assertEqual(receipt["questionnaire"]["delimiter"], ",")
            self.assertEqual(receipt["signal"]["delimiter"], ",")
            self.assertEqual(receipt["questionnaire"]["data_rows_read"], 0)
            self.assertEqual(receipt["signal"]["data_rows_read"], 0)
            self.assertEqual(
                receipt["questionnaire"]["columns"],
                ["__index__", "ID", "VALENCE", "AROUSAL", "EMOTION"],
            )
            self.assertEqual(
                receipt["signal"]["columns"],
                ["Time", "TP9", "AF7", "AF8", "TP10"],
            )
            self.assertIn("/clean-signals/0.0078125S/", receipt["signal"]["member"])
            self.assertNotIn("unclean-signals", receipt["signal"]["member"])

    def test_selection_requires_registered_branches(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "questionnaire"):
            select_members(["only/signal.csv"])


if __name__ == "__main__":
    unittest.main()
