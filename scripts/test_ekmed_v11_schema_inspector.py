from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from inspect_ekmed_v11_archive_schema import build_manifest


class EKMArchiveInspectorTest(unittest.TestCase):
    def test_records_central_directory_without_member_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "synthetic.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "questionnaires/preprocessed/sam.csv",
                    "subject,valence\n1,7\n",
                )
                archive.writestr(
                    "muse_wearable_data/preprocessed/clean-signals/muse/0.0078125/1/ANGER.csv",
                    "TP9,AF7,AF8,TP10\n0,0,0,0\n",
                )
            manifest = build_manifest(
                archive_path,
                expected_bytes=archive_path.stat().st_size,
                expected_md5=None,
                compute_hashes=False,
            )
            central = manifest["central_directory"]
            self.assertEqual(manifest["participant_member_opened"], False)
            self.assertEqual(central["member_count"], 2)
            self.assertEqual(central["suffix_counts"], {".csv": 2})
            self.assertEqual(len(central["members"]), 2)
            self.assertTrue(all("crc32" in row for row in central["members"]))

    def test_rejects_byte_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "synthetic.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("a.txt", "x")
            with self.assertRaisesRegex(RuntimeError, "byte size mismatch"):
                build_manifest(
                    archive_path,
                    expected_bytes=archive_path.stat().st_size + 1,
                    expected_md5=None,
                    compute_hashes=False,
                )


if __name__ == "__main__":
    unittest.main()
