from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

from inspect_emognition_v11_archive_schema import build_manifest


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "synthetic_study_data.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            handle.writestr("22/MUSE/22_amusement.csv", "secret participant values")
            handle.writestr("22/QUESTIONNAIRES/22_amusement.json", '{"VALENCE": 8}')
            handle.writestr("README.txt", "synthetic schema fixture")

        with patch.object(
            zipfile.ZipFile,
            "open",
            side_effect=AssertionError("Participant member content was opened"),
        ):
            manifest = build_manifest(archive, require_official_checksum=False)

    assert manifest["zip_entry_count"] == 3
    assert manifest["suffix_counts"] == {".csv": 1, ".json": 1, ".txt": 1}
    assert manifest["official_dataverse_v6_md5_match"] is False
    assert "no member was opened or extracted" in manifest["inspection_method"]
    assert all("path" in entry and "crc32" in entry for entry in manifest["entries"])
    print("Emognition schema inspector synthetic test passed without member reads")


if __name__ == "__main__":
    main()
