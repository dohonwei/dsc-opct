from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inspect_eegemotions27_v11_repository.py"
CHANNELS = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="eegemotions27_metadata_fixture_") as directory:
        fixture_root = Path(directory) / "repository"
        raw_root = fixture_root / "eeg_raw"
        raw_root.mkdir(parents=True)
        (fixture_root / "emotivX_channels_location.ced").write_text(
            "\n".join(
                f"{index + 1}\t{channel}" for index, channel in enumerate(CHANNELS)
            ),
            encoding="utf-8",
        )
        for participant in range(1, 31):
            for emotion_id in (4, 5, 6, 13, 17, 18, 20, 22, 24, 25):
                (raw_root / f"{participant}_{emotion_id}.0.txt").touch()

        output = Path(directory) / "manifest.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--data-root",
                str(fixture_root),
                "--output",
                str(output),
                "--allow-non-git-fixture",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert manifest["raw_filename_count"] == 300
        assert manifest["participant_count_encoded_in_filenames"] == 30
        assert manifest["trials_per_participant_min_from_filenames"] == 10
        assert manifest["channels_from_metadata_file"] == list(CHANNELS)
        assert "signal files were not opened" in manifest["inspection_method"]
        assert "without opening EEG signal files" in completed.stdout
    print("EEGEmotions-27 repository inspector synthetic test passed")


if __name__ == "__main__":
    main()
