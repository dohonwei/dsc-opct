from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/inspect_amigos_v11_schema.py"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="amigos_schema_fixture_") as directory:
        fixture_root = Path(directory) / "authorized_archive"
        fixture_root.mkdir()
        savemat(
            fixture_root / "Data_Preprocessed_P01.mat",
            {
                "joined_data": np.zeros((16, 128, 17), dtype=np.float32),
                "labels_selfassessment": np.zeros((16, 12), dtype=np.float32),
            },
        )
        output = Path(directory) / "schema_manifest.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--data-root",
                str(fixture_root),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert manifest["mat_file_count"] == 1
        assert manifest["schema_signature_count"] == 1
        assert manifest["inspection_method"].startswith("scipy.io.whosmat")
        variables = {item["name"]: item for item in manifest["files"][0]["variables"]}
        assert variables["joined_data"]["shape"] == [16, 128, 17]
        assert variables["labels_selfassessment"]["shape"] == [16, 12]
        assert "without loading participant arrays" in completed.stdout
    print("AMIGOS schema inspector synthetic test passed")


if __name__ == "__main__":
    main()
