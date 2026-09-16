from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiofuse.data import load_eppvr


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "physiofuse.yaml").read_text(encoding="utf-8"))
    data = load_eppvr(cfg["paths"]["eppvr_root"], cfg["paths"]["eppvr_metadata_root"])
    print(json.dumps({
        "dataset": data.name, "subjects": data.n_subjects, "sessions_per_subject": 1,
        "trials_per_subject": data.n_trials, "windows_per_trial": data.eeg.shape[2],
        "eeg_features": data.eeg.shape[3], "pps_features": data.pps.shape[3],
        "arousal_counts": {str(i): int((data.arousal == i).sum()) for i in (0, 1)},
        "valence_counts": {str(i): int((data.valence == i).sum()) for i in (0, 1)},
    }, indent=2))


if __name__ == "__main__":
    main()
