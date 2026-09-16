from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scipy.io import whosmat
from tqdm.auto import tqdm

from amigos_v11_contract import DEFAULT_DATA_ROOT, ROOT, sha256, verify_preaccess_contract


DEFAULT_OUTPUT = ROOT / "outputs/amigos_v11_preaccess/schema_only_acquisition_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record AMIGOS file hashes and MAT variable schemas without loading "
            "participant arrays or values."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def inspect_mat_file(path: Path) -> dict:
    variables = [
        {"name": name, "shape": list(shape), "mat_type": mat_type}
        for name, shape, mat_type in whosmat(path)
    ]
    return {
        "relative_path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "variables": variables,
    }


def build_manifest(data_root: Path) -> dict:
    reservation, _, _ = verify_preaccess_contract()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Authorized AMIGOS data root not found: {data_root}. "
            "Obtain the official archive under its EULA first."
        )
    files = sorted(data_root.rglob("*.mat"), key=lambda path: path.as_posix().lower())
    if not files:
        raise FileNotFoundError(f"No MAT files found under authorized root: {data_root}")

    entries = []
    signatures: Counter[str] = Counter()
    for path in tqdm(files, desc="AMIGOS schema-only MAT audit", unit="file", dynamic_ncols=True):
        entry = inspect_mat_file(path)
        entry["relative_path"] = path.relative_to(data_root).as_posix()
        entries.append(entry)
        signature = json.dumps(entry["variables"], sort_keys=True, separators=(",", ":"))
        signatures[signature] += 1

    return {
        "status": "amigos_schema_only_acquisition_manifest_complete",
        "dataset": "AMIGOS",
        "phase": "pre-participant-value schema inspection",
        "data_root": str(data_root.resolve()),
        "mat_file_count": len(entries),
        "schema_signature_count": len(signatures),
        "schema_signature_frequencies": sorted(signatures.values(), reverse=True),
        "files": entries,
        "inspection_method": "scipy.io.whosmat plus byte hashing; scipy.io.loadmat was not called",
        "reservation_id": reservation["registration_id"],
        "claim_boundary": (
            "This manifest proves only archive presence, stable bytes, and variable schemas. "
            "It does not establish structural eligibility, signal quality, endpoint variation, "
            "effectiveness, or safety."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite schema manifest: {args.output}")
    manifest = build_manifest(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Recorded {manifest['mat_file_count']} MAT schemas in {args.output} "
        "without loading participant arrays"
    )


if __name__ == "__main__":
    main()
