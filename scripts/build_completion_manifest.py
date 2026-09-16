from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a non-destructive SHA-256 manifest for a completed experiment directory."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completion manifest: {output}")
    if output == root or root in output.parents:
        raise ValueError("Completion manifest must be written outside the hashed experiment directory")

    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "status": "completed_artifacts_hashed_without_modification",
        "experiment": args.experiment,
        "root": root.as_posix(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in manifest if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
