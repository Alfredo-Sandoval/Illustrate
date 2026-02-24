#!/usr/bin/env python3
"""Generate latest.json update manifest from release artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(path_value: str) -> Path:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate desktop updater latest.json manifest")
    parser.add_argument("--version", required=True, help="Release version (e.g. 1.2.3)")
    parser.add_argument("--tag", required=True, help="Git tag used in release (e.g. v1.2.3)")
    parser.add_argument("--repo-slug", default="Alfredo-Sandoval/Illustrate", help="GitHub slug owner/repo")
    parser.add_argument("--macos-file", type=_required_file, required=True, help="macOS artifact path (.dmg)")
    parser.add_argument("--windows-file", type=_required_file, required=True, help="Windows artifact path (.zip)")
    parser.add_argument("--out", default="dist/package/latest.json", help="Output path for manifest JSON")
    args = parser.parse_args()

    base_url = f"https://github.com/{args.repo_slug}/releases/download/{args.tag}"
    mac_name = args.macos_file.name
    win_name = args.windows_file.name

    payload = {
        "version": str(args.version),
        "tag": str(args.tag),
        "released_at": datetime.now(timezone.utc).isoformat(),
        "channels": {
            "stable": {
                "macos": {
                    "url": f"{base_url}/{mac_name}",
                    "sha256": _sha256(args.macos_file),
                    "filename": mac_name,
                },
                "windows": {
                    "url": f"{base_url}/{win_name}",
                    "sha256": _sha256(args.windows_file),
                    "filename": win_name,
                },
            }
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
