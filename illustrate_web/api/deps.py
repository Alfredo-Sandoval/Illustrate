"""Shared backend dependency helpers."""

from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path

_UPLOAD_ROOT = Path(tempfile.gettempdir()) / "illustrate_uploads"
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def upload_root() -> Path:
    return _UPLOAD_ROOT


def register_upload(data: bytes, suffix: str = ".pdb") -> str:
    pdb_id = uuid.uuid4().hex
    path = _UPLOAD_ROOT / f"{pdb_id}{suffix}"
    path.write_bytes(data)
    return pdb_id


def get_upload_path(pdb_id: str) -> Path:
    token = pdb_id.strip().lower()
    if not _UPLOAD_ID_RE.fullmatch(token):
        raise FileNotFoundError(f"Unknown pdb_id: {pdb_id}")

    matches = [path for path in _UPLOAD_ROOT.iterdir() if path.is_file() and path.stem.lower() == token]
    if len(matches) != 1:
        raise FileNotFoundError(f"Unknown pdb_id: {pdb_id}")
    return matches[0]
