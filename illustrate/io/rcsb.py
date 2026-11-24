"""Fetch PDB files from RCSB by ID."""

from __future__ import annotations

import http.client
import os
import re
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_CACHE_DIR = Path.home() / ".cache" / "illustrate"
_PDB_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")


def _fetch_timeout_seconds() -> float:
    raw = os.environ.get("ILLUSTRATE_RCSB_FETCH_TIMEOUT_SECONDS")
    if raw is None or raw.strip() == "":
        return 30.0
    value = float(raw)
    if value <= 0.0:
        raise ValueError("ILLUSTRATE_RCSB_FETCH_TIMEOUT_SECONDS must be > 0")
    return value


def _is_complete_pdb(path: Path) -> bool:
    """Return whether *path* ends with the ``END`` record that closes a PDB file."""
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - 256))
            tail = handle.read()
    except OSError:
        return False
    lines = tail.rstrip().splitlines()
    return bool(lines) and lines[-1].strip() == b"END"


def _write_atomically(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def fetch_pdb(pdb_id: str) -> Path:
    """Get a four-character PDB ID from the cache or download its PDB file.

    Strip surrounding whitespace and use an uppercase cache filename. Reuse
    the cached file only if it ends with an ``END`` record.

    Raises
    ------
    ValueError
        If the ID is invalid or the timeout setting is nonnumeric or
        nonpositive when downloading.
    FileNotFoundError
        If RCSB returns 404 for the PDB file.
    ConnectionError
        On network failure, including timeouts and interrupted downloads.
    """
    pdb_id = pdb_id.strip()
    if not _PDB_ID_RE.match(pdb_id):
        raise ValueError(
            f"PDB ID must be exactly 4 alphanumeric characters, got {pdb_id!r}"
        )

    pdb_id_upper = pdb_id.upper()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _CACHE_DIR / f"{pdb_id_upper}.pdb"

    if _is_complete_pdb(cached):
        return cached

    url = _RCSB_URL.format(pdb_id=pdb_id_upper)
    try:
        with urlopen(url, timeout=_fetch_timeout_seconds()) as resp:  # noqa: S310
            data = resp.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(
                f"PDB ID {pdb_id_upper!r} not found on RCSB"
            ) from exc
        raise ConnectionError(
            f"RCSB returned HTTP {exc.code} for {pdb_id_upper}"
        ) from exc
    except URLError as exc:
        raise ConnectionError(
            f"Could not connect to RCSB: {exc.reason}"
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        # Timeouts and dropped connections during the read are not URLErrors.
        raise ConnectionError(
            f"Download from RCSB failed for {pdb_id_upper}: {exc}"
        ) from exc

    # Write through a temporary file so an interrupted write never leaves a
    # partial structure in the cache.
    _write_atomically(cached, data)
    return cached
