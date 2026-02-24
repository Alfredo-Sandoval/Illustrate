"""Fetch PDB files from RCSB by ID."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_CACHE_DIR = Path.home() / ".cache" / "illustrate"
_PDB_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")


def fetch_pdb(pdb_id: str) -> Path:
    """Download a PDB file from RCSB, returning the local cached path.

    Parameters
    ----------
    pdb_id : str
        A 4-character alphanumeric PDB identifier (e.g. ``"2hhb"``).

    Returns
    -------
    Path
        Path to the cached ``.pdb`` file.

    Raises
    ------
    ValueError
        If *pdb_id* is not exactly 4 alphanumeric characters.
    FileNotFoundError
        If RCSB returns 404 (structure does not exist).
    ConnectionError
        On network failure.
    """
    pdb_id = pdb_id.strip()
    if not _PDB_ID_RE.match(pdb_id):
        raise ValueError(
            f"PDB ID must be exactly 4 alphanumeric characters, got {pdb_id!r}"
        )

    pdb_id_upper = pdb_id.upper()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = _CACHE_DIR / f"{pdb_id_upper}.pdb"

    if cached.exists() and cached.stat().st_size > 0:
        return cached

    url = _RCSB_URL.format(pdb_id=pdb_id_upper)
    try:
        with urlopen(url, timeout=30) as resp:  # noqa: S310
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

    cached.write_bytes(data)
    return cached
