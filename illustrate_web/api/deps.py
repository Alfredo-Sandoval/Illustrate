"""Shared backend dependency helpers."""

from __future__ import annotations

from collections import deque
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import UploadFile

_UPLOAD_ROOT = Path(tempfile.gettempdir()) / "illustrate_uploads"
_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_UPLOAD_READ_CHUNK_SIZE = 1024 * 1024
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_STATE: dict[tuple[str, str], deque[float]] = {}
_DEFAULT_RATE_LIMITS: dict[str, tuple[int, float]] = {
    "render": (30, 60.0),
    "upload": (60, 60.0),
    "fetch": (30, 60.0),
    "suggest": (120, 60.0),
}


class RateLimitExceeded(RuntimeError):
    def __init__(self, *, scope: str, limit: int, window_seconds: float, retry_after_seconds: float) -> None:
        super().__init__(f"rate limit exceeded for {scope}: {limit} requests per {window_seconds:.0f}s")
        self.retry_after_seconds = max(1, int(retry_after_seconds))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = float(raw)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def upload_ttl_seconds() -> float:
    return _env_float("ILLUSTRATE_UPLOAD_TTL_SECONDS", 24.0 * 60.0 * 60.0)


def upload_max_bytes() -> int:
    value = _env_int("ILLUSTRATE_API_UPLOAD_MAX_BYTES", 16 * 1024 * 1024)
    if value <= 0:
        raise ValueError("ILLUSTRATE_API_UPLOAD_MAX_BYTES must be > 0")
    return value


def cleanup_uploads(*, now: float | None = None) -> int:
    ttl_seconds = upload_ttl_seconds()
    if ttl_seconds <= 0.0:
        return 0

    current_time = time.time() if now is None else now
    cutoff = current_time - ttl_seconds
    removed = 0
    for path in list(_UPLOAD_ROOT.iterdir()):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def _rate_limit_config(scope: str) -> tuple[int, float]:
    default_limit, default_window = _DEFAULT_RATE_LIMITS.get(scope, (0, 0.0))
    limit = _env_int(f"ILLUSTRATE_API_{scope.upper()}_RATE_LIMIT", default_limit)
    window = _env_float(f"ILLUSTRATE_API_{scope.upper()}_RATE_WINDOW_SECONDS", default_window)
    return limit, window


def enforce_rate_limit(scope: str, *, client_host: str | None) -> None:
    limit, window_seconds = _rate_limit_config(scope)
    if limit <= 0 or window_seconds <= 0.0:
        return

    host_key = (client_host or "local").strip() or "local"
    state_key = (scope, host_key)
    now = time.monotonic()
    cutoff = now - window_seconds

    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_STATE.setdefault(state_key, deque())
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = window_seconds - (now - bucket[0])
            raise RateLimitExceeded(
                scope=scope,
                limit=limit,
                window_seconds=window_seconds,
                retry_after_seconds=max(retry_after, 1.0),
            )
        bucket.append(now)


def upload_root() -> Path:
    return _UPLOAD_ROOT


def register_upload(data: bytes, suffix: str = ".pdb") -> str:
    cleanup_uploads()
    pdb_id = uuid.uuid4().hex
    path = _UPLOAD_ROOT / f"{pdb_id}{suffix}"
    path.write_bytes(data)
    return pdb_id


async def store_upload_file(upload: UploadFile, suffix: str = ".pdb") -> str:
    cleanup_uploads()
    pdb_id = uuid.uuid4().hex
    path = _UPLOAD_ROOT / f"{pdb_id}{suffix}"
    max_bytes = upload_max_bytes()
    written = 0
    try:
        with path.open("wb") as handle:
            while True:
                chunk = await upload.read(_UPLOAD_READ_CHUNK_SIZE)
                if not chunk:
                    break
                next_written = written + len(chunk)
                if next_written > max_bytes:
                    raise ValueError(f"uploaded file exceeds {max_bytes} bytes")
                handle.write(chunk)
                written = next_written
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return pdb_id


def get_upload_path(pdb_id: str) -> Path:
    cleanup_uploads()
    token = pdb_id.strip().lower()
    if not _UPLOAD_ID_RE.fullmatch(token):
        raise FileNotFoundError(f"Unknown pdb_id: {pdb_id}")

    matches = [path for path in _UPLOAD_ROOT.iterdir() if path.is_file() and path.stem.lower() == token]
    if len(matches) != 1:
        raise FileNotFoundError(f"Unknown pdb_id: {pdb_id}")
    return matches[0]
