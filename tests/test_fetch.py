from __future__ import annotations

from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import illustrate.io.rcsb as fetch_module
from illustrate.io.rcsb import fetch_pdb


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _install_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    monkeypatch.setattr(fetch_module, "_CACHE_DIR", cache)
    return cache


@pytest.mark.parametrize("bad_id", ["", "abc", "abcde", "ab cd", "ab-d", "ab\nd"])
def test_fetch_pdb_rejects_non_four_alphanumeric_id(bad_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_cache(monkeypatch, tmp_path)

    def fail_urlopen(url: str, timeout: float) -> None:
        raise AssertionError("urlopen must not be called for invalid IDs")

    monkeypatch.setattr(fetch_module, "urlopen", fail_urlopen)

    with pytest.raises(ValueError, match="PDB ID must be exactly 4 alphanumeric characters"):
        fetch_pdb(bad_id)


def test_fetch_pdb_normalizes_lowercase_id_and_writes_uppercase_cache_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _install_cache(monkeypatch, tmp_path)
    captured_urls: list[str] = []

    def fake_urlopen(url: str, timeout: float):
        captured_urls.append(url)
        return _Response(b"ATOM      1  CA  GLY A   1       0.000   0.000  -5.000\nEND\n")

    monkeypatch.setattr(fetch_module, "urlopen", fake_urlopen)

    path = fetch_pdb("2hhb")

    assert path == cache / "2HHB.pdb"
    assert captured_urls == ["https://files.rcsb.org/download/2HHB.pdb"]
    assert path.read_bytes().startswith(b"ATOM")


def test_fetch_pdb_returns_cached_file_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _install_cache(monkeypatch, tmp_path)
    cache.mkdir(parents=True)
    cached = cache / "1ABC.pdb"
    content = b"ATOM      1  CA  GLY A   1       0.000   0.000  -5.000\nEND   \n"
    cached.write_bytes(content)

    def fail_urlopen(url: str, timeout: float) -> None:
        raise AssertionError("urlopen must not be called when the cached file is complete")

    monkeypatch.setattr(fetch_module, "urlopen", fail_urlopen)

    path = fetch_pdb("1abc")

    assert path == cached
    assert path.read_bytes() == content


def test_fetch_pdb_redownloads_truncated_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _install_cache(monkeypatch, tmp_path)
    cache.mkdir(parents=True)
    (cache / "1ABC.pdb").write_bytes(b"ATOM      1  CA  GLY A   1       0.0")
    fresh = b"ATOM      1  CA  GLY A   1       0.000   0.000  -5.000\nEND\n"

    def fake_urlopen(url: str, timeout: float):
        return _Response(fresh)

    monkeypatch.setattr(fetch_module, "urlopen", fake_urlopen)

    path = fetch_pdb("1abc")

    assert path.read_bytes() == fresh
    assert sorted(p.name for p in cache.iterdir()) == ["1ABC.pdb"]


def test_fetch_pdb_maps_read_timeout_to_connection_error_without_caching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _install_cache(monkeypatch, tmp_path)

    class _StalledResp:
        def read(self) -> bytes:
            raise TimeoutError("The read operation timed out")

        def __enter__(self):
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def fake_urlopen(url: str, timeout: float) -> _StalledResp:
        return _StalledResp()

    monkeypatch.setattr(fetch_module, "urlopen", fake_urlopen)

    with pytest.raises(ConnectionError, match="Download from RCSB failed for 1ABC"):
        fetch_pdb("1abc")

    assert list(cache.iterdir()) == []


def test_fetch_pdb_redownloads_when_cached_file_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _install_cache(monkeypatch, tmp_path)
    cache.mkdir(parents=True)
    (cache / "1ABC.pdb").write_bytes(b"")

    def fake_urlopen(url: str, timeout: float):
        return _Response(b"FRESH")

    monkeypatch.setattr(fetch_module, "urlopen", fake_urlopen)

    path = fetch_pdb("1abc")

    assert path.read_bytes() == b"FRESH"


def test_fetch_pdb_maps_http_404_to_file_not_found_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_cache(monkeypatch, tmp_path)

    def fake_urlopen(url: str, timeout: float):
        raise HTTPError(url, 404, "Not Found", hdrs=Message(), fp=None)

    monkeypatch.setattr(fetch_module, "urlopen", fake_urlopen)

    with pytest.raises(FileNotFoundError, match="'9ZZZ' not found on RCSB"):
        fetch_pdb("9zzz")


def test_fetch_pdb_maps_http_500_to_connection_error_with_status_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_cache(monkeypatch, tmp_path)

    def fake_urlopen(url: str, timeout: float):
        raise HTTPError(url, 500, "Server Error", hdrs=Message(), fp=None)

    monkeypatch.setattr(fetch_module, "urlopen", fake_urlopen)

    with pytest.raises(ConnectionError, match=r"RCSB returned HTTP 500 for 2HHB"):
        fetch_pdb("2hhb")


def test_fetch_pdb_maps_url_error_to_connection_error_preserving_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_cache(monkeypatch, tmp_path)

    def fake_urlopen(url: str, timeout: float):
        raise URLError("name resolution failed")

    monkeypatch.setattr(fetch_module, "urlopen", fake_urlopen)

    with pytest.raises(ConnectionError, match="name resolution failed"):
        fetch_pdb("2hhb")


@pytest.mark.parametrize(("configured_timeout", "expected"), [(None, 30.0), ("   ", 30.0), ("7.5", 7.5)])
def test_fetch_pdb_passes_configured_timeout_to_urlopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_timeout: str | None, expected: float,
) -> None:
    _install_cache(monkeypatch, tmp_path)
    if configured_timeout is None:
        monkeypatch.delenv("ILLUSTRATE_RCSB_FETCH_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("ILLUSTRATE_RCSB_FETCH_TIMEOUT_SECONDS", configured_timeout)
    captured: dict[str, float] = {}

    def fake_urlopen(url: str, timeout: float):
        captured["timeout"] = timeout
        return _Response(b"X")

    monkeypatch.setattr(fetch_module, "urlopen", fake_urlopen)

    fetch_pdb("2hhb")

    assert captured["timeout"] == expected


@pytest.mark.parametrize("configured_timeout", ["0", "-5.0"])
def test_fetch_timeout_seconds_rejects_non_positive_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_timeout: str,
) -> None:
    _install_cache(monkeypatch, tmp_path)
    monkeypatch.setenv("ILLUSTRATE_RCSB_FETCH_TIMEOUT_SECONDS", configured_timeout)
    with pytest.raises(ValueError, match="ILLUSTRATE_RCSB_FETCH_TIMEOUT_SECONDS must be > 0"):
        fetch_pdb("2hhb")
