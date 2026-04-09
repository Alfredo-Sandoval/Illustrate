from __future__ import annotations

import importlib
from io import BytesIO
import os
import time
from typing import Generator

from fastapi.testclient import TestClient
from PIL import Image as PILImage
import pytest

from illustrate_web.api import deps
from illustrate_web.api.main import app
from illustrate_web.api.routes import render as render_route
from illustrate_web.api.routes import suggest as suggest_route


def _upload_minimal_pdb() -> str:
    atom_line = (
        f"{'ATOM':<6}{1:5d} {'CA':<4}{' ':1}{'GLY':>3} {'A':1}{1:4d}    "
        f"{0.000:8.3f}{0.000:8.3f}{-5.000:8.3f}"
    )
    payload = atom_line + "\nEND\n"

    response = client.post(
        "/api/upload-pdb",
        files={"file": ("mini.pdb", payload, "text/plain")},
    )
    assert response.status_code == 200
    return response.json()["pdb_id"]


def _cleanup_upload(pdb_id: str) -> None:
    for file_path in deps.upload_root().glob(f"{pdb_id}*"):
        file_path.unlink(missing_ok=True)


client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_api_rate_limits() -> Generator[None, None, None]:
    with deps._RATE_LIMIT_LOCK:
        deps._RATE_LIMIT_STATE.clear()
    yield
    with deps._RATE_LIMIT_LOCK:
        deps._RATE_LIMIT_STATE.clear()


def test_health_ok() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_presets_endpoint_returns_list() -> None:
    response = client.get("/api/presets")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) > 0


def test_pdb_suggest_endpoint_returns_results(monkeypatch) -> None:
    expected = [
        {"pdb_id": "2HHB", "title": "Hemoglobin"},
        {"pdb_id": "1A3N", "title": "Hemoglobin beta"},
    ]

    def fake_suggest(query: str) -> list[dict[str, str]]:
        assert query == "2hhb"
        return expected

    monkeypatch.setattr(suggest_route, "_suggest_pdb", fake_suggest)

    response = client.get("/api/pdb-suggest", params={"q": "2hhb"})
    assert response.status_code == 200
    assert response.json() == expected


def test_pdb_suggest_endpoint_rejects_short_query() -> None:
    response = client.get("/api/pdb-suggest", params={"q": "a"})
    assert response.status_code == 422


def test_suggest_rate_limit_ignores_forwarded_header_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ILLUSTRATE_API_TRUST_PROXY_HEADERS", raising=False)
    monkeypatch.setenv("ILLUSTRATE_API_SUGGEST_RATE_LIMIT", "1")
    monkeypatch.setenv("ILLUSTRATE_API_SUGGEST_RATE_WINDOW_SECONDS", "60")
    monkeypatch.setattr(suggest_route, "_suggest_pdb", lambda _query: [])

    first = client.get("/api/pdb-suggest", params={"q": "2hhb"}, headers={"X-Forwarded-For": "1.1.1.1"})
    second = client.get("/api/pdb-suggest", params={"q": "2hhb"}, headers={"X-Forwarded-For": "2.2.2.2"})

    assert first.status_code == 200
    assert second.status_code == 429


def test_suggest_rate_limit_uses_forwarded_header_when_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILLUSTRATE_API_TRUST_PROXY_HEADERS", "1")
    monkeypatch.setenv("ILLUSTRATE_API_SUGGEST_RATE_LIMIT", "1")
    monkeypatch.setenv("ILLUSTRATE_API_SUGGEST_RATE_WINDOW_SECONDS", "60")
    monkeypatch.setattr(suggest_route, "_suggest_pdb", lambda _query: [])

    first = client.get("/api/pdb-suggest", params={"q": "2hhb"}, headers={"X-Forwarded-For": "1.1.1.1"})
    second = client.get("/api/pdb-suggest", params={"q": "2hhb"}, headers={"X-Forwarded-For": "2.2.2.2"})

    assert first.status_code == 200
    assert second.status_code == 200


def test_upload_then_render_invalid_output_format_returns_400() -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.5, 0.0],
                    "radius": 1.5,
                },
            ],
            "transform": {
                "scale": 10.0,
            },
            "world": {
                "width": 20,
                "height": 20,
            },
            "outlines": {
                "enabled": False,
            },
            "output_format": "gif",
        }

        response = client.post("/api/render", json=render_payload)

        assert response.status_code == 400
        assert "Unsupported output format" in response.json()["detail"]
    finally:
        _cleanup_upload(pdb_id)


def test_upload_pdb_writes_file_to_upload_root() -> None:
    atom_line = (
        f"{'ATOM':<6}{1:5d} {'CA':<4}{' ':1}{'GLY':>3} {'A':1}{1:4d}    "
        f"{0.000:8.3f}{0.000:8.3f}{-5.000:8.3f}"
    )
    payload = atom_line + "\nEND\n"

    response = client.post(
        "/api/upload-pdb",
        files={"file": ("mini.pdb", payload, "text/plain")},
    )
    assert response.status_code == 200

    pdb_id = response.json()["pdb_id"]
    found = list(deps.upload_root().glob(f"{pdb_id}*"))
    assert len(found) == 1
    assert found[0].name.startswith(f"{pdb_id}.")
    _cleanup_upload(pdb_id)


def test_upload_pdb_rejects_payloads_over_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILLUSTRATE_API_UPLOAD_MAX_BYTES", "16")

    response = client.post(
        "/api/upload-pdb",
        files={"file": ("mini.pdb", "x" * 17, "text/plain")},
    )

    assert response.status_code == 413
    assert "uploaded file exceeds" in response.json()["detail"]


def test_render_unknown_pdb_id_returns_400() -> None:
    response = client.post(
        "/api/render",
        json={
            "pdb_id": "does-not-exist",
            "rules": [],
            "output_format": "png",
        },
    )

    assert response.status_code == 400
    assert "Unknown pdb_id" in response.json()["detail"]


def test_render_rejects_non_exact_or_pattern_pdb_ids() -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        short_id_response = client.post(
            "/api/render",
            json={
                "pdb_id": pdb_id[:8],
                "rules": [],
                "output_format": "png",
            },
        )
        assert short_id_response.status_code == 400
        assert "Unknown pdb_id" in short_id_response.json()["detail"]

        wildcard_response = client.post(
            "/api/render",
            json={
                "pdb_id": "*",
                "rules": [],
                "output_format": "png",
            },
        )
        assert wildcard_response.status_code == 400
        assert "Unknown pdb_id" in wildcard_response.json()["detail"]
    finally:
        _cleanup_upload(pdb_id)


def test_render_rejects_translate_with_wrong_arity() -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.2, 0.2],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "transform": {
                "scale": 12.0,
                "translate": [0.0],
            },
            "output_format": "png",
        }

        response = client.post("/api/render", json=render_payload)
        assert response.status_code == 422
    finally:
        _cleanup_upload(pdb_id)


def test_render_rejects_invalid_autocenter_with_422() -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.2, 0.2],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "transform": {
                "autocenter": "banana",
            },
            "output_format": "png",
        }

        response = client.post("/api/render", json=render_payload)
        assert response.status_code == 422
        assert "unsupported autocenter mode" in response.json()["detail"]
    finally:
        _cleanup_upload(pdb_id)


def test_render_with_valid_payload_returns_image_bytes() -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.2, 0.2],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "transform": {
                "scale": 12.0,
            },
            "outlines": {
                "enabled": False,
            },
            "output_format": "png",
        }

        response = client.post("/api/render", json=render_payload)

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert isinstance(response.content, (bytes, bytearray))
        assert response.content.startswith(b"\x89PNG")
    finally:
        _cleanup_upload(pdb_id)


def test_render_rejects_rules_with_invalid_color_tuple() -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.5],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "output_format": "png",
        }

        response = client.post("/api/render", json=render_payload)

        assert response.status_code == 400
        assert "Each rule must provide a three-element color tuple." in response.json()["detail"]
    finally:
        _cleanup_upload(pdb_id)


def test_render_short_descriptor_no_longer_crashes_server() -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.2, 0.2],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "output_format": "png",
        }

        response = client.post("/api/render", json=render_payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
    finally:
        _cleanup_upload(pdb_id)


def test_render_accepts_portable_pixmap_alias() -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.2, 0.2],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "transform": {
                "scale": 12.0,
            },
            "output_format": "image/x-portable-pixmap",
        }

        response = client.post("/api/render", json=render_payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/x-portable-pixmap"
        assert isinstance(response.content, (bytes, bytearray))
    finally:
        _cleanup_upload(pdb_id)


def test_render_png_response_preserves_alpha_channel() -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.2, 0.2],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "transform": {
                "scale": 12.0,
            },
            "outlines": {
                "enabled": False,
            },
            "output_format": "png",
        }

        response = client.post("/api/render", json=render_payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        with PILImage.open(BytesIO(response.content)) as image:
            assert image.mode == "RGBA"
    finally:
        _cleanup_upload(pdb_id)


def test_render_reuses_cached_atoms_for_identical_rules(monkeypatch) -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        with render_route._ATOM_CACHE_LOCK:
            render_route._ATOM_CACHE.clear()

        calls = {"count": 0}
        real_load_pdb = render_route._load_pdb

        def counting_load_pdb(pdb_path: str, rules):
            calls["count"] += 1
            return real_load_pdb(pdb_path, rules)

        monkeypatch.setattr(render_route, "_load_pdb", counting_load_pdb)

        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.2, 0.2],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "transform": {
                "scale": 12.0,
            },
            "output_format": "png",
        }

        response_one = client.post("/api/render", json=render_payload)
        response_two = client.post("/api/render", json=render_payload)
        assert response_one.status_code == 200
        assert response_two.status_code == 200
        assert calls["count"] == 1
    finally:
        with render_route._ATOM_CACHE_LOCK:
            render_route._ATOM_CACHE.clear()
        _cleanup_upload(pdb_id)


def test_render_reloads_cached_atoms_when_upload_changes(monkeypatch) -> None:
    pdb_id = _upload_minimal_pdb()
    try:
        with render_route._ATOM_CACHE_LOCK:
            render_route._ATOM_CACHE.clear()

        calls = {"count": 0}
        real_load_pdb = render_route._load_pdb

        def counting_load_pdb(pdb_path: str, rules):
            calls["count"] += 1
            return real_load_pdb(pdb_path, rules)

        monkeypatch.setattr(render_route, "_load_pdb", counting_load_pdb)

        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.2, 0.2],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "transform": {
                "scale": 12.0,
            },
            "output_format": "png",
        }

        response_one = client.post("/api/render", json=render_payload)
        assert response_one.status_code == 200

        upload_path = deps.get_upload_path(pdb_id)
        upload_path.write_text(
            (
                f"{'ATOM':<6}{1:5d} {'CA':<4}{' ':1}{'GLY':>3} {'A':1}{1:4d}    "
                f"{0.000:8.3f}{0.000:8.3f}{-6.000:8.3f}\nEND\n"
            ),
            encoding="utf-8",
        )

        response_two = client.post("/api/render", json=render_payload)
        assert response_two.status_code == 200
        assert calls["count"] == 2
    finally:
        with render_route._ATOM_CACHE_LOCK:
            render_route._ATOM_CACHE.clear()
        _cleanup_upload(pdb_id)


def test_upload_cleanup_prunes_expired_files(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(deps, "_UPLOAD_ROOT", tmp_path)
    monkeypatch.setenv("ILLUSTRATE_UPLOAD_TTL_SECONDS", "1")

    expired = tmp_path / ("a" * 32 + ".pdb")
    expired.write_bytes(b"old")
    stale_time = time.time() - 10.0
    os.utime(expired, (stale_time, stale_time))

    new_id = deps.register_upload(b"fresh", ".pdb")

    assert not expired.exists()
    assert len(list(tmp_path.glob(f"{new_id}*"))) == 1


def test_render_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILLUSTRATE_API_RENDER_RATE_LIMIT", "1")
    monkeypatch.setenv("ILLUSTRATE_API_RENDER_RATE_WINDOW_SECONDS", "60")

    pdb_id = _upload_minimal_pdb()
    try:
        render_payload = {
            "pdb_id": pdb_id,
            "rules": [
                {
                    "record_name": "ATOM",
                    "descriptor": "----------",
                    "res_low": 0,
                    "res_high": 9999,
                    "color": [1.0, 0.2, 0.2],
                    "radius": 1.5,
                },
            ],
            "world": {
                "width": 20,
                "height": 20,
            },
            "output_format": "png",
        }

        first = client.post("/api/render", json=render_payload)
        second = client.post("/api/render", json=render_payload)

        assert first.status_code == 200
        assert second.status_code == 429
        assert "rate limit exceeded" in second.json()["detail"]
        assert int(second.headers["retry-after"]) >= 1
    finally:
        _cleanup_upload(pdb_id)


def test_timeout_helpers_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch_module = importlib.import_module("illustrate.fetch")

    monkeypatch.setenv("ILLUSTRATE_RCSB_FETCH_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("ILLUSTRATE_RCSB_SUGGEST_TIMEOUT_SECONDS", "3.5")

    assert fetch_module._fetch_timeout_seconds() == 12.5
    assert suggest_route._suggest_timeout_seconds() == 3.5
