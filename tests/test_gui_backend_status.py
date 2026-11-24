from __future__ import annotations

import importlib.util
import os

import pytest

if importlib.util.find_spec("PySide6") is None:
    raise pytest.skip.Exception(
        "PySide6 not installed; skipping GUI backend status tests",
        allow_module_level=True,
    )

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import illustrate.gui.widgets.backend_status as backend_module
from illustrate.gui.widgets.backend_status import (
    BackendAvailability,
    BackendStatus,
    BackendStatusLabel,
    BackendStatusWorker,
    collect_backend_status,
    format_backend_label,
    format_backend_tooltip,
)
from illustrate.rendering.pipeline import RenderCancellationToken, RenderCancelled


def _qapp() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


def test_backend_status_formats_auto_available_backend() -> None:
    status = collect_backend_status(
        environ={},
        supported=lambda: ("numpy", "mlx"),
        available=lambda name: name in {"numpy", "mlx"},
        resolve=lambda _backend: "mlx",
    )

    assert status.active == "mlx"
    assert status.source == "auto"
    assert status.active_available is True
    assert format_backend_label(status) == "Backend: MLX"

    tooltip = format_backend_tooltip(status)
    assert "Active render backend: MLX (available, host auto-selection)." in tooltip
    assert "Available: NumPy, MLX" in tooltip


def test_backend_status_marks_unavailable_env_override() -> None:
    status = collect_backend_status(
        environ={"ILLUSTRATE_RENDER_BACKEND": " mlx "},
        supported=lambda: ("numpy", "mlx"),
        available=lambda name: name == "numpy",
        resolve=lambda _backend: "mlx",
    )

    assert status.active == "mlx"
    assert status.source == "environment"
    assert status.env_override == "mlx"
    assert status.active_available is False
    assert format_backend_label(status) == "Backend: MLX unavailable"

    tooltip = format_backend_tooltip(status)
    assert "ILLUSTRATE_RENDER_BACKEND=mlx" in tooltip
    assert "Active render backend: MLX (unavailable, ILLUSTRATE_RENDER_BACKEND override)." in tooltip
    assert "Available: NumPy" in tooltip
    assert "Unavailable: MLX" in tooltip


def test_backend_status_reports_invalid_env_without_raising() -> None:
    def reject(_backend: str | None) -> str:
        raise ValueError("Unsupported render backend: 'bad'")

    status = collect_backend_status(
        environ={"ILLUSTRATE_RENDER_BACKEND": "bad"},
        supported=lambda: ("numpy", "mlx"),
        available=lambda name: name == "numpy",
        resolve=reject,
    )

    assert status.active is None
    assert status.error == "Unsupported render backend: 'bad'"
    assert format_backend_label(status) == "Backend: invalid"

    tooltip = format_backend_tooltip(status)
    assert "Active render backend: unavailable." in tooltip
    assert "Selection error: Unsupported render backend: 'bad'" in tooltip
    assert "Available: NumPy" in tooltip


def test_backend_status_label_displays_probe_result() -> None:
    _qapp()
    status = BackendStatus(
        active="numpy",
        source="auto",
        env_override=None,
        backends=(
            BackendAvailability(name="numpy", label="NumPy", available=True),
            BackendAvailability(name="mlx", label="MLX", available=False),
        ),
    )
    label = BackendStatusLabel()
    label.show_status(status)

    assert label.text() == "Backend: NumPy"
    assert "Active render backend: NumPy" in label.toolTip()
    assert "Unavailable: MLX" in label.toolTip()


@pytest.mark.parametrize("override", [None, "numba"])
@pytest.mark.parametrize("build_fails", [False, True])
def test_numba_startup_reports_preparation_result(
    monkeypatch: pytest.MonkeyPatch, override: str | None, build_fails: bool
) -> None:
    _qapp()
    if override is None:
        monkeypatch.delenv("ILLUSTRATE_RENDER_BACKEND", raising=False)
    else:
        monkeypatch.setenv("ILLUSTRATE_RENDER_BACKEND", override)
    status = BackendStatus(
        active="numba",
        source="auto" if override is None else "environment",
        env_override=override,
        backends=(BackendAvailability("numpy", "NumPy", True), BackendAvailability("numba", "Numba", True)),
    )
    activity: list[str] = []

    class Token:
        def __init__(self, cancelled: bool) -> None:
            self.cancelled = cancelled

        def is_cancelled(self) -> bool:
            return self.cancelled

    def collect_status(**_kwargs: object) -> BackendStatus:
        return status

    def backend_available(_name: str) -> bool:
        return True

    def load_cache(_token: RenderCancellationToken) -> None:
        activity.append("load")

    monkeypatch.setattr(backend_module, "collect_backend_status", collect_status)
    monkeypatch.setattr(backend_module, "host_backend_candidates", lambda: ("numba", "numpy"))
    monkeypatch.setattr(backend_module, "backend_available", backend_available)
    worker = BackendStatusWorker()
    labels: list[str] = []
    token = Token(False)

    def record_status(value: BackendStatus) -> None:
        labels.append(format_backend_label(value))

    worker.result_ready.connect(record_status)

    def build_cache() -> None:
        if override is None:
            assert worker.render_backend(token) == "numpy"
            assert labels == ["Backend: NumPy · preparing Numba"]
        else:
            assert labels == ["Backend: Numba · preparing"]
            with pytest.raises(RenderCancelled):
                worker.render_backend(Token(True))
        activity.append("build")
        if build_fails:
            raise RuntimeError("Cannot write the compilation cache")

    monkeypatch.setattr(worker, "_build_numba_cache", build_cache)
    monkeypatch.setattr(backend_module, "prepare_numba_backend", load_cache)
    worker.run()

    if build_fails:
        if override is None:
            assert worker.render_backend(token) == "numpy"
            assert labels[-1] == "Backend: NumPy · Numba preparation failed"
        else:
            with pytest.raises(RuntimeError, match="Cannot write the compilation cache"):
                worker.render_backend(token)
            assert labels[-1] == "Backend: Numba · Numba preparation failed"
        assert activity == ["build"]
    else:
        assert worker.render_backend(token) is None
        assert labels[-1] == "Backend: Numba"
        assert activity == ["build", "load"]
    assert len(labels) == 2
