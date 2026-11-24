"""Compact renderer backend status for the desktop toolbar."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QProcess, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QLabel, QWidget

from illustrate.gui.main import PREPARE_NUMBA_ARGUMENT
from illustrate.rendering.api import host_backend_candidates, prepare_numba_backend, resolve_render_backend
from illustrate.rendering.kernels import backend_available, supported_backends
from illustrate.rendering.pipeline import RenderCancellationToken, RenderCancelled

_BACKEND_ENV_VAR = "ILLUSTRATE_RENDER_BACKEND"

_BACKEND_LABELS = {
    "numpy": "NumPy",
    "numba": "Numba",
    "mlx": "MLX",
}


@dataclass(frozen=True)
class BackendAvailability:
    name: str
    label: str
    available: bool


@dataclass(frozen=True)
class BackendStatus:
    active: str | None
    source: str
    env_override: str | None
    backends: tuple[BackendAvailability, ...]
    error: str | None = None
    preparing_numba: bool = False

    @property
    def active_available(self) -> bool:
        if self.active is None:
            return False
        return any(
            backend.name == self.active and backend.available
            for backend in self.backends
        )


def _backend_display_name(name: str) -> str:
    normalized = str(name).strip().lower()
    return _BACKEND_LABELS.get(normalized, normalized or "unknown")


def _availability_rows(
    names: Iterable[str],
    available: Callable[[str], bool],
) -> tuple[BackendAvailability, ...]:
    rows: list[BackendAvailability] = []
    for raw_name in names:
        name = str(raw_name).strip().lower()
        if not name:
            continue
        try:
            is_available = bool(available(name))
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError):
            is_available = False
        rows.append(
            BackendAvailability(
                name=name,
                label=_backend_display_name(name),
                available=is_available,
            )
        )
    return tuple(rows)


def collect_backend_status(
    *,
    environ: Mapping[str, str] | None = None,
    supported: Callable[[], Iterable[str]] = supported_backends,
    available: Callable[[str], bool] = backend_available,
    resolve: Callable[[str | None], str] = resolve_render_backend,
) -> BackendStatus:
    env = os.environ if environ is None else environ
    raw_override = env.get(_BACKEND_ENV_VAR)
    env_override = raw_override.strip() if raw_override and raw_override.strip() else None
    source = "environment" if env_override is not None else "auto"

    rows = _availability_rows(supported(), available)
    try:
        active = str(resolve(env_override)).strip().lower()
    except ValueError as exc:
        return BackendStatus(
            active=None,
            source=source,
            env_override=env_override,
            backends=rows,
            error=str(exc),
        )
    except (RuntimeError, TypeError) as exc:
        active = env_override.strip().lower() if env_override is not None else None
        return BackendStatus(
            active=active,
            source=source,
            env_override=env_override,
            backends=rows,
            error=str(exc),
        )

    return BackendStatus(
        active=active,
        source=source,
        env_override=env_override,
        backends=rows,
        error=None,
    )


def format_backend_label(status: BackendStatus) -> str:
    if status.active is None:
        if status.env_override is not None:
            return "Backend: invalid"
        return "Backend: unavailable"

    label = _backend_display_name(status.active)
    if not status.active_available:
        return f"Backend: {label} unavailable"
    if status.preparing_numba:
        preparation = "preparing" if status.active == "numba" else "preparing Numba"
        return f"Backend: {label} · {preparation}"
    if status.error and status.error.startswith("Numba preparation failed:"):
        return f"Backend: {label} · Numba preparation failed"
    return f"Backend: {label}"


def _join_labels(backends: Iterable[BackendAvailability], *, available: bool) -> str:
    labels = [backend.label for backend in backends if backend.available is available]
    return ", ".join(labels) if labels else "none"


def format_backend_tooltip(status: BackendStatus) -> str:
    lines: list[str] = []
    if status.active is None:
        lines.append("Active render backend: unavailable.")
    else:
        availability = "available" if status.active_available else "unavailable"
        source = (
            f"{_BACKEND_ENV_VAR} override"
            if status.source == "environment"
            else "host auto-selection"
        )
        lines.append(
            "Active render backend: "
            f"{_backend_display_name(status.active)} ({availability}, {source})."
        )

    if status.env_override is not None:
        lines.append(f"{_BACKEND_ENV_VAR}={status.env_override}")
    if status.error:
        lines.append(f"Selection error: {status.error}")
    if status.preparing_numba:
        if status.active == "numpy":
            lines.append("Using NumPy while the Numba kernels are prepared.")
        else:
            lines.append("Rendering starts after the Numba kernels are prepared.")

    lines.append(f"Available: {_join_labels(status.backends, available=True)}")
    lines.append(f"Unavailable: {_join_labels(status.backends, available=False)}")
    return "\n".join(lines)


class BackendStatusWorker(QThread):
    """Resolve the desktop backend and prepare the Numba kernels."""

    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._preparing_numba = Event()
        self._preparation_finished = Event()
        self._preparation_error: str | None = None

    def startup_backend(self) -> str | None:
        """Choose the portable CPU backend while automatic Numba starts."""
        if os.environ.get(_BACKEND_ENV_VAR, "").strip():
            return None
        if self._preparation_finished.is_set():
            return "numpy" if self._preparation_error is not None else None
        if self._preparing_numba.is_set():
            return "numpy"
        candidate = next((name for name in host_backend_candidates() if backend_available(name)), None)
        return "numpy" if candidate == "numba" else None

    def render_backend(self, cancellation_token: RenderCancellationToken) -> str | None:
        """Let the render worker wait for explicit Numba without holding up Qt."""
        if os.environ.get(_BACKEND_ENV_VAR, "").strip().lower() != "numba":
            return self.startup_backend()
        while not self._preparation_finished.is_set():
            if cancellation_token.is_cancelled():
                raise RenderCancelled("render cancelled while preparing Numba")
            self._preparation_finished.wait(0.05)
        if cancellation_token.is_cancelled():
            raise RenderCancelled("render cancelled while preparing Numba")
        if self._preparation_error is not None:
            raise RuntimeError(self._preparation_error)
        return None

    def is_cancelled(self) -> bool:
        return self.isInterruptionRequested()

    def _build_numba_cache(self) -> None:
        # LLVM compilation can hold Python's GIL long enough to stall Qt
        # even when the compiler is called from a QThread.
        args = [PREPARE_NUMBA_ARGUMENT]
        if not getattr(sys, "frozen", False):
            # Numba keys the cache by the source path. Use the parent's
            # import root even if the current directory has a symlink alias.
            args[:0] = [
                "-B", "-c",
                "import sys; sys.path[0] = sys.argv.pop(1); "
                "from illustrate.gui.main import main; raise SystemExit(main())",
                str(Path(__file__).parents[3]),
            ]
        child = QProcess()
        child.start(sys.executable, args)
        if not child.waitForStarted():
            raise RuntimeError(f"Could not start Numba preparation: {child.errorString()}")
        while child.state() != QProcess.ProcessState.NotRunning:
            if self.is_cancelled():
                child.kill()
                child.waitForFinished()
                raise RenderCancelled("Numba preparation cancelled")
            child.waitForFinished(50)
        if child.exitStatus() != QProcess.ExitStatus.NormalExit or child.exitCode() != 0:
            error = bytes(child.readAllStandardError().data()).decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or f"Numba preparation exited with code {child.exitCode()}")

    def run(self) -> None:
        status: BackendStatus | None = None
        try:
            # The child verifies the provisional Numba choice and imports it
            # away from the desktop interpreter before loading cached code.
            if self.startup_backend() == "numpy":
                status = collect_backend_status(resolve=lambda override: override or "numba")
            else:
                status = collect_backend_status()
            if status.active == "numba" and status.active_available:
                self._preparing_numba.set()
                initial_backend = "numpy" if status.source == "auto" else "numba"
                self.result_ready.emit(replace(status, active=initial_backend, preparing_numba=True))
                self._build_numba_cache()
                prepare_numba_backend(self)
        except RenderCancelled:
            return
        except Exception as exc:
            if status is None or not self._preparing_numba.is_set():
                self.failed.emit(str(exc))
                return
            self._preparation_error = f"Numba preparation failed: {exc}"
            active = "numpy" if status.source == "auto" else "numba"
            status = replace(status, active=active, error=self._preparation_error)
        finally:
            self._preparation_finished.set()
        self.result_ready.emit(status)


class BackendStatusLabel(QLabel):
    """Show the selected backend and put availability details in the tooltip."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("backendStatusLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setAccessibleName("Render backend status")
        self.setText("Backend: checking…")

    @Slot(object)
    def show_status(self, status: BackendStatus) -> None:
        self.setText(format_backend_label(status))
        tooltip = format_backend_tooltip(status)
        self.setToolTip(tooltip)
        self.setAccessibleDescription(tooltip)

    @Slot(str)
    def show_error(self, message: str) -> None:
        self.setText("Backend: unavailable")
        self.setToolTip(message)
        self.setAccessibleDescription(message)


__all__ = [
    "BackendAvailability",
    "BackendStatus",
    "BackendStatusLabel",
    "BackendStatusWorker",
    "collect_backend_status",
    "format_backend_label",
    "format_backend_tooltip",
]
