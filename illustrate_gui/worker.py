"""Background render worker for the desktop app."""

from __future__ import annotations

import platform
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable

from illustrate import AtomTable, RenderParams, RenderResult, render_from_atoms
from illustrate.fetch import fetch_pdb
from illustrate.pdb import load_pdb


def _desktop_render_backend() -> str | None:
    if sys.platform == "darwin" and platform.machine().strip().lower() in {"arm64", "aarch64"}:
        return "mlx"
    return None


@dataclass
class RenderRequest:
    params: RenderParams
    atoms: AtomTable
    request_id: int = 0
    interactive: bool = False


@dataclass(frozen=True)
class RenderJobResult:
    request_id: int
    interactive: bool
    result: RenderResult


@dataclass(frozen=True)
class RenderJobFailure:
    request_id: int
    interactive: bool
    message: str


@dataclass(frozen=True)
class LoadJobResult:
    request_id: int
    rules_signature: str
    render_after_load: bool
    atoms: AtomTable


@dataclass(frozen=True)
class LoadJobFailure:
    request_id: int
    rules_signature: str
    message: str


@dataclass
class LoadRequest:
    pdb_path: str
    rules: list[Any]
    rules_signature: str
    request_id: int = 0
    render_after_load: bool = False


class _CallbackSignal:
    def __init__(self) -> None:
        self._callbacks: list[Callable[..., None]] = []

    def connect(self, callback: Callable[..., None]) -> None:
        self._callbacks.append(callback)

    def emit(self, *args: Any) -> None:
        for callback in list(self._callbacks):
            callback(*args)


try:
    from PySide6.QtCore import QThread, Signal
    _HAS_QT_THREAD = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_QT_THREAD = False


if not _HAS_QT_THREAD:

    class RenderWorker:
        """No-op synchronous worker when Qt is unavailable."""

        def __init__(self) -> None:
            self.finished = _CallbackSignal()
            self.failed = _CallbackSignal()

        def submit(self, request: RenderRequest) -> None:
            self.start(request)

        def start(self, request: RenderRequest) -> None:
            try:
                result = render_from_atoms(
                    request.atoms,
                    request.params,
                    backend=_desktop_render_backend(),
                )
            except Exception as exc:  # pragma: no cover - depends on input validity
                if request.request_id != 0:
                    self.failed.emit(
                        RenderJobFailure(
                            request_id=request.request_id,
                            interactive=bool(request.interactive),
                            message=str(exc),
                        )
                    )
                else:
                    self.failed.emit(str(exc))
            else:
                payload: RenderResult | RenderJobResult
                if request.request_id != 0:
                    payload = RenderJobResult(
                        request_id=request.request_id,
                        interactive=bool(request.interactive),
                        result=result,
                    )
                else:
                    payload = result
                self.finished.emit(payload)

        def isRunning(self) -> bool:
            return False

        def wait(self) -> None:
            return None

    class LoadWorker:
        """No-op synchronous worker when Qt is unavailable."""

        def __init__(self) -> None:
            self.finished = _CallbackSignal()
            self.failed = _CallbackSignal()

        def submit(self, request: LoadRequest) -> None:
            self.start(request)

        def start(self, request: LoadRequest) -> None:
            try:
                atoms = load_pdb(request.pdb_path, request.rules)
            except Exception as exc:  # pragma: no cover - depends on input validity
                self.failed.emit(
                    LoadJobFailure(
                        request_id=request.request_id,
                        rules_signature=request.rules_signature,
                        message=str(exc),
                    )
                )
            else:
                self.finished.emit(
                    LoadJobResult(
                        request_id=request.request_id,
                        rules_signature=request.rules_signature,
                        render_after_load=bool(request.render_after_load),
                        atoms=atoms,
                    )
                )

        def isRunning(self) -> bool:
            return False

        def wait(self) -> None:
            return None

else:

    class RenderWorker(QThread):
        """Threaded render worker for async GUI rendering."""

        finished = Signal(object)
        failed = Signal(str)

        def __init__(self) -> None:
            super().__init__()
            self._request: RenderRequest | None = None
            self._request_lock = threading.Lock()

        def submit(self, request: RenderRequest) -> None:
            with self._request_lock:
                self._request = request
                should_start = not self.isRunning()
            if should_start:
                self.start()

        def _pop_request(self) -> RenderRequest | None:
            with self._request_lock:
                request = self._request
                self._request = None
            return request

        def run(self) -> None:
            while True:
                request = self._pop_request()
                if request is None:
                    return
                try:
                    result = render_from_atoms(
                        request.atoms,
                        request.params,
                        backend=_desktop_render_backend(),
                    )
                except Exception as exc:
                    if request.request_id != 0:
                        self.failed.emit(
                            RenderJobFailure(
                                request_id=request.request_id,
                                interactive=bool(request.interactive),
                                message=str(exc),
                            )
                        )
                    else:
                        self.failed.emit(str(exc))
                    continue
                if request.request_id != 0:
                    self.finished.emit(
                        RenderJobResult(
                            request_id=request.request_id,
                            interactive=bool(request.interactive),
                            result=result,
                        )
                    )
                else:
                    self.finished.emit(result)

    class LoadWorker(QThread):
        """Threaded structure load worker for async GUI structure loading."""

        finished = Signal(object)
        failed = Signal(object)

        def __init__(self) -> None:
            super().__init__()
            self._request: LoadRequest | None = None
            self._request_lock = threading.Lock()

        def submit(self, request: LoadRequest) -> None:
            with self._request_lock:
                self._request = request
                should_start = not self.isRunning()
            if should_start:
                self.start()

        def _pop_request(self) -> LoadRequest | None:
            with self._request_lock:
                request = self._request
                self._request = None
            return request

        def run(self) -> None:
            while True:
                request = self._pop_request()
                if request is None:
                    return
                try:
                    atoms = load_pdb(request.pdb_path, request.rules)
                except Exception as exc:
                    self.failed.emit(
                        LoadJobFailure(
                            request_id=request.request_id,
                            rules_signature=request.rules_signature,
                            message=str(exc),
                        )
                    )
                    continue
                self.finished.emit(
                    LoadJobResult(
                        request_id=request.request_id,
                        rules_signature=request.rules_signature,
                        render_after_load=bool(request.render_after_load),
                        atoms=atoms,
                    )
                )


if not _HAS_QT_THREAD:

    class FetchWorker:
        """No-op synchronous worker when Qt is unavailable."""

        def __init__(self) -> None:
            self.fetched = _CallbackSignal()
            self.failed = _CallbackSignal()

        def submit(self, pdb_id: str) -> None:
            try:
                path = fetch_pdb(pdb_id)
            except Exception as exc:  # pragma: no cover - depends on input validity
                self.failed.emit(str(exc))
            else:
                self.fetched.emit(str(path), str(pdb_id).strip().upper())

        def isRunning(self) -> bool:
            return False

        def wait(self) -> None:
            return None

else:

    class FetchWorker(QThread):
        """Threaded PDB fetch worker for async GUI fetches."""

        fetched = Signal(str, str)
        failed = Signal(str)

        def __init__(self) -> None:
            super().__init__()
            self._pdb_id: str | None = None
            self._request_lock = threading.Lock()

        def submit(self, pdb_id: str) -> None:
            normalized = str(pdb_id).strip()
            if normalized == "":
                raise ValueError("pdb_id must not be empty")
            with self._request_lock:
                if self.isRunning():
                    raise RuntimeError("fetch already in progress")
                self._pdb_id = normalized
            self.start()

        def _pop_request(self) -> str | None:
            with self._request_lock:
                pdb_id = self._pdb_id
                self._pdb_id = None
            return pdb_id

        def run(self) -> None:
            pdb_id = self._pop_request()
            if pdb_id is None:
                return
            try:
                path = fetch_pdb(pdb_id)
            except Exception as exc:
                self.failed.emit(str(exc))
                return
            self.fetched.emit(str(path), pdb_id.upper())
