"""Background render worker for the desktop app."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from illustrate import RenderParams, RenderResult, render_from_atoms


@dataclass
class RenderRequest:
    params: RenderParams
    atoms: object
    request_id: int = 0


try:
    from PySide6.QtCore import QThread, Signal
except Exception:  # pragma: no cover - optional dependency
    QThread = object  # type: ignore
    Signal = None


if QThread is object:

    class RenderWorker:
        """No-op synchronous worker when Qt is unavailable."""

        def __init__(
            self,
            on_done: Callable[[RenderResult], None],
            on_error: Callable[[str], None] | None = None,
        ) -> None:
            self.on_done = on_done
            self.on_error = on_error

        def start(self, request: RenderRequest) -> None:
            try:
                result = render_from_atoms(request.atoms, request.params)
            except Exception as exc:  # pragma: no cover - depends on input validity
                if self.on_error is not None:
                    self.on_error(str(exc))
                else:
                    raise
            else:
                self.on_done(result)

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
                    result = render_from_atoms(request.atoms, request.params)
                except Exception as exc:
                    self.failed.emit(str(exc))
                    continue
                if request.request_id != 0:
                    self.finished.emit((request.request_id, result))
                else:
                    self.finished.emit(result)
