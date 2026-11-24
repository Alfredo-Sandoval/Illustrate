from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QThread, Signal

from illustrate.io.image_export import export_png, export_svg


ExportKind = Literal["png", "svg"]


@dataclass(frozen=True)
class ExportRequest:
    path: str
    kind: ExportKind
    rgb: NDArray[np.uint8]
    opacity: NDArray[np.uint8] | None


@dataclass(frozen=True)
class ExportJobResult:
    path: str
    kind: ExportKind


@dataclass(frozen=True)
class ExportJobFailure:
    path: str
    kind: ExportKind
    message: str


class ExportWorker(QThread):
    """Write one image at a time and reject submissions while busy."""

    finished = Signal(object)
    failed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._request: ExportRequest | None = None

    def submit(self, request: ExportRequest) -> bool:
        if self.isRunning():
            return False
        self._request = request
        self.start()
        return True

    def run(self) -> None:
        request = self._request
        self._request = None
        if request is None:
            return
        try:
            if request.kind == "png":
                export_png(request.path, request.rgb, request.opacity)
            else:
                export_svg(request.path, request.rgb, request.opacity)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.failed.emit(
                ExportJobFailure(
                    path=request.path,
                    kind=request.kind,
                    message=str(exc),
                )
            )
            return
        self.finished.emit(ExportJobResult(path=request.path, kind=request.kind))


__all__ = [
    "ExportJobFailure",
    "ExportJobResult",
    "ExportKind",
    "ExportRequest",
    "ExportWorker",
]
