from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, QThread, Signal

from illustrate import RenderParams
from illustrate.io.params_json import load_settings, save_settings


SettingsIOKind = Literal["load", "save"]


@dataclass(frozen=True)
class SettingsIORequest:
    path: Path
    kind: SettingsIOKind
    params: RenderParams | None = None


@dataclass(frozen=True)
class SettingsIOResult:
    path: Path
    kind: SettingsIOKind
    params: RenderParams | None = None
    resolved_pdb: Path | None = None


@dataclass(frozen=True)
class SettingsIOFailure:
    path: Path
    kind: SettingsIOKind
    message: str


def _resolve_saved_structure(params: RenderParams, settings_path: Path) -> Path | None:
    """Resolve a saved PDB path relative to its settings file and check existence."""
    saved_path = params.pdb_path.strip()
    if not saved_path:
        return None
    candidate = Path(saved_path)
    if not candidate.is_absolute():
        candidate = (settings_path.parent / candidate).resolve()
    return candidate if candidate.exists() else None


class SettingsIOWorker(QThread):
    """Perform one settings read or write at a time, rejecting work while busy."""

    finished = Signal(object)
    failed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._request: SettingsIORequest | None = None

    def submit(self, request: SettingsIORequest) -> bool:
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
            if request.kind == "save":
                if request.params is None:
                    raise ValueError("settings save request is missing params")
                output_path = save_settings(request.params, request.path)
                self.finished.emit(SettingsIOResult(path=output_path, kind="save"))
                return
            params = load_settings(request.path)
            resolved_pdb = _resolve_saved_structure(params, request.path)
        except (OSError, TypeError, KeyError, ValueError) as exc:
            self.failed.emit(
                SettingsIOFailure(
                    path=request.path,
                    kind=request.kind,
                    message=str(exc),
                )
            )
            return
        self.finished.emit(
            SettingsIOResult(
                path=request.path, kind="load", params=params, resolved_pdb=resolved_pdb
            )
        )


__all__ = [
    "SettingsIOFailure",
    "SettingsIOKind",
    "SettingsIORequest",
    "SettingsIOResult",
    "SettingsIOWorker",
]
