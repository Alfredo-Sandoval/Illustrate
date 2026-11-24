from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from illustrate.gui.updater import check_for_updates


@dataclass(frozen=True)
class UpdateCheckFailure:
    message: str


class UpdateCheckWorker(QThread):
    """Fetch the release-check result in a worker thread and emit it to Qt."""

    finished = Signal(object)
    failed = Signal(object)

    def run(self) -> None:
        try:
            result = check_for_updates()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.failed.emit(UpdateCheckFailure(message=str(exc)))
            return
        self.finished.emit(result)


__all__ = ["UpdateCheckFailure", "UpdateCheckWorker"]
