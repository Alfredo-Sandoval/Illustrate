from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QMessageBox

from illustrate import RenderParams
from illustrate.gui.updater import RELEASES_PAGE_URL, UpdateCheckResult
from illustrate.gui.workers import (
    SettingsIOFailure,
    SettingsIORequest,
    SettingsIOResult,
    SettingsIOWorker,
    UpdateCheckFailure,
    UpdateCheckWorker,
)

if TYPE_CHECKING:
    from illustrate.gui.main_window import MainWindow

class SettingsController(QObject):
    """Connect settings and release-check workers to desktop dialogs and actions."""

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self.settings_worker = SettingsIOWorker(self)
        self.settings_worker.finished.connect(self._on_settings_io_done)
        self.settings_worker.failed.connect(self._on_settings_io_failed)
        self.update_worker = UpdateCheckWorker(self)
        self.update_worker.finished.connect(self._on_update_check_done)
        self.update_worker.failed.connect(self._on_update_check_failed)

    # Settings IO

    def save_settings(self) -> None:
        win = self._window
        if win.pdb_path is None:
            win.statusBar().showMessage("Load a PDB before saving settings.")
            return
        try:
            params = win.build_params()
        except (TypeError, KeyError, ValueError) as exc:
            win.statusBar().showMessage(str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(
            win,
            "Save settings",
            str(Path.home() / "illustrate_settings.json"),
            "JSON settings (*.json)",
        )
        if not path:
            return
        if self.settings_worker.isRunning():
            win.statusBar().showMessage("Settings I/O already in progress...")
            return
        if self.settings_worker.submit(
            SettingsIORequest(path=Path(path), kind="save", params=params)
        ):
            win.main_actions.save_settings.setEnabled(False)
            win.main_actions.load_settings.setEnabled(False)
            win.statusBar().showMessage(f"Saving settings {path}...")

    def load_settings(self) -> None:
        win = self._window
        path, _ = QFileDialog.getOpenFileName(
            win,
            "Load settings",
            str(Path.home()),
            "JSON settings (*.json)",
        )
        if not path:
            return
        source = Path(path)
        if self.settings_worker.isRunning():
            win.statusBar().showMessage("Settings I/O already in progress...")
            return
        if self.settings_worker.submit(SettingsIORequest(path=source, kind="load")):
            win.main_actions.save_settings.setEnabled(False)
            win.main_actions.load_settings.setEnabled(False)
            win.statusBar().showMessage(f"Loading settings {source.name}...")

    def _on_settings_io_done(self, payload: object) -> None:
        win = self._window
        win.main_actions.save_settings.setEnabled(True)
        win.main_actions.load_settings.setEnabled(True)
        if not isinstance(payload, SettingsIOResult):
            return
        if payload.kind == "save":
            win.statusBar().showMessage(f"Saved settings {payload.path}")
            return
        if payload.params is None:
            QMessageBox.critical(
                win, "Load settings failed", "Settings payload was empty."
            )
            return
        try:
            self.apply_loaded_params(
                payload.params, source=payload.path, resolved_pdb=payload.resolved_pdb
            )
        except ValueError as exc:
            QMessageBox.critical(win, "Load settings failed", str(exc))
            win.statusBar().showMessage(f"Load settings failed: {exc}")

    def _on_settings_io_failed(self, payload: object) -> None:
        win = self._window
        win.main_actions.save_settings.setEnabled(True)
        win.main_actions.load_settings.setEnabled(True)
        if isinstance(payload, SettingsIOFailure):
            title = (
                "Save settings failed"
                if payload.kind == "save"
                else "Load settings failed"
            )
            QMessageBox.critical(win, title, payload.message)
            win.statusBar().showMessage(f"{title}: {payload.message}")
            return
        win.statusBar().showMessage("Settings I/O failed.")

    def check_for_updates(self) -> None:
        win = self._window
        if self.update_worker.isRunning():
            win.statusBar().showMessage("Update check already in progress...")
            return
        win.main_actions.check_updates.setEnabled(False)
        win.statusBar().showMessage("Checking for updates...")
        self.update_worker.start()

    def _on_update_check_done(self, result: UpdateCheckResult) -> None:
        win = self._window
        win.main_actions.check_updates.setEnabled(True)
        if result.status == "error":
            QMessageBox.information(
                win,
                "Check Updates",
                f"Could not check for updates.\n\n{result.message}",
            )
            return
        if result.status == "up_to_date":
            QMessageBox.information(win, "Check Updates", result.message)
            return

        latest = result.latest_version or "(unknown)"
        target_url = result.download_url or RELEASES_PAGE_URL
        reply = QMessageBox.question(
            win,
            "Update Available",
            (
                f"A newer version is available: {latest}\n"
                f"Current version: {result.current_version}\n\n"
                "Open the download page now?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            if not QDesktopServices.openUrl(target_url):
                win.statusBar().showMessage("Could not open the download page.")

    def _on_update_check_failed(self, payload: object) -> None:
        win = self._window
        win.main_actions.check_updates.setEnabled(True)
        message = payload.message if isinstance(payload, UpdateCheckFailure) else "unknown error"
        QMessageBox.information(
            win,
            "Check Updates",
            f"Could not check for updates.\n\n{message}",
        )

    def apply_loaded_params(
        self, params: RenderParams, *, source: Path, resolved_pdb: Path | None = None
    ) -> None:
        win = self._window
        win.apply_params_to_panels(params)

        loaded_pdb = str(params.pdb_path).strip()
        if resolved_pdb is not None:
            win.sync_render_size_controls()
            win.sync_viewport_orientation()
            win.structure_controller.activate_loaded_structure(
                path=str(resolved_pdb),
                status_message=f"Loaded settings from {source.name}",
                render_after_load=True,
            )
            return

        win.sync_render_size_controls()
        win.sync_viewport_orientation()
        win.schedule_render_dimensions_update()
        if loaded_pdb:
            win.statusBar().showMessage(
                f"Loaded settings from {source.name} (PDB not found: {loaded_pdb})"
            )
            if win.pdb_path is None:
                win.set_loaded_model_label(None)
        else:
            win.statusBar().showMessage(f"Loaded settings from {source.name}")


__all__ = ["SettingsController"]
