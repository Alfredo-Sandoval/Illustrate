from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QMessageBox

from illustrate.core.types import RenderResult
from illustrate.gui.workers import (
    RenderJobFailure,
    RenderJobResult,
    RenderRequest,
)
from illustrate.gui.workers.render import RenderJobStale

if TYPE_CHECKING:
    from illustrate.gui.main_window import MainWindow


class RenderController(QObject):
    """Submit full and interactive renders and reject stale worker results."""

    def __init__(self, window: MainWindow) -> None:
        super().__init__(window)
        self._window = window
        self._active_full_request_id: int | None = None
        # Viewport drags keep one render in flight. Moves that arrive meanwhile
        # collapse into one follow-up render of the latest controls, so every
        # frame completes instead of being cancelled by the next mouse event.
        self._drag_request_id: int | None = None
        self._drag_render_pending = False
        window.worker.stale.connect(self.on_render_stale)

    # Submission

    def sync_render_action(self) -> None:
        """Enable Render unless atoms are loading or a full render is running."""
        win = self._window
        win.main_actions.render.setEnabled(
            not win.atoms_loading and self._active_full_request_id is None
        )

    def invalidate(self) -> None:
        """Discard full and interactive renders belonging to the previous state."""
        win = self._window
        win.render_request_id += 1
        self._active_full_request_id = None
        self._drag_request_id = None
        self._drag_render_pending = False
        win.worker.cancel_pending()
        win.set_render_busy(False)
        self.sync_render_action()

    def request_drag_render(self) -> None:
        """Request a frame and coalesce further edits while one is in flight."""
        if self._drag_request_id is not None:
            self._drag_render_pending = True
            return
        self.render(interactive=True)

    def _finish_drag_frame(self, request_id: int | None) -> None:
        if request_id is None or request_id != self._drag_request_id:
            return
        self._drag_request_id = None
        if self._drag_render_pending:
            self._drag_render_pending = False
            self.render(interactive=True)

    def render(self, _checked: bool = False, *, interactive: bool = False) -> None:
        win = self._window
        if win.pdb_path is None:
            win.statusBar().showMessage("Load a PDB first.")
            return
        params = win.build_params()

        atoms_ready = win.load_atoms_if_needed(
            status_message="Loading structure before render...",
            render_after_load=True,
        )
        if atoms_ready is False or win.atoms is None:
            if win.atoms_loading:
                win.statusBar().showMessage("Structure is still loading...")
            else:
                win.statusBar().showMessage("Could not load atoms from PDB.")
            return

        win.render_request_id += 1
        request_id = int(win.render_request_id)
        if interactive:
            self._drag_request_id = request_id
        else:
            self._active_full_request_id = request_id
            self._drag_request_id = None
            self._drag_render_pending = False
            win.set_render_busy(True)
            self.sync_render_action()
        win.worker.submit(
            RenderRequest(
                params=params,
                atoms=win.atoms,
                request_id=request_id,
                interactive=interactive,
            )
        )
        if not interactive:
            win.statusBar().showMessage("Rendering...")

    # Worker callbacks

    def on_render_done(self, payload: RenderJobResult | RenderResult) -> None:
        win = self._window
        if isinstance(payload, RenderJobResult):
            render_mode = "interactive" if payload.interactive else "full"
            request_id = payload.request_id
            frame = payload.frame
            result = payload.result
        else:
            render_mode = "full"
            request_id = None
            frame = None
            result = payload
        if request_id is not None and self._is_stale(request_id):
            self._handle_stale_render(request_id, render_mode)
            return
        elapsed = win.render_elapsed_suffix()
        if render_mode == "full":
            self._active_full_request_id = None
            win.set_render_busy(False)
            self.sync_render_action()
        # Drag frames are full-quality renders of the controls they were
        # submitted with, so they commit exactly like a Render click.
        win.main_actions.copy_to_clipboard.setEnabled(True)
        win.set_render_outcome(current=True)
        win.last_result = result
        if frame is not None:
            win.display_image_frame(frame)
        else:
            win.display_image(result.rgb, result.opacity)
        win.schedule_render_dimensions_update()
        win.statusBar().showMessage(f"Rendered {result.width}x{result.height}{elapsed}")
        self._finish_drag_frame(request_id)

    def on_render_failed(self, payload: RenderJobFailure | str) -> None:
        win = self._window
        render_mode = "full"
        if isinstance(payload, RenderJobFailure):
            render_mode = "interactive" if payload.interactive else "full"
            if self._is_stale(payload.request_id):
                self._handle_stale_render(payload.request_id, render_mode)
                return
            message = payload.message
        else:
            message = payload
        elapsed = win.render_elapsed_suffix()
        if render_mode == "full":
            self._active_full_request_id = None
            win.set_render_busy(False)
            self.sync_render_action()
        # The view still shows the previous image, which no longer matches the
        # controls, so say why instead of reporting the render as up to date.
        win.set_render_outcome(current=False, error=str(message))
        win.statusBar().showMessage(f"Render failed: {message}{elapsed}")
        if render_mode == "full":
            QMessageBox.critical(win, "Render failed", str(message))
        if isinstance(payload, RenderJobFailure):
            self._finish_drag_frame(payload.request_id)

    def on_render_stale(self, payload: RenderJobStale) -> None:
        render_mode = "interactive" if payload.interactive else "full"
        self._handle_stale_render(payload.request_id, render_mode)

    def _is_stale(self, request_id: int) -> bool:
        # Every submission and invalidation raises the window's request ID.
        return int(request_id) < self._window.render_request_id

    def _handle_stale_render(self, request_id: int, render_mode: str) -> None:
        if render_mode != "full":
            self._finish_drag_frame(request_id)
            return
        if self._active_full_request_id != int(request_id):
            return
        win = self._window
        self._active_full_request_id = None
        win.set_render_busy(False)
        self.sync_render_action()
        win.update_render_button_style()
        win.statusBar().showMessage("Render superseded.")


__all__ = ["RenderController"]
