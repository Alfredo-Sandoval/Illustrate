from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QFileDialog

from illustrate.core.types import SelectionRule
from illustrate.gui.render_params import rules_signature as default_rules_signature
from illustrate.gui.workers import (
    FetchJobFailure,
    FetchJobResult,
    FetchRequest,
    LoadJobFailure,
    LoadJobResult,
    LoadRequest,
)

if TYPE_CHECKING:
    from illustrate.gui.main_window import MainWindow


@dataclass(frozen=True)
class _AtomLoad:
    """The atom load in flight and what to do once its atoms arrive."""

    request_id: int
    rules_signature: str
    render_after: bool = False
    # Fit the view before that render; set only together with render_after.
    fit_after: bool = False


class StructureController(QObject):
    """Activate local or fetched structures and load their atoms for rendering.

    The window's atoms are current when ``window.atoms_signature`` matches the
    rules. Otherwise one load is in flight (``loading``) or, after a failure,
    none is and the atoms stay missing until something asks again.
    """

    def __init__(
        self,
        window: MainWindow,
        *,
        rules_signature: Callable[[list[SelectionRule]], str] = default_rules_signature,
    ) -> None:
        super().__init__(window)
        self._window = window
        self._rules_signature = rules_signature
        self._fetch_request_id = 0
        self._active_fetch_request_id: int | None = None
        self._load_request_id = 0
        self._load: _AtomLoad | None = None

    @property
    def loading(self) -> bool:
        return self._load is not None

    # Open / fetch

    def open_pdb(self) -> None:
        win = self._window
        path, _ = QFileDialog.getOpenFileName(
            win,
            "Open PDB",
            str(Path.home()),
            "PDB Files (*.pdb *.ent);;All Files (*)",
        )
        if not path:
            return
        self.activate_loaded_structure(
            path=path,
            title_name=Path(path).name,
            status_message=f"Loaded {Path(path).name}. Rendering...",
            render_after_load=True,
            fit_view=True,
        )

    def fetch_pdb(self) -> None:
        win = self._window
        pdb_id = win.pdb_id_input.text().strip()
        if not pdb_id:
            win.statusBar().showMessage("Enter a PDB ID first.")
            return
        if self._active_fetch_request_id is not None or win.fetch_worker.isRunning():
            win.statusBar().showMessage("Fetch already in progress...")
            return
        win.main_actions.fetch_pdb.setEnabled(False)
        win.statusBar().showMessage(f"Fetching {pdb_id}...")
        self._fetch_request_id += 1
        self._active_fetch_request_id = self._fetch_request_id
        win.fetch_worker.submit(FetchRequest(request_id=self._fetch_request_id, pdb_id=pdb_id))

    def abandon_fetch(self) -> None:
        """Ignore the result of any fetch still in flight (used when closing)."""
        self._active_fetch_request_id = None

    def _accept_fetch_result(self, request_id: int) -> bool:
        if self._active_fetch_request_id is None:
            self._window.main_actions.fetch_pdb.setEnabled(True)
            return False
        if request_id != self._active_fetch_request_id:
            return False
        self._active_fetch_request_id = None
        self._window.main_actions.fetch_pdb.setEnabled(True)
        return True

    def on_fetch_done(self, payload: FetchJobResult) -> None:
        if not self._accept_fetch_result(payload.request_id):
            return
        self.activate_loaded_structure(
            path=payload.path,
            title_name=f"{payload.pdb_id}.pdb",
            status_message=f"Fetched {payload.pdb_id}.pdb. Rendering...",
            render_after_load=True,
            fit_view=True,
        )

    def on_fetch_failed(self, payload: FetchJobFailure) -> None:
        win = self._window
        if self._accept_fetch_result(payload.request_id):
            win.statusBar().showMessage(f"Fetch failed: {payload.message}")

    def on_suggestion_selected(self, pdb_id: str) -> None:
        win = self._window
        win.pdb_id_input.setText(pdb_id)
        self.fetch_pdb()

    # Async load completion

    def _take_load(self, request_id: int) -> _AtomLoad | None:
        load = self._load
        if load is None or request_id != load.request_id:
            return None
        self._load = None
        self._window.sync_render_action()
        return load

    def on_atoms_loaded(self, payload: object) -> None:
        win = self._window
        if not isinstance(payload, LoadJobResult):
            return
        load = self._take_load(payload.request_id)
        if load is None:
            return
        win.atoms = payload.atoms
        win.atoms_signature = payload.rules_signature
        win.atom_match_counts = payload.match_counts
        win.update_rule_match_counts()
        win.schedule_render_dimensions_update()
        if load.fit_after:
            win.render_size.fit_view(render=True)
        elif load.render_after:
            win.render_frame()

    def on_atoms_failed(self, payload: object) -> None:
        if not isinstance(payload, LoadJobFailure):
            return
        if self._take_load(payload.request_id) is None:
            return
        self._window.statusBar().showMessage(f"Structure load failed: {payload.message}")

    # Activation / loading

    def activate_loaded_structure(
        self,
        *,
        path: str,
        title_name: str | None = None,
        status_message: str | None = None,
        render_after_load: bool = False,
        fit_view: bool = False,
    ) -> None:
        win = self._window
        self._active_fetch_request_id = None
        win.pdb_path = path
        win.setWindowTitle(title_name or Path(path).name)
        win.set_loaded_model_label(path)
        win.clear_last_result()
        win.atoms = None
        win.atoms_signature = ""
        win.atom_match_counts = None
        self._load = None
        self._ensure_atoms(
            status_message=status_message or "Loading structure...",
            render_after=render_after_load,
            fit_after=fit_view and render_after_load,
        )

    def load_atoms_if_needed(
        self,
        *,
        status_message: str | None = None,
        render_after_load: bool = False,
    ) -> bool:
        """Return whether the atoms match the current rules, starting a load if not.

        With ``render_after_load``, render when the load for these rules,
        whether started here or already running, delivers its atoms.
        """
        return self._ensure_atoms(
            status_message=status_message or "Updating structure...",
            render_after=render_after_load,
        )

    def _ensure_atoms(
        self,
        *,
        status_message: str,
        render_after: bool = False,
        fit_after: bool = False,
    ) -> bool:
        win = self._window
        if win.pdb_path is None:
            self._load = None
            win.sync_render_action()
            win.rule_panel.set_match_counts(None)
            return False

        rules = win.current_rules()
        signature = self._rules_signature(rules)
        if win.atoms is not None and signature == win.atoms_signature:
            # A load still running for other rules is no longer wanted.
            self._load = None
            win.sync_render_action()
            win.update_rule_match_counts()
            win.schedule_render_dimensions_update()
            return True

        previous = self._load
        if previous is not None and previous.rules_signature == signature:
            # Submitting it again would discard the running load, and the
            # output-size refresh asks every few milliseconds, so a load slower
            # than that would never finish.
            self._load = replace(
                previous,
                render_after=previous.render_after or render_after,
                fit_after=previous.fit_after or fit_after,
            )
            return False

        # A render or fit asked of a load these rules supersede still applies.
        self._load_request_id += 1
        self._load = _AtomLoad(
            request_id=self._load_request_id,
            rules_signature=signature,
            render_after=render_after or (previous is not None and previous.render_after),
            fit_after=fit_after or (previous is not None and previous.fit_after),
        )
        win.sync_render_action()
        win.statusBar().showMessage(status_message)
        win.schedule_render_dimensions_update()
        win.load_worker.submit(
            LoadRequest(
                pdb_path=win.pdb_path,
                rules=rules,
                rules_signature=signature,
                request_id=self._load_request_id,
            )
        )
        return False


__all__ = ["StructureController"]
