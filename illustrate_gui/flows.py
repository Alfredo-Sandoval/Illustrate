"""Focused Qt flow controllers used by the desktop main window."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from illustrate.pdb import load_pdb
from illustrate.types import RenderParams
from illustrate_gui.worker import (
    LoadJobFailure,
    LoadJobResult,
    LoadRequest,
    RenderJobFailure,
    RenderJobResult,
    RenderRequest,
)

try:
    from PySide6.QtCore import QObject, QTimer
    from PySide6.QtWidgets import QFileDialog, QMessageBox
except Exception as exc:  # pragma: no cover
    raise SystemExit("PySide6 is required to run the desktop app") from exc


class StructureFlowController(QObject):
    """Own the file/fetch/model-loading orchestration for MainWindow."""

    def __init__(
        self,
        window: Any,
        *,
        atom_loader: Callable[[str, list[Any]], Any] = load_pdb,
        rules_signature: Callable[[list[Any]], str],
    ) -> None:
        super().__init__(window)
        self._window = window
        self._atom_loader = atom_loader
        self._rules_signature = rules_signature

    def open_pdb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Open PDB",
            str(Path.home()),
            "PDB Files (*.pdb *.ent);;All Files (*)",
        )
        if not path:
            return
        self.activate_loaded_structure(
            path=path,
            title_name=Path(path).name,
            status_message=f"Loaded {Path(path).name}. Click Render.",
        )

    def fetch_pdb(self) -> None:
        window = self._window
        pdb_id = window.pdb_id_input.text().strip()
        if not pdb_id:
            window.statusBar().showMessage("Enter a PDB ID first.")
            return
        worker = window.fetch_worker
        if hasattr(worker, "isRunning") and worker.isRunning():
            window.statusBar().showMessage("Fetch already in progress...")
            return
        window.fetch_action.setEnabled(False)
        window.statusBar().showMessage(f"Fetching {pdb_id}...")
        try:
            worker.submit(pdb_id)
        except Exception as exc:
            window.fetch_action.setEnabled(True)
            window.statusBar().showMessage(f"Fetch failed: {exc}")

    def on_fetch_done(self, path: str, pdb_id: str) -> None:
        self._window.fetch_action.setEnabled(True)
        self.activate_loaded_structure(
            path=path,
            title_name=f"{pdb_id}.pdb",
            status_message=f"Fetched {pdb_id}.pdb. Rendering...",
            render_after_load=True,
        )

    def on_fetch_failed(self, message: str) -> None:
        self._window.fetch_action.setEnabled(True)
        self._window.statusBar().showMessage(f"Fetch failed: {message}")

    def on_suggestion_selected(self, pdb_id: str) -> None:
        self._window.pdb_id_input.setText(pdb_id)
        self.fetch_pdb()

    def activate_loaded_structure(
        self,
        *,
        path: str,
        title_name: str | None = None,
        status_message: str | None = None,
        render_after_load: bool = False,
    ) -> None:
        window = self._window
        window.pdb_path = path
        window.setWindowTitle(f"Illustrate - {title_name or Path(path).name}")
        window._set_loaded_model_label(path)
        window._clear_last_result()
        window._atoms = None
        window._atoms_signature = ""
        window.viewport.set_preview_scene(None, None, None)
        window._clear_interactive_preview()
        self.request_atoms_load(status_message=status_message, render_after_load=render_after_load)

    def load_atoms_if_needed(self) -> None:
        window = self._window
        if window.pdb_path is None:
            window._atoms_loading = False
            window.rule_panel.set_match_counts(None)
            return
        if getattr(window, "_atoms_loading", False) and window._atoms is None:
            return
        rules = window._current_rules()
        signature = self._rules_signature(rules)
        if window._atoms is not None and signature == window._atoms_signature:
            window._atoms_loading = False
            window._push_preview_scene()
            window._update_rule_match_counts()
            window._schedule_render_dimensions_update()
            return
        window._atoms = self._atom_loader(window.pdb_path, rules)
        window._atoms_signature = signature
        window._atoms_loading = False
        window._push_preview_scene()
        window._update_rule_match_counts()
        window._schedule_render_dimensions_update()

    def request_atoms_load(
        self,
        *,
        status_message: str | None = None,
        render_after_load: bool = False,
    ) -> None:
        window = self._window
        if window.pdb_path is None:
            window.rule_panel.set_match_counts(None)
            return

        rules = window._current_rules()
        signature = self._rules_signature(rules)
        if window._atoms is not None and signature == window._atoms_signature:
            window._atoms_loading = False
            window._push_preview_scene()
            window._update_rule_match_counts()
            window._schedule_render_dimensions_update()
            window._load_render_after_load = False
            if render_after_load:
                window._render()
            return

        worker = window.load_worker
        if hasattr(worker, "isRunning") and worker.isRunning():
            window.statusBar().showMessage("Structure load already in progress...")
        window._atoms_loading = True
        window._load_render_after_load = bool(render_after_load)
        window._load_request_id += 1
        request_id = window._load_request_id
        window._latest_load_request_id = request_id
        window.render_action.setEnabled(False)
        if status_message:
            window.statusBar().showMessage(status_message)
        else:
            window.statusBar().showMessage("Loading structure...")
        window._schedule_render_dimensions_update()
        try:
            worker.submit(
                LoadRequest(
                    pdb_path=window.pdb_path,
                    rules=rules,
                    rules_signature=signature,
                    request_id=request_id,
                    render_after_load=render_after_load,
                )
            )
        except Exception as exc:
            window._atoms_loading = False
            window.render_action.setEnabled(True)
            window.statusBar().showMessage(f"Structure load failed: {exc}")
            return

    def on_atoms_loaded(self, payload: object) -> None:
        if not isinstance(payload, LoadJobResult):
            return

        window = self._window
        if payload.request_id != window._latest_load_request_id:
            return

        window._atoms_loading = False
        window.render_action.setEnabled(True)
        window._atoms = payload.atoms
        window._atoms_signature = payload.rules_signature
        window._push_preview_scene()
        window._update_rule_match_counts()
        window._schedule_render_dimensions_update()
        if payload.render_after_load or window._load_render_after_load:
            window._load_render_after_load = False
            window._render()

    def on_atoms_failed(self, payload: object) -> None:
        if not isinstance(payload, LoadJobFailure):
            return

        window = self._window
        if payload.request_id != window._latest_load_request_id:
            return

        window._atoms_loading = False
        window._load_render_after_load = False
        window.render_action.setEnabled(True)
        window.statusBar().showMessage(f"Structure load failed: {payload.message}")

    def apply_loaded_params(self, params: RenderParams, *, source: Path) -> None:
        window = self._window
        window._suspend_panel_callbacks = True
        try:
            transform = params.transform
            window.transform_panel.set_value(
                scale=transform.scale,
                xrot=transform.rotations[2][1] if len(transform.rotations) > 2 else 0.0,
                yrot=transform.rotations[1][1] if len(transform.rotations) > 1 else 0.0,
                zrot=transform.rotations[0][1] if len(transform.rotations) > 0 else 90.0,
                xtran=transform.translate[0],
                ytran=transform.translate[1],
                ztran=transform.translate[2],
            )
            world = params.world
            window.world_panel.set_value(
                background="#%02x%02x%02x" % tuple(int(c * 255) for c in world.background),
                fog="#%02x%02x%02x" % tuple(int(c * 255) for c in world.fog_color),
                fog_front=world.fog_front,
                fog_back=world.fog_back,
                shadows=world.shadows,
                shadow_strength=world.shadow_strength,
                shadow_angle=world.shadow_angle,
                shadow_min_z=world.shadow_min_z,
                shadow_max_dark=world.shadow_max_dark,
            )
            outlines = params.outlines
            window.outline_panel.set_value(
                enabled=outlines.enabled,
                contour_low=outlines.contour_low,
                contour_high=outlines.contour_high,
                kernel=outlines.kernel,
                z_diff_min=outlines.z_diff_min,
                z_diff_max=outlines.z_diff_max,
                subunit_low=outlines.subunit_low,
                subunit_high=outlines.subunit_high,
                residue_low=outlines.residue_low,
                residue_high=outlines.residue_high,
                residue_diff=outlines.residue_diff,
            )
            window.rule_panel.set_value(params.rules)
            if params.world.width > 0 and params.world.height > 0:
                window.render_size_mode_combo.setCurrentText("Custom")
                window.render_width_spin.setValue(int(params.world.width))
                window.render_height_spin.setValue(int(params.world.height))
            else:
                window.render_size_mode_combo.setCurrentText("Auto")
        finally:
            window._suspend_panel_callbacks = False

        loaded_pdb = str(params.pdb_path).strip()
        loaded_file = None
        if loaded_pdb:
            candidate = Path(loaded_pdb)
            if not candidate.is_absolute():
                candidate = (source.parent / candidate).resolve()
            if candidate.exists():
                loaded_file = candidate

        if loaded_file is not None:
            window.pdb_path = str(loaded_file)
            window.setWindowTitle(f"Illustrate - {loaded_file.name}")
            window._set_loaded_model_label(window.pdb_path)
            window._clear_last_result()
            window._atoms = None
            window._atoms_signature = ""
            window._atoms_loading = False
            window._load_render_after_load = True
            window.viewport.set_preview_scene(None, None, None)
            window._clear_interactive_preview()
            window._sync_render_size_controls()
            window._sync_preview_transform()
            window._sync_preview_style()
            self.request_atoms_load(status_message=f"Loaded settings from {source.name}", render_after_load=True)
            return

        window._sync_render_size_controls()
        window._sync_preview_transform()
        window._sync_preview_style()
        window._request_preview_render()
        window._schedule_render_dimensions_update()
        if loaded_file is None and loaded_pdb:
            window.statusBar().showMessage(f"Loaded settings from {source.name} (PDB not found: {loaded_pdb})")
            if window.pdb_path is None:
                window._set_loaded_model_label(None)
        else:
            window.statusBar().showMessage(f"Loaded settings from {source.name}")


class PreviewFlowController(QObject):
    """Own the coalesced preview render loop for MainWindow."""

    def __init__(self, window: Any, *, interactive_settle_ms: int) -> None:
        super().__init__(window)
        self._window = window
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.submit_pending_render)
        self.interactive_settle_timer = QTimer(self)
        self.interactive_settle_timer.setSingleShot(True)
        self.interactive_settle_timer.timeout.connect(self.on_interactive_settle_timeout)
        self._interactive_settle_ms = interactive_settle_ms

    def clear_interactive_preview(self) -> None:
        window = self._window
        window._preview_request_id += 1
        window._latest_preview_request_id = window._preview_request_id
        window._preview_pending = False
        self.preview_timer.stop()
        window.viewport.update_preview_image(None)

    def request_render(self) -> None:
        window = self._window
        window._preview_pending = True
        if self.preview_timer.isActive():
            return
        mode = window._preview_quality_mode()
        delay_ms = 14 if mode == "fast" else (24 if mode == "high" else 20)
        self.preview_timer.start(delay_ms)

    def submit_pending_render(self) -> None:
        window = self._window
        if not window._preview_pending:
            return
        window._preview_pending = False
        if window.pdb_path is None:
            return
        if window._atoms is None:
            window._load_atoms_if_needed()
        if window._atoms is None:
            return
        try:
            params = window._build_preview_params()
        except Exception:
            return
        window._preview_request_id += 1
        request_id = window._preview_request_id
        window._latest_preview_request_id = request_id
        request = RenderRequest(params=params, atoms=window._atoms, request_id=request_id)
        worker = window.preview_worker
        if hasattr(worker, "submit"):
            worker.submit(request)
        else:
            worker.start(request)

    def push_preview_scene(self) -> None:
        window = self._window
        if window._atoms is None or int(getattr(window._atoms, "n", 0)) <= 0:
            window.viewport.set_preview_scene(None, None, None)
            window._clear_interactive_preview()
            return

        n = int(window._atoms.n)
        coords = np.asarray(window._atoms.coord[1 : n + 1], dtype=np.float32)
        atom_types = np.asarray(window._atoms.type_idx[1 : n + 1], dtype=np.int32)
        if coords.shape[0] == 0:
            window.viewport.set_preview_scene(None, None, None)
            window._clear_interactive_preview()
            return

        rules = window._current_rules()
        colortype = np.full((len(rules) + 1, 3), 0.74, dtype=np.float32)
        radtype = np.full((len(rules) + 1,), 1.2, dtype=np.float32)
        for idx, rule in enumerate(rules, start=1):
            colortype[idx] = np.asarray(rule.color, dtype=np.float32)
            radtype[idx] = max(0.0, float(rule.radius))

        atom_types = np.clip(atom_types, 0, len(rules))
        colors = colortype[atom_types]
        radii = radtype[atom_types]
        visible = np.isfinite(coords).all(axis=1) & np.isfinite(radii) & (radii > 0.01)
        if not np.any(visible):
            window.viewport.set_preview_scene(None, None, None)
            window._clear_interactive_preview()
            return

        window.viewport.set_preview_scene(coords[visible], colors[visible], radii[visible])
        window._clear_interactive_preview()

    def schedule_interactive_settle_render(self) -> None:
        if not self._window.auto_render_on_drag.isChecked():
            return
        self.interactive_settle_timer.start(self._interactive_settle_ms)

    def on_interactive_settle_timeout(self) -> None:
        if not self._window.auto_render_on_drag.isChecked():
            return
        self._window._render(interactive=False)

    def on_preview_done(self, payload: object) -> None:
        if isinstance(payload, RenderJobResult):
            request_id = int(payload.request_id)
            result = payload.result
        elif isinstance(payload, tuple) and len(payload) == 2:
            request_id_raw, result = payload
            if not isinstance(request_id_raw, (int, np.integer, str)):
                return
            request_id = int(request_id_raw)
        else:
            return
        if request_id != self._window._latest_preview_request_id:
            return
        if not hasattr(result, "rgb"):
            return
        rgb = np.asarray(result.rgb, dtype=np.uint8)
        opacity = np.asarray(result.opacity, dtype=np.uint8) if hasattr(result, "opacity") else None
        self._window._display_preview_image(rgb, opacity)

    def on_preview_failed(self, payload: object) -> None:
        if isinstance(payload, RenderJobFailure):
            if payload.request_id != self._window._latest_preview_request_id:
                return
        self._window.viewport.update_preview_image(None)


class RenderFlowController(QObject):
    """Own the asynchronous render submission/completion flow for MainWindow."""

    def __init__(self, window: Any) -> None:
        super().__init__(window)
        self._window = window

    def render_interactive(self) -> None:
        self._window._render(interactive=True)
        self._window._schedule_interactive_settle_render()

    def render(self, *, interactive: bool = False) -> None:
        window = self._window
        if window.pdb_path is None:
            window.statusBar().showMessage("Load a PDB first.")
            return
        if not interactive and window._interactive_settle_timer.isActive():
            window._interactive_settle_timer.stop()
        try:
            params = window._build_interactive_rerender_params() if interactive else window._build_params()
        except Exception as exc:
            window.statusBar().showMessage(str(exc))
            return

        window._load_atoms_if_needed()
        if window._atoms is None:
            if getattr(window, "_atoms_loading", False):
                window.statusBar().showMessage("Structure is still loading...")
            else:
                window.statusBar().showMessage("Could not load atoms from PDB.")
            return

        if not interactive:
            window.render_action.setEnabled(False)
            window._set_render_busy(True)

        window._render_request_id += 1
        request = RenderRequest(
            params=params,
            atoms=window._atoms,
            request_id=window._render_request_id,
            interactive=interactive,
        )
        worker = window.worker
        if hasattr(worker, "submit"):
            worker.submit(request)
        else:
            worker.start(request)
        window.statusBar().showMessage("Re-rendering..." if interactive else "Rendering...")

    def on_render_done(self, result: object) -> None:
        window = self._window
        render_mode = "full"
        request_id: int | None = None
        if isinstance(result, RenderJobResult):
            render_mode = "interactive" if result.interactive else "full"
            request_id = int(result.request_id)
            result = result.result
        elif isinstance(result, tuple) and len(result) == 2:
            _request_id, payload = result
            if isinstance(_request_id, (int, np.integer, str)):
                request_id = int(_request_id)
            result = payload
        if request_id is not None and request_id != window._render_request_id:
            return
        if render_mode == "full":
            window.render_action.setEnabled(True)
        elapsed = window._render_elapsed_suffix()
        if render_mode == "full":
            window._set_render_busy(False)
        window.copy_action.setEnabled(True)
        if render_mode == "full":
            window._params_dirty = False
        window._update_render_btn_style()
        if not hasattr(result, "rgb"):
            window.statusBar().showMessage(f"Render returned no image{elapsed}.")
            return
        if render_mode == "full":
            window._last_result = result
        rgb = np.asarray(result.rgb, dtype=np.uint8)
        opacity = np.asarray(result.opacity, dtype=np.uint8) if hasattr(result, "opacity") else None
        window._display_image(rgb, opacity)
        window._clear_interactive_preview()
        window._schedule_render_dimensions_update()
        verb = "Re-rendered" if render_mode == "interactive" else "Rendered"
        width = int(getattr(result, "width", 0))
        height = int(getattr(result, "height", 0))
        window.statusBar().showMessage(f"{verb} {width}x{height}{elapsed}")

    def on_render_failed(self, payload: object) -> None:
        window = self._window
        render_mode = "full"
        message = str(payload)
        if isinstance(payload, RenderJobFailure):
            if payload.request_id != window._render_request_id:
                return
            render_mode = "interactive" if payload.interactive else "full"
            message = payload.message
        if render_mode == "full":
            window.render_action.setEnabled(True)
        elapsed = window._render_elapsed_suffix()
        if render_mode == "full":
            window._set_render_busy(False)
            QMessageBox.critical(window, "Render failed", message)
        window.statusBar().showMessage(f"Render failed{elapsed}")
