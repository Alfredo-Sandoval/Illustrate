"""Desktop PySide6 application for interactive rendering."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from illustrate import RenderParams
from illustrate.core.types import AtomTable, RenderResult, SelectionRule
from illustrate.rendering.api import max_render_scale
from illustrate.presets import default_rules
from illustrate.gui.controllers import (
    RenderController,
    SettingsController,
    StructureController,
)
from illustrate.gui.image_frame import (
    GuiImageFrame,
    pixmap_from_frame,
    pixmap_from_image,
)
from illustrate.gui.panels.outlines import OutlinesPanel
from illustrate.gui.panels.rules import RulePanel
from illustrate.gui.panels.transform import TransformPanel
from illustrate.gui.panels.world import WorldPanel
from illustrate.gui.render_params import (
    rules_signature,
    transform_panel_values,
    render_params_from_panels,
)
from illustrate.gui.pdb_completer import PdbCompleter
from illustrate.gui.actions import MainActions, build_main_actions
from illustrate.gui.themes import apply_appearance
from illustrate.gui.viewport import RenderViewport
from illustrate.gui.widgets import (
    RenderBar,
    build_main_menus,
    build_output_size_controls,
    fill_navigation,
)
from illustrate.gui.widgets.backend_status import BackendStatusLabel, BackendStatusWorker
from illustrate.gui.widgets.render_size_toolbar import RenderSizeToolbar
from illustrate.gui.workers import (
    ExportKind,
    ExportJobFailure,
    ExportJobResult,
    ExportRequest,
    ExportWorker,
    FetchWorker,
    LoadWorker,
    RenderWorker,
)
from illustrate.gui.workers.presets import PresetSaveRequest, PresetWorker
from illustrate.io.preset_store import (
    CustomPresetLoadResult,
    builtin_preset_items,
    format_custom_preset_load_issues,
)


from illustrate.gui.kit import AppShell, set_attention
from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QCloseEvent, QGuiApplication, QIcon, QKeySequence, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QWidget,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QInputDialog,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
)


_NO_STRUCTURE_TITLE = "No structure loaded"


def _check_control_range(name: str, control: object, value: float | str) -> None:
    if isinstance(control, (QDoubleSpinBox, QSpinBox)):
        number = float(value)
        if not math.isfinite(number) or not control.minimum() <= number <= control.maximum():
            raise ValueError(
                f"GUI settings require {name} between {control.minimum():g} and {control.maximum():g}; got {value}."
            )
    elif isinstance(control, QComboBox) and control.findText(str(value)) < 0:
        raise ValueError(f"GUI settings do not support {name}={value}.")


class MainWindow(AppShell):
    def create_actions(self) -> MainActions:
        actions = build_main_actions(self, {
            "open_pdb": self.structure_controller.open_pdb,
            "fetch_pdb": self.structure_controller.fetch_pdb,
            "save_preset": self._save_custom_preset,
            "load_settings": self._settings_controller.load_settings,
            "save_settings": self._settings_controller.save_settings,
            "check_updates": self._settings_controller.check_for_updates,
            "render": self.render_frame,
            "fit_view": self._fit_view,
            "export_png": self._export_png,
            "export_svg": self._export_svg,
            "copy_to_clipboard": self._copy_to_clipboard,
        })
        # Nothing to copy until the first render.
        actions.copy_to_clipboard.setEnabled(False)
        return actions

    def __init__(self) -> None:
        super().__init__("Illustrate")
        self.setWindowTitle("Illustrate")
        self.setMinimumSize(1100, 700)
        self.resize(1360, 860)

        icons = Path(__file__).parent / "icons"
        icon_path = icons / ("icon.icns" if sys.platform == "darwin" else "icon.png")
        if icon_path.exists():
            app_icon = QIcon(str(icon_path))
            self.setWindowIcon(app_icon)
            app_instance = QApplication.instance()
            if isinstance(app_instance, QApplication):
                app_instance.setWindowIcon(app_icon)

        self.pdb_path: str | None = None
        self.atoms: AtomTable | None = None
        self.atoms_signature = ""
        self.last_result: RenderResult | None = None
        # True while a viewport drag step is writing the transform panel.
        self._viewport_drag_change = False
        # True while the window, not the user, is writing the panels.
        self.panel_callbacks_suspended = True
        # Set only through set_render_outcome.
        self.params_dirty = False
        self.render_error: str | None = None
        self.atom_match_counts: list[int] | None = None
        self._render_btn: QPushButton | None = None
        self._render_started_at: float | None = None
        self.render_request_id = 0
        self._last_pixmap = None
        self._preset_items: list[tuple[str, RenderParams]] = []
        self._builtin_presets = builtin_preset_items()
        self._custom_presets: list[tuple[str, RenderParams]] = []
        self._startup_started = False
        # Workers
        self.fetch_worker = FetchWorker(self)

        self.export_worker = ExportWorker(self)
        self.export_worker.finished.connect(self._on_export_done)
        self.export_worker.failed.connect(self._on_export_failed)

        self.load_worker = LoadWorker(self)
        self._backend_status_worker = BackendStatusWorker(self)
        self.worker = RenderWorker(self, startup_backend=self._backend_status_worker.render_backend)

        # Controllers own the flow logic. They read the workers above, and the
        # workers report straight to them.
        self.structure_controller = StructureController(
            self, rules_signature=rules_signature
        )
        self._render_controller = RenderController(self)
        self._settings_controller = SettingsController(self)
        self.fetch_worker.fetched.connect(self.structure_controller.on_fetch_done)
        self.fetch_worker.failed.connect(self.structure_controller.on_fetch_failed)
        self.load_worker.result_ready.connect(self.structure_controller.on_atoms_loaded)
        self.load_worker.failed.connect(self.structure_controller.on_atoms_failed)
        self.worker.result_ready.connect(self._render_controller.on_render_done)
        self.worker.failed.connect(self._render_controller.on_render_failed)
        # Retries a deferred close until every worker thread has stopped.
        self._closing = False
        self._close_retry_timer = QTimer(self)
        self._close_retry_timer.setInterval(100)
        self._close_retry_timer.timeout.connect(self.close)

        self.main_actions = self.create_actions()
        self.main_actions.save_preset.setEnabled(False)
        self._preset_worker = PresetWorker(self)
        self._preset_worker.loaded.connect(self._on_custom_presets_loaded)
        self._preset_worker.saved.connect(self._on_custom_preset_saved)
        self._preset_worker.failed.connect(self._on_preset_io_failed)
        self._preset_worker.finished.connect(self._on_preset_io_finished)

        # The shared shell: navigation dock on the left, the viewport workspace
        # with its render bar in the center, controls on the right.
        self.viewport = RenderViewport()
        self.viewport.rotation_requested.connect(self._on_viewport_rotation)

        self._navigation = fill_navigation(self.navigation, self.main_actions)
        self.pdb_id_input = self._navigation.pdb_search
        self._pdb_completer = PdbCompleter(self.pdb_id_input, parent=self)
        self._pdb_completer.activated.connect(self.structure_controller.on_suggestion_selected)
        self.preset_list = self._navigation.preset_list
        self.preset_list.currentRowChanged.connect(self._apply_preset)
        self._refresh_preset_list(default_index=3)
        self.toggle_sidebar_action = self.navigation.toggleViewAction()
        self.toggle_sidebar_action.setText("Sidebar")
        self.toggle_sidebar_action.setShortcut(QKeySequence("Ctrl+B"))

        self.render_bar = RenderBar(self.main_actions, self)
        self._render_btn = self.render_bar.render_button
        self.auto_render_on_drag = self.render_bar.auto_render_checkbox
        self.render_dims_label = self.render_bar.detail

        output = build_output_size_controls()
        self.render_size_mode_combo = output.mode_combo
        self.render_width_spin = output.width_spin
        self.render_height_spin = output.height_spin
        self.render_size = RenderSizeToolbar(
            self,
            mode_combo=output.mode_combo,
            width_spin=output.width_spin,
            height_spin=output.height_spin,
            dims_label=self.render_dims_label,
            build_params=self.build_params,
        )
        for size_signal in (
            output.mode_combo.currentTextChanged,
            output.width_spin.valueChanged,
            output.height_spin.valueChanged,
        ):
            size_signal.connect(self.render_size.on_size_changed)

        self.transform_panel = TransformPanel(on_changed=self._transform_changed)
        self.world_panel = WorldPanel(on_changed=self._panel_changed)
        self.outline_panel = OutlinesPanel(on_changed=self._panel_changed)
        self.inspector.add_section("Transform", self.transform_panel)
        self.inspector.add_section("World / Lighting", self.world_panel, expanded=False)
        self.inspector.add_section("Outlines", self.outline_panel, expanded=False)
        self.inspector.add_section("Output", output.panel)
        self.toggle_inspector_action = self.inspector.toggleViewAction()
        self.toggle_inspector_action.setText("Controls")
        self.toggle_inspector_action.setShortcut(QKeySequence("Ctrl+I"))

        # Selection Rules dock (bottom, hidden by default). The seven-column
        # table needs the full window width. The panel draws its own heading
        # and close button, so the dock has no title bar.
        self.rule_panel = RulePanel(default_rules(), on_changed=self._panel_changed)
        self.rules_dock = QDockWidget("Selection Rules", self)
        self.rules_dock.setObjectName("rulesDock")
        self.rules_dock.setWidget(self.rule_panel)
        self.rules_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.rules_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.rules_dock.setTitleBarWidget(QWidget(self.rules_dock))
        self.rule_panel.btn_close.clicked.connect(self.rules_dock.hide)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.rules_dock)
        self.resizeDocks([self.rules_dock], [260], Qt.Orientation.Vertical)
        self.toggle_rules_action = self.rules_dock.toggleViewAction()
        self.toggle_rules_action.setText("Selection Rules")
        self.rules_dock.hide()

        self.workspace.set_title(_NO_STRUCTURE_TITLE)
        self.workspace.add_header_action(self.main_actions.fit_view, icon="fit")
        self.workspace.add_header_action(self.toggle_rules_action, icon="list", text="Rules")
        self.add_inspector_toggle()
        self.workspace.set_body(self.viewport)
        self.workspace.set_action_bar(self.render_bar)
        self.resizeDocks(
            [self.navigation, self.inspector], [240, 340], Qt.Orientation.Horizontal
        )
        app_instance = QApplication.instance()
        if not isinstance(app_instance, QApplication):
            raise RuntimeError("MainWindow requires a QApplication.")
        build_main_menus(self, app_instance)

        status = self.statusBar()
        status.setObjectName("mainStatusBar")
        status.setSizeGripEnabled(False)
        # Inset the permanent widgets from the window edge; Qt already insets
        # the message on the left.
        status.setContentsMargins(0, 0, 10, 0)
        self.backend_status_label = BackendStatusLabel(self)
        self.backend_status_label.setObjectName("backendStatus")
        self._backend_status_worker.result_ready.connect(self.backend_status_label.show_status)
        self._backend_status_worker.failed.connect(self.backend_status_label.show_error)
        self._render_progress = QProgressBar(self)
        self._render_progress.setRange(0, 0)
        self._render_progress.setFixedWidth(140)
        self._render_progress.setTextVisible(False)
        self._render_progress.hide()
        status.addPermanentWidget(self.backend_status_label)
        status.addPermanentWidget(self._render_progress)
        status.showMessage("Ready")
        self.panel_callbacks_suspended = False

        # Start on the chain-colored preset.
        self._apply_preset(self.preset_list.currentRow())
        self.render_size.sync_controls_enabled()
        self.sync_viewport_orientation()
        self.render_size.schedule_dimensions_update()
        self._refresh_render_heading()
        # Keep the PDB search field from taking keyboard focus at startup.
        self.viewport.setFocus()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._startup_started:
            return
        self._startup_started = True
        self._preset_worker.start()
        self._backend_status_worker.start()

    # Panel callbacks

    def _panel_changed(self, _value: object) -> None:
        if isinstance(_value, list) and hasattr(self, "transform_panel"):
            # Rules set the largest atom, which caps Scale. Apply the cap even
            # while callbacks are suspended so loaded settings and presets get it.
            rules = cast(list[SelectionRule], _value)
            self.transform_panel.set_scale_limit(max_render_scale(rules))
        if self.panel_callbacks_suspended:
            return
        if isinstance(_value, list) and hasattr(self, "rule_panel"):
            self.rule_panel.set_match_counts(None)
        if hasattr(self, "transform_panel") and hasattr(self, "viewport"):
            self.sync_viewport_orientation()
            self.schedule_render_dimensions_update()
        if self._viewport_drag_change:
            # The drag render that follows replaces the image; invalidating here
            # would cancel the frame already in flight.
            return
        self.mark_params_dirty()
        self.statusBar().showMessage("Controls changed. Click Render.")

    def _transform_changed(self, value: object) -> None:
        # Editing a transform field is the same operation as dragging the view,
        # so it follows the same live-render setting instead of waiting for Render.
        live = (
            not self.panel_callbacks_suspended
            and not self._viewport_drag_change
            and self.pdb_path is not None
            and self.auto_render_on_drag.isChecked()
        )
        if not live:
            self._panel_changed(value)
            return
        with self._viewport_drag_step():
            self._panel_changed(value)
        self._render_controller.request_drag_render()

    @contextmanager
    def suspended_panel_callbacks(self) -> Generator[None, None, None]:
        """Write the panels without treating the change as a user edit."""
        previous = self.panel_callbacks_suspended
        self.panel_callbacks_suspended = True
        try:
            yield
        finally:
            self.panel_callbacks_suspended = previous

    @contextmanager
    def _viewport_drag_step(self, active: bool = True) -> Generator[None, None, None]:
        previous = self._viewport_drag_change
        self._viewport_drag_change = active
        try:
            yield
        finally:
            self._viewport_drag_change = previous

    @property
    def atoms_loading(self) -> bool:
        return self.structure_controller.loading

    def sync_render_action(self) -> None:
        self._render_controller.sync_render_action()

    def mark_params_dirty(self) -> None:
        self._render_controller.invalidate()
        self.set_render_outcome(current=False)

    def set_render_outcome(self, *, current: bool, error: str | None = None) -> None:
        """Record whether the image matches the controls; a failed render never does.

        ``error`` says why the latest render failed and lasts until a render
        succeeds or the controls change.
        """
        self.params_dirty = not current or error is not None
        self.render_error = error
        self.update_render_button_style()

    def update_render_button_style(self) -> None:
        self._refresh_render_heading()
        button = self._render_btn
        if button is None:
            return
        # Nothing to re-render until a structure is loaded.
        set_attention(button, bool(self.params_dirty) and self.pdb_path is not None)

    def _refresh_render_heading(self) -> None:
        if not hasattr(self, "render_bar"):
            return
        detail = ""
        if self._render_started_at is not None:
            heading = "Rendering…"
        elif self.pdb_path is None:
            heading = "Open or fetch a structure"
        elif self.render_error is not None:
            # Lead with the error's first sentence; the tooltip holds all of it.
            heading = f"Render failed: {self.render_error.split('. ', 1)[0].rstrip('.')}"
            detail = self.render_error
        elif self.params_dirty:
            heading = "Controls changed — render to update"
        else:
            heading = "Render is up to date"
        self.render_bar.set_heading(heading)
        self.render_bar.heading.setToolTip(detail)

    # Presets

    def _refresh_preset_list(
        self, *, default_index: int | None = None, preferred_label: str | None = None
    ) -> None:
        items = list(self._builtin_presets)
        items.extend(
            (f"Custom: {name}", params) for name, params in self._custom_presets
        )
        self._preset_items = items
        presets = self.preset_list
        if preferred_label is not None:
            previous_text = preferred_label
        elif presets.currentRow() >= 0:
            previous_text = presets.currentItem().text()
        else:
            previous_text = ""
        block = presets.blockSignals(True)
        try:
            presets.clear()
            for label, _params in items:
                presets.addItem(QListWidgetItem(label))
            selected = -1
            if previous_text:
                matches = presets.findItems(previous_text, Qt.MatchFlag.MatchExactly)
                if matches:
                    selected = presets.row(matches[0])
            if (
                selected < 0
                and default_index is not None
                and 0 <= default_index < presets.count()
            ):
                selected = default_index
            if selected < 0 and presets.count() > 0:
                selected = 0
            if selected >= 0:
                presets.setCurrentRow(selected)
        finally:
            presets.blockSignals(block)

    @Slot(object)
    def _on_custom_presets_loaded(self, result: CustomPresetLoadResult) -> None:
        self._custom_presets = result.presets
        self._refresh_preset_list()
        if self._closing or not result.issues:
            return
        issue_count = len(result.issues)
        first_issue = format_custom_preset_load_issues(result.issues[:1])
        if issue_count == 1:
            self.statusBar().showMessage(f"Custom preset issue: {first_issue}")
            return
        self.statusBar().showMessage(
            f"{issue_count} custom preset issues; first: {first_issue}"
        )

    def _save_custom_preset(self) -> None:
        if not self.main_actions.save_preset.isEnabled() or self._closing:
            return
        if self.pdb_path is None:
            self.statusBar().showMessage("Load a PDB before saving a custom preset.")
            return
        try:
            params = self.build_params()
        except (TypeError, KeyError, ValueError) as exc:
            self.statusBar().showMessage(str(exc))
            return
        name, ok = QInputDialog.getText(self, "Save Custom Preset", "Preset name:")
        if not ok:
            return
        preset_name = str(name).strip()
        if not preset_name:
            self.statusBar().showMessage("Preset save canceled: name is empty.")
            return

        custom_idx = next(
            (
                i
                for i, (existing, _params) in enumerate(self._custom_presets)
                if existing == preset_name
            ),
            -1,
        )
        if custom_idx >= 0:
            overwrite = QMessageBox.question(
                self,
                "Overwrite preset?",
                f"A custom preset named '{preset_name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if overwrite != QMessageBox.StandardButton.Yes:
                return

        request = PresetSaveRequest(
            name=preset_name,
            params=params,
            presets=tuple(self._custom_presets),
            replace_index=custom_idx,
        )
        if self._preset_worker.submit_save(request):
            self.main_actions.save_preset.setEnabled(False)
            self.statusBar().showMessage(f"Saving custom preset '{preset_name}'...")

    @Slot(str, list)
    def _on_custom_preset_saved(
        self, preset_name: str, presets: list[tuple[str, RenderParams]]
    ) -> None:
        self._custom_presets = presets
        if self._closing:
            return
        self._refresh_preset_list(preferred_label=f"Custom: {preset_name}")
        self.statusBar().showMessage(f"Saved custom preset '{preset_name}'.")

    @Slot(str, str)
    def _on_preset_io_failed(self, operation: str, message: str) -> None:
        if self._closing:
            return
        if operation == "save":
            QMessageBox.critical(self, "Save preset failed", message)
        else:
            self.statusBar().showMessage(f"Could not load custom presets: {message}")

    @Slot()
    def _on_preset_io_finished(self) -> None:
        self.main_actions.save_preset.setEnabled(not self._closing)

    # Status / progress / labels

    def set_loaded_model_label(self, path: str | None) -> None:
        self.workspace.set_title(Path(path).name if path else _NO_STRUCTURE_TITLE)
        # On macOS this gives the title bar the file's proxy icon.
        self.setWindowFilePath(path or "")
        self.update_render_button_style()

    def clear_last_result(self) -> None:
        self._render_controller.invalidate()
        self.last_result = None
        self._last_pixmap = None
        self.main_actions.copy_to_clipboard.setEnabled(False)

    def set_render_busy(self, busy: bool) -> None:
        if busy:
            self._render_started_at = time.perf_counter()
            self._render_progress.show()
        else:
            self._render_progress.hide()
            self._render_started_at = None
        self._refresh_render_heading()

    def render_elapsed_suffix(self) -> str:
        if self._render_started_at is None:
            return ""
        elapsed = max(0.0, time.perf_counter() - self._render_started_at)
        return f" in {elapsed:.2f}s"

    def update_rule_match_counts(self) -> None:
        if self.atoms is None:
            self.rule_panel.set_match_counts(None)
            return
        rules = self.current_rules()
        if len(rules) == 0:
            self.rule_panel.set_match_counts([])
            return
        if self.atom_match_counts is not None and len(self.atom_match_counts) == len(
            rules
        ):
            self.rule_panel.set_match_counts(list(self.atom_match_counts))
            return
        self.rule_panel.set_match_counts(None)

    def current_rules(self) -> list[SelectionRule]:
        return list(self.rule_panel.value)

    # Viewport sync

    def sync_viewport_orientation(self) -> None:
        transform = self.transform_panel.value
        self.viewport.set_orientation(
            xrot=float(transform["xrot"]),
            yrot=float(transform["yrot"]),
            zrot=float(transform["zrot"]),
        )

    # Render-size delegation

    def sync_render_size_controls(self) -> None:
        self.render_size.sync_controls_enabled()

    def schedule_render_dimensions_update(self) -> None:
        self.render_size.schedule_dimensions_update()

    def _fit_view(self) -> None:
        self.viewport.reset_view()
        self.render_size.fit_view()

    # Settings IO

    def apply_params_to_panels(self, params: RenderParams) -> None:
        transform_values = transform_panel_values(params.transform)
        world = params.world
        world_values = {
            "background": "#%02x%02x%02x" % tuple(int(c * 255) for c in world.background),
            "fog": "#%02x%02x%02x" % tuple(int(c * 255) for c in world.fog_color),
            "fog_front": world.fog_front, "fog_back": world.fog_back,
            "shadows": world.shadows, "shadow_strength": world.shadow_strength,
            "shadow_angle": world.shadow_angle, "shadow_min_z": world.shadow_min_z,
            "shadow_max_dark": world.shadow_max_dark,
        }
        outline_values = {name: getattr(params.outlines, name) for name in self.outline_panel.value}
        panels = (
            ("transform", self.transform_panel, transform_values),
            ("world", self.world_panel, world_values),
            ("outlines", self.outline_panel, outline_values),
        )
        for group, panel, values in panels:
            for name, value in values.items():
                _check_control_range(f"{group}.{name}", getattr(panel, name, None), value)
        if world.width > 0 and world.height > 0:
            _check_control_range("world.width", self.render_width_spin, world.width)
            _check_control_range("world.height", self.render_height_spin, world.height)
        with self.suspended_panel_callbacks():
            self.transform_panel.set_value(transform_values)
            self.world_panel.set_value(world_values)
            self.outline_panel.set_value(outline_values)
            self.rule_panel.set_value(params.rules)
            if params.world.width > 0 and params.world.height > 0:
                self.render_size_mode_combo.setCurrentText("Custom")
                self.render_width_spin.setValue(int(params.world.width))
                self.render_height_spin.setValue(int(params.world.height))
            else:
                self.render_size_mode_combo.setCurrentText("Auto")
        self.mark_params_dirty()

    # Render parameters

    def build_params(self) -> RenderParams:
        if self.pdb_path is None:
            raise RuntimeError("no pdb selected")
        render_w, render_h = self.render_size.selected_dimensions()
        return render_params_from_panels(
            self.pdb_path,
            self.current_rules(),
            self.transform_panel.value,
            self.world_panel.value,
            self.outline_panel.value,
            render_width=render_w,
            render_height=render_h,
        )

    def load_atoms_if_needed(
        self,
        *,
        status_message: str | None = None,
        render_after_load: bool = False,
    ) -> bool:
        return self.structure_controller.load_atoms_if_needed(
            status_message=status_message,
            render_after_load=render_after_load,
        )

    # Render submission/completion

    def render_frame(self, _checked: bool = False, *, interactive: bool = False) -> None:
        self._render_controller.render(interactive=interactive)

    # Image display + export

    def display_image_frame(self, frame: GuiImageFrame) -> None:
        qpix = pixmap_from_frame(frame)
        if qpix is None:
            return
        self._last_pixmap = qpix
        self.viewport.update_image(qpix)

    def display_image(
        self, rgb: NDArray[np.uint8], opacity: NDArray[np.uint8] | None = None
    ) -> None:
        qpix = pixmap_from_image(rgb, opacity)
        if qpix is None:
            return
        self._last_pixmap = qpix
        self.viewport.update_image(qpix)

    def _export_png(self) -> None:
        if self.last_result is None:
            self.statusBar().showMessage("Render first before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", str(Path.home() / "render.png"), "PNG image (*.png)"
        )
        if not path:
            return
        self._submit_export(
            path=path,
            kind="png",
            rgb=self.last_result.rgb,
            opacity=self.last_result.opacity
            if hasattr(self.last_result, "opacity")
            else None,
        )

    def _export_svg(self) -> None:
        if self.last_result is None:
            self.statusBar().showMessage("Render first before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", str(Path.home() / "render.svg"), "SVG image (*.svg)"
        )
        if not path:
            return
        self._submit_export(
            path=path,
            kind="svg",
            rgb=self.last_result.rgb,
            opacity=self.last_result.opacity
            if hasattr(self.last_result, "opacity")
            else None,
        )

    def _submit_export(
        self,
        *,
        path: str,
        kind: ExportKind,
        rgb: NDArray[np.uint8],
        opacity: NDArray[np.uint8] | None,
    ) -> None:
        if self.export_worker.isRunning():
            self.statusBar().showMessage("Export already in progress...")
            return
        accepted = self.export_worker.submit(
            ExportRequest(path=path, kind=kind, rgb=rgb, opacity=opacity)
        )
        if accepted:
            self.main_actions.export_png.setEnabled(False)
            self.main_actions.export_svg.setEnabled(False)
            self.statusBar().showMessage(f"Exporting {path}...")

    def _on_export_done(self, payload: object) -> None:
        self.main_actions.export_png.setEnabled(True)
        self.main_actions.export_svg.setEnabled(True)
        if isinstance(payload, ExportJobResult):
            self.statusBar().showMessage(f"Saved {payload.path}")

    def _on_export_failed(self, payload: object) -> None:
        self.main_actions.export_png.setEnabled(True)
        self.main_actions.export_svg.setEnabled(True)
        if isinstance(payload, ExportJobFailure):
            QMessageBox.critical(self, "Export failed", payload.message)
            self.statusBar().showMessage(f"Export failed: {payload.message}")
            return
        self.statusBar().showMessage("Export failed.")

    def _copy_to_clipboard(self) -> None:
        if self.last_result is None:
            return
        qpix = self._last_pixmap
        if qpix is None:
            rgb = self.last_result.rgb
            opacity = (
                self.last_result.opacity
                if hasattr(self.last_result, "opacity")
                else None
            )
            qpix = pixmap_from_image(rgb, opacity)
        if qpix is None:
            return
        QGuiApplication.clipboard().setPixmap(qpix)
        self.statusBar().showMessage("Copied to clipboard")

    # Presets / viewport callbacks

    def _apply_preset(self, index: int) -> None:
        presets = self._preset_items
        if index < 0 or index >= len(presets):
            return
        _name, preset = presets[index]
        # A preset restyles the structure on screen: with one loaded, keep the
        # view (scale, rotation, translation) the user set instead of jumping
        # to the preset's own.
        view = dict(self.transform_panel.value) if self.pdb_path is not None else None
        try:
            self.apply_params_to_panels(preset)
        except ValueError as exc:
            QMessageBox.critical(self, "Apply preset failed", str(exc))
            return
        if view is not None:
            with self.suspended_panel_callbacks():
                self.transform_panel.set_value(view)
        self.sync_viewport_orientation()
        self.rule_panel.set_match_counts(None)
        self.schedule_render_dimensions_update()
        self.render_frame()

    def _on_viewport_rotation(self, x_delta: float, y_delta: float) -> None:
        self.apply_viewport_transform(
            xrot=float(self.transform_panel.value["xrot"]) + x_delta,
            yrot=float(self.transform_panel.value["yrot"]) + y_delta,
        )

    def apply_viewport_transform(self, **values: float) -> None:
        auto_render = bool(self.auto_render_on_drag.isChecked())
        with self._viewport_drag_step(auto_render):
            # set_value replaces the whole transform, so keep the other fields.
            self.transform_panel.set_value({**self.transform_panel.value, **values})
        if auto_render:
            self._render_controller.request_drag_render()

    def closeEvent(self, event: QCloseEvent) -> None:
        # Stop new work and cancel what can be cancelled. This repeats on every
        # retry because result handlers may still schedule work.
        self.render_size.stop()
        self._pdb_completer.stop()
        self.structure_controller.abandon_fetch()
        self.worker.cancel_pending()
        self.load_worker.cancel_pending()
        self._backend_status_worker.requestInterruption()

        busy = self._busy_worker_label()
        if busy is not None:
            # Keep the event loop running until the workers can be destroyed.
            event.ignore()
            if not self._closing:
                self._closing = True
                self.setEnabled(False)
                self._close_retry_timer.start()
            self.statusBar().showMessage(f"Closing after the {busy} finishes...")
            return
        self._close_retry_timer.stop()
        super().closeEvent(event)

    def _busy_worker_label(self) -> str | None:
        settings = self._settings_controller
        for label, worker in (
            ("structure download", self.fetch_worker),
            ("export", self.export_worker),
            ("structure load", self.load_worker),
            ("render", self.worker),
            ("output size estimate", self.render_size.estimate_worker),
            ("settings file", settings.settings_worker),
            ("update check", settings.update_worker),
        ):
            if worker.isRunning():
                return label
        if self._pdb_completer.is_running():
            return "PDB search"
        if self._preset_worker.isRunning():
            return "custom preset file"
        if self._backend_status_worker.isRunning():
            return "backend check"
        return None


__all__ = ["MainWindow", "run"]


def run() -> int:
    app = QApplication([])
    app.setApplicationDisplayName("Illustrate")
    apply_appearance("light", app)
    main = MainWindow()
    main.show()
    main.raise_()
    main.activateWindow()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
