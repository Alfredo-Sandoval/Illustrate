"""Desktop PySide6 application for interactive rendering."""

from __future__ import annotations

import json
import hashlib
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image as PILImage

from illustrate import (
    OutlineParams,
    RenderParams,
    SelectionRule,
    Transform,
    WorldParams,
    estimate_render_size,
    params_from_json,
    params_to_json,
)
from illustrate.fetch import fetch_pdb
from illustrate.io import write_png, write_svg
from illustrate.pdb import load_pdb
from illustrate.presets import default_rules, preset_library
from illustrate_gui.panels.outlines import OutlinesPanel
from illustrate_gui.panels.rules import RulePanel
from illustrate_gui.panels.transform import TransformPanel
from illustrate_gui.panels.world import WorldPanel
from illustrate_gui.autocomplete import PdbCompleter
from illustrate_gui.updater import RELEASES_PAGE_URL, check_for_updates
from illustrate_gui.viewport import RenderViewport
from illustrate_gui.worker import RenderRequest, RenderWorker

_AUTO_FRAME_PAD = 30
_INTERACTIVE_SETTLE_MS = 180


try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QGuiApplication, QIcon, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDockWidget,
        QFileDialog,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QScrollArea,
        QSplitter,
        QSpinBox,
        QToolBar,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit("PySide6 is required to run the desktop app") from exc


# ── Dark theme ───────────────────────────────────────────────────────

DARK_STYLE = """
/* ── Base ── */
QMainWindow, QWidget {
    background-color: #1c1c1e;
    color: #d1d1d6;
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 13px;
}

/* ── Toolbar ── */
QToolBar {
    background-color: #2c2c2e;
    border-bottom: 1px solid #38383a;
    spacing: 8px;
    padding: 6px 10px;
}
QToolBar QToolButton {
    background-color: #3a3a3c;
    border: 1px solid #48484a;
    border-radius: 6px;
    padding: 5px 14px;
    color: #e5e5ea;
    font-weight: 500;
    font-size: 12px;
    min-height: 16px;
}
QToolBar QToolButton:hover {
    background-color: #48484a;
    border-color: #636366;
}
QToolBar QToolButton:pressed {
    background-color: #545456;
}
QToolBar QToolButton:disabled {
    color: #48484a;
    background-color: #2c2c2e;
    border-color: #38383a;
}

/* ── Inputs ── */
QComboBox {
    background-color: #2c2c2e;
    border: 1px solid #48484a;
    border-radius: 6px;
    padding: 5px 10px;
    color: #e5e5ea;
    min-height: 22px;
    font-size: 12px;
}
QComboBox:hover { border-color: #636366; }
QComboBox:focus { border-color: #0a84ff; }
QComboBox::drop-down {
    border: none;
    width: 24px;
    subcontrol-position: center right;
    padding-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #2c2c2e;
    color: #e5e5ea;
    selection-background-color: #0a84ff;
    selection-color: #ffffff;
    border: 1px solid #48484a;
    border-radius: 6px;
    padding: 2px;
    outline: 0;
}
QDoubleSpinBox, QSpinBox {
    background-color: #2c2c2e;
    border: 1px solid #48484a;
    border-radius: 6px;
    padding: 4px 8px;
    color: #e5e5ea;
    min-height: 22px;
    font-size: 12px;
    selection-background-color: #0a84ff;
}
QDoubleSpinBox:hover, QSpinBox:hover { border-color: #636366; }
QDoubleSpinBox:focus, QSpinBox:focus { border-color: #0a84ff; }
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {
    background-color: transparent;
    border: none;
    width: 16px;
}
QCheckBox {
    color: #d1d1d6;
    spacing: 8px;
    padding: 4px 0px;
    font-size: 12px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #48484a;
    background-color: #2c2c2e;
}
QCheckBox::indicator:checked {
    background-color: #0a84ff;
    border-color: #0a84ff;
}
QCheckBox::indicator:hover { border-color: #636366; }
QPushButton {
    background-color: #3a3a3c;
    border: 1px solid #48484a;
    border-radius: 6px;
    padding: 5px 14px;
    color: #e5e5ea;
    font-size: 12px;
    min-height: 16px;
}
QPushButton:hover { background-color: #48484a; border-color: #636366; }
QPushButton:pressed { background-color: #545456; }
QPushButton:disabled { color: #48484a; background-color: #2c2c2e; border-color: #38383a; }
QLineEdit {
    background-color: #2c2c2e;
    border: 1px solid #48484a;
    border-radius: 6px;
    padding: 5px 10px;
    color: #e5e5ea;
    font-size: 12px;
    min-height: 22px;
    selection-background-color: #0a84ff;
}
QLineEdit:hover { border-color: #636366; }
QLineEdit:focus { border-color: #0a84ff; }

/* ── Toolbar inline widgets ── */
QToolBar QLineEdit, QToolBar QSpinBox, QToolBar QDoubleSpinBox, QToolBar QComboBox {
    background-color: #1c1c1e;
    border: 1px solid #48484a;
    border-radius: 6px;
    color: #e5e5ea;
}
QToolBar QLabel {
    background: transparent;
    color: #c7c7cc;
    font-size: 12px;
    padding: 0 4px;
}

/* ── Layout ── */
QScrollArea { border: none; background-color: #1c1c1e; }
QSplitter::handle { background-color: #38383a; width: 1px; }

/* ── Form labels ── */
QLabel {
    color: #98989d;
    font-size: 12px;
}
QFormLayout { spacing: 6px; }

/* ── Table ── */
QTableWidget {
    background-color: #2c2c2e;
    alternate-background-color: #323234;
    color: #e5e5ea;
    gridline-color: #38383a;
    border: 1px solid #38383a;
    border-radius: 6px;
    selection-background-color: #0a84ff;
}
QTableWidget::item { padding: 4px 6px; }
QTableWidget::item:selected { background-color: #0a84ff; color: #ffffff; }
QHeaderView::section {
    background-color: #2c2c2e;
    color: #98989d;
    border: none;
    border-right: 1px solid #38383a;
    border-bottom: 1px solid #38383a;
    padding: 6px 8px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}

/* ── Status bar ── */
QStatusBar {
    background-color: #2c2c2e;
    color: #98989d;
    border-top: 1px solid #38383a;
    font-size: 11px;
    padding: 2px 10px;
}

/* ── Scrollbar ── */
QScrollBar:vertical {
    background-color: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #48484a;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background-color: #636366; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""

LIGHT_STYLE = """
/* ── Base ── */
QMainWindow, QWidget {
    background-color: #f4f4f6;
    color: #1f2125;
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 13px;
}

/* ── Toolbar ── */
QToolBar {
    background-color: #f7f7f9;
    border-bottom: 1px solid #d6d8de;
    spacing: 8px;
    padding: 6px 10px;
}
QToolBar QToolButton {
    background-color: #ffffff;
    border: 1px solid #c8ccd4;
    border-radius: 6px;
    padding: 5px 14px;
    color: #1f2125;
    font-weight: 500;
    font-size: 12px;
    min-height: 16px;
}
QToolBar QToolButton:hover {
    background-color: #eef1f6;
    border-color: #aeb5c2;
}
QToolBar QToolButton:pressed {
    background-color: #e2e7ef;
}
QToolBar QToolButton:disabled {
    color: #8a90a0;
    background-color: #eef1f5;
    border-color: #d5d9e2;
}

/* ── Inputs ── */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #c8ccd4;
    border-radius: 6px;
    padding: 5px 10px;
    color: #1f2125;
    min-height: 22px;
    font-size: 12px;
}
QComboBox:hover { border-color: #aeb5c2; }
QComboBox:focus { border-color: #1f78d1; }
QComboBox::drop-down {
    border: none;
    width: 24px;
    subcontrol-position: center right;
    padding-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #1f2125;
    selection-background-color: #d6e7ff;
    selection-color: #102a4d;
    border: 1px solid #c8ccd4;
    border-radius: 6px;
    padding: 2px;
    outline: 0;
}
QDoubleSpinBox, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #c8ccd4;
    border-radius: 6px;
    padding: 4px 8px;
    color: #1f2125;
    min-height: 22px;
    font-size: 12px;
    selection-background-color: #d6e7ff;
}
QDoubleSpinBox:hover, QSpinBox:hover { border-color: #aeb5c2; }
QDoubleSpinBox:focus, QSpinBox:focus { border-color: #1f78d1; }
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {
    background-color: transparent;
    border: none;
    width: 16px;
}
QCheckBox {
    color: #1f2125;
    spacing: 8px;
    padding: 4px 0px;
    font-size: 12px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #b7bcc7;
    background-color: #ffffff;
}
QCheckBox::indicator:checked {
    background-color: #1f78d1;
    border-color: #1f78d1;
}
QCheckBox::indicator:hover { border-color: #97a0b1; }
QPushButton {
    background-color: #ffffff;
    border: 1px solid #c8ccd4;
    border-radius: 6px;
    padding: 5px 14px;
    color: #1f2125;
    font-size: 12px;
    min-height: 16px;
}
QPushButton:hover { background-color: #eef1f6; border-color: #aeb5c2; }
QPushButton:pressed { background-color: #e2e7ef; }
QPushButton:disabled { color: #8a90a0; background-color: #eef1f5; border-color: #d5d9e2; }
QLineEdit {
    background-color: #ffffff;
    border: 1px solid #c8ccd4;
    border-radius: 6px;
    padding: 5px 10px;
    color: #1f2125;
    font-size: 12px;
    min-height: 22px;
    selection-background-color: #d6e7ff;
}
QLineEdit:hover { border-color: #aeb5c2; }
QLineEdit:focus { border-color: #1f78d1; }

/* ── Toolbar inline widgets ── */
QToolBar QLineEdit, QToolBar QSpinBox, QToolBar QDoubleSpinBox, QToolBar QComboBox {
    background-color: #ffffff;
    border: 1px solid #c8ccd4;
    border-radius: 6px;
    color: #1f2125;
}
QToolBar QLabel {
    background: transparent;
    color: #4f5665;
    font-size: 12px;
    padding: 0 4px;
}

/* ── Layout ── */
QScrollArea { border: none; background-color: #f4f4f6; }
QSplitter::handle { background-color: #d0d4dc; width: 1px; }

/* ── Form labels ── */
QLabel {
    color: #505666;
    font-size: 12px;
}
QFormLayout { spacing: 6px; }

/* ── Table ── */
QTableWidget {
    background-color: #ffffff;
    alternate-background-color: #f7f8fa;
    color: #1f2125;
    gridline-color: #d8dce5;
    border: 1px solid #d8dce5;
    border-radius: 6px;
    selection-background-color: #d6e7ff;
}
QTableWidget::item { padding: 4px 6px; }
QTableWidget::item:selected { background-color: #d6e7ff; color: #1f2125; }
QHeaderView::section {
    background-color: #eef1f6;
    color: #4a5160;
    border: none;
    border-right: 1px solid #d8dce5;
    border-bottom: 1px solid #d8dce5;
    padding: 6px 8px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}

/* ── Status bar ── */
QStatusBar {
    background-color: #f7f7f9;
    color: #545b6b;
    border-top: 1px solid #d6d8de;
    font-size: 11px;
    padding: 2px 10px;
}

/* ── Scrollbar ── */
QScrollBar:vertical {
    background-color: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background-color: #c4cad6;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background-color: #aeb6c4; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""


# ── Collapsible section widget ───────────────────────────────────────


class CollapsibleSection(QWidget):
    """A header button that shows/hides a content widget."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False) -> None:
        super().__init__()
        self._toggle = QToolButton(self)
        self._toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: 600; font-size: 12px;"
            " padding: 8px 4px; background: transparent; }"
        )
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setArrowType(self._arrow_for_state(expanded))

        self._content = content
        self._content.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(0)
        layout.addWidget(self._toggle)
        layout.addWidget(self._content)

        self._toggle.toggled.connect(self._on_toggle)

    def _on_toggle(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setArrowType(self._arrow_for_state(checked))

    @staticmethod
    def _arrow_for_state(expanded: bool) -> Qt.ArrowType:
        return Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow


# ── Helpers ──────────────────────────────────────────────────────────


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    hex_value = value.lstrip("#")
    if len(hex_value) != 6:
        return (1.0, 1.0, 1.0)
    return (
        int(hex_value[0:2], 16) / 255.0,
        int(hex_value[2:4], 16) / 255.0,
        int(hex_value[4:6], 16) / 255.0,
    )


def _rules_signature(rules: list[SelectionRule]) -> str:
    payload = []
    for rule in rules:
        payload.append((rule.record_name, rule.descriptor, rule.res_low, rule.res_high, rule.color, rule.radius))
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def _to_render_params(
    pdb_path: str,
    rules: list[SelectionRule],
    transform_state: Mapping[str, float],
    world_state: Mapping[str, float | bool | str],
    outline_state: Mapping[str, float | int | bool],
    *,
    render_width: int,
    render_height: int,
) -> RenderParams:
    return RenderParams(
        pdb_path=pdb_path,
        rules=rules,
        transform=Transform(
            scale=float(transform_state.get("scale", 12.0)),
            translate=(
                float(transform_state.get("xtran", 0.0)),
                float(transform_state.get("ytran", 0.0)),
                float(transform_state.get("ztran", 0.0)),
            ),
            rotations=[
                ("z", float(transform_state.get("zrot", 90.0))),
                ("y", float(transform_state.get("yrot", 0.0))),
                ("x", float(transform_state.get("xrot", 0.0))),
            ],
            autocenter="auto",
        ),
        world=WorldParams(
            background=_hex_to_rgb(str(world_state.get("background", "#ffffff"))),
            fog_color=_hex_to_rgb(str(world_state.get("fog", "#ffffff"))),
            fog_front=float(world_state.get("fog_front", 1.0)),
            fog_back=float(world_state.get("fog_back", 1.0)),
            shadows=bool(world_state.get("shadows", False)),
            shadow_strength=float(world_state.get("shadow_strength", 0.0023)),
            shadow_angle=float(world_state.get("shadow_angle", 2.0)),
            shadow_min_z=float(world_state.get("shadow_min_z", 1.0)),
            shadow_max_dark=float(world_state.get("shadow_max_dark", 0.2)),
            width=int(render_width),
            height=int(render_height),
        ),
        outlines=OutlineParams(
            enabled=bool(outline_state.get("enabled", True)),
            contour_low=float(outline_state.get("contour_low", 3.0)),
            contour_high=float(outline_state.get("contour_high", 10.0)),
            kernel=int(outline_state.get("kernel", 4)),
            z_diff_min=float(outline_state.get("z_diff_min", 0.0)),
            z_diff_max=float(outline_state.get("z_diff_max", 5.0)),
            subunit_low=float(outline_state.get("subunit_low", 3.0)),
            subunit_high=float(outline_state.get("subunit_high", 10.0)),
            residue_low=float(outline_state.get("residue_low", 3.0)),
            residue_high=float(outline_state.get("residue_high", 8.0)),
            residue_diff=float(outline_state.get("residue_diff", 6000.0)),
        ),
    )


def _builtin_preset_items() -> list[tuple[str, RenderParams]]:
    from illustrate.presets import PRESET_NAMES
    presets = preset_library(default_rules())
    return [(name, preset) for name, preset in zip(PRESET_NAMES, presets)]


def _runtime_data_dir() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if isinstance(bundle_root, str) and bundle_root:
        return Path(bundle_root) / "data"
    return Path(__file__).resolve().parent.parent / "data"


# ── Main window ──────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    _fetch_done_signal = Signal(str, str)  # (path, pdb_id)
    _fetch_failed_signal = Signal(str)  # (error message)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Illustrate")
        self.resize(1400, 900)
        _data_dir = _runtime_data_dir()
        _icon_path = _data_dir / "icon.icns"
        if not _icon_path.exists():
            _icon_path = _data_dir / "icon.png"
        if _icon_path.exists():
            _app_icon = QIcon(str(_icon_path))
            self.setWindowIcon(_app_icon)
            QApplication.instance().setWindowIcon(_app_icon)

        self.pdb_path: str | None = None
        self._atoms: Any = None
        self._atoms_signature = ""
        self._last_result = None
        self._preview_request_id = 0
        self._latest_preview_request_id = 0
        self._preview_pending = False
        self._suppress_preview_render_once = False
        self._suspend_panel_callbacks = False
        self._params_dirty = False
        self._render_btn: QToolButton | None = None
        self._render_started_at: float | None = None
        self._render_request_id = 0
        self._preset_items: list[tuple[str, RenderParams]] = []
        self._builtin_presets = _builtin_preset_items()
        self._custom_presets: list[tuple[str, RenderParams]] = self._load_custom_presets()

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._submit_preview_render)
        self._interactive_settle_timer = QTimer(self)
        self._interactive_settle_timer.setSingleShot(True)
        self._interactive_settle_timer.timeout.connect(self._on_interactive_settle_timeout)
        self._dimensions_timer = QTimer(self)
        self._dimensions_timer.setSingleShot(True)
        self._dimensions_timer.timeout.connect(self._update_render_dimensions_label)

        # ── Fetch signals (thread-safe) ──
        self._fetch_done_signal.connect(self._on_fetch_done)
        self._fetch_failed_signal.connect(self._on_fetch_failed)

        # ── Worker ──
        self.worker = RenderWorker()
        if hasattr(self.worker, "finished"):
            self.worker.finished.connect(self._on_render_done)
            self.worker.failed.connect(self._on_render_failed)

        # Preview renders run on a separate coalescing worker to keep drag
        # interactions responsive while preserving final renderer look.
        self.preview_worker = RenderWorker()
        if hasattr(self.preview_worker, "finished"):
            self.preview_worker.finished.connect(self._on_preview_done)
            self.preview_worker.failed.connect(self._on_preview_failed)

        # ── Toolbar ──
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = toolbar.addAction("Open PDB")
        open_action.triggered.connect(self._open_pdb)

        toolbar.addSeparator()

        # Search / fetch PDB
        self.pdb_id_input = QLineEdit()
        self.pdb_id_input.setPlaceholderText("Search PDB (ID or name)...")
        self.pdb_id_input.setFixedWidth(250)
        toolbar.addWidget(self.pdb_id_input)
        self._pdb_completer = PdbCompleter(self.pdb_id_input, parent=self)
        self._pdb_completer.activated.connect(self._on_suggestion_selected)
        self.fetch_action = toolbar.addAction("Fetch")
        self.fetch_action.triggered.connect(self._fetch_pdb)
        self.loaded_model_label = QLabel("Model: (none)")
        self.loaded_model_label.setStyleSheet("QLabel { font-weight: 600; }")
        toolbar.addWidget(self.loaded_model_label)

        toolbar.addSeparator()
        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(self._apply_preset)
        toolbar.addWidget(self.preset_combo)
        self._refresh_preset_combo(default_index=3)
        self.save_preset_action = toolbar.addAction("Save Preset")
        self.save_preset_action.triggered.connect(self._save_custom_preset)

        toolbar.addSeparator()
        self.load_settings_action = toolbar.addAction("Load Settings")
        self.load_settings_action.triggered.connect(self._load_settings)
        self.save_settings_action = toolbar.addAction("Save Settings")
        self.save_settings_action.triggered.connect(self._save_settings)
        self.check_updates_action = toolbar.addAction("Check Updates")
        self.check_updates_action.triggered.connect(self._check_for_updates)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Theme"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        toolbar.addWidget(self.theme_combo)

        toolbar.addSeparator()
        self.preview_quality_combo = QComboBox()
        self.preview_quality_combo.addItems(["Fast", "Balanced", "High"])
        self.preview_quality_combo.setCurrentText("Fast")
        self.preview_quality_combo.setToolTip("Interactive preview quality")
        self.preview_quality_combo.currentTextChanged.connect(self._on_preview_quality_changed)
        toolbar.addWidget(self.preview_quality_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Render Size"))
        self.render_size_mode_combo = QComboBox()
        self.render_size_mode_combo.addItems(["Auto", "Custom"])
        self.render_size_mode_combo.currentTextChanged.connect(self._on_render_size_changed)
        toolbar.addWidget(self.render_size_mode_combo)
        self.render_width_spin = QSpinBox()
        self.render_width_spin.setRange(64, 3000)
        self.render_width_spin.setValue(2000)
        self.render_width_spin.setFixedWidth(84)
        self.render_width_spin.valueChanged.connect(self._on_render_size_changed)
        toolbar.addWidget(self.render_width_spin)
        toolbar.addWidget(QLabel("\u00d7"))
        self.render_height_spin = QSpinBox()
        self.render_height_spin.setRange(64, 3000)
        self.render_height_spin.setValue(2000)
        self.render_height_spin.setFixedWidth(84)
        self.render_height_spin.valueChanged.connect(self._on_render_size_changed)
        toolbar.addWidget(self.render_height_spin)
        self.render_dims_label = QLabel("Output: auto")
        toolbar.addWidget(self.render_dims_label)

        toolbar.addSeparator()
        self.render_action = toolbar.addAction("Render")
        self.render_action.triggered.connect(self._render)
        self.fit_view_action = toolbar.addAction("Fit View")
        self.fit_view_action.triggered.connect(self._fit_view)
        # Style the Render button as primary action
        render_btn = toolbar.widgetForAction(self.render_action)
        if render_btn is not None:
            self._render_btn = render_btn
            render_btn.setStyleSheet(self._render_btn_style(dirty=False))

        self.export_action = toolbar.addAction("Export PNG")
        self.export_action.triggered.connect(self._export_png)
        self.export_svg_action = toolbar.addAction("Export SVG")
        self.export_svg_action.triggered.connect(self._export_svg)

        self.copy_action = toolbar.addAction("Copy to Clipboard")
        self.copy_action.triggered.connect(self._copy_to_clipboard)
        self.copy_action.setEnabled(False)

        # ── Viewport ──
        self.viewport = RenderViewport()
        self.viewport.rotation_requested.connect(self._on_viewport_rotation)
        self.viewport.zoom_requested.connect(self._on_viewport_zoom)

        # ── Sidebar ──
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(2)

        self.auto_render_on_drag = QCheckBox("Render on drag/zoom")
        self.auto_render_on_drag.setChecked(True)
        sidebar_layout.addWidget(self.auto_render_on_drag)

        self.transform_panel = TransformPanel(on_changed=self._panel_changed)
        sidebar_layout.addWidget(CollapsibleSection("Transform", self.transform_panel, expanded=True))

        self.world_panel = WorldPanel(on_changed=self._panel_changed)
        sidebar_layout.addWidget(CollapsibleSection("World / Lighting", self.world_panel))

        self.outline_panel = OutlinesPanel(on_changed=self._panel_changed)
        sidebar_layout.addWidget(CollapsibleSection("Outlines", self.outline_panel))

        sidebar_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(400)

        # ── Splitter (sidebar left, viewport right) ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self.viewport)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # ── Selection Rules dock (bottom, minimized by default) ──
        self.rule_panel = RulePanel(default_rules(), on_changed=self._panel_changed)
        self.rules_dock = QDockWidget("Selection Rules", self)
        self.rules_dock.setWidget(self.rule_panel)
        self.rules_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.rules_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.rules_dock)
        self.toggle_rules_action = self.rules_dock.toggleViewAction()
        self.toggle_rules_action.setText("Selection Rules")
        toolbar.addAction(self.toggle_rules_action)
        self.rules_dock.hide()
        self._render_progress = QProgressBar(self)
        self._render_progress.setRange(0, 0)
        self._render_progress.setFixedWidth(140)
        self._render_progress.setTextVisible(False)
        self._render_progress.hide()
        self.statusBar().addPermanentWidget(self._render_progress)

        self.statusBar().showMessage("Ready")
        self.theme_combo.setCurrentText("Dark")
        self._apply_theme("Dark", announce=False)
        # Start on chain-colored dark preset by default.
        self._apply_preset(self.preset_combo.currentIndex())
        self._sync_render_size_controls()
        self._sync_preview_transform()
        self._sync_preview_style()
        self._schedule_render_dimensions_update()

    # ── Callbacks ──

    def _panel_changed(self, _value: object) -> None:
        if self._suspend_panel_callbacks:
            return
        skip_preview_render = self._suppress_preview_render_once
        self._suppress_preview_render_once = False
        if isinstance(_value, list) and hasattr(self, "rule_panel"):
            self.rule_panel.set_match_counts(None)
        if hasattr(self, "transform_panel") and hasattr(self, "viewport"):
            self._sync_preview_transform()
            self._sync_preview_style()
            if not skip_preview_render:
                self._request_preview_render()
            self._schedule_render_dimensions_update()
        self._params_dirty = True
        self._update_render_btn_style()
        self.statusBar().showMessage("Controls changed. Click Render.")

    def _apply_theme(self, theme: str, *, announce: bool = True) -> None:
        selection = str(theme).strip().lower()
        stylesheet = DARK_STYLE if selection != "light" else LIGHT_STYLE
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet)
        self._update_render_btn_style()
        if announce:
            label = "Light" if selection == "light" else "Dark"
            self.statusBar().showMessage(f"Theme: {label}")

    def _on_theme_changed(self, theme: str) -> None:
        self._apply_theme(theme, announce=True)

    def _render_btn_style(self, *, dirty: bool) -> str:
        light_theme = hasattr(self, "theme_combo") and self.theme_combo.currentText() == "Light"
        disabled_bg = "#eef1f5" if light_theme else "#2c2c2e"
        disabled_border = "#d5d9e2" if light_theme else "#38383a"
        disabled_fg = "#8a90a0" if light_theme else "#48484a"
        if dirty:
            return (
                "QToolButton { background-color: #ff9f0a; border: 1px solid #ff9f0a;"
                " border-radius: 6px; padding: 5px 20px; color: #1c1c1e; font-weight: 700;"
                " font-size: 12px; }"
                "QToolButton:hover { background-color: #ffb340; border-color: #ffb340; }"
                "QToolButton:pressed { background-color: #dd8800; border-color: #dd8800; }"
                f"QToolButton:disabled {{ background-color: {disabled_bg}; border-color: {disabled_border}; color: {disabled_fg}; }}"
            )
        return (
            "QToolButton { background-color: #0a84ff; border: 1px solid #0a84ff;"
            " border-radius: 6px; padding: 5px 20px; color: #ffffff; font-weight: 600;"
            " font-size: 12px; }"
            "QToolButton:hover { background-color: #409cff; border-color: #409cff; }"
            "QToolButton:pressed { background-color: #0064d2; border-color: #0064d2; }"
            f"QToolButton:disabled {{ background-color: {disabled_bg}; border-color: {disabled_border}; color: {disabled_fg}; }}"
        )

    def _update_render_btn_style(self) -> None:
        if self._render_btn is None:
            return
        self._render_btn.setStyleSheet(self._render_btn_style(dirty=self._params_dirty))

    @staticmethod
    def _custom_preset_store_path() -> Path:
        return Path.home() / ".illustrate_presets.json"

    def _load_custom_presets(self) -> list[tuple[str, RenderParams]]:
        path = self._custom_preset_store_path()
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        items: list[tuple[str, RenderParams]] = []
        presets_payload = payload.get("presets", [])
        if not isinstance(presets_payload, list):
            return items
        for entry in presets_payload:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            params_payload = entry.get("params")
            if not name or not isinstance(params_payload, dict):
                continue
            try:
                params = params_from_json(json.dumps(params_payload))
            except Exception:
                continue
            items.append((name, params))
        return items

    def _write_custom_presets(self) -> None:
        payload = {
            "version": 1,
            "presets": [
                {"name": name, "params": json.loads(params_to_json(params))}
                for name, params in self._custom_presets
            ],
        }
        path = self._custom_preset_store_path()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _refresh_preset_combo(self, *, default_index: int | None = None, preferred_label: str | None = None) -> None:
        items = list(self._builtin_presets)
        items.extend((f"Custom: {name}", params) for name, params in self._custom_presets)
        self._preset_items = items
        previous_text = preferred_label if preferred_label is not None else self.preset_combo.currentText()
        block = self.preset_combo.blockSignals(True)
        try:
            self.preset_combo.clear()
            for label, _params in items:
                self.preset_combo.addItem(label)
            selected = -1
            if previous_text:
                selected = self.preset_combo.findText(previous_text)
            if selected < 0 and default_index is not None and 0 <= default_index < self.preset_combo.count():
                selected = default_index
            if selected < 0 and self.preset_combo.count() > 0:
                selected = 0
            if selected >= 0:
                self.preset_combo.setCurrentIndex(selected)
        finally:
            self.preset_combo.blockSignals(block)

    def _save_custom_preset(self) -> None:
        if self.pdb_path is None:
            self.statusBar().showMessage("Load a PDB before saving a custom preset.")
            return
        try:
            params = self._build_params()
        except Exception as exc:
            self.statusBar().showMessage(str(exc))
            return
        name, ok = QInputDialog.getText(self, "Save Custom Preset", "Preset name:")
        if not ok:
            return
        preset_name = str(name).strip()
        if not preset_name:
            self.statusBar().showMessage("Preset save canceled: name is empty.")
            return

        custom_idx = next((i for i, (existing, _params) in enumerate(self._custom_presets) if existing == preset_name), -1)
        if custom_idx >= 0:
            overwrite = QMessageBox.question(
                self,
                "Overwrite preset?",
                f"A custom preset named '{preset_name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if overwrite != QMessageBox.StandardButton.Yes:
                return

        serial = params_from_json(params_to_json(params))
        serial.pdb_path = ""
        if custom_idx >= 0:
            self._custom_presets[custom_idx] = (preset_name, serial)
        else:
            self._custom_presets.append((preset_name, serial))
        try:
            self._write_custom_presets()
        except Exception as exc:
            QMessageBox.critical(self, "Save preset failed", str(exc))
            return
        combo_label = f"Custom: {preset_name}"
        self._refresh_preset_combo(preferred_label=combo_label)
        self.statusBar().showMessage(f"Saved custom preset '{preset_name}'.")

    def _set_loaded_model_label(self, path: str | None) -> None:
        if not path:
            self.loaded_model_label.setText("Model: (none)")
            return
        self.loaded_model_label.setText(f"Model: {Path(path).name}")

    def _set_render_busy(self, busy: bool) -> None:
        if busy:
            self._render_started_at = time.perf_counter()
            self._render_progress.show()
            return
        self._render_progress.hide()
        self._render_started_at = None

    def _render_elapsed_suffix(self) -> str:
        if self._render_started_at is None:
            return ""
        elapsed = max(0.0, time.perf_counter() - self._render_started_at)
        return f" in {elapsed:.2f}s"

    def _update_rule_match_counts(self) -> None:
        if self._atoms is None:
            self.rule_panel.set_match_counts(None)
            return
        rules = self._current_rules()
        if len(rules) == 0:
            self.rule_panel.set_match_counts([])
            return
        counts = [0] * len(rules)
        atom_types = np.asarray(self._atoms.type_idx, dtype=np.int32)
        if atom_types.size > 1:
            hist = np.bincount(atom_types[1:], minlength=len(rules) + 1)
            for index in range(len(rules)):
                counts[index] = int(hist[index + 1]) if index + 1 < len(hist) else 0
        self.rule_panel.set_match_counts(counts)

    def _current_rules(self) -> list[SelectionRule]:
        return list(self.rule_panel.value)

    def _sync_preview_transform(self) -> None:
        transform = self.transform_panel.value
        self.viewport.set_preview_transform(
            scale=float(transform["scale"]),
            xrot=float(transform["xrot"]),
            yrot=float(transform["yrot"]),
            zrot=float(transform["zrot"]),
            xtran=float(transform["xtran"]),
            ytran=float(transform["ytran"]),
            ztran=float(transform["ztran"]),
        )

    def _sync_preview_style(self) -> None:
        world = self.world_panel.value
        self.viewport.set_preview_world(
            background=_hex_to_rgb(str(world.get("background", "#0b0e14"))),
            fog_color=_hex_to_rgb(str(world.get("fog", "#ffffff"))),
            fog_front=float(world.get("fog_front", 1.0)),
            fog_back=float(world.get("fog_back", 1.0)),
        )

    def _selected_render_dimensions(self) -> tuple[int, int]:
        if self.render_size_mode_combo.currentText() != "Custom":
            # Match Fortran-style auto framing: negative values request
            # auto-size with extra border to avoid edge clipping.
            return (-_AUTO_FRAME_PAD, -_AUTO_FRAME_PAD)
        return (int(self.render_width_spin.value()), int(self.render_height_spin.value()))

    def _sync_render_size_controls(self) -> None:
        custom = self.render_size_mode_combo.currentText() == "Custom"
        self.render_width_spin.setEnabled(custom)
        self.render_height_spin.setEnabled(custom)

    def _on_render_size_changed(self, _value: object) -> None:
        self._sync_render_size_controls()
        if self._suspend_panel_callbacks:
            return
        self._schedule_render_dimensions_update()
        self.statusBar().showMessage("Render size updated.")

    def _schedule_render_dimensions_update(self) -> None:
        if self._dimensions_timer.isActive():
            self._dimensions_timer.stop()
        self._dimensions_timer.start(60)

    def _update_render_dimensions_label(self) -> None:
        width, height = self._selected_render_dimensions()
        if width > 0 and height > 0:
            self.render_dims_label.setText(f"Output: {width}\u00d7{height}")
            return

        if self.pdb_path is None:
            self.render_dims_label.setText("Output: auto")
            return
        if self._atoms is None:
            try:
                self._load_atoms_if_needed()
            except Exception:
                self.render_dims_label.setText("Output: auto")
                return
        if self._atoms is None:
            self.render_dims_label.setText("Output: auto")
            return
        try:
            est_w, est_h = estimate_render_size(self._atoms, self._build_params())
        except Exception:
            self.render_dims_label.setText("Output: auto")
            return
        if est_w > 0 and est_h > 0:
            self.render_dims_label.setText(f"Output: {est_w}\u00d7{est_h}")
        else:
            self.render_dims_label.setText("Output: auto")

    def _fit_view(self) -> None:
        if self.pdb_path is None:
            self.statusBar().showMessage("Load a PDB first.")
            return
        self._load_atoms_if_needed()
        if self._atoms is None or int(getattr(self._atoms, "n", 0)) <= 0:
            self.statusBar().showMessage("No atoms available to fit.")
            return

        view_w = max(320, int(self.viewport.width()))
        view_h = max(240, int(self.viewport.height()))

        try:
            params = self._build_params()
        except Exception as exc:
            self.statusBar().showMessage(str(exc))
            return
        params.world.width = 0
        params.world.height = 0
        params.transform.translate = (0.0, 0.0, 0.0)

        try:
            est_w, est_h = estimate_render_size(self._atoms, params)
        except Exception as exc:
            self.statusBar().showMessage(f"Fit to view failed: {exc}")
            return
        if est_w <= 0 or est_h <= 0:
            self.statusBar().showMessage("Fit to view failed: invalid bounds.")
            return

        current_scale = float(self.transform_panel.value["scale"])
        fit_factor = min((0.92 * float(view_w)) / float(est_w), (0.92 * float(view_h)) / float(est_h))
        fit_factor = max(0.05, min(20.0, fit_factor))
        next_scale = max(1.0, min(500.0, current_scale * fit_factor))
        self.transform_panel.set_value(scale=next_scale, xtran=0.0, ytran=0.0, ztran=0.0)
        self.statusBar().showMessage(f"Fitted view (scale {next_scale:.2f})")

    def _clear_interactive_preview(self) -> None:
        self._preview_request_id += 1
        self._latest_preview_request_id = self._preview_request_id
        self._preview_pending = False
        if hasattr(self, "_preview_timer"):
            self._preview_timer.stop()
        self.viewport.update_preview_image(None)

    def _preview_quality_mode(self) -> str:
        if not hasattr(self, "preview_quality_combo"):
            return "balanced"
        mode = str(self.preview_quality_combo.currentText()).strip().lower()
        if mode not in {"fast", "balanced", "high"}:
            return "balanced"
        return mode

    def _on_preview_quality_changed(self, _value: str) -> None:
        self._request_preview_render()
        self.statusBar().showMessage(f"Preview quality: {self.preview_quality_combo.currentText()}")

    def _preview_target_side(self, atom_count: int, mode: str) -> float:
        if mode == "fast":
            if atom_count >= 90000:
                return 190.0
            if atom_count >= 60000:
                return 240.0
            if atom_count >= 30000:
                return 300.0
            if atom_count >= 12000:
                return 370.0
            return 480.0

        if mode == "high":
            if atom_count >= 90000:
                return 280.0
            if atom_count >= 60000:
                return 360.0
            if atom_count >= 30000:
                return 450.0
            if atom_count >= 12000:
                return 560.0
            return 700.0

        # balanced
        if atom_count >= 90000:
            return 240.0
        if atom_count >= 60000:
            return 300.0
        if atom_count >= 30000:
            return 380.0
        if atom_count >= 12000:
            return 460.0
        return 560.0

    def _preview_dimensions(self) -> tuple[int, int]:
        view_w = max(1, int(self.viewport.width()))
        view_h = max(1, int(self.viewport.height()))
        max_side = max(view_w, view_h)
        if max_side <= 0:
            return (320, 240)

        atom_count = int(getattr(self._atoms, "n", 0)) if self._atoms is not None else 0
        mode = self._preview_quality_mode()
        target_side = self._preview_target_side(atom_count, mode)

        scale = min(1.0, target_side / float(max_side))
        width = max(160, int(round(view_w * scale)))
        height = max(120, int(round(view_h * scale)))
        width = max(2, (width // 2) * 2)
        height = max(2, (height // 2) * 2)
        return (width, height)

    def _preview_reference_dimensions(self, params: RenderParams) -> tuple[int, int]:
        if params.world.width > 0 and params.world.height > 0:
            return (int(params.world.width), int(params.world.height))

        if self._last_result is not None:
            last_w = int(getattr(self._last_result, "width", 0))
            last_h = int(getattr(self._last_result, "height", 0))
            if last_w > 0 and last_h > 0:
                return (last_w, last_h)

        if self._atoms is None:
            return (max(1, int(self.viewport.width())), max(1, int(self.viewport.height())))

        try:
            est_w, est_h = estimate_render_size(self._atoms, params)
        except Exception:
            return (max(1, int(self.viewport.width())), max(1, int(self.viewport.height())))
        if est_w > 0 and est_h > 0:
            return (int(est_w), int(est_h))
        return (max(1, int(self.viewport.width())), max(1, int(self.viewport.height())))

    def _build_preview_params(self) -> RenderParams:
        params = self._build_params()
        ref_w, ref_h = self._preview_reference_dimensions(params)
        width, height = self._preview_dimensions()
        params.world.width = int(width)
        params.world.height = int(height)
        # Interactive previews render at reduced resolution. Keep framing stable
        # by scaling atom size relative to the full render target dimensions.
        downsample = min(float(width) / float(max(1, ref_w)), float(height) / float(max(1, ref_h)))
        params.transform.scale = float(params.transform.scale) * max(0.01, min(1.0, downsample))

        atom_count = int(getattr(self._atoms, "n", 0)) if self._atoms is not None else 0
        mode = self._preview_quality_mode()

        # Shadows/outlines are the heaviest post-passes.
        # "Fast" mode prioritizes interaction speed over preview fidelity.
        if mode == "fast":
            params.world.shadows = False
            params.outlines.enabled = False
        elif mode == "balanced" and atom_count >= 18000:
            params.world.shadows = False
            if atom_count >= 50000:
                params.outlines.enabled = False
        elif mode == "high" and atom_count >= 90000:
            params.world.shadows = False
        if mode != "fast" and atom_count >= 110000:
            params.outlines.enabled = False
        return params

    def _build_interactive_rerender_params(self) -> RenderParams:
        """Build drag/zoom params that preserve style while trimming heavy passes."""
        params = self._build_params()
        # Keep outlines and framing stable during interaction while dropping the
        # most expensive post-pass. A full render with shadows is submitted on
        # settle.
        params.world.shadows = False
        return params

    def _schedule_interactive_settle_render(self) -> None:
        if not self.auto_render_on_drag.isChecked():
            return
        self._interactive_settle_timer.start(_INTERACTIVE_SETTLE_MS)

    def _on_interactive_settle_timeout(self) -> None:
        if not self.auto_render_on_drag.isChecked():
            return
        self._render(interactive=False)

    def _request_preview_render(self) -> None:
        self._preview_pending = True
        if not self._preview_timer.isActive():
            # Coalesce bursty UI updates to avoid oversubmitting preview jobs.
            mode = self._preview_quality_mode()
            delay_ms = 14 if mode == "fast" else (24 if mode == "high" else 20)
            self._preview_timer.start(delay_ms)

    def _submit_preview_render(self) -> None:
        if not self._preview_pending:
            return
        self._preview_pending = False
        if self.pdb_path is None:
            return
        if self._atoms is None:
            self._load_atoms_if_needed()
        if self._atoms is None:
            return
        try:
            params = self._build_preview_params()
        except Exception:
            return
        self._preview_request_id += 1
        request_id = self._preview_request_id
        self._latest_preview_request_id = request_id
        request = RenderRequest(params=params, atoms=self._atoms, request_id=request_id)
        if hasattr(self.preview_worker, "submit"):
            self.preview_worker.submit(request)
        else:
            self.preview_worker.start(request)

    def _push_preview_scene(self) -> None:
        if self._atoms is None or int(getattr(self._atoms, "n", 0)) <= 0:
            self.viewport.set_preview_scene(None, None, None)
            self._clear_interactive_preview()
            return

        n = int(self._atoms.n)
        coords = np.asarray(self._atoms.coord[1 : n + 1], dtype=np.float32)
        atom_types = np.asarray(self._atoms.type_idx[1 : n + 1], dtype=np.int32)
        if coords.shape[0] == 0:
            self.viewport.set_preview_scene(None, None, None)
            self._clear_interactive_preview()
            return

        rules = self._current_rules()
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
            self.viewport.set_preview_scene(None, None, None)
            self._clear_interactive_preview()
            return

        self.viewport.set_preview_scene(coords[visible], colors[visible], radii[visible])
        self._clear_interactive_preview()

    def _save_settings(self) -> None:
        if self.pdb_path is None:
            self.statusBar().showMessage("Load a PDB before saving settings.")
            return
        try:
            params = self._build_params()
        except Exception as exc:
            self.statusBar().showMessage(str(exc))
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save settings",
            str(Path.home() / "illustrate_settings.json"),
            "JSON settings (*.json)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.write_text(params_to_json(params), encoding="utf-8")
        self.statusBar().showMessage(f"Saved settings {output_path}")

    def _check_for_updates(self) -> None:
        result = check_for_updates()
        if result.status == "error":
            QMessageBox.information(
                self,
                "Check Updates",
                f"Could not check for updates.\n\n{result.message}",
            )
            return
        if result.status == "up_to_date":
            QMessageBox.information(
                self,
                "Check Updates",
                result.message,
            )
            return

        latest = result.latest_version or "(unknown)"
        target_url = result.download_url or RELEASES_PAGE_URL
        reply = QMessageBox.question(
            self,
            "Update Available",
            (
                f"A newer version is available: {latest}\n"
                f"Current version: {result.current_version}\n\n"
                "Open the download page now?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            webbrowser.open(target_url)

    def _load_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load settings",
            str(Path.home()),
            "JSON settings (*.json)",
        )
        if not path:
            return
        source = Path(path)
        try:
            params = params_from_json(source.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Load settings failed", str(exc))
            return

        self._apply_loaded_params(params, source=source)

    def _apply_loaded_params(self, params: RenderParams, *, source: Path) -> None:
        self._suspend_panel_callbacks = True
        try:
            transform = params.transform
            self.transform_panel.set_value(
                scale=transform.scale,
                xrot=transform.rotations[2][1] if len(transform.rotations) > 2 else 0.0,
                yrot=transform.rotations[1][1] if len(transform.rotations) > 1 else 0.0,
                zrot=transform.rotations[0][1] if len(transform.rotations) > 0 else 90.0,
                xtran=transform.translate[0],
                ytran=transform.translate[1],
                ztran=transform.translate[2],
            )
            world = params.world
            self.world_panel.set_value(
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
            self.outline_panel.set_value(
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
            self.rule_panel.set_value(params.rules)
            if params.world.width > 0 and params.world.height > 0:
                self.render_size_mode_combo.setCurrentText("Custom")
                self.render_width_spin.setValue(int(params.world.width))
                self.render_height_spin.setValue(int(params.world.height))
            else:
                self.render_size_mode_combo.setCurrentText("Auto")
        finally:
            self._suspend_panel_callbacks = False

        loaded_pdb = str(params.pdb_path).strip()
        loaded_file = None
        if loaded_pdb:
            candidate = Path(loaded_pdb)
            if not candidate.is_absolute():
                candidate = (source.parent / candidate).resolve()
            if candidate.exists():
                loaded_file = candidate

        if loaded_file is not None:
            self.pdb_path = str(loaded_file)
            self.setWindowTitle(f"Illustrate \u2014 {loaded_file.name}")
            self._set_loaded_model_label(self.pdb_path)
            self._atoms = None
            self._atoms_signature = ""
            self.viewport.set_preview_scene(None, None, None)
            self._clear_interactive_preview()
            self._load_atoms_if_needed()

        self._sync_render_size_controls()
        self._sync_preview_transform()
        self._sync_preview_style()
        self._request_preview_render()
        self._schedule_render_dimensions_update()
        if loaded_file is None and loaded_pdb:
            self.statusBar().showMessage(f"Loaded settings from {source.name} (PDB not found: {loaded_pdb})")
            if self.pdb_path is None:
                self._set_loaded_model_label(None)
        else:
            self.statusBar().showMessage(f"Loaded settings from {source.name}")

    def _open_pdb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PDB",
            str(Path.home()),
            "PDB Files (*.pdb *.ent);;All Files (*)",
        )
        if not path:
            return
        self.pdb_path = path
        self.setWindowTitle(f"Illustrate \u2014 {Path(path).name}")
        self._set_loaded_model_label(path)
        self._atoms = None
        self._atoms_signature = ""
        self.viewport.set_preview_scene(None, None, None)
        self._clear_interactive_preview()
        self._load_atoms_if_needed()
        self._schedule_render_dimensions_update()
        self.statusBar().showMessage(f"Loaded {Path(path).name}. Click Render.")

    def _fetch_pdb(self) -> None:
        pdb_id = self.pdb_id_input.text().strip()
        if not pdb_id:
            self.statusBar().showMessage("Enter a PDB ID first.")
            return
        self.fetch_action.setEnabled(False)
        self.statusBar().showMessage(f"Fetching {pdb_id}\u2026")

        def _do_fetch():
            try:
                path = fetch_pdb(pdb_id)
                self._fetch_done_signal.emit(str(path), pdb_id.upper())
            except Exception as exc:
                self._fetch_failed_signal.emit(str(exc))

        threading.Thread(target=_do_fetch, daemon=True).start()

    def _on_fetch_done(self, path: str, pdb_id: str) -> None:
        self.fetch_action.setEnabled(True)
        self.pdb_path = path
        self.setWindowTitle(f"Illustrate \u2014 {pdb_id}.pdb")
        self._set_loaded_model_label(path)
        self._atoms = None
        self._atoms_signature = ""
        self.viewport.set_preview_scene(None, None, None)
        self._clear_interactive_preview()
        self._load_atoms_if_needed()
        self._schedule_render_dimensions_update()
        self.statusBar().showMessage(f"Fetched {pdb_id}.pdb. Rendering\u2026")
        self._render()

    def _on_fetch_failed(self, message: str) -> None:
        self.fetch_action.setEnabled(True)
        self.statusBar().showMessage(f"Fetch failed: {message}")

    def _on_suggestion_selected(self, pdb_id: str) -> None:
        self.pdb_id_input.setText(pdb_id)
        self._fetch_pdb()

    def _build_params(self) -> RenderParams:
        if self.pdb_path is None:
            raise RuntimeError("no pdb selected")
        render_w, render_h = self._selected_render_dimensions()
        return _to_render_params(
            self.pdb_path,
            self._current_rules(),
            self.transform_panel.value,
            self.world_panel.value,
            self.outline_panel.value,
            render_width=render_w,
            render_height=render_h,
        )

    def _load_atoms_if_needed(self) -> None:
        if self.pdb_path is None:
            self.rule_panel.set_match_counts(None)
            return
        rules = self._current_rules()
        signature = _rules_signature(rules)
        if self._atoms is not None and signature == self._atoms_signature:
            self._push_preview_scene()
            self._update_rule_match_counts()
            self._schedule_render_dimensions_update()
            return
        self._atoms = load_pdb(self.pdb_path, rules)
        self._atoms_signature = signature
        self._push_preview_scene()
        self._update_rule_match_counts()
        self._schedule_render_dimensions_update()

    def _render_interactive(self) -> None:
        self._render(interactive=True)
        self._schedule_interactive_settle_render()

    def _render(self, _checked: bool = False, *, interactive: bool = False) -> None:
        if self.pdb_path is None:
            self.statusBar().showMessage("Load a PDB first.")
            return
        if not interactive and self._interactive_settle_timer.isActive():
            self._interactive_settle_timer.stop()
        try:
            params = self._build_interactive_rerender_params() if interactive else self._build_params()
        except Exception as exc:
            self.statusBar().showMessage(str(exc))
            return

        self._load_atoms_if_needed()
        if self._atoms is None:
            self.statusBar().showMessage("Could not load atoms from PDB.")
            return

        if not interactive:
            self.render_action.setEnabled(False)
            self._set_render_busy(True)

        self._render_request_id += 1
        request_id = -self._render_request_id if interactive else self._render_request_id
        request = RenderRequest(params=params, atoms=self._atoms, request_id=request_id)
        if hasattr(self.worker, "submit"):
            self.worker.submit(request)
        else:
            self.worker.start(request)
        self.statusBar().showMessage("Re-rendering\u2026" if interactive else "Rendering\u2026")

    def _on_render_done(self, result) -> None:
        render_mode = "full"
        if isinstance(result, tuple) and len(result) == 2:
            request_id, payload = result
            result = payload
            render_mode = "interactive" if int(request_id) < 0 else "full"
        if render_mode == "full":
            self.render_action.setEnabled(True)
        elapsed = self._render_elapsed_suffix()
        if render_mode == "full":
            self._set_render_busy(False)
        self.copy_action.setEnabled(True)
        if render_mode == "full":
            self._params_dirty = False
        self._update_render_btn_style()
        if not hasattr(result, "rgb"):
            self.statusBar().showMessage(f"Render returned no image{elapsed}.")
            return
        if render_mode == "full":
            self._last_result = result
        rgb = np.asarray(result.rgb, dtype=np.uint8)
        opacity = np.asarray(result.opacity, dtype=np.uint8) if hasattr(result, "opacity") else None
        self._display_image(rgb, opacity)
        self._clear_interactive_preview()
        self._schedule_render_dimensions_update()
        verb = "Re-rendered" if render_mode == "interactive" else "Rendered"
        self.statusBar().showMessage(f"{verb} {result.width}\u00d7{result.height}{elapsed}")

    def _on_render_failed(self, message: str) -> None:
        self.render_action.setEnabled(True)
        elapsed = self._render_elapsed_suffix()
        self._set_render_busy(False)
        QMessageBox.critical(self, "Render failed", message)
        self.statusBar().showMessage(f"Render failed{elapsed}")

    def _on_preview_done(self, payload) -> None:
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        request_id, result = payload
        if int(request_id) != self._latest_preview_request_id:
            return
        if not hasattr(result, "rgb"):
            return
        rgb = np.asarray(result.rgb, dtype=np.uint8)
        opacity = np.asarray(result.opacity, dtype=np.uint8) if hasattr(result, "opacity") else None
        self._display_preview_image(rgb, opacity)

    def _on_preview_failed(self, _message: str) -> None:
        # Keep interactive viewport responsive if preview rendering fails.
        self.viewport.update_preview_image(None)

    @staticmethod
    def _to_rgba(rgb: np.ndarray, opacity: np.ndarray | None) -> np.ndarray | None:
        if opacity is None:
            return None
        if opacity.ndim != 2:
            return None
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None
        if opacity.shape != rgb.shape[:2]:
            return None
        return np.dstack((rgb, opacity))

    def _pixmap_from_image(self, rgb: np.ndarray, opacity: np.ndarray | None = None) -> QPixmap | None:
        if rgb.size == 0:
            return None

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None

        # Renderer outputs row-major image buffers ([height, width, channels]),
        # matching QImage expectations.
        rgb_u8 = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        height, width = rgb_u8.shape[:2]

        if opacity is not None and opacity.ndim == 2 and opacity.shape == (height, width):
            rgba_u8 = np.empty((height, width, 4), dtype=np.uint8)
            rgba_u8[:, :, :3] = rgb_u8
            rgba_u8[:, :, 3] = np.ascontiguousarray(np.asarray(opacity, dtype=np.uint8))
            qimage = QImage(
                rgba_u8.data,
                width,
                height,
                int(rgba_u8.strides[0]),
                QImage.Format.Format_RGBA8888,
            )
            return QPixmap.fromImage(qimage.copy())

        # Backward-compatibility for any legacy x-major opacity buffers.
        if opacity is not None and opacity.ndim == 2 and opacity.shape == (width, height):
            opacity_u8 = np.ascontiguousarray(np.asarray(opacity, dtype=np.uint8).transpose(1, 0))
            rgba_u8 = np.empty((height, width, 4), dtype=np.uint8)
            rgba_u8[:, :, :3] = rgb_u8
            rgba_u8[:, :, 3] = opacity_u8
            qimage = QImage(
                rgba_u8.data,
                width,
                height,
                int(rgba_u8.strides[0]),
                QImage.Format.Format_RGBA8888,
            )
            return QPixmap.fromImage(qimage.copy())

        qimage = QImage(
            rgb_u8.data,
            width,
            height,
            int(rgb_u8.strides[0]),
            QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(qimage.copy())

    def _display_image(self, rgb: np.ndarray, opacity: np.ndarray | None = None) -> None:
        qpix = self._pixmap_from_image(rgb, opacity)
        if qpix is None:
            return
        self.viewport.update_image(qpix)

    def _display_preview_image(self, rgb: np.ndarray, opacity: np.ndarray | None = None) -> None:
        qpix = self._pixmap_from_image(rgb, opacity)
        if qpix is None:
            return
        self.viewport.update_preview_image(qpix)

    def _export_png(self) -> None:
        if self._last_result is None:
            self.statusBar().showMessage("Render first before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", str(Path.home() / "render.png"), "PNG image (*.png)"
        )
        if not path:
            return
        rgb = np.asarray(self._last_result.rgb, dtype=np.uint8)
        opacity = np.asarray(self._last_result.opacity, dtype=np.uint8) if hasattr(self._last_result, "opacity") else None
        rgba = self._to_rgba(rgb, opacity)
        if rgba is None:
            write_png(path, self._last_result.rgb)
        else:
            PILImage.fromarray(rgba, mode="RGBA").save(path, format="PNG")
        self.statusBar().showMessage(f"Saved {path}")

    def _export_svg(self) -> None:
        if self._last_result is None:
            self.statusBar().showMessage("Render first before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", str(Path.home() / "render.svg"), "SVG image (*.svg)"
        )
        if not path:
            return
        rgb = np.asarray(self._last_result.rgb, dtype=np.uint8)
        opacity = np.asarray(self._last_result.opacity, dtype=np.uint8) if hasattr(self._last_result, "opacity") else None
        rgba = self._to_rgba(rgb, opacity)
        write_svg(path, rgba if rgba is not None else rgb)
        self.statusBar().showMessage(f"Saved {path}")

    def _copy_to_clipboard(self) -> None:
        if self._last_result is None:
            return
        rgb = np.asarray(self._last_result.rgb, dtype=np.uint8)
        opacity = np.asarray(self._last_result.opacity, dtype=np.uint8) if hasattr(self._last_result, "opacity") else None
        qpix = self._pixmap_from_image(rgb, opacity)
        if qpix is None:
            return
        QGuiApplication.clipboard().setPixmap(qpix)
        self.statusBar().showMessage("Copied to clipboard")

    def _apply_preset(self, index: int) -> None:
        presets = self._preset_items
        if index < 0 or index >= len(presets):
            return
        _name, preset = presets[index]
        world = preset.world
        outlines = preset.outlines
        transform = preset.transform
        self._suspend_panel_callbacks = True
        try:
            self.transform_panel.set_value(
                scale=transform.scale,
                xrot=transform.rotations[2][1] if len(transform.rotations) > 2 else 0.0,
                yrot=transform.rotations[1][1] if len(transform.rotations) > 1 else 0.0,
                zrot=transform.rotations[0][1] if len(transform.rotations) > 0 else 90.0,
                xtran=transform.translate[0],
                ytran=transform.translate[1],
                ztran=transform.translate[2],
            )
            self.world_panel.set_value(
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
            self.outline_panel.set_value(
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
            self.rule_panel.set_value(preset.rules)
            if world.width > 0 and world.height > 0:
                self.render_size_mode_combo.setCurrentText("Custom")
                self.render_width_spin.setValue(int(world.width))
                self.render_height_spin.setValue(int(world.height))
            else:
                self.render_size_mode_combo.setCurrentText("Auto")
        finally:
            self._suspend_panel_callbacks = False
        self._sync_preview_transform()
        self._sync_preview_style()
        self._request_preview_render()
        self.rule_panel.set_match_counts(None)
        self._schedule_render_dimensions_update()
        self._render()

    def _on_viewport_rotation(self, x_delta: float, y_delta: float) -> None:
        self._suppress_preview_render_once = True
        self.transform_panel.set_value(
            xrot=float(self.transform_panel.value["xrot"]) + y_delta,
            yrot=float(self.transform_panel.value["yrot"]) + x_delta,
        )
        if self.auto_render_on_drag.isChecked():
            self._render_interactive()

    def _on_viewport_zoom(self, factor: float) -> None:
        self._suppress_preview_render_once = True
        self.transform_panel.set_value(
            scale=max(1.0, float(self.transform_panel.value["scale"]) * float(factor))
        )
        if self.auto_render_on_drag.isChecked():
            self._render_interactive()

    def closeEvent(self, event) -> None:
        self._pdb_completer.cleanup()
        self._preview_timer.stop()
        self._interactive_settle_timer.stop()
        self._dimensions_timer.stop()
        if hasattr(self.worker, "isRunning") and self.worker.isRunning():
            self.statusBar().showMessage("Waiting for render worker to finish...")
            self.worker.wait()
        if hasattr(self.preview_worker, "isRunning") and self.preview_worker.isRunning():
            self.statusBar().showMessage("Waiting for preview worker to finish...")
            self.preview_worker.wait()
        super().closeEvent(event)


def run() -> int:
    app = QApplication([])
    app.setStyleSheet(DARK_STYLE)
    main = MainWindow()
    main.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
