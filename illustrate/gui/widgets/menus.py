"""Menu setup and the inspector's auto and custom output-size controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from illustrate.gui.kit import add_appearance_menu
from PySide6.QtWidgets import QApplication, QComboBox, QFormLayout, QSpinBox, QWidget


if TYPE_CHECKING:
    from illustrate.gui.main_window import MainWindow


def build_main_menus(window: MainWindow, app: QApplication) -> None:
    actions = window.main_actions
    file_menu = window.menuBar().addMenu("&File")
    file_menu.addAction(actions.open_pdb)
    file_menu.addAction(actions.fetch_pdb)
    file_menu.addSeparator()
    file_menu.addAction(actions.load_settings)
    file_menu.addAction(actions.save_settings)
    file_menu.addAction(actions.save_preset)
    file_menu.addSeparator()
    file_menu.addAction(actions.export_png)
    file_menu.addAction(actions.export_svg)

    edit_menu = window.menuBar().addMenu("&Edit")
    edit_menu.addAction(actions.copy_to_clipboard)

    view_menu = window.menuBar().addMenu("&View")
    view_menu.addAction(window.toggle_sidebar_action)
    view_menu.addAction(window.toggle_inspector_action)
    view_menu.addAction(window.toggle_rules_action)
    view_menu.addSeparator()
    view_menu.addAction(actions.fit_view)
    view_menu.addSeparator()
    add_appearance_menu(view_menu.addMenu("&Appearance"), app)

    render_menu = window.menuBar().addMenu("&Render")
    render_menu.addAction(actions.render)

    help_menu = window.menuBar().addMenu("&Help")
    help_menu.addAction(actions.check_updates)


@dataclass(frozen=True, slots=True)
class OutputSizeControls:
    panel: QWidget
    mode_combo: QComboBox
    width_spin: QSpinBox
    height_spin: QSpinBox


def build_output_size_controls() -> OutputSizeControls:
    panel = QWidget()
    form = QFormLayout(panel)

    mode_combo = QComboBox(panel)
    mode_combo.addItems(["Auto", "Custom"])
    form.addRow("Output size", mode_combo)

    width_spin = QSpinBox(panel)
    width_spin.setRange(64, 3000)
    width_spin.setValue(2000)
    width_spin.setSuffix(" px")
    form.addRow("Width", width_spin)
    height_spin = QSpinBox(panel)
    height_spin.setRange(64, 3000)
    height_spin.setValue(2000)
    height_spin.setSuffix(" px")
    form.addRow("Height", height_spin)

    # Width and height apply only to a custom size; Auto sizes the image to
    # the structure, and the render bar reports the result.
    def show_custom_size(mode: str) -> None:
        for spin in (width_spin, height_spin):
            form.setRowVisible(spin, mode == "Custom")

    mode_combo.currentTextChanged.connect(show_custom_size)
    show_custom_size(mode_combo.currentText())
    return OutputSizeControls(panel, mode_combo, width_spin, height_spin)
