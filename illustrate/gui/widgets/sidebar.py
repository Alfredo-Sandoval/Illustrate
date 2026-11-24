"""Navigation dock contents: open or fetch a structure, pick a preset, manage settings files."""

from __future__ import annotations

from dataclasses import dataclass

from illustrate.gui.kit import NavigationDock, action_button
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QLineEdit, QListWidget

from illustrate.gui.actions import MainActions


@dataclass(frozen=True, slots=True)
class NavigationControls:
    pdb_search: QLineEdit
    preset_list: QListWidget


def fill_navigation(dock: NavigationDock, actions: MainActions) -> NavigationControls:
    """Add Illustrate's Structure, Presets, and Settings groups to ``dock``."""

    panel = dock.panel
    dock.add_section("Structure")
    search_row = QHBoxLayout()
    search_row.setContentsMargins(0, 0, 0, 4)
    search_row.setSpacing(4)
    pdb_search = QLineEdit(panel)
    pdb_search.setPlaceholderText("PDB ID or name")
    pdb_search.setAccessibleName("Search PDB")
    search_row.addWidget(pdb_search, 1)
    fetch_button = action_button(actions.fetch_pdb, "icon", panel, icon="download")
    fetch_button.setFixedSize(30, 30)
    search_row.addWidget(fetch_button)
    dock.add_layout(search_row)
    dock.add_action(actions.open_pdb, icon="folder", text="Open PDB file…")

    dock.add_section("Presets")
    preset_list = QListWidget(panel)
    preset_list.setAccessibleName("Presets")
    preset_list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    preset_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    preset_list.setTextElideMode(Qt.TextElideMode.ElideRight)
    preset_list.setSpacing(1)
    dock.add_widget(preset_list, 1)
    dock.add_action(actions.save_preset, icon="bookmark", text="Save as preset…")

    dock.add_section("Settings")
    dock.add_action(actions.load_settings, icon="file", text="Load settings…")
    dock.add_action(actions.save_settings, icon="save", text="Save settings…")
    return NavigationControls(pdb_search, preset_list)
