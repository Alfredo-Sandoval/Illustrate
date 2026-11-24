"""Render and export commands on the workspace's action bar."""

from __future__ import annotations

from illustrate.gui.kit import ActionBar, action_button, role_button
from PySide6.QtWidgets import QCheckBox, QMenu, QWidget

from illustrate.gui.actions import MainActions


class RenderBar(ActionBar):
    def __init__(self, actions: MainActions, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.copy_button = action_button(actions.copy_to_clipboard, "icon", self.surface, icon="copy")
        self.copy_button.setFixedSize(36, 36)
        self.add_trailing(self.copy_button)

        export_menu = QMenu(self)
        export_menu.addAction(actions.export_png)
        export_menu.addAction(actions.export_svg)
        self.export_button = role_button(
            "Export", "secondary", self.surface, tooltip="Export the last full render as PNG or SVG."
        )
        self.export_button.setObjectName("exportButton")
        self.export_button.setFixedHeight(36)
        self.export_button.setMenu(export_menu)
        self.add_trailing(self.export_button)

        self.render_button = action_button(actions.render, "primary", self.surface)
        self.render_button.setFixedHeight(36)
        self.render_button.setMinimumWidth(112)
        self.add_trailing(self.render_button)

        self.auto_render_checkbox = QCheckBox("Render while dragging", self)
        self.auto_render_checkbox.setChecked(True)
        self.auto_render_checkbox.setToolTip(
            "Re-render at full quality while you drag to rotate, or when you change Transform values."
        )
        self.add_footer(self.auto_render_checkbox)
        self.detail.setText("Output: auto")
