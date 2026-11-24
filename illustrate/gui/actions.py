"""Shared QAction definitions for the desktop shell."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QWidget


ActionHandler = Callable[..., None]


@dataclass(frozen=True)
class MainActions:
    open_pdb: QAction
    fetch_pdb: QAction
    save_preset: QAction
    load_settings: QAction
    save_settings: QAction
    check_updates: QAction
    render: QAction
    fit_view: QAction
    export_png: QAction
    export_svg: QAction
    copy_to_clipboard: QAction


def _action(
    parent: QWidget,
    text: str,
    handler: ActionHandler,
    *,
    shortcut: str | None = None,
    status_tip: str | None = None,
) -> QAction:
    action = QAction(text, parent)
    if shortcut:
        action.setShortcut(QKeySequence(shortcut))
    if status_tip:
        action.setStatusTip(status_tip)
        action.setToolTip(status_tip)
    action.triggered.connect(handler)
    return action


def build_main_actions(parent: QWidget, handlers: dict[str, ActionHandler]) -> MainActions:
    """Create one QAction per command for menus, toolbars, and shortcuts."""

    return MainActions(
        open_pdb=_action(
            parent,
            "Open PDB",
            handlers["open_pdb"],
            shortcut="Ctrl+O",
            status_tip="Open a local PDB or ENT file.",
        ),
        fetch_pdb=_action(
            parent,
            "Fetch",
            handlers["fetch_pdb"],
            shortcut="Ctrl+L",
            status_tip="Fetch the typed PDB ID.",
        ),
        save_preset=_action(
            parent,
            "Save Preset",
            handlers["save_preset"],
            status_tip="Save the current controls as a custom preset.",
        ),
        load_settings=_action(
            parent,
            "Load Settings",
            handlers["load_settings"],
            shortcut="Ctrl+Shift+O",
            status_tip="Load render settings from JSON.",
        ),
        save_settings=_action(
            parent,
            "Save Settings",
            handlers["save_settings"],
            shortcut="Ctrl+S",
            status_tip="Save render settings to JSON.",
        ),
        check_updates=_action(
            parent,
            "Check Updates",
            handlers["check_updates"],
            status_tip="Check the release page for a newer desktop build.",
        ),
        render=_action(
            parent,
            "Render",
            handlers["render"],
            shortcut="Ctrl+R",
            status_tip="Render the current view.",
        ),
        fit_view=_action(
            parent,
            "Fit View",
            handlers["fit_view"],
            shortcut="Ctrl+0",
            status_tip="Fit the loaded structure to the viewport.",
        ),
        export_png=_action(
            parent,
            "Export PNG",
            handlers["export_png"],
            shortcut="Ctrl+E",
            status_tip="Export the last full render as a PNG image.",
        ),
        export_svg=_action(
            parent,
            "Export SVG",
            handlers["export_svg"],
            status_tip="Export the last full render as an SVG image.",
        ),
        copy_to_clipboard=_action(
            parent,
            "Copy to Clipboard",
            handlers["copy_to_clipboard"],
            status_tip="Copy the last full render to the clipboard.",
        ),
    )
