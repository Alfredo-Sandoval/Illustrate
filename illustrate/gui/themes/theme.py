"""Illustrate-specific styles layered over the standard desktop themes.

The local stylesheets use tokens defined by ``illustrate.gui.kit``.
"""

from __future__ import annotations

from importlib import resources

from illustrate.gui import kit
from PySide6.QtWidgets import QApplication

APP_STYLESHEETS = {
    theme: resources.files("illustrate.gui.themes").joinpath(f"{theme}.qss").read_text(encoding="utf-8")
    for theme in ("light", "dark")
}


def apply_appearance(appearance: kit.Appearance, app: QApplication | None = None) -> None:
    """Apply the chosen appearance and register Illustrate's stylesheet overrides.

    Use the active QApplication when ``app`` is omitted.
    """
    app_instance = QApplication.instance() if app is None else app
    if not isinstance(app_instance, QApplication):
        raise RuntimeError("No active QApplication is available for theme application.")
    kit.apply_appearance(app_instance, appearance, extra_stylesheet=APP_STYLESHEETS)
