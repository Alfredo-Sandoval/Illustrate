from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if importlib.util.find_spec("PySide6") is None:
    raise pytest.skip.Exception(
        "PySide6 not installed; skipping GUI theme tests",
        allow_module_level=True,
    )

from illustrate.gui import kit
from PySide6.QtWidgets import QApplication

from illustrate.gui.themes import APP_STYLESHEETS, apply_appearance


@pytest.fixture
def app() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


@pytest.mark.parametrize("name", ["dark", "light"])
def test_app_rules_resolve_in_their_theme(name: str) -> None:
    # Non-shared tokens raise, so each stylesheet must use only tokens its theme defines.
    assert "@" not in kit.load_stylesheet(name, APP_STYLESHEETS[name])


@pytest.mark.parametrize("appearance", ["dark", "light"])
def test_apply_appearance_installs_theme_with_app_rules(
    app: QApplication, appearance: kit.Appearance
) -> None:
    apply_appearance(appearance, app)

    assert kit.current_theme() == appearance
    assert app.styleSheet() == kit.load_stylesheet(appearance, APP_STYLESHEETS[appearance])
