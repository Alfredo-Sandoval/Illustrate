from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMenu

from illustrate.gui.kit import add_appearance_menu, appearance_controller, apply_appearance, current_theme


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


def test_dark_appearance_uses_the_configured_dark_theme(app: QApplication) -> None:
    apply_appearance(app, "dark", dark_theme="midnight")
    assert current_theme() == "midnight"

    # Later switches keep the configured themes without repeating them.
    apply_appearance(app, "light")
    apply_appearance(app, "dark")
    assert current_theme() == "midnight"


def test_system_appearance_follows_the_color_scheme(app: QApplication) -> None:
    controller = apply_appearance(app, "system", dark_theme="dark")
    expected = "dark" if app.styleHints().colorScheme() == Qt.ColorScheme.Dark else "light"
    assert current_theme() == expected

    other = Qt.ColorScheme.Light if expected == "dark" else Qt.ColorScheme.Dark
    controller._system_changed(other)  # pyright: ignore[reportPrivateUsage]

    assert current_theme() == ("light" if expected == "dark" else "dark")


def test_menu_actions_follow_the_controller(app: QApplication) -> None:
    menu = QMenu()
    group = add_appearance_menu(menu, app)

    appearance_controller(app).apply("light")

    checked = [action.data() for action in group.actions() if action.isChecked()]
    assert checked == ["light"]


def test_each_theme_installs_its_own_application_rules(app: QApplication) -> None:
    rules = {"light": "QLabel#probe { margin: 1px; }", "dark": "QLabel#probe { margin: 2px; }"}

    apply_appearance(app, "light", dark_theme="dark", extra_stylesheet=rules)
    assert rules["light"] in app.styleSheet()
    assert rules["dark"] not in app.styleSheet()

    apply_appearance(app, "dark")
    assert rules["dark"] in app.styleSheet()
    assert rules["light"] not in app.styleSheet()


def test_a_theme_without_application_rules_is_rejected(app: QApplication) -> None:
    with pytest.raises(ValueError, match="No application stylesheet was given for the 'dark' theme"):
        apply_appearance(app, "dark", dark_theme="dark", extra_stylesheet={"light": ""})
