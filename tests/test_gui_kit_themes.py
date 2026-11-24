from __future__ import annotations

import pytest
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QPushButton

from illustrate.gui.kit import THEMES, apply_theme, colors, current_theme, load_stylesheet, theme_events
from illustrate.gui.kit.icons import set_button_icon


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


@pytest.mark.parametrize("name", THEMES)
def test_every_theme_resolves_its_widget_rules(name: str) -> None:
    # The widget rules ride along with every theme, so each theme must define
    # every token they use; a leftover at-sign would reach Qt as invalid QSS.
    stylesheet = load_stylesheet(name, 'QLabel#appRule { color: @accent; }')

    assert "@" not in stylesheet
    assert "QFrame#actionSurface" in stylesheet
    assert "QLabel#appRule" in stylesheet


def test_application_rules_may_only_use_shared_tokens() -> None:
    with pytest.raises(ValueError, match="non-shared tokens"):
        load_stylesheet("dark", "QLabel { color: @neutral-light; }")


def test_apply_theme_installs_palette_and_announces_the_switch(app: QApplication) -> None:
    seen: list[str] = []
    theme_events.changed.connect(seen.append)
    try:
        apply_theme(app, "midnight")
    finally:
        theme_events.changed.disconnect(seen.append)

    assert seen == ["midnight"]
    assert current_theme() == "midnight"
    window = app.palette().color(QPalette.ColorRole.Window).name()
    assert window == colors("midnight")["surface-window"]


def test_theme_switch_redraws_interface_icons(app: QApplication) -> None:
    button = QPushButton()
    set_button_icon(button, "folder")
    apply_theme(app, "light")
    light_icon = button.icon().pixmap(20, 20).toImage()

    apply_theme(app, "dark")

    assert button.icon().pixmap(20, 20).toImage() != light_icon
