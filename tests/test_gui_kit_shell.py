from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QSpinBox, QWidget

from illustrate.gui.kit import AppShell, CollapsibleSection, action_button, apply_theme


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


@pytest.fixture
def shell(app: QApplication) -> AppShell:
    apply_theme(app, "light")
    window = AppShell("Demo")
    window.resize(1100, 700)
    window.show()
    app.processEvents()
    return window


def test_regions_sit_on_their_sides(shell: AppShell) -> None:
    assert shell.dockWidgetArea(shell.navigation) == Qt.DockWidgetArea.LeftDockWidgetArea
    assert shell.dockWidgetArea(shell.inspector) == Qt.DockWidgetArea.RightDockWidgetArea
    assert shell.centralWidget() is shell.workspace


def test_dock_toggles_hide_and_show_the_docks(shell: AppShell, app: QApplication) -> None:
    shell.navigation.toggleViewAction().trigger()
    shell.inspector.toggleViewAction().trigger()
    app.processEvents()
    assert not shell.navigation.isVisible()
    assert not shell.inspector.isVisible()

    shell.navigation.toggleViewAction().trigger()
    app.processEvents()
    assert shell.navigation.isVisible()


def test_inspector_sections_keep_order_and_get_form_spacing(shell: AppShell) -> None:
    first = QWidget()
    form = QFormLayout(first)
    form.addRow("Scale", QSpinBox())
    shell.inspector.add_section("Transform", first)
    shell.inspector.add_section("Lighting", QLabel("body"), expanded=False)

    sections = shell.inspector.findChildren(CollapsibleSection)
    assert [section.toggle.text() for section in sections] == ["Transform", "Lighting"]
    assert form.verticalSpacing() == 8
    assert not sections[1].content.isVisibleTo(shell)


def test_action_buttons_mirror_their_action(app: QApplication) -> None:
    action = QAction("Render")
    button = action_button(action, "primary")

    action.setEnabled(False)
    assert not button.isEnabled()

    action.setEnabled(True)
    triggered: list[bool] = []
    action.triggered.connect(lambda: triggered.append(True))
    button.click()
    assert triggered == [True]


def test_toasts_keep_at_most_four_visible(shell: AppShell) -> None:
    for index in range(6):
        shell.toasts.show_toast(f"Message {index}", "info")

    assert shell.toasts.visible_count() == 4
    shell.toasts.shutdown()
