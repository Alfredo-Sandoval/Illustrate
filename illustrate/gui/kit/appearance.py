"""Light, dark, or system appearance, following the OS when set to system."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QApplication, QMenu

from illustrate.gui.kit.themes import THEMES, apply_theme

__all__ = [
    "APPEARANCES",
    "Appearance",
    "AppearanceController",
    "add_appearance_menu",
    "appearance_controller",
    "apply_appearance",
]

Appearance = Literal["system", "light", "dark"]
APPEARANCES: tuple[Appearance, ...] = ("system", "light", "dark")


class AppearanceController(QObject):
    """Owns the application's appearance and the theme each mode installs.

    One controller lives on the ``QApplication``; ``apply_appearance`` finds
    or creates it. ``changed`` carries the appearance after every switch.
    """

    changed = Signal(str)

    def __init__(
        self,
        app: QApplication,
        *,
        dark_theme: str = "dark",
        light_theme: str = "light",
        extra_stylesheet: str | Mapping[str, str] = "",
    ) -> None:
        super().__init__(app)
        self.app = app
        self.dark_theme = dark_theme
        self.light_theme = light_theme
        self.extra_stylesheet = extra_stylesheet
        self.appearance: Appearance = "system"
        self.theme: str | None = None
        # True while installing a theme, whose own color-scheme signal is ignored.
        self._installing = False
        app.styleHints().colorSchemeChanged.connect(self._system_changed)

    def apply(self, appearance: Appearance) -> None:
        if appearance not in APPEARANCES:
            raise ValueError(f"Unknown appearance {appearance!r}; expected one of {APPEARANCES}")
        self.appearance = appearance
        if appearance == "system":
            hints = self.app.styleHints()
            hints.unsetColorScheme()
            self._system_changed(hints.colorScheme(), force=True)
        else:
            self._install(self.dark_theme if appearance == "dark" else self.light_theme)
        self.changed.emit(appearance)

    def reinstall(self) -> None:
        """Apply the current theme again, after the application rules changed."""

        if self.theme is not None:
            self._install(self.theme)

    def _install(self, theme: str) -> None:
        rules = self.extra_stylesheet
        if not isinstance(rules, str):
            if theme not in rules:
                raise ValueError(f"No application stylesheet was given for the {theme!r} theme")
            rules = rules[theme]
        apply_theme(self.app, theme, rules)
        self.theme = theme

    @Slot(Qt.ColorScheme)
    def _system_changed(self, scheme: Qt.ColorScheme, force: bool = False) -> None:
        if self.appearance != "system" or self._installing:
            return
        theme = self.dark_theme if scheme == Qt.ColorScheme.Dark else self.light_theme
        if theme == self.theme and not force:
            return
        # apply_theme pins Qt's scheme. Ignore its signals during installation,
        # then release the override so the next OS change reaches us.
        self._installing = True
        try:
            self._install(theme)
        finally:
            self.app.styleHints().unsetColorScheme()
            self._installing = False


def apply_appearance(
    app: QApplication,
    appearance: Appearance,
    *,
    dark_theme: str | None = None,
    light_theme: str | None = None,
    extra_stylesheet: str | Mapping[str, str] | None = None,
) -> AppearanceController:
    """Apply ``appearance`` and return the application's controller.

    Theme names and application rules given here are kept for later switches,
    so a menu can call ``apply_appearance(app, "dark")`` without repeating them.
    ``extra_stylesheet`` is one stylesheet for every theme, or a mapping from
    theme name to that theme's stylesheet.
    """

    controller = appearance_controller(app)
    for theme in (dark_theme, light_theme):
        if theme is not None and theme not in THEMES:
            raise ValueError(f"Unknown theme {theme!r}; expected one of {THEMES}")
    if dark_theme is not None:
        controller.dark_theme = dark_theme
    if light_theme is not None:
        controller.light_theme = light_theme
    if extra_stylesheet is not None:
        controller.extra_stylesheet = extra_stylesheet
    controller.apply(appearance)
    return controller


def add_appearance_menu(menu: QMenu, app: QApplication) -> QActionGroup:
    """Add exclusive System, Light, and Dark actions to ``menu``.

    The actions call ``apply_appearance`` and follow the controller, so a
    settings dialog that switches appearance keeps the menu in step.
    """

    controller = appearance_controller(app)
    group = QActionGroup(menu)
    group.setExclusive(True)
    for appearance in APPEARANCES:
        action = QAction(appearance.capitalize(), menu)
        action.setCheckable(True)
        action.setChecked(appearance == controller.appearance)
        action.setData(appearance)
        group.addAction(action)
        menu.addAction(action)

    def choose(action: QAction) -> None:
        controller.apply(cast(Appearance, action.data()))

    group.triggered.connect(choose)

    def follow(appearance: str) -> None:
        for action in group.actions():
            action.setChecked(action.data() == appearance)

    controller.changed.connect(follow)
    return group


def appearance_controller(app: QApplication) -> AppearanceController:
    """Return the application's controller, creating it on first use."""

    controller = app.findChild(
        AppearanceController, options=Qt.FindChildOption.FindDirectChildrenOnly
    )
    return controller if controller is not None else AppearanceController(app)
