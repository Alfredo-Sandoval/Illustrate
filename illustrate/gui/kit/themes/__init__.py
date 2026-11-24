"""Standard Fusion themes: token-resolved QSS, palette, and icons.

A theme is a color table over one of the two base schemes. ``light`` and
``dark`` are the bases; ``midnight`` recolors the dark scheme in navy and shares
its stylesheet and icons. Every theme also carries ``widgets.qss``, the rules
for this package's shell and widgets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict, cast

from PySide6.QtCore import QDir, QObject, Qt, Signal
from PySide6.QtGui import QColor, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from illustrate.gui.kit.icons import set_icon_color

__all__ = [
    "SHARED_TOKENS",
    "THEMES",
    "apply_theme",
    "colors",
    "current_theme",
    "load_stylesheet",
    "theme_events",
]

THEMES = ("dark", "light", "midnight")
# Every theme defines these, so application rules written with them work under
# every theme. Other tokens are internal to one theme's stylesheet.
SHARED_TOKENS = (
    "surface-window",
    "surface-panel",
    "surface-raised",
    "surface-control",
    "surface-hover",
    "surface-selected",
    "surface-sunken",
    "border-subtle",
    "border-default",
    "border-input",
    "border-strong",
    "border-focus",
    "text-primary",
    "text-secondary",
    "text-muted",
    "text-disabled",
    "text-selected",
    "text-link",
    "text-on-accent",
    "selection-background",
    "accent",
    "accent-hover",
    "success",
    "warning",
    "error",
    "surface-overlay",
    "border-overlay",
    "surface-tint",
)
_ROOT = Path(__file__).resolve().parent
_TOKEN = re.compile(r"@([a-z][a-z0-9-]*)")
_SCHEMES = {"dark": Qt.ColorScheme.Dark, "light": Qt.ColorScheme.Light}
_WIDGET_RULES = (_ROOT / "widgets.qss").read_text(encoding="utf-8")


class _ThemeSpec(TypedDict):
    scheme: str
    source: str
    colors: dict[str, str]
    palette: dict[str, str]
    disabled_text: str


class _ThemeEvents(QObject):
    """Announces theme switches to custom painters and plots."""

    changed = Signal(str)


theme_events = _ThemeEvents()
"""``theme_events.changed`` carries the theme name after ``apply_theme``."""

_current: str | None = None


def _spec(name: str) -> _ThemeSpec:
    if name not in THEMES:
        raise ValueError(f"Unknown theme {name!r}; expected one of {THEMES}")
    text = (_ROOT / name / "colors.json").read_text(encoding="utf-8")
    spec = cast(_ThemeSpec, json.loads(text))
    if spec["scheme"] not in _SCHEMES:
        raise ValueError(f"Theme {name!r} has unknown scheme {spec['scheme']!r}")
    return spec


def _scheme_dir(spec: _ThemeSpec) -> Path:
    """Return the base scheme's folder, which owns the stylesheet and icons."""

    return _ROOT / spec["scheme"]


def colors(name: str) -> dict[str, str]:
    """Return the theme's color tokens, for custom painters and plots."""

    return dict(_spec(name)["colors"])


def current_theme() -> str | None:
    """Return the name of the last applied theme, or None before the first."""

    return _current


def load_stylesheet(name: str, extra_stylesheet: str = "") -> str:
    """Return the theme QSS, the widget rules, and application rules, resolved.

    The widget rules and application rules may use only ``SHARED_TOKENS``, so
    the same rules work under every theme. An unknown token raises instead of
    reaching Qt.
    """

    spec = _spec(name)
    table = spec["colors"]
    layered = f"{_WIDGET_RULES}\n{extra_stylesheet}"
    foreign = sorted(set(_TOKEN.findall(layered)) - set(SHARED_TOKENS))
    if foreign:
        raise ValueError(f"Application stylesheet uses non-shared tokens: {foreign}")

    def color(match: re.Match[str]) -> str:
        token = match.group(1)
        if token not in table:
            raise ValueError(f"Unknown color @{token} for the {name} theme")
        return table[token]

    base = (_scheme_dir(spec) / "theme.qss").read_text(encoding="utf-8")
    return _TOKEN.sub(color, f"{base}\n{layered}")


def _palette(spec: _ThemeSpec) -> QPalette:
    table = spec["colors"]
    palette = QPalette()
    for role, token in spec["palette"].items():
        palette.setColor(QPalette.ColorRole[role], QColor(table[token]))
    disabled = QColor(table[spec["disabled_text"]])
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return palette


def apply_theme(app: QApplication, name: str, extra_stylesheet: str = "") -> None:
    """Install Fusion, the 13 px system font, the palette, icons, and the QSS.

    Call again with another name to switch themes at runtime. Stylesheet icons
    resolve through the ``theme:`` search prefix; interface icons on buttons
    are redrawn in the new theme's secondary text color.
    """

    global _current
    spec = _spec(name)
    stylesheet = load_stylesheet(name, extra_stylesheet)
    if app.setStyle("Fusion") is None:
        raise RuntimeError("Qt Fusion style is unavailable")
    app.styleHints().setColorScheme(_SCHEMES[spec["scheme"]])
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    font.setPixelSize(13)
    app.setFont(font)
    app.setPalette(_palette(spec))
    QDir.setSearchPaths("theme", [str(_scheme_dir(spec) / "icons")])
    app.setStyleSheet(stylesheet)
    set_icon_color(spec["colors"]["text-secondary"])
    _current = name
    theme_events.changed.emit(name)
