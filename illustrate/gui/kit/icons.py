"""Outline interface icons for buttons, drawn in the active theme's color.

Icons are 24-unit SVG strokes. ``set_button_icon`` tags a button with an icon
name; ``apply_theme`` redraws every tagged button in the new theme's color, so
applications never recolor icons by hand. ``register_icon`` adds an
application glyph drawn the same way.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractButton, QApplication

__all__ = [
    "ICON_NAMES",
    "interface_icon",
    "register_icon",
    "set_button_icon",
    "set_icon_color",
]

_PATHS: dict[str, str] = {
    "bookmark": '<path d="M6 3h12a1 1 0 0 1 1 1v17l-7-4-7 4V4a1 1 0 0 1 1-1z"/>',
    "close": '<path d="m6 6 12 12M18 6 6 18"/>',
    "copy": (
        '<rect x="8" y="8" width="12" height="12" rx="2"/>'
        '<path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>'
    ),
    "download": '<path d="M12 4v11m-5-5 5 5 5-5M5 20h14"/>',
    "file": (
        '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z'
        'M14 3v6h6M8 13h8M8 17h5"/>'
    ),
    "fit": (
        '<path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4'
        'M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/>'
    ),
    "folder": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5'
        'a2 2 0 0 1-2-2zM3 9h18"/>'
    ),
    "list": (
        '<path d="M9 6h11M9 12h11M9 18h11"/>'
        '<circle cx="4.5" cy="6" r="1"/><circle cx="4.5" cy="12" r="1"/>'
        '<circle cx="4.5" cy="18" r="1"/>'
    ),
    "panel": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
    "save": (
        '<path d="M5 3h11l5 5v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>'
        '<path d="M7 3v5h8M7 21v-7h10v7"/>'
    ),
    "sliders": (
        '<path d="M4 7h7m4 0h5M4 17h3m4 0h9"/>'
        '<circle cx="13" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>'
    ),
}
ICON_NAMES = tuple(_PATHS)
"""Built-in icon names; ``register_icon`` adds more at runtime."""

# Buttons remember their icon name so a theme switch can redraw them.
_ICON_PROPERTY = "interfaceIcon"
# Drawn before the first theme is applied; apply_theme replaces it.
_color = "#85857e"


def register_icon(name: str, svg_body: str) -> None:
    """Add an application icon: SVG elements on a 24-unit stroked canvas."""

    if name in _PATHS:
        raise ValueError(f"Icon {name!r} is already defined")
    _PATHS[name] = svg_body


def interface_icon(name: str, color: str | None = None) -> QIcon:
    """Draw icon ``name`` in ``color``, or in the current theme's icon color."""

    if name not in _PATHS:
        raise ValueError(f"Unknown icon {name!r}; register it with register_icon")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color or _color}" stroke-width="1.6" '
        f'stroke-linecap="round" stroke-linejoin="round">{_PATHS[name]}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    # Rendered at 2x so the icon stays sharp on HiDPI displays.
    pixmap = QPixmap(40, 40)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    return QIcon(pixmap)


def set_button_icon(button: QAbstractButton, name: str) -> None:
    """Show icon ``name`` on ``button`` and keep it in the theme's color."""

    icon = interface_icon(name)
    button.setProperty(_ICON_PROPERTY, name)
    button.setIcon(icon)


def set_icon_color(color: str) -> None:
    """Redraw every tagged button in the application in ``color``."""

    global _color
    _color = color
    if QApplication.instance() is None:
        return
    for widget in QApplication.allWidgets():
        if isinstance(widget, QAbstractButton):
            name = widget.property(_ICON_PROPERTY)
            if isinstance(name, str) and name in _PATHS:
                widget.setIcon(interface_icon(name))
