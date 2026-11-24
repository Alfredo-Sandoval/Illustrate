"""Push buttons with a theme role, optionally presenting a window QAction.

Both themes style four ``buttonRole`` values: ``primary`` for the one main
command of a surface, ``secondary`` for other commands, ``navigation`` for
full-width sidebar rows, and ``icon`` for square icon-only buttons.
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QPushButton, QWidget

from illustrate.gui.kit.icons import set_button_icon

__all__ = ["ButtonRole", "action_button", "role_button", "set_attention"]

ButtonRole = Literal["primary", "secondary", "navigation", "icon"]


def role_button(
    text: str,
    role: ButtonRole,
    parent: QWidget | None = None,
    *,
    icon: str | None = None,
    tooltip: str = "",
) -> QPushButton:
    """Return a push button styled for ``role``, with an optional theme icon."""

    button = QPushButton(text, parent)
    button.setProperty("buttonRole", role)
    if icon is not None:
        set_button_icon(button, icon)
        button.setIconSize(QSize(18, 18) if role == "navigation" else QSize(16, 16))
    if role in ("navigation", "icon"):
        button.setCursor(Qt.CursorShape.PointingHandCursor)
    if role == "navigation":
        button.setFixedHeight(36)
    if tooltip:
        button.setToolTip(tooltip)
    if role == "icon":
        button.setAccessibleName(tooltip or text)
    return button


def action_button(
    action: QAction,
    role: ButtonRole,
    parent: QWidget | None = None,
    *,
    icon: str | None = None,
    text: str | None = None,
) -> QPushButton:
    """Return a button that triggers ``action`` and mirrors its state.

    The action stays the owner: controllers enable, disable, and check the
    action, and the button follows. Icon buttons show no text; their tooltip
    carries the action's name and shortcut.
    """

    label = "" if role == "icon" else (text if text is not None else action.text())
    tooltip = action.toolTip() or action.text()
    shortcut = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
    if shortcut:
        tooltip = f"{tooltip} · {shortcut}"
    button = role_button(label, role, parent, icon=icon, tooltip=tooltip)
    button.setCheckable(action.isCheckable())
    button.setChecked(action.isChecked())
    button.setEnabled(action.isEnabled())
    # Signal-to-slot connections between two Qt objects, so Qt drops them when
    # either is destroyed during window teardown.
    action.enabledChanged.connect(button.setEnabled)
    action.toggled.connect(button.setChecked)
    if action.isCheckable():
        # The action owns the checked state; a click toggles it once.
        button.clicked.connect(lambda _checked=False: action.trigger())
    else:
        button.clicked.connect(action.trigger)
    return button


def set_attention(button: QPushButton, attention: bool) -> None:
    """Mark a primary button whose result is out of date, such as after edits."""

    if bool(button.property("attention")) == attention:
        return
    button.setProperty("attention", attention)
    # Qt restyles on a dynamic property change only after a re-polish.
    style = button.style()
    style.unpolish(button)
    style.polish(button)
