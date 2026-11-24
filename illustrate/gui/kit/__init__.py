"""Standard look and layout for the Illustrate desktop application.

Themes (``light``, ``dark``, ``midnight``), the system-following appearance,
the application shell, and the shared widgets.
"""

from illustrate.gui.kit.appearance import (
    APPEARANCES,
    Appearance,
    AppearanceController,
    add_appearance_menu,
    appearance_controller,
    apply_appearance,
)
from illustrate.gui.kit.buttons import ButtonRole, action_button, role_button, set_attention
from illustrate.gui.kit.icons import ICON_NAMES, interface_icon, register_icon, set_button_icon
from illustrate.gui.kit.sections import CollapsibleSection, configure_form, form_label
from illustrate.gui.kit.shell import ActionBar, AppShell, InspectorDock, NavigationDock, Workspace
from illustrate.gui.kit.themes import (
    SHARED_TOKENS,
    THEMES,
    apply_theme,
    colors,
    current_theme,
    load_stylesheet,
    theme_events,
)
from illustrate.gui.kit.toasts import ToastKind, ToastManager

__all__ = [
    "APPEARANCES",
    "ICON_NAMES",
    "SHARED_TOKENS",
    "THEMES",
    "ActionBar",
    "AppShell",
    "Appearance",
    "AppearanceController",
    "ButtonRole",
    "CollapsibleSection",
    "InspectorDock",
    "NavigationDock",
    "ToastKind",
    "ToastManager",
    "Workspace",
    "action_button",
    "add_appearance_menu",
    "appearance_controller",
    "apply_appearance",
    "apply_theme",
    "colors",
    "configure_form",
    "current_theme",
    "form_label",
    "interface_icon",
    "load_stylesheet",
    "register_icon",
    "role_button",
    "set_attention",
    "set_button_icon",
    "theme_events",
]
