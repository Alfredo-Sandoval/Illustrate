"""The standard main window: navigation dock, workspace, and inspector dock.

``AppShell`` lays out the three regions; the application fills them::

    shell = AppShell("Illustrate")
    shell.navigation.add_section("Structure")
    shell.navigation.add_widget(search_field)
    shell.workspace.set_body(viewport)
    shell.workspace.set_action_bar(render_bar)
    shell.inspector.add_section("Transform", transform_panel)
    shell.add_view_menu(app)
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from illustrate.gui.kit.appearance import add_appearance_menu
from illustrate.gui.kit.buttons import ButtonRole, action_button, role_button
from illustrate.gui.kit.sections import CollapsibleSection, configure_form
from illustrate.gui.kit.toasts import ToastManager

__all__ = ["ActionBar", "AppShell", "InspectorDock", "NavigationDock", "Workspace"]


def _bare_dock(dock: QDockWidget, area: Qt.DockWidgetArea) -> None:
    """Pin a dock to one side, closable through its toggle action, with no title bar."""

    dock.setAllowedAreas(area)
    # Closable so the toggle action works; the empty title bar shows no button.
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
    dock.setTitleBarWidget(QWidget(dock))


class NavigationDock(QDockWidget):
    """Left dock: the application's name, then labelled groups of controls."""

    def __init__(self, brand: str, parent: QWidget | None = None) -> None:
        super().__init__("Sidebar", parent)
        self.setObjectName("navigationDock")
        _bare_dock(self, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setMinimumWidth(220)
        self.setMaximumWidth(300)

        self.panel = QWidget(self)
        self.panel.setObjectName("navigationPanel")
        self.panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._layout = QVBoxLayout(self.panel)
        self._layout.setContentsMargins(12, 18, 12, 12)
        self._layout.setSpacing(4)
        self.brand = QLabel(brand, self.panel)
        self.brand.setObjectName("navigationBrand")
        self._layout.addWidget(self.brand)
        self._layout.addSpacing(18)
        self._sections = 0
        self.setWidget(self.panel)

    def add_section(self, title: str) -> QLabel:
        """Start a labelled group; later widgets belong to it."""

        if self._sections:
            self._layout.addSpacing(20)
        self._sections += 1
        label = QLabel(title, self.panel)
        label.setObjectName("navigationSection")
        self._layout.addWidget(label)
        self._layout.addSpacing(2)
        return label

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self._layout.addWidget(widget, stretch)

    def add_layout(self, layout: QLayout) -> None:
        self._layout.addLayout(layout)

    def add_action(
        self, action: QAction, *, icon: str | None = None, text: str | None = None
    ) -> QPushButton:
        """Add a full-width navigation row that triggers ``action``."""

        button = action_button(action, "navigation", self.panel, icon=icon, text=text)
        self._layout.addWidget(button)
        return button

    def add_detail(self, text: str) -> QLabel:
        """Add a muted, wrapping line of plain text, such as the open file's name."""

        label = QLabel(text, self.panel)
        label.setObjectName("navigationDetail")
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        self._layout.addWidget(label)
        return label

    def add_stretch(self) -> None:
        self._layout.addStretch(1)


class ActionBar(QWidget):
    """Rounded command surface below the workspace body.

    One row: the heading (a short status line) and any leading widgets on the
    left, trailing widgets (the primary command last) on the right. A footer
    row below the surface holds options and a detail label.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMaximumWidth(760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.surface = QFrame(self)
        self.surface.setObjectName("actionSurface")
        row = QHBoxLayout(self.surface)
        row.setContentsMargins(16, 10, 10, 10)
        row.setSpacing(8)
        self.heading = QLabel("", self.surface)
        self.heading.setObjectName("actionHeading")
        row.addWidget(self.heading, 1)
        self._leading = QHBoxLayout()
        self._leading.setSpacing(8)
        self._trailing = QHBoxLayout()
        self._trailing.setSpacing(8)
        row.addLayout(self._leading)
        row.addLayout(self._trailing)
        layout.addWidget(self.surface)

        footer = QWidget(self)
        footer.setObjectName("actionFooter")
        footer_row = QHBoxLayout(footer)
        footer_row.setContentsMargins(4, 0, 4, 0)
        self._options = QHBoxLayout()
        footer_row.addLayout(self._options)
        footer_row.addStretch(1)
        self.detail = QLabel("", footer)
        self.detail.setObjectName("actionDetail")
        footer_row.addWidget(self.detail)
        layout.addWidget(footer)

    def add_leading(self, widget: QWidget) -> None:
        self._leading.addWidget(widget)

    def add_trailing(self, widget: QWidget) -> None:
        self._trailing.addWidget(widget)

    def add_footer(self, widget: QWidget) -> None:
        """Add an option, such as a checkbox, to the left of the detail label."""

        self._options.addWidget(widget)

    def set_heading(self, text: str) -> None:
        self.heading.setText(text)


class Workspace(QWidget):
    """Central region: a header with title and commands, the body, and an action bar."""

    def __init__(
        self, *, sidebar_action: QAction | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("workspace")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        header = QWidget(self)
        header.setObjectName("workspaceHeader")
        self._header = QHBoxLayout(header)
        self._header.setContentsMargins(18, 12, 18, 8)
        self._header.setSpacing(8)
        if sidebar_action is not None:
            toggle = action_button(sidebar_action, "icon", header, icon="panel")
            toggle.setFixedSize(30, 30)
            self._header.addWidget(toggle)
        self.title = QLabel("", header)
        self.title.setObjectName("workspaceTitle")
        self._header.addWidget(self.title)
        self._header.addStretch(1)
        self._layout.addWidget(header)

        self._body = QHBoxLayout()
        self._body.setContentsMargins(18, 4, 18, 0)
        self._layout.addLayout(self._body, 1)

        self._footer = QHBoxLayout()
        self._footer.setContentsMargins(24, 12, 24, 16)
        self._layout.addLayout(self._footer)

    def set_title(self, text: str) -> None:
        self.title.setText(text)

    def add_header_action(
        self,
        action: QAction,
        role: ButtonRole = "secondary",
        *,
        icon: str | None = None,
        text: str | None = None,
    ) -> QPushButton:
        button = action_button(action, role, self, icon=icon, text=text)
        self._header.addWidget(button)
        return button

    def set_body(self, widget: QWidget) -> None:
        if self._body.count():
            raise RuntimeError("The workspace body is already set")
        self._body.addWidget(widget)

    def set_action_bar(self, bar: QWidget) -> None:
        if self._footer.count():
            raise RuntimeError("The workspace action bar is already set")
        self._footer.addStretch(1)
        self._footer.addWidget(bar, 8)
        self._footer.addStretch(1)


class InspectorDock(QDockWidget):
    """Right dock: a heading with a close button over scrolling collapsible sections."""

    def __init__(self, title: str = "Controls", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setObjectName("inspectorDock")
        _bare_dock(self, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(320)
        self.setMaximumWidth(420)

        body = QWidget(self)
        body.setObjectName("inspectorPanel")
        body.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(16, 18, 12, 12)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 4, 0)
        self.heading = QLabel(title, body)
        self.heading.setObjectName("inspectorHeading")
        header.addWidget(self.heading, 1)
        close = role_button("", "icon", body, icon="close", tooltip=f"Close {title.lower()}")
        close.setFixedSize(28, 28)
        close.clicked.connect(self.hide)
        header.addWidget(close)
        layout.addLayout(header)

        scroll = QScrollArea(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(scroll)
        content.setObjectName("inspectorContent")
        self._sections = QVBoxLayout(content)
        self._sections.setContentsMargins(0, 0, 4, 0)
        self._sections.setSpacing(4)
        self._sections.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        self.setWidget(body)

    def add_section(
        self, title: str, panel: QWidget, *, expanded: bool = True
    ) -> CollapsibleSection:
        """Append a collapsible section; a ``QFormLayout`` panel gets the standard form spacing."""

        form = panel.layout()
        if isinstance(form, QFormLayout):
            configure_form(form)
        section = CollapsibleSection(title, panel, expanded=expanded)
        self._sections.insertWidget(self._sections.count() - 1, section)
        return section


class AppShell(QMainWindow):
    """Main window with the standard regions, toasts, and a status bar.

    ``brand`` names the application in the navigation dock. The workspace
    header starts with the sidebar toggle; add application header commands,
    then ``add_inspector_toggle`` to end the header with the inspector toggle.
    """

    def __init__(self, brand: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setDockNestingEnabled(False)
        self.navigation = NavigationDock(brand, self)
        self.inspector = InspectorDock("Controls", self)
        self.workspace = Workspace(sidebar_action=self.navigation.toggleViewAction(), parent=self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.navigation)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector)
        self.setCentralWidget(self.workspace)
        self.toasts = ToastManager(self.workspace)
        self.statusBar()

    def add_inspector_toggle(self) -> QPushButton:
        """Add the inspector toggle to the workspace header; call after other header commands."""

        return self.workspace.add_header_action(
            self.inspector.toggleViewAction(), "secondary", icon="sliders"
        )

    def add_view_menu(self, app: QApplication, title: str = "&View") -> QMenu:
        """Add a menu with the dock toggles and an Appearance submenu."""

        menu = self.menuBar().addMenu(title)
        menu.addAction(self.navigation.toggleViewAction())
        menu.addAction(self.inspector.toggleViewAction())
        menu.addSeparator()
        add_appearance_menu(menu.addMenu("&Appearance"), app)
        return menu

    def closeEvent(self, event: QCloseEvent) -> None:
        self.toasts.shutdown()
        super().closeEvent(event)
