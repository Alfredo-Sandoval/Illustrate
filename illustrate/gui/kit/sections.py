"""Collapsible sections and the form layout shared by sidebar and inspector panels."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["CollapsibleSection", "configure_form", "form_label"]

FORM_LABEL_MIN_WIDTH = 132


class CollapsibleSection(QWidget):
    """A section below a rule line whose header button shows or hides its content."""

    def __init__(
        self,
        title: str,
        content: QWidget,
        *,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        rule = QFrame(self)
        rule.setObjectName("sectionRule")
        rule.setFixedHeight(1)
        layout.addWidget(rule)

        self.toggle = QToolButton(self)
        self.toggle.setObjectName("sectionToggle")
        header_font = QApplication.font()
        header_font.setBold(True)
        self.toggle.setFont(header_font)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.toggled.connect(self.set_expanded)
        layout.addWidget(self.toggle)
        layout.addWidget(content)
        self.set_expanded(expanded)

    def set_expanded(self, expanded: bool) -> None:
        if self.toggle.isChecked() != expanded:
            self.toggle.setChecked(expanded)
            return
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(expanded)


def configure_form(layout: QFormLayout) -> None:
    """Apply the standard label column, spacing, and field growth to a form."""

    layout.setContentsMargins(0, 0, 0, 8)
    layout.setHorizontalSpacing(10)
    layout.setVerticalSpacing(8)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)


def form_label(text: str, parent: QWidget | None = None) -> QLabel:
    """Return a right-aligned form label with the standard minimum width.

    Use it when several forms stacked in one column must line up.
    """

    label = QLabel(text, parent)
    label.setObjectName("formLabel")
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    label.setMinimumWidth(FORM_LABEL_MIN_WIDTH)
    return label
