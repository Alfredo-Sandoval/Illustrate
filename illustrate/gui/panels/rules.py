from __future__ import annotations

from typing import Callable

from illustrate.core.types import SelectionRule

from illustrate.gui.kit import role_button
from PySide6.QtCore import QAbstractItemModel, QLocale, QModelIndex, QPersistentModelIndex, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSpinBox,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def _default_rule() -> SelectionRule:
    return SelectionRule(
        record_name="ATOM  ",
        descriptor="----------",
        res_low=0,
        res_high=9999,
        color=(1.0, 0.7, 0.5),
        radius=1.5,
    )


def _to_hex(color: tuple[float, float, float]) -> str:
    return f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"


def _from_hex(value: str) -> tuple[float, float, float]:
    v = value.lstrip("#")
    if len(v) != 6:
        return (1.0, 1.0, 1.0)
    r = int(v[0:2], 16) / 255.0
    g = int(v[2:4], 16) / 255.0
    b = int(v[4:6], 16) / 255.0
    return (r, g, b)


_ROW_HEIGHT = 32
(
    COLOR_COLUMN,
    RECORD_COLUMN,
    ATOM_COLUMN,
    RESIDUE_COLUMN,
    CHAIN_COLUMN,
    RES_LOW_COLUMN,
    RES_HIGH_COLUMN,
    RADIUS_COLUMN,
    MATCHES_COLUMN,
) = range(9)
_HEADERS = ["Color", "Record", "Atom", "Residue", "Chain", "Res Low", "Res High", "Radius", "Matches"]
_HEADER_TOOLTIPS = {
    COLOR_COLUMN: "Double-click to choose the color.",
    RECORD_COLUMN: "PDB record type: ATOM for polymer atoms, HETATM for ligands, ions, and water.",
    ATOM_COLUMN: "Atom name, PDB columns 13-16. A dash matches any character.",
    RESIDUE_COLUMN: "Residue name, PDB columns 18-20, such as HOH or HEM.",
    CHAIN_COLUMN: "Chain identifier, PDB column 22.",
    RES_LOW_COLUMN: "First residue number the rule applies to.",
    RES_HIGH_COLUMN: "Last residue number the rule applies to.",
    RADIUS_COLUMN: "Sphere radius in angstroms. Zero hides the matched atoms.",
    MATCHES_COLUMN: "Atoms of the loaded structure that take this rule. An atom takes the first rule it matches.",
}
# Matches takes the remaining width.
_COLUMN_WIDTHS = {
    COLOR_COLUMN: 116,
    RECORD_COLUMN: 104,
    ATOM_COLUMN: 88,
    RESIDUE_COLUMN: 88,
    CHAIN_COLUMN: 80,
    RES_LOW_COLUMN: 88,
    RES_HIGH_COLUMN: 88,
    RADIUS_COLUMN: 96,
}
# Where each pattern column sits in the 10-character descriptor (PDB columns 13-22).
_PATTERN_SLICES = {ATOM_COLUMN: (0, 4), RESIDUE_COLUMN: (5, 8), CHAIN_COLUMN: (9, 10)}
_DESCRIPTOR_ROLE = Qt.ItemDataRole.UserRole
_ModelIndex = QModelIndex | QPersistentModelIndex

def _normalize_descriptor(descriptor: str) -> str:
    return f"{descriptor.upper():-<10}"[:10]

def _normalize_pattern(text: str, width: int) -> str:
    """Normalize to a fixed-width PDB pattern, with blank or Any as all wildcards."""
    value = text.strip().upper()
    if value == "ANY":
        value = ""
    return f"{value:-<{width}}"[:width]

def _swatch(hex_color: str) -> QIcon:
    pixmap = QPixmap(28, 28)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(hex_color))
    painter.setPen(QPen(QColor(0, 0, 0, 50), 1))
    painter.drawRoundedRect(QRectF(0.5, 0.5, 13, 13), 3.5, 3.5)
    painter.end()
    return QIcon(pixmap)

class _RuleDelegate(QStyledItemDelegate):
    """Display wildcard fields as Any and zero radii as Hidden."""

    def displayText(self, value: object, locale: QLocale | QLocale.Language) -> str:
        if isinstance(value, float):
            return "Hidden" if value == 0.0 else f"{value:.2f} Å"
        if isinstance(value, str) and value and set(value) == {"-"}:
            return "Any"
        return str(value)

    def initStyleOption(self, option: QStyleOptionViewItem, index: _ModelIndex) -> None:
        super().initStyleOption(option, index)
        if index.column() in (RES_LOW_COLUMN, RES_HIGH_COLUMN, RADIUS_COLUMN):
            option.displayAlignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        wildcard = getattr(option, "text", "") in ("Any", "Hidden")
        if wildcard or index.column() == MATCHES_COLUMN:
            muted = option.palette.color(QPalette.ColorRole.PlaceholderText)
            option.palette.setColor(QPalette.ColorRole.Text, muted)

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: _ModelIndex
    ) -> QWidget:
        column = index.column()
        if column == RECORD_COLUMN:
            combo = QComboBox(parent)
            combo.addItems(["ATOM", "HETATM"])

            def commit(_text: str) -> None:
                self.commitData.emit(combo)

            combo.textActivated.connect(commit)
            return combo
        if column in (RES_LOW_COLUMN, RES_HIGH_COLUMN):
            spin = QSpinBox(parent)
            spin.setRange(0, 9999)
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            return spin
        if column == RADIUS_COLUMN:
            radius = QDoubleSpinBox(parent)
            radius.setRange(0.0, 20.0)
            radius.setSingleStep(0.1)
            radius.setSuffix(" Å")
            radius.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            return radius
        edit = QLineEdit(parent)
        start, stop = _PATTERN_SLICES[column]
        edit.setMaxLength(stop - start)
        return edit

    def setEditorData(self, editor: QWidget, index: _ModelIndex) -> None:
        value = index.data(Qt.ItemDataRole.EditRole)
        if isinstance(editor, QComboBox):
            editor.setCurrentText(str(value))
        elif isinstance(editor, QLineEdit):
            editor.setText(str(value).rstrip("-"))
        else:
            super().setEditorData(editor, index)

    def setModelData(
        self, editor: QWidget, model: QAbstractItemModel, index: _ModelIndex
    ) -> None:
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
        elif isinstance(editor, QLineEdit):
            start, stop = _PATTERN_SLICES[index.column()]
            model.setData(
                index, _normalize_pattern(editor.text(), stop - start), Qt.ItemDataRole.EditRole
            )
        else:
            super().setModelData(editor, model, index)

class RulePanel(QWidget):
    """Editable rule table in first-match order."""

    def __init__(
        self,
        rules: list[SelectionRule],
        on_changed: Callable[[list[SelectionRule]], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_changed = on_changed
        self._rules = list(rules) if rules else [_default_rule()]
        self._build_ui()
        self.set_rules(self._rules)

    @property
    def value(self) -> list[SelectionRule]:
        return list(self._rules)

    def set_value(self, rules: list[SelectionRule]) -> None:
        self.set_rules(rules)

    def set_match_counts(self, counts: list[int] | None) -> None:
        was_blocked = self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, MATCHES_COLUMN)
            if item is None:
                continue
            if counts is None or row >= len(counts):
                item.setText("-")
            else:
                item.setText(str(int(counts[row])))
        self.table.blockSignals(was_blocked)

    def set_rules(self, rules: list[SelectionRule]) -> None:
        self._rules = list(rules) if rules else [_default_rule()]
        was_blocked = self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(self._rules))
            for row, rule in enumerate(self._rules):
                self._show_rule(row, rule)
        finally:
            self.table.blockSignals(was_blocked)
        self.set_match_counts(None)
        self._refresh_buttons()
        self._notify()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("Selection Rules", self)
        title.setObjectName("rulesHeading")
        header.addWidget(title)
        hint = QLabel("An atom takes the first rule it matches. Double-click a value to edit it.", self)
        hint.setObjectName("rulesHint")
        header.addSpacing(8)
        header.addWidget(hint, 1)

        self.btn_add = role_button("Add", "secondary", self, tooltip="Add a rule at the end")
        self.btn_remove = role_button("Remove", "secondary", self, tooltip="Remove the selected rule")
        self.btn_up = role_button("Up", "secondary", self, tooltip="Move the selected rule earlier")
        self.btn_down = role_button("Down", "secondary", self, tooltip="Move the selected rule later")
        for button in (self.btn_add, self.btn_remove, self.btn_up, self.btn_down):
            button.setFixedHeight(28)
            header.addWidget(button)
        self.btn_close = role_button("", "icon", self, icon="close", tooltip="Close selection rules")
        self.btn_close.setFixedSize(28, 28)
        header.addWidget(self.btn_close)
        layout.addLayout(header)

        self.table = QTableWidget(0, len(_HEADERS), self)
        self.table.setObjectName("rulesTable")
        self.table.setHorizontalHeaderLabels(_HEADERS)
        for column, tooltip in _HEADER_TOOLTIPS.items():
            header_item = self.table.horizontalHeaderItem(column)
            if header_item is not None:
                header_item.setToolTip(tooltip)
                if column in (RES_LOW_COLUMN, RES_HIGH_COLUMN, RADIUS_COLUMN):
                    header_item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
        self.table.setItemDelegate(_RuleDelegate(self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.setMinimumHeight(120)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(_ROW_HEIGHT)
        self.table.setShowGrid(False)
        header_view = self.table.horizontalHeader()
        header_view.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header_view.setStretchLastSection(True)
        header_view.setHighlightSections(False)
        for column, width in _COLUMN_WIDTHS.items():
            self.table.setColumnWidth(column, width)
        layout.addWidget(self.table, 1)

        self.btn_add.clicked.connect(self._add_blank_rule)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_up.clicked.connect(self._move_up)
        self.btn_down.clicked.connect(self._move_down)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.table.itemSelectionChanged.connect(self._refresh_buttons)

    def _item(self, row: int, column: int) -> QTableWidgetItem:
        item = self.table.item(row, column)
        if item is None:
            raise RuntimeError(f"rule table has no item at row {row}, column {column}")
        return item

    def _show_rule(self, row: int, rule: SelectionRule) -> None:
        """Populate a row with the table's signals blocked by the caller."""
        descriptor = _normalize_descriptor(str(rule.descriptor))
        hex_color = _to_hex(rule.color)
        values: dict[int, object] = {
            COLOR_COLUMN: hex_color,
            RECORD_COLUMN: rule.record_name.strip(),
            RES_LOW_COLUMN: int(rule.res_low),
            RES_HIGH_COLUMN: int(rule.res_high),
            RADIUS_COLUMN: float(rule.radius),
        }
        for column, (start, stop) in _PATTERN_SLICES.items():
            values[column] = descriptor[start:stop]
        for column, value in values.items():
            item = self.table.item(row, column)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(row, column, item)
            item.setData(Qt.ItemDataRole.EditRole, value)
        color_item = self._item(row, COLOR_COLUMN)
        color_item.setIcon(_swatch(hex_color))
        color_item.setFlags(color_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        # Keeps the descriptor characters that no column shows (PDB columns 17 and 21).
        self._item(row, ATOM_COLUMN).setData(_DESCRIPTOR_ROLE, descriptor)
        if self.table.item(row, MATCHES_COLUMN) is None:
            matches = QTableWidgetItem("-")
            matches.setFlags(matches.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, MATCHES_COLUMN, matches)

    def _rule_from_row(self, row: int) -> SelectionRule:
        def edit_value(column: int) -> object:
            return self._item(row, column).data(Qt.ItemDataRole.EditRole)

        descriptor = list(str(self._item(row, ATOM_COLUMN).data(_DESCRIPTOR_ROLE)))
        for column, (start, stop) in _PATTERN_SLICES.items():
            descriptor[start:stop] = _normalize_pattern(str(edit_value(column)), stop - start)
        previous = self._rules[row]
        try:
            res_low = int(str(edit_value(RES_LOW_COLUMN)))
            res_high = int(str(edit_value(RES_HIGH_COLUMN)))
            radius = float(str(edit_value(RADIUS_COLUMN)))
        except ValueError:
            res_low, res_high, radius = previous.res_low, previous.res_high, previous.radius
        return SelectionRule(
            record_name=str(edit_value(RECORD_COLUMN)).strip().upper().ljust(6),
            descriptor="".join(descriptor),
            res_low=res_low,
            res_high=res_high,
            color=_from_hex(str(edit_value(COLOR_COLUMN))),
            radius=radius,
        )

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if not 0 <= row < len(self._rules):
            return
        rule = self._rule_from_row(row)
        self._rules[row] = rule
        was_blocked = self.table.blockSignals(True)
        try:
            self._show_rule(row, rule)
        finally:
            self.table.blockSignals(was_blocked)
        self._notify()

    def _notify(self) -> None:
        if self._on_changed is not None:
            self._on_changed(list(self._rules))

    def _refresh_buttons(self) -> None:
        row = self.table.currentRow() if self.table.selectedItems() else -1
        self.btn_remove.setEnabled(row >= 0)
        self.btn_up.setEnabled(row > 0)
        self.btn_down.setEnabled(0 <= row < self.table.rowCount() - 1)

    def _add_blank_rule(self) -> None:
        self.set_rules([*self._rules, _default_rule()])
        self.table.selectRow(len(self._rules) - 1)
        self.table.scrollToBottom()

    def _remove_selected(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self._rules):
            return
        rules = list(self._rules)
        del rules[row]
        self.set_rules(rules)
        self.table.selectRow(min(row, len(self._rules) - 1))

    def _move_up(self) -> None:
        self._move_selected(-1)

    def _move_down(self) -> None:
        self._move_selected(1)

    def _move_selected(self, offset: int) -> None:
        source = self.table.currentRow()
        target = source + offset
        if not (0 <= source < len(self._rules) and 0 <= target < len(self._rules)):
            return
        rules = list(self._rules)
        rules[source], rules[target] = rules[target], rules[source]
        self.set_rules(rules)
        self.table.selectRow(target)

    def _on_cell_double_clicked(self, row: int, column: int) -> None:
        if column != COLOR_COLUMN or not 0 <= row < len(self._rules):
            return
        red, green, blue = self._rules[row].color
        selected = QColorDialog.getColor(
            QColor(int(red * 255), int(green * 255), int(blue * 255)), self
        )
        if not selected.isValid():
            return
        self._item(row, COLOR_COLUMN).setData(
            Qt.ItemDataRole.EditRole,
            _to_hex((selected.red() / 255.0, selected.green() / 255.0, selected.blue() / 255.0)),
        )
