"""Rule editor panel."""

from __future__ import annotations

from typing import Callable

from illustrate.types import SelectionRule


def _default_rule() -> SelectionRule:
    return SelectionRule(
        record_name="ATOM  ",
        descriptor="----------",
        res_low=0,
        res_high=9999,
        color=(1.0, 0.7, 0.5),
        radius=1.5,
    )


try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDoubleSpinBox,
        QGridLayout,
        QHeaderView,
        QLineEdit,
        QPushButton,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QWidget,
        QColorDialog,
    )
except Exception:  # pragma: no cover - optional dependency
    QWidget = object  # type: ignore
    QColor = None  # type: ignore


def _to_hex(color: tuple[float, float, float]) -> str:
    return f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"


def _contrast_text(hex_color: str) -> str:
    """Return 'black' or 'white' for readable text on the given background."""
    v = hex_color.lstrip("#")
    if len(v) != 6:
        return "black"
    r, g, b = int(v[0:2], 16) / 255.0, int(v[2:4], 16) / 255.0, int(v[4:6], 16) / 255.0
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.5 else "white"


def _from_hex(value: str) -> tuple[float, float, float]:
    v = value.lstrip("#")
    if len(v) != 6:
        return (1.0, 1.0, 1.0)
    r = int(v[0:2], 16) / 255.0
    g = int(v[2:4], 16) / 255.0
    b = int(v[4:6], 16) / 255.0
    return (r, g, b)


def _serialize_rule(rule: SelectionRule) -> tuple[SelectionRule, str]:
    return (rule, _to_hex(rule.color))


if QColor is None:

    class RulePanel(QWidget):
        """Fallback rule panel."""

        def __init__(
            self,
            rules: list[SelectionRule],
            on_changed: Callable[[list[SelectionRule]], None] | None = None,
        ) -> None:
            self._rules = list(rules) if rules else [_default_rule()]
            self._on_changed = on_changed

        @property
        def value(self) -> list[SelectionRule]:
            return list(self._rules)

        def widget(self):
            return None

        def set_value(self, rules: list[SelectionRule]) -> None:
            self._rules = list(rules) if rules else [_default_rule()]
            if self._on_changed is not None:
                self._on_changed(self._rules)

        def set_match_counts(self, counts: list[int] | None) -> None:
            return None

else:

    class RulePanel(QWidget):
        """Editable table of atom selection rules."""

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
                item = self.table.item(row, 6)
                if item is None:
                    item = QTableWidgetItem("-")
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row, 6, item)
                if counts is None or row >= len(counts):
                    item.setText("-")
                else:
                    item.setText(str(int(counts[row])))
            self.table.blockSignals(was_blocked)

        def set_rules(self, rules: list[SelectionRule]) -> None:
            rows = rules if rules else [_default_rule()]
            self._rules = list(rows)
            self.table.blockSignals(True)
            try:
                self.table.setRowCount(0)
                for rule in rows:
                    self._add_row(rule)
            finally:
                self.table.blockSignals(False)
            self.set_match_counts(None)
            if self._on_changed is not None:
                self._on_changed(self._rules)

        def _build_ui(self) -> None:
            layout = QGridLayout(self)
            self.table = QTableWidget(0, 7, self)
            self.table.setHorizontalHeaderLabels(
                ["Record", "Descriptor", "Res Low", "Res High", "Color", "Radius", "Matches"]
            )
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.table.setMinimumHeight(200)
            self.table.verticalHeader().setVisible(False)
            self.table.setAlternatingRowColors(False)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

            self.btn_add = QPushButton("Add", self)
            self.btn_remove = QPushButton("Remove", self)
            self.btn_up = QPushButton("Up", self)
            self.btn_down = QPushButton("Down", self)

            self.btn_add.clicked.connect(self._add_blank_rule)
            self.btn_remove.clicked.connect(self._remove_selected)
            self.btn_up.clicked.connect(self._move_up)
            self.btn_down.clicked.connect(self._move_down)

            layout.addWidget(self.table, 0, 0, 1, 4)
            layout.addWidget(self.btn_add, 1, 0)
            layout.addWidget(self.btn_remove, 1, 1)
            layout.addWidget(self.btn_up, 1, 2)
            layout.addWidget(self.btn_down, 1, 3)
            margins = layout.contentsMargins()
            layout.setContentsMargins(margins.left(), margins.top(), margins.right(), 16)

            self.table.itemChanged.connect(self._emit)

        def _read_current_rules(self) -> list[SelectionRule]:
            rules: list[SelectionRule] = []
            for row in range(self.table.rowCount()):
                rec = self.table.cellWidget(row, 0)
                descriptor = self.table.cellWidget(row, 1)
                lo = self.table.cellWidget(row, 2)
                hi = self.table.cellWidget(row, 3)
                color_btn = self.table.cellWidget(row, 4)
                radius = self.table.cellWidget(row, 5)
                if (
                    not isinstance(rec, QComboBox)
                    or not isinstance(descriptor, QLineEdit)
                    or not isinstance(lo, QSpinBox)
                    or not isinstance(hi, QSpinBox)
                    or not isinstance(color_btn, QPushButton)
                    or not isinstance(radius, QDoubleSpinBox)
                ):
                    continue
                descriptor_value = f"{descriptor.text().upper():-<10}"[:10]
                rules.append(
                    SelectionRule(
                        record_name=str(rec.currentText()).ljust(6),
                        descriptor=descriptor_value,
                        res_low=int(lo.value()),
                        res_high=int(hi.value()),
                        color=_from_hex(color_btn.text()),
                        radius=float(radius.value()),
                    )
                )
            return rules

        def _emit(self, _item: QTableWidgetItem | None = None) -> None:
            self._rules = self._read_current_rules()
            if self._on_changed is not None:
                self._on_changed(self._rules)

        def _add_row(self, rule: SelectionRule) -> None:
            row = self.table.rowCount()
            self.table.insertRow(row)

            _cell_style = "background: transparent; border: none;"

            rec = QComboBox(self.table)
            rec.addItems(["ATOM", "HETATM"])
            rec.setCurrentText(rule.record_name.strip())
            rec.setStyleSheet(_cell_style)
            rec.currentTextChanged.connect(self._emit)

            descriptor = QLineEdit(self.table)
            descriptor.setMaxLength(10)
            descriptor.setText(str(rule.descriptor)[:10])
            descriptor.setPlaceholderText("----------")
            descriptor.setStyleSheet(_cell_style)
            descriptor.textChanged.connect(self._emit)

            lo = QSpinBox(self.table)
            lo.setRange(0, 9999)
            lo.setValue(rule.res_low)
            lo.setFrame(False)
            lo.setStyleSheet(_cell_style)
            lo.valueChanged.connect(self._emit)

            hi = QSpinBox(self.table)
            hi.setRange(0, 9999)
            hi.setValue(rule.res_high)
            hi.setFrame(False)
            hi.setStyleSheet(_cell_style)
            hi.valueChanged.connect(self._emit)

            hex_c = _to_hex(rule.color)
            color_button = QPushButton(hex_c, self.table)
            color_button.setStyleSheet(
                f"background:{hex_c}; color:{_contrast_text(hex_c)}; border: none; border-radius: 3px;"
            )
            color_button.clicked.connect(self._pick_color)

            r = QDoubleSpinBox(self.table)
            r.setRange(0.0, 20.0)
            r.setSingleStep(0.1)
            r.setValue(rule.radius)
            r.setFrame(False)
            r.setStyleSheet(_cell_style)
            r.valueChanged.connect(self._emit)

            self.table.setCellWidget(row, 0, rec)
            self.table.setCellWidget(row, 1, descriptor)
            self.table.setCellWidget(row, 2, lo)
            self.table.setCellWidget(row, 3, hi)
            self.table.setCellWidget(row, 4, color_button)
            self.table.setCellWidget(row, 5, r)
            match_item = QTableWidgetItem("-")
            match_item.setFlags(match_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            match_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 6, match_item)

        def _add_blank_rule(self) -> None:
            self._add_row(_default_rule())
            self._emit()

        def _remove_selected(self) -> None:
            row = self.table.currentRow()
            if row >= 0:
                self.table.removeRow(row)
            if self.table.rowCount() == 0:
                self._add_row(_default_rule())
            self._emit()

        def _move_up(self) -> None:
            row = self.table.currentRow()
            if row <= 0:
                return
            self._swap_rows(row, row - 1)

        def _move_down(self) -> None:
            row = self.table.currentRow()
            if row < 0 or row >= self.table.rowCount() - 1:
                return
            self._swap_rows(row, row + 1)

        def _swap_rows(self, source: int, target: int) -> None:
            rules = self._read_current_rules()
            if not (0 <= source < len(rules) and 0 <= target < len(rules)):
                return
            rules[source], rules[target] = rules[target], rules[source]
            self.set_rules(rules)
            self.table.selectRow(target)
            self._emit()

        def _pick_color(self) -> None:
            sender = self.sender()
            if not isinstance(sender, QPushButton):
                return
            current = _from_hex(sender.text())
            qcolor = QColor(int(current[0] * 255), int(current[1] * 255), int(current[2] * 255))
            selected = QColorDialog.getColor(qcolor, self)
            if not selected.isValid():
                return
            hex_color = _to_hex((selected.red() / 255.0, selected.green() / 255.0, selected.blue() / 255.0))
            sender.setText(hex_color)
            sender.setStyleSheet(
                f"background:{hex_color}; color:{_contrast_text(hex_color)}; border: none; border-radius: 3px;"
            )
            self._emit()
