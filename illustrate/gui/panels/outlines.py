from __future__ import annotations

from typing import Callable, Mapping

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QRadioButton, QWidget


def _default_outlines() -> dict[str, float | int | bool]:
    return {
        "enabled": True,
        "contour_low": 3.0,
        "contour_high": 10.0,
        "kernel": 4,
        "z_diff_min": 0.0,
        "z_diff_max": 5.0,
        "subunit_low": 3.0,
        "subunit_high": 10.0,
        "residue_low": 3.0,
        "residue_high": 8.0,
        "residue_diff": 6000.0,
    }


def _float_or_default(value: float | str | None, default: float) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _normalize(values: Mapping[str, float | int | bool | str] | None, **overrides: float | int | bool | str) -> dict[str, float | int | bool]:
    merged: dict[str, float | int | bool | str] = dict(_default_outlines())
    if values:
        merged.update(values)
    for key, value in overrides.items():
        merged[key] = value
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["contour_low"] = _float_or_default(merged.get("contour_low"), 3.0)
    merged["contour_high"] = _float_or_default(merged.get("contour_high"), 10.0)
    merged["kernel"] = int(merged.get("kernel", 4))
    merged["z_diff_min"] = _float_or_default(merged.get("z_diff_min"), 0.0)
    merged["z_diff_max"] = _float_or_default(merged.get("z_diff_max"), 5.0)
    merged["subunit_low"] = _float_or_default(merged.get("subunit_low"), 3.0)
    merged["subunit_high"] = _float_or_default(merged.get("subunit_high"), 10.0)
    merged["residue_low"] = _float_or_default(merged.get("residue_low"), 3.0)
    merged["residue_high"] = _float_or_default(merged.get("residue_high"), 8.0)
    merged["residue_diff"] = _float_or_default(merged.get("residue_diff"), 6000.0)
    return {key: value for key, value in merged.items() if not isinstance(value, str)}


class OutlinesPanel(QWidget):
    def __init__(self, on_changed: Callable[[dict[str, float | int | bool]], None] | None = None) -> None:
        super().__init__()
        self._on_changed = on_changed
        self._value = _default_outlines()

        form = QFormLayout(self)
        enabled_layout = QHBoxLayout()
        self.enabled_on = QRadioButton("Enabled", self)
        self.enabled_off = QRadioButton("Disabled", self)
        self.enabled_on.setChecked(True)
        self.enabled_on.toggled.connect(self._read_controls)
        enabled_layout.addWidget(self.enabled_on)
        enabled_layout.addWidget(self.enabled_off)
        form.addRow("Outlines", enabled_layout)

        self.contour_low = QDoubleSpinBox(self)
        self.contour_low.setRange(0.0, 50.0)
        self.contour_low.setSingleStep(0.5)
        self.contour_low.setValue(3.0)

        self.contour_high = QDoubleSpinBox(self)
        self.contour_high.setRange(0.0, 50.0)
        self.contour_high.setSingleStep(0.5)
        self.contour_high.setValue(10.0)

        self.kernel = QComboBox(self)
        self.kernel.addItems(["1", "2", "3", "4"])
        self.kernel.setCurrentText("4")

        self.z_diff_min = QDoubleSpinBox(self)
        self.z_diff_min.setRange(0.0, 50.0)
        self.z_diff_min.setSingleStep(0.1)

        self.z_diff_max = QDoubleSpinBox(self)
        self.z_diff_max.setRange(0.0, 50.0)
        self.z_diff_max.setSingleStep(0.1)
        self.z_diff_max.setValue(5.0)

        self.subunit_low = QDoubleSpinBox(self)
        self.subunit_low.setRange(0.0, 50.0)
        self.subunit_low.setSingleStep(0.5)
        self.subunit_low.setValue(3.0)
        self.subunit_high = QDoubleSpinBox(self)
        self.subunit_high.setRange(0.0, 50.0)
        self.subunit_high.setSingleStep(0.5)
        self.subunit_high.setValue(10.0)

        self.residue_low = QDoubleSpinBox(self)
        self.residue_low.setRange(0.0, 50.0)
        self.residue_low.setSingleStep(0.5)
        self.residue_low.setValue(3.0)
        self.residue_high = QDoubleSpinBox(self)
        self.residue_high.setRange(0.0, 50.0)
        self.residue_high.setSingleStep(0.5)
        self.residue_high.setValue(8.0)
        self.residue_diff = QDoubleSpinBox(self)
        self.residue_diff.setRange(0.0, 10000.0)
        self.residue_diff.setSingleStep(1.0)
        self.residue_diff.setValue(6000.0)

        form.addRow("Contour low", self.contour_low)
        form.addRow("Contour high", self.contour_high)
        form.addRow("Kernel", self.kernel)
        form.addRow("Z diff min", self.z_diff_min)
        form.addRow("Z diff max", self.z_diff_max)
        form.addRow("Subunit low", self.subunit_low)
        form.addRow("Subunit high", self.subunit_high)
        form.addRow("Residue low", self.residue_low)
        form.addRow("Residue high", self.residue_high)
        form.addRow("Residue diff", self.residue_diff)

        for widget in (
            self.contour_low,
            self.contour_high,
            self.z_diff_min,
            self.z_diff_max,
            self.subunit_low,
            self.subunit_high,
            self.residue_low,
            self.residue_high,
            self.residue_diff,
        ):
            widget.valueChanged.connect(self._read_controls)
        self.kernel.currentTextChanged.connect(self._read_controls)
        self.enabled_off.toggled.connect(self._read_controls)

    @property
    def value(self) -> dict[str, float | int | bool]:
        return self._value

    def set_value(self, values: Mapping[str, float | int | bool | str] | None = None, **kwargs: float | int | bool | str) -> None:
        self._value = _normalize(values, **kwargs)
        blockers = [QSignalBlocker(widget) for widget in (
            self.enabled_on, self.enabled_off, self.contour_low, self.contour_high,
            self.kernel, self.z_diff_min, self.z_diff_max, self.subunit_low,
            self.subunit_high, self.residue_low, self.residue_high, self.residue_diff,
        )]
        try:
            self.enabled_on.setChecked(bool(self._value["enabled"]))
            self.enabled_off.setChecked(not bool(self._value["enabled"]))
            self.contour_low.setValue(float(self._value["contour_low"]))
            self.contour_high.setValue(float(self._value["contour_high"]))
            self.kernel.setCurrentText(str(int(self._value["kernel"])))
            self.z_diff_min.setValue(float(self._value["z_diff_min"]))
            self.z_diff_max.setValue(float(self._value["z_diff_max"]))
            self.subunit_low.setValue(float(self._value["subunit_low"]))
            self.subunit_high.setValue(float(self._value["subunit_high"]))
            self.residue_low.setValue(float(self._value["residue_low"]))
            self.residue_high.setValue(float(self._value["residue_high"]))
            self.residue_diff.setValue(float(self._value["residue_diff"]))
        finally:
            for blocker in blockers:
                blocker.unblock()
        self._read_controls()

    def _read_controls(self) -> None:
        self._value = _normalize(
            {
                "enabled": self.enabled_on.isChecked(),
                "contour_low": self.contour_low.value(),
                "contour_high": self.contour_high.value(),
                "kernel": int(self.kernel.currentText()),
                "z_diff_min": self.z_diff_min.value(),
                "z_diff_max": self.z_diff_max.value(),
                "subunit_low": self.subunit_low.value(),
                "subunit_high": self.subunit_high.value(),
                "residue_low": self.residue_low.value(),
                "residue_high": self.residue_high.value(),
                "residue_diff": self.residue_diff.value(),
            }
        )
        if self._on_changed is not None:
            self._on_changed(self._value)
