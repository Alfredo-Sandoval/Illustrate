from __future__ import annotations

import math
from typing import Callable, Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QPushButton, QWidget


def _default_transform() -> dict[str, float]:
    return {
        "scale": 12.0,
        "xrot": 0.0,
        "yrot": 0.0,
        "zrot": 90.0,
        "xtran": 0.0,
        "ytran": 0.0,
        "ztran": 0.0,
    }


# Upper bound of the Scale control when the atom radii allow more.
_SCALE_CEILING = 500.0


def _scale_maximum(limit: float) -> float:
    # Round down to the control's two decimals so the cap never exceeds the limit.
    return max(1.0, min(_SCALE_CEILING, math.floor(float(limit) * 100.0) / 100.0))


def _float_or_default(value: float | str | None, default: float) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _normalize(values: Mapping[str, float | str | None] | None, **overrides: float) -> dict[str, float]:
    base = _default_transform()
    if values:
        for key in base:
            if key in values:
                base[key] = _float_or_default(values[key], base[key])
    for key, value in overrides.items():
        base[key] = _float_or_default(value, base[key])
    return base


class _AngleSpinBox(QDoubleSpinBox):
    """Degree spin box that keeps 6 decimals but hides trailing zeros."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # Keep equivalent Euler angles precise when loading arbitrary rotations.
        self.setDecimals(6)
        self.setRange(-180.0, 180.0)
        self.setSingleStep(1.0)
        self.setSuffix("°")

    def textFromValue(self, value: float) -> str:
        locale = self.locale()
        # Normalize -0.0 so a cleared angle does not display as "-0".
        text = locale.toString(float(value) + 0.0, "f", self.decimals())
        point = locale.decimalPoint()
        if point in text:
            text = text.rstrip("0").removesuffix(point)
        return text

class TransformPanel(QWidget):
    def __init__(self, on_changed: Callable[[dict[str, float]], None] | None = None) -> None:
        super().__init__()
        self._on_changed = on_changed
        self._value = _default_transform()
        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.scale = QDoubleSpinBox(self)
        self.scale.setRange(1.0, _SCALE_CEILING)
        self.scale.setSingleStep(0.5)
        self.scale.setValue(12.0)
        # Scale is image pixels per ångström of the structure.
        self.scale.setSuffix(" px/Å")

        self.xrot = _AngleSpinBox(self)
        self.yrot = _AngleSpinBox(self)
        self.zrot = _AngleSpinBox(self)
        self.zrot.setValue(90.0)

        self.xtran = QDoubleSpinBox(self)
        self.xtran.setRange(-10000.0, 10000.0)
        self.xtran.setSingleStep(0.5)

        self.ytran = QDoubleSpinBox(self)
        self.ytran.setRange(-10000.0, 10000.0)
        self.ytran.setSingleStep(0.5)

        self.ztran = QDoubleSpinBox(self)
        self.ztran.setRange(-10000.0, 10000.0)
        self.ztran.setSingleStep(0.5)

        for translation in (self.xtran, self.ytran, self.ztran):
            translation.setSuffix(" Å")

        layout.addRow("Scale", self.scale)
        layout.addRow("X rotation", self.xrot)
        layout.addRow("Y rotation", self.yrot)
        layout.addRow("Z rotation", self.zrot)
        layout.addRow("X translation", self.xtran)
        layout.addRow("Y translation", self.ytran)
        layout.addRow("Z translation", self.ztran)

        reset_btn = QPushButton("Reset", self)
        reset_btn.clicked.connect(self._reset)
        layout.addRow(reset_btn)

        for widget in (self.scale, self.xrot, self.yrot, self.zrot, self.xtran, self.ytran, self.ztran):
            widget.valueChanged.connect(self._read_controls)
        self._read_controls()

    @property
    def value(self) -> dict[str, float]:
        return self._value

    def set_value(self, values: Mapping[str, float] | None = None, **kwargs: float) -> None:
        self._value = _normalize(values, **kwargs)
        updates = (
            (self.scale, "scale"),
            (self.xrot, "xrot"),
            (self.yrot, "yrot"),
            (self.zrot, "zrot"),
            (self.xtran, "xtran"),
            (self.ytran, "ytran"),
            (self.ztran, "ztran"),
        )
        for widget, key in updates:
            if abs(float(widget.value()) - float(self._value[key])) < 1e-9:
                continue
            previous = widget.blockSignals(True)
            widget.setValue(self._value[key])
            widget.blockSignals(previous)
        self._read_controls()

    def set_scale_limit(self, limit: float) -> None:
        """Apply the renderer's scale limit within the spin box's range and precision.

        Lowering the cap below the current value clamps it and reports a change.
        """
        self.scale.setMaximum(_scale_maximum(limit))

    def _reset(self) -> None:
        self.set_value(_default_transform())

    def _read_controls(self) -> None:
        self._value = {
            "scale": float(self.scale.value()),
            "xrot": float(self.xrot.value()),
            "yrot": float(self.yrot.value()),
            "zrot": float(self.zrot.value()),
            "xtran": float(self.xtran.value()),
            "ytran": float(self.ytran.value()),
            "ztran": float(self.ztran.value()),
        }
        if self._on_changed is not None:
            self._on_changed(self._value)
