"""Transform panel controls."""

from __future__ import annotations

from typing import Any, Callable, Mapping


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


def _clamp_numeric(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


try:
    from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QPushButton, QWidget
except Exception:  # pragma: no cover - optional dependency
    QDoubleSpinBox = None  # type: ignore
    QWidget = object  # type: ignore


def _normalize(values: Mapping[str, Any] | None, **overrides: float) -> dict[str, float]:
    base = _default_transform()
    if values:
        for key in base:
            if key in values:
                base[key] = _clamp_numeric(values[key], base[key])
    for key, value in overrides.items():
        base[key] = _clamp_numeric(value, base[key])
    return base


if QDoubleSpinBox is None:

    class TransformPanel:
        """Fallback transform panel."""

        def __init__(self, on_changed: Callable[[dict[str, float]], None] | None = None) -> None:
            self._value = _default_transform()
            self._on_changed = on_changed

        @property
        def value(self) -> dict[str, float]:
            return self._value

        def widget(self):
            return None

        def set_value(self, values: Mapping[str, float] | None = None, **kwargs: float) -> None:
            self._value = _normalize(values, **kwargs)
            if self._on_changed is not None:
                self._on_changed(self._value)

else:

    class TransformPanel(QWidget):
        """Transform controls with scale/rotation/translation spinboxes."""

        def __init__(self, on_changed: Callable[[dict[str, float]], None] | None = None) -> None:
            super().__init__()
            self._on_changed = on_changed
            self._value = _default_transform()
            layout = QFormLayout(self)

            self.scale = QDoubleSpinBox(self)
            self.scale.setRange(1.0, 500.0)
            self.scale.setSingleStep(0.5)
            self.scale.setValue(12.0)

            self.xrot = QDoubleSpinBox(self)
            self.xrot.setRange(-180.0, 180.0)
            self.xrot.setSingleStep(1.0)

            self.yrot = QDoubleSpinBox(self)
            self.yrot.setRange(-180.0, 180.0)
            self.yrot.setSingleStep(1.0)

            self.zrot = QDoubleSpinBox(self)
            self.zrot.setRange(-180.0, 180.0)
            self.zrot.setValue(90.0)
            self.zrot.setSingleStep(1.0)

            self.xtran = QDoubleSpinBox(self)
            self.xtran.setRange(-10000.0, 10000.0)
            self.xtran.setSingleStep(0.5)

            self.ytran = QDoubleSpinBox(self)
            self.ytran.setRange(-10000.0, 10000.0)
            self.ytran.setSingleStep(0.5)

            self.ztran = QDoubleSpinBox(self)
            self.ztran.setRange(-10000.0, 10000.0)
            self.ztran.setSingleStep(0.5)

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
                widget.valueChanged.connect(self._sync)
            self._sync()

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
                previous = widget.blockSignals(True)
                widget.setValue(self._value[key])
                widget.blockSignals(previous)
            self._sync()

        def _reset(self) -> None:
            self.set_value(_default_transform())

        def _sync(self) -> None:
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
