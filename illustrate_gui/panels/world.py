"""World panel controls."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def _default_world() -> dict[str, float | str | bool]:
    return {
        "background": "#ffffff",
        "fog": "#ffffff",
        "fog_front": 1.0,
        "fog_back": 1.0,
        "shadows": True,
        "shadow_strength": 0.0023,
        "shadow_angle": 2.0,
        "shadow_min_z": 1.0,
        "shadow_max_dark": 0.2,
    }


def _coerce(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QColorDialog,
        QCheckBox,
        QDoubleSpinBox,
        QFormLayout,
        QPushButton,
        QWidget,
    )
except Exception:  # pragma: no cover - optional dependency
    QColor = None  # type: ignore
    QWidget = object  # type: ignore


def _contrast_text(hex_color: str) -> str:
    """Return 'black' or 'white' for readable text on the given background."""
    v = hex_color.lstrip("#")
    if len(v) != 6:
        return "black"
    r, g, b = int(v[0:2], 16) / 255.0, int(v[2:4], 16) / 255.0, int(v[4:6], 16) / 255.0
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.5 else "white"


def _normalize(values: Mapping[str, Any] | None, **overrides: Any) -> dict[str, float | str | bool]:
    merged = dict(_default_world())
    if values:
        merged.update(values)
    for key, value in overrides.items():
        merged[key] = value
    for key in ("fog_front", "fog_back", "shadow_strength", "shadow_angle", "shadow_min_z", "shadow_max_dark"):
        merged[key] = _coerce(merged.get(key), float(_default_world()[key]))
    merged["shadows"] = bool(merged.get("shadows", _default_world()["shadows"]))
    merged["background"] = str(merged.get("background", "#ffffff"))
    merged["fog"] = str(merged.get("fog", "#ffffff"))
    return merged


if QColor is None:

    class WorldPanel:
        """Fallback world panel."""

        def __init__(self, on_changed: Callable[[dict[str, float | str | bool]], None] | None = None) -> None:
            self._value = _default_world()
            self._on_changed = on_changed

        @property
        def value(self) -> dict[str, float | str | bool]:
            return self._value

        def widget(self):
            return None

        def set_value(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
            self._value = _normalize(values, **kwargs)
            if self._on_changed is not None:
                self._on_changed(self._value)

else:

    class WorldPanel(QWidget):
        """World controls: background, fog, shadows."""

        def __init__(self, on_changed: Callable[[dict[str, float | str | bool]], None] | None = None) -> None:
            super().__init__()
            self._on_changed = on_changed
            self._value = _default_world()

            form = QFormLayout(self)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            self.bg = QPushButton("#ffffff", self)
            self.bg.setStyleSheet("background:#ffffff; color:black")
            self.bg.clicked.connect(self._pick_background)

            self.fog = QPushButton("#ffffff", self)
            self.fog.setStyleSheet("background:#ffffff; color:black")
            self.fog.clicked.connect(self._pick_fog)

            self.fog_front = QDoubleSpinBox(self)
            self.fog_front.setRange(0.0, 1.0)
            self.fog_front.setDecimals(2)
            self.fog_front.setSingleStep(0.01)
            self.fog_front.setValue(1.0)
            self.fog_front.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

            self.fog_back = QDoubleSpinBox(self)
            self.fog_back.setRange(0.0, 1.0)
            self.fog_back.setDecimals(2)
            self.fog_back.setSingleStep(0.01)
            self.fog_back.setValue(1.0)
            self.fog_back.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

            self.shadows = QCheckBox(self)
            self.shadows.setChecked(True)

            self.shadow_strength = QDoubleSpinBox(self)
            self.shadow_strength.setRange(0.0, 1.0)
            self.shadow_strength.setDecimals(4)
            self.shadow_strength.setSingleStep(0.0001)
            self.shadow_strength.setValue(0.0023)
            self.shadow_strength.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

            self.shadow_angle = QDoubleSpinBox(self)
            self.shadow_angle.setRange(0.0, 10.0)
            self.shadow_angle.setDecimals(2)
            self.shadow_angle.setSingleStep(0.1)
            self.shadow_angle.setValue(2.0)
            self.shadow_angle.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

            self.shadow_min_z = QDoubleSpinBox(self)
            self.shadow_min_z.setRange(0.0, 20.0)
            self.shadow_min_z.setDecimals(2)
            self.shadow_min_z.setSingleStep(0.1)
            self.shadow_min_z.setValue(1.0)
            self.shadow_min_z.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

            self.shadow_max_dark = QDoubleSpinBox(self)
            self.shadow_max_dark.setRange(0.0, 1.0)
            self.shadow_max_dark.setDecimals(3)
            self.shadow_max_dark.setSingleStep(0.01)
            self.shadow_max_dark.setValue(0.2)
            self.shadow_max_dark.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

            form.addRow("Background", self.bg)
            form.addRow("Fog", self.fog)
            form.addRow("Fog front", self.fog_front)
            form.addRow("Fog back", self.fog_back)
            form.addRow("Enable shadows", self.shadows)
            form.addRow("Shadow strength", self.shadow_strength)
            form.addRow("Shadow angle", self.shadow_angle)
            form.addRow("Shadow start Z", self.shadow_min_z)
            form.addRow("Max shadow", self.shadow_max_dark)

            self.fog_front.valueChanged.connect(self._sync)
            self.fog_back.valueChanged.connect(self._sync)
            self.shadows.toggled.connect(self._sync)
            self.shadow_strength.valueChanged.connect(self._sync)
            self.shadow_angle.valueChanged.connect(self._sync)
            self.shadow_min_z.valueChanged.connect(self._sync)
            self.shadow_max_dark.valueChanged.connect(self._sync)

        @property
        def value(self) -> dict[str, float | str | bool]:
            return self._value

        def set_value(self, values: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
            self._value = _normalize(values, **kwargs)
            bg_hex = str(self._value["background"])
            self.bg.setText(bg_hex)
            self.bg.setStyleSheet(f"background:{bg_hex}; color:{_contrast_text(bg_hex)}")
            fog_hex = str(self._value["fog"])
            self.fog.setText(fog_hex)
            self.fog.setStyleSheet(f"background:{fog_hex}; color:{_contrast_text(fog_hex)}")
            self.fog_front.setValue(float(self._value["fog_front"]))
            self.fog_back.setValue(float(self._value["fog_back"]))
            self.shadows.setChecked(bool(self._value["shadows"]))
            self.shadow_strength.setValue(float(self._value["shadow_strength"]))
            self.shadow_angle.setValue(float(self._value["shadow_angle"]))
            self.shadow_min_z.setValue(float(self._value["shadow_min_z"]))
            self.shadow_max_dark.setValue(float(self._value["shadow_max_dark"]))
            self._sync()

        def _pick_background(self) -> None:
            selected = QColorDialog.getColor(parent=self, title="Background")
            if not selected.isValid():
                return
            value = f"#{selected.red():02x}{selected.green():02x}{selected.blue():02x}"
            self._value["background"] = value
            self.bg.setText(value)
            self.bg.setStyleSheet(f"background:{value}; color:{_contrast_text(value)}")
            self._sync()

        def _pick_fog(self) -> None:
            selected = QColorDialog.getColor(parent=self, title="Fog color")
            if not selected.isValid():
                return
            value = f"#{selected.red():02x}{selected.green():02x}{selected.blue():02x}"
            self._value["fog"] = value
            self.fog.setText(value)
            self.fog.setStyleSheet(f"background:{value}; color:{_contrast_text(value)}")
            self._sync()

        def _sync(self) -> None:
            self._value = _normalize(
                {
                    "background": self._value["background"],
                    "fog": self._value["fog"],
                    "fog_front": self.fog_front.value(),
                    "fog_back": self.fog_back.value(),
                    "shadows": self.shadows.isChecked(),
                    "shadow_strength": self.shadow_strength.value(),
                    "shadow_angle": self.shadow_angle.value(),
                    "shadow_min_z": self.shadow_min_z.value(),
                    "shadow_max_dark": self.shadow_max_dark.value(),
                }
            )
            if self._on_changed is not None:
                self._on_changed(self._value)
