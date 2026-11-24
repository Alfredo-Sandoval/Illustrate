"""Render display, image zoom and pan, and the molecular orientation gizmo."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from illustrate.gui import kit

_HAS_OPENGL_WIDGETS = False
if TYPE_CHECKING:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
else:
    try:
        from PySide6.QtOpenGLWidgets import QOpenGLWidget
    except ImportError:  # pragma: no cover - optional OpenGL backend
        _HAS_OPENGL_WIDGETS = False
    else:
        _HAS_OPENGL_WIDGETS = True

_DisplayKey = tuple[int, int, int]
_Vector = NDArray[np.float32]
_RGBA_TOKEN = re.compile(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")


def _token_color(value: str) -> QColor:
    """Accept CSS-style rgba tokens alongside Qt color names and hex strings."""
    match = _RGBA_TOKEN.fullmatch(value.strip())
    if match is not None:
        red, green, blue, alpha = (int(part) for part in match.groups())
        return QColor(red, green, blue, alpha)
    color = QColor(value)
    if not color.isValid():
        raise ValueError(f"Unsupported theme color {value!r}")
    return color


# Orientation gizmo: an axis triad in the lower-left corner, in the conventional
# X/Y/Z = red/green/blue. Mid tones read on light and dark renders alike, and a
# halo in the theme's overlay color keeps the strokes legible over the image.
_GIZMO_AXES: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("X", "#e5484d", (1.0, 0.0, 0.0)),
    ("Y", "#2f9e5b", (0.0, 1.0, 0.0)),
    ("Z", "#3d7fe6", (0.0, 0.0, 1.0)),
)
_GIZMO_MARGIN = 14.0
_GIZMO_AXIS_LENGTH = 30.0
_GIZMO_LABEL_GAP = 9.0
_GIZMO_LABEL_CLEARANCE = 12.0
_GIZMO_LINE_WIDTH = 2.0
_GIZMO_HALO_WIDTH = 2.5
_GIZMO_FONT_PIXELS = 11
# Below this projected length an axis points almost straight at or away from
# the viewer, so it is drawn end-on: a circled dot (toward) or cross (away).
_GIZMO_END_ON = 0.3
_GIZMO_END_ON_RADIUS = 5.0
_GIZMO_AWAY_ALPHA = 0.55

# The wheel magnifies the displayed render, up to this factor over the
# fitted view. It never changes the render itself.
_MAX_VIEW_ZOOM = 8.0


def _gizmo_offset(vector: _Vector) -> QPointF:
    """Return where a rotated unit axis points on screen, in widget pixels.

    As in the original Illustrate, renders place rotated +x toward the bottom
    of the image and rotated +y toward the right, with +z toward the viewer
    (a right-handed frame), so the gizmo does the same.
    """
    return QPointF(float(vector[1]), float(vector[0])) * _GIZMO_AXIS_LENGTH


def _should_use_opengl_backend() -> bool:
    if not _HAS_OPENGL_WIDGETS:
        return False

    disable_flag = os.environ.get("ILLUSTRATE_DISABLE_OPENGL", "").strip().lower()
    if disable_flag in {"1", "true", "yes", "on"}:
        return False

    # QOpenGLWidget is not reliable under purely offscreen backends.
    platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if platform in {"offscreen", "minimal", "headless"}:
        return False
    return True


class _ViewportBehavior:
    _pixmap: QPixmap | None
    _last_point: QPoint | None
    # The mouse gesture in progress and the button that started it.
    _gesture: Literal["rotate", "pan"] | None
    _gesture_button: Qt.MouseButton
    _view_zoom: float
    _view_pan: QPointF
    _xrot: float
    _yrot: float
    _zrot: float
    _base_display_pixmap: QPixmap | None
    _base_display_key: _DisplayKey | None
    _checkerboard_brush: QBrush
    _empty_bg: QColor
    _gizmo_halo: QColor
    _gizmo_basis_key: tuple[float, float, float] | None
    _gizmo_basis: list[tuple[str, QColor, _Vector]] | None

    def _widget(self) -> QWidget:
        return cast(QWidget, self)

    def _init_view(self) -> None:
        widget = self._widget()
        self._pixmap = None
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        widget.setMinimumSize(320, 240)
        widget.setMouseTracking(True)
        # Clicking the image takes keyboard focus away from text fields.
        widget.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._last_point = None
        self._gesture = None
        self._gesture_button = Qt.MouseButton.NoButton
        self._view_zoom = 1.0
        self._view_pan = QPointF()
        self._xrot = 0.0
        self._yrot = 0.0
        self._zrot = 90.0
        self._base_display_pixmap = None
        self._base_display_key = None
        self._gizmo_basis_key = None
        self._gizmo_basis = None
        self.set_theme_colors(kit.colors(kit.current_theme() or "light"))
        kit.theme_events.changed.connect(self._follow_theme)

    def _follow_theme(self, name: str) -> None:
        self.set_theme_colors(kit.colors(name))

    def set_theme_colors(self, colors: Mapping[str, str]) -> None:
        self._empty_bg = _token_color(colors["surface-sunken"])
        # Transparent renders sit on a low-contrast checkerboard: the theme's
        # translucent tint over the sunken surface, so it reads as background.
        self._checkerboard_brush = self._make_checkerboard_brush(
            self._empty_bg, _token_color(colors["surface-tint"])
        )
        self._gizmo_halo = _token_color(colors["surface-overlay"])
        self._widget().update()

    @staticmethod
    def _make_checkerboard_brush(base: QColor, tint: QColor) -> QBrush:
        tile = 16
        pixmap = QPixmap(tile * 2, tile * 2)
        pixmap.fill(base)
        painter = QPainter(pixmap)
        painter.fillRect(tile, 0, tile, tile, tint)
        painter.fillRect(0, tile, tile, tile, tint)
        painter.end()
        return QBrush(pixmap)

    def _display_key(self, pixmap: QPixmap) -> _DisplayKey | None:
        # Key on the widget's device-pixel size so moving to a screen with a
        # different pixel ratio invalidates the scaled copy.
        device_size = self._widget().size() * self._widget().devicePixelRatioF()
        width = int(device_size.width())
        height = int(device_size.height())
        if pixmap.isNull() or width <= 0 or height <= 0:
            return None
        return (int(pixmap.cacheKey()), width, height)

    def _scale_pixmap_to_view(self, pixmap: QPixmap) -> QPixmap | None:
        if self._display_key(pixmap) is None:
            return None
        # Show at most one render pixel per device pixel and only shrink renders
        # larger than the view: upscaling blurs the image on HiDPI screens.
        widget = self._widget()
        device_pixel_ratio = widget.devicePixelRatioF()
        device_size = widget.size() * device_pixel_ratio
        if pixmap.width() <= device_size.width() and pixmap.height() <= device_size.height():
            display = QPixmap(pixmap)
        else:
            display = pixmap.scaled(
                device_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        display.setDevicePixelRatio(device_pixel_ratio)
        return display

    def _prepare_base_display(self) -> None:
        if self._pixmap is None:
            self._base_display_pixmap = None
            self._base_display_key = None
            return
        key = self._display_key(self._pixmap)
        if key is None:
            self._base_display_pixmap = None
            self._base_display_key = None
            return
        if self._base_display_pixmap is not None and self._base_display_key == key:
            return
        self._base_display_pixmap = self._scale_pixmap_to_view(self._pixmap)
        self._base_display_key = key if self._base_display_pixmap is not None else None

    def _base_display(self) -> QPixmap | None:
        self._prepare_base_display()
        return self._base_display_pixmap

    def _draw_centered_pixmap(self, painter: QPainter, pixmap: QPixmap | None) -> None:
        if pixmap is None or pixmap.isNull():
            return
        widget = self._widget()
        size = pixmap.deviceIndependentSize()
        x = int((widget.width() - size.width()) // 2)
        y = int((widget.height() - size.height()) // 2)
        painter.drawPixmap(x, y, pixmap)

    def update_image(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._prepare_base_display()
        self._widget().update()

    @property
    def view_zoom(self) -> float:
        """Image magnification relative to the fitted display."""
        return self._view_zoom

    def reset_view(self) -> None:
        """Show the whole render again, undoing wheel zoom and panning."""
        if self._view_zoom == 1.0 and self._view_pan.isNull():
            return
        self._view_zoom = 1.0
        self._view_pan = QPointF()
        self._widget().update()

    def _zoomed_size(self, base: QPixmap) -> tuple[float, float]:
        size = base.deviceIndependentSize()
        return (size.width() * self._view_zoom, size.height() * self._view_zoom)

    def _clamp_pan(self, base: QPixmap) -> None:
        # Along an axis where the zoomed render is larger than the view, keep the
        # view covered; along one where it fits, keep it centered.
        widget = self._widget()
        width, height = self._zoomed_size(base)
        limit_x = max(0.0, (width - widget.width()) / 2.0)
        limit_y = max(0.0, (height - widget.height()) / 2.0)
        self._view_pan = QPointF(
            min(limit_x, max(-limit_x, self._view_pan.x())),
            min(limit_y, max(-limit_y, self._view_pan.y())),
        )

    def _draw_zoomed_pixmap(self, painter: QPainter, base: QPixmap) -> None:
        if self._pixmap is None:
            return
        self._clamp_pan(base)
        widget = self._widget()
        width, height = self._zoomed_size(base)
        center = QPointF(widget.width() / 2.0, widget.height() / 2.0) + self._view_pan
        target = QRectF(center.x() - width / 2.0, center.y() - height / 2.0, width, height)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        # Sample the full-resolution render rather than the copy fitted to the view.
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        painter.restore()

    def _rotation_matrix(self) -> _Vector:
        """Apply the panel rotations in X, Y, Z order to column vectors."""
        xrad = np.deg2rad(self._xrot)
        yrad = np.deg2rad(self._yrot)
        zrad = np.deg2rad(self._zrot)

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(xrad), -np.sin(xrad)],
                [0.0, np.sin(xrad), np.cos(xrad)],
            ],
            dtype=np.float32,
        )
        ry = np.array(
            [
                [np.cos(yrad), 0.0, np.sin(yrad)],
                [0.0, 1.0, 0.0],
                [-np.sin(yrad), 0.0, np.cos(yrad)],
            ],
            dtype=np.float32,
        )
        rz = np.array(
            [
                [np.cos(zrad), -np.sin(zrad), 0.0],
                [np.sin(zrad), np.cos(zrad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return np.asarray(rz @ ry @ rx, dtype=np.float32)

    def set_orientation(self, *, xrot: float, yrot: float, zrot: float) -> None:
        """Set the rotation shown by the orientation gizmo, in degrees."""
        next_values = (float(xrot), float(yrot), float(zrot))
        if next_values == (self._xrot, self._yrot, self._zrot):
            return
        self._xrot, self._yrot, self._zrot = next_values
        self._gizmo_basis_key = None
        self._gizmo_basis = None
        self._widget().update()

    def _orientation_basis(self) -> list[tuple[str, QColor, _Vector]]:
        key = (self._xrot, self._yrot, self._zrot)
        if self._gizmo_basis_key == key and self._gizmo_basis is not None:
            return self._gizmo_basis

        rotation = self._rotation_matrix()
        basis = [
            (name, QColor(color), np.asarray(rotation @ np.array(axis, dtype=np.float32), dtype=np.float32))
            for name, color, axis in _GIZMO_AXES
        ]
        # Back to front, so axes nearer the viewer draw on top.
        basis.sort(key=lambda item: float(item[2][2]))
        self._gizmo_basis_key = key
        self._gizmo_basis = basis
        return basis

    def _draw_orientation_gizmo(self, painter: QPainter) -> None:
        widget = self._widget()
        if widget.width() < 120 or widget.height() < 120:
            return

        basis = self._orientation_basis()
        reach = _GIZMO_MARGIN + _GIZMO_AXIS_LENGTH + _GIZMO_LABEL_GAP
        center = QPointF(reach, widget.height() - reach)
        offsets = [_gizmo_offset(vector) for _name, _color, vector in basis]
        lengths = [max(float(np.hypot(offset.x(), offset.y())), 1e-6) for offset in offsets]
        end_on = [length < _GIZMO_END_ON * _GIZMO_AXIS_LENGTH for length in lengths]
        # With an end-on marker at the origin, the other axes start at its rim.
        hub = _GIZMO_END_ON_RADIUS + _GIZMO_HALO_WIDTH if any(end_on) else 0.0

        labels: list[QPointF] = []
        for index, offset in enumerate(offsets):
            if end_on[index]:
                # Label an end-on axis on the side away from the other two.
                others = QPointF()
                for other_index, other in enumerate(offsets):
                    if other_index != index:
                        others += other
                spread = max(float(np.hypot(others.x(), others.y())), 1e-6)
                labels.append(center - others * ((_GIZMO_END_ON_RADIUS + _GIZMO_LABEL_GAP) / spread))
            else:
                labels.append(center + offset * ((lengths[index] + _GIZMO_LABEL_GAP) / lengths[index]))
        # When two axes line up on screen, move the farther one's label aside.
        for far in range(len(labels)):
            for near in range(far + 1, len(labels)):
                gap = labels[far] - labels[near]
                if float(np.hypot(gap.x(), gap.y())) < _GIZMO_LABEL_CLEARANCE:
                    along = offsets[far] * (1.0 / lengths[far])
                    labels[far] += QPointF(-along.y(), along.x()) * _GIZMO_LABEL_CLEARANCE

        font = QFont(widget.font())
        font.setPixelSize(_GIZMO_FONT_PIXELS)
        font.setBold(True)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Back to front, so axes nearer the viewer draw on top.
        for index, (name, color, vector) in enumerate(basis):
            ink = QColor(color)
            if float(vector[2]) < 0.0:
                ink.setAlphaF(_GIZMO_AWAY_ALPHA)
            offset = offsets[index]
            if end_on[index]:
                self._draw_end_on_axis(painter, center, ink, toward_viewer=float(vector[2]) > 0.0)
            elif lengths[index] > hub:
                start = center + offset * (hub / lengths[index])
                tip = center + offset
                self._stroke_gizmo(painter, ink, lambda: painter.drawLine(start, tip))
            label = QPainterPath()
            label.addText(0.0, 0.0, font, name)
            label.translate(labels[index] - label.boundingRect().center())
            painter.strokePath(label, self._gizmo_halo_pen(_GIZMO_HALO_WIDTH * 2.0))
            painter.fillPath(label, ink)
        painter.restore()

    def _gizmo_halo_pen(self, width: float) -> QPen:
        return QPen(
            self._gizmo_halo,
            width,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )

    def _stroke_gizmo(self, painter: QPainter, ink: QColor, draw: Callable[[], None]) -> None:
        """Draw a halo under each axis to keep it readable over the image."""
        painter.setPen(self._gizmo_halo_pen(_GIZMO_LINE_WIDTH + 2.0 * _GIZMO_HALO_WIDTH))
        draw()
        painter.setPen(
            QPen(ink, _GIZMO_LINE_WIDTH, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        draw()

    def _draw_end_on_axis(
        self, painter: QPainter, center: QPointF, ink: QColor, *, toward_viewer: bool
    ) -> None:
        """Draw a circled dot for an axis pointing toward the viewer, or a cross for one pointing away."""
        radius = _GIZMO_END_ON_RADIUS
        arm = radius * 0.55

        def draw() -> None:
            painter.drawEllipse(center, radius, radius)
            if toward_viewer:
                painter.drawEllipse(center, 1.0, 1.0)
            else:
                painter.drawLine(center + QPointF(-arm, -arm), center + QPointF(arm, arm))
                painter.drawLine(center + QPointF(-arm, arm), center + QPointF(arm, -arm))

        self._stroke_gizmo(painter, ink, draw)

    def _paint_content(self, painter: QPainter) -> None:
        widget = self._widget()
        # Show transparency with a checkerboard when the rendered pixmap carries alpha.
        draw_checkerboard = self._pixmap is not None and self._pixmap.hasAlphaChannel()
        if draw_checkerboard:
            painter.fillRect(widget.rect(), self._checkerboard_brush)
        else:
            painter.fillRect(widget.rect(), self._empty_bg)
        if self._pixmap is None:
            return
        base = self._base_display()
        if self._view_zoom == 1.0 or base is None:
            self._draw_centered_pixmap(painter, base)
        else:
            self._draw_zoomed_pixmap(painter, base)
        self._draw_orientation_gizmo(painter)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._gesture is not None:
            # Another button pressed mid-gesture neither starts nor ends one.
            return
        button = event.button()
        # Middle-drag, or Shift-drag, pans a zoomed view; a plain drag rotates.
        pan = button == Qt.MouseButton.MiddleButton or (
            button == Qt.MouseButton.LeftButton
            and bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        )
        if pan and self._view_zoom > 1.0:
            self._gesture = "pan"
            self._widget().setCursor(Qt.CursorShape.ClosedHandCursor)
        elif button == Qt.MouseButton.LeftButton:
            self._gesture = "rotate"
        else:
            return
        self._gesture_button = button
        self._last_point = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._gesture is None or self._last_point is None:
            return
        current = event.position().toPoint()
        dx = current.x() - self._last_point.x()
        dy = current.y() - self._last_point.y()
        self._last_point = current
        if self._gesture == "pan":
            self._view_pan += QPointF(dx, dy)
            self._widget().update()
            return
        getattr(self, "rotation_requested").emit(float(dy * 0.5), float(dx * 0.5))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._gesture is None or event.button() != self._gesture_button:
            return
        if self._gesture == "pan":
            self._widget().unsetCursor()
        self._gesture = None
        self._gesture_button = Qt.MouseButton.NoButton
        self._last_point = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        base = self._base_display()
        if delta == 0 or base is None:
            event.ignore()
            return
        # One mouse-wheel notch (120 units) zooms 10%. Trackpads send many small
        # deltas per gesture, so the step must scale with the delta.
        zoom = min(_MAX_VIEW_ZOOM, max(1.0, self._view_zoom * 1.1 ** (delta / 120.0)))
        if abs(zoom - 1.0) < 1e-3:
            zoom = 1.0
        if zoom != self._view_zoom:
            # Keep the part of the render under the pointer where it is.
            widget = self._widget()
            anchor = event.position() - QPointF(widget.width() / 2.0, widget.height() / 2.0)
            self._view_pan = anchor - (anchor - self._view_pan) * (zoom / self._view_zoom)
            self._view_zoom = zoom
            self._clamp_pan(base)
            widget.update()
        event.accept()


class _RasterViewport(_ViewportBehavior, QWidget):
    rotation_requested = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._init_view()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        self._paint_content(painter)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._prepare_base_display()
        super().resizeEvent(event)


if _HAS_OPENGL_WIDGETS or TYPE_CHECKING:

    class _OpenGLViewport(_ViewportBehavior, QOpenGLWidget):
        rotation_requested = Signal(float, float)

        def __init__(self) -> None:
            super().__init__()
            self._init_view()

        def paintGL(self) -> None:
            painter = QPainter(self)
            self._paint_content(painter)
            painter.end()

        def resizeEvent(self, event: QResizeEvent) -> None:
            self._prepare_base_display()
            super().resizeEvent(event)


else:
    _OpenGLViewport = _RasterViewport


if TYPE_CHECKING:

    class RenderViewport(_RasterViewport):
        """Shared viewport interface for static type checking."""


elif _should_use_opengl_backend():

    class RenderViewport(_OpenGLViewport):
        """Viewport selected when Qt OpenGL widgets are available and enabled."""

else:

    class RenderViewport(_RasterViewport):
        """Raster-painted viewport for headless runs or when OpenGL is disabled."""
