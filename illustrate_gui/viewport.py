"""Viewport widget for desktop image display."""

from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import QPoint, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except Exception:  # pragma: no cover - optional OpenGL backend
    QOpenGLWidget = None  # type: ignore[assignment]


def _should_use_opengl_backend() -> bool:
    if QOpenGLWidget is None:
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
    _preview_pixmap: QPixmap | None
    _has_preview_scene: bool
    _last_point: QPoint | None
    _dragging: bool
    _preview_coords: np.ndarray | None
    _preview_colors: np.ndarray | None
    _preview_radii: np.ndarray | None
    _preview_center: np.ndarray | None
    _preview_extent: float
    _preview_scale: float
    _preview_xrot: float
    _preview_yrot: float
    _preview_zrot: float
    _preview_xtran: float
    _preview_ytran: float
    _preview_ztran: float
    _preview_until: float
    _preview_bg: QColor
    _preview_bg_color: tuple[int, int, int]
    _preview_fog_color: tuple[float, float, float]
    _preview_fog_front: float
    _preview_fog_back: float

    def _init_view(self) -> None:
        self._pixmap = None
        self._preview_pixmap = None
        self._has_preview_scene = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)  # type: ignore[attr-defined]
        self.setMinimumSize(320, 240)  # type: ignore[attr-defined]
        self.setMouseTracking(True)  # type: ignore[attr-defined]
        self._last_point = None
        self._dragging = False
        self._preview_coords = None
        self._preview_colors = None
        self._preview_radii = None
        self._preview_center = None
        self._preview_extent = 1.0
        self._preview_scale = 12.0
        self._preview_xrot = 0.0
        self._preview_yrot = 0.0
        self._preview_zrot = 90.0
        self._preview_xtran = 0.0
        self._preview_ytran = 0.0
        self._preview_ztran = 0.0
        self._preview_until = 0.0
        self._preview_bg = QColor(11, 14, 20)
        self._preview_bg_color = (11, 14, 20)
        self._preview_fog_color = (1.0, 1.0, 1.0)
        self._preview_fog_front = 1.0
        self._preview_fog_back = 1.0

    def update_image(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._preview_pixmap = None
        self.update()  # type: ignore[attr-defined]

    def update_preview_image(self, pixmap: QPixmap | None) -> None:
        self._preview_pixmap = pixmap
        self.update()  # type: ignore[attr-defined]

    def set_preview_background(self, rgb: tuple[float, float, float]) -> None:
        self._preview_bg = QColor(
            int(max(0.0, min(1.0, rgb[0])) * 255.0),
            int(max(0.0, min(1.0, rgb[1])) * 255.0),
            int(max(0.0, min(1.0, rgb[2])) * 255.0),
        )
        self.update()  # type: ignore[attr-defined]

    def set_preview_world(
        self,
        *,
        background: tuple[float, float, float],
        fog_color: tuple[float, float, float],
        fog_front: float,
        fog_back: float,
    ) -> None:
        self._preview_bg_color = (
            int(max(0.0, min(1.0, background[0])) * 255.0),
            int(max(0.0, min(1.0, background[1])) * 255.0),
            int(max(0.0, min(1.0, background[2])) * 255.0),
        )
        self._preview_bg = QColor(*self._preview_bg_color)
        self._preview_fog_color = fog_color
        self._preview_fog_front = fog_front
        self._preview_fog_back = fog_back
        self.update()  # type: ignore[attr-defined]

    def set_preview_scene(
        self,
        coords: np.ndarray | None,
        colors: np.ndarray | None,
        radii: np.ndarray | None,
    ) -> None:
        del colors, radii

        if coords is None:
            self._has_preview_scene = False
            self._preview_coords = None
            self._preview_colors = None
            self._preview_radii = None
            self._preview_center = None
            self._preview_extent = 1.0
            self._preview_pixmap = None
            self.update()  # type: ignore[attr-defined]
            return

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("preview coords must be Nx3")

        finite_mask = np.isfinite(coords).all(axis=1)
        if not np.any(finite_mask):
            self._has_preview_scene = False
            self._preview_coords = None
            self._preview_colors = None
            self._preview_radii = None
            self._preview_center = None
            self._preview_extent = 1.0
            self._preview_pixmap = None
            self.update()  # type: ignore[attr-defined]
            return

        # Scene metadata is used only as an availability flag; renderer-backed
        # preview frames provide the actual pixels.
        self._has_preview_scene = True
        self._preview_coords = np.empty((1, 3), dtype=np.float32)
        self._preview_colors = np.empty((1, 3), dtype=np.float32)
        self._preview_radii = np.empty((1,), dtype=np.float32)
        self._preview_center = np.zeros((3,), dtype=np.float32)
        self._preview_extent = 1.0
        self._preview_pixmap = None
        self.update()  # type: ignore[attr-defined]

    def set_preview_transform(
        self,
        *,
        scale: float,
        xrot: float,
        yrot: float,
        zrot: float,
        xtran: float,
        ytran: float,
        ztran: float,
    ) -> None:
        self._preview_scale = float(scale)
        self._preview_xrot = float(xrot)
        self._preview_yrot = float(yrot)
        self._preview_zrot = float(zrot)
        self._preview_xtran = float(xtran)
        self._preview_ytran = float(ytran)
        self._preview_ztran = float(ztran)
        self.update()  # type: ignore[attr-defined]

    def _touch_preview_window(self, duration_s: float = 0.25) -> None:
        self._preview_until = max(self._preview_until, time.monotonic() + duration_s)
        QTimer.singleShot(int(duration_s * 1000) + 30, self.update)  # type: ignore[arg-type]

    def _draw_orientation_gizmo(self, painter: QPainter) -> None:
        if self.width() < 120 or self.height() < 120:  # type: ignore[attr-defined]
            return

        xrad = np.deg2rad(self._preview_xrot)
        yrad = np.deg2rad(self._preview_yrot)
        zrad = np.deg2rad(self._preview_zrot)

        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, np.cos(xrad), -np.sin(xrad)], [0.0, np.sin(xrad), np.cos(xrad)]],
            dtype=np.float32,
        )
        ry = np.array(
            [[np.cos(yrad), 0.0, np.sin(yrad)], [0.0, 1.0, 0.0], [-np.sin(yrad), 0.0, np.cos(yrad)]],
            dtype=np.float32,
        )
        rz = np.array(
            [[np.cos(zrad), -np.sin(zrad), 0.0], [np.sin(zrad), np.cos(zrad), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        rotation = rz @ ry @ rx
        basis = [
            ("X", QColor(255, 70, 70), rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("Y", QColor(100, 255, 120), rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            ("Z", QColor(90, 160, 255), rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ]
        basis.sort(key=lambda item: float(item[2][2]))

        origin = QPoint(56, self.height() - 56)  # type: ignore[attr-defined]
        radius = 30.0
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 115))
        painter.drawEllipse(origin, 36, 36)

        for axis_name, axis_color, vector in basis:
            end = QPoint(
                int(round(origin.x() + float(vector[0]) * radius)),
                int(round(origin.y() - float(vector[1]) * radius)),
            )
            pen = QPen(axis_color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(origin, end)
            painter.drawText(end + QPoint(4, -4), axis_name)

    def _paint_content(self, painter: QPainter) -> None:
        def _paint_checkerboard() -> None:
            tile = 18
            rect = self.rect()  # type: ignore[attr-defined]
            color_a = QColor(48, 48, 52)
            color_b = QColor(64, 64, 70)
            for y in range(0, rect.height(), tile):
                y_index = (y // tile) % 2
                for x in range(0, rect.width(), tile):
                    x_index = (x // tile) % 2
                    painter.fillRect(x, y, tile, tile, color_a if (x_index + y_index) % 2 == 0 else color_b)

        show_preview = self._has_preview_scene and (
            self._pixmap is None or time.monotonic() < self._preview_until
        )
        if show_preview:
            active_preview = self._preview_pixmap if self._preview_pixmap is not None else self._pixmap
            if active_preview is not None and active_preview.hasAlphaChannel():
                _paint_checkerboard()
            else:
                painter.fillRect(self.rect(), self._preview_bg)  # type: ignore[attr-defined]

            if self._preview_pixmap is not None:
                scaled = self._preview_pixmap.scaled(
                    self.size(),  # type: ignore[attr-defined]
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                x = (self.width() - scaled.width()) // 2  # type: ignore[attr-defined]
                y = (self.height() - scaled.height()) // 2  # type: ignore[attr-defined]
                painter.drawPixmap(x, y, scaled)
            elif self._pixmap is not None:
                scaled = self._pixmap.scaled(
                    self.size(),  # type: ignore[attr-defined]
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                x = (self.width() - scaled.width()) // 2  # type: ignore[attr-defined]
                y = (self.height() - scaled.height()) // 2  # type: ignore[attr-defined]
                painter.drawPixmap(x, y, scaled)
            self._draw_orientation_gizmo(painter)
            return

        # Show transparency with a checkerboard when the rendered pixmap carries alpha.
        draw_checkerboard = self._pixmap is not None and self._pixmap.hasAlphaChannel()
        if draw_checkerboard:
            _paint_checkerboard()
        else:
            painter.fillRect(self.rect(), QColor(43, 43, 43))  # type: ignore[attr-defined]
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),  # type: ignore[attr-defined]
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2  # type: ignore[attr-defined]
        y = (self.height() - scaled.height()) // 2  # type: ignore[attr-defined]
        painter.drawPixmap(x, y, scaled)
        self._draw_orientation_gizmo(painter)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_point = event.position().toPoint()
            self._touch_preview_window()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging or self._last_point is None:
            return
        current = event.position().toPoint()
        dx = current.x() - self._last_point.x()
        dy = current.y() - self._last_point.y()
        self._last_point = current
        self.rotation_requested.emit(float(dy * 0.5), float(dx * 0.5))  # type: ignore[attr-defined]
        self._touch_preview_window()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False
        self._last_point = None
        self._touch_preview_window(0.35)
        super().mouseReleaseEvent(event)  # type: ignore[misc]

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.angleDelta().y() == 0:
            return
        factor = 1.0 + (0.1 if event.angleDelta().y() > 0 else -0.1)
        self.zoom_requested.emit(float(factor))  # type: ignore[attr-defined]
        self._touch_preview_window()
        event.accept()


class _RasterViewport(_ViewportBehavior, QWidget):
    rotation_requested = Signal(float, float)
    zoom_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self._init_view()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        self._paint_content(painter)


if QOpenGLWidget is not None:

    class _OpenGLViewport(_ViewportBehavior, QOpenGLWidget):
        rotation_requested = Signal(float, float)
        zoom_requested = Signal(float)

        def __init__(self) -> None:
            super().__init__()
            self._init_view()

        def paintGL(self) -> None:
            painter = QPainter(self)
            self._paint_content(painter)
            painter.end()


else:
    _OpenGLViewport = _RasterViewport


if _should_use_opengl_backend():

    class RenderViewport(_OpenGLViewport):
        """Viewport used by desktop UI with OpenGL acceleration when available."""

else:

    class RenderViewport(_RasterViewport):
        """Viewport used by desktop UI with raster fallback rendering."""


def is_opengl_viewport(widget: QWidget) -> bool:
    return QOpenGLWidget is not None and isinstance(widget, QOpenGLWidget)
