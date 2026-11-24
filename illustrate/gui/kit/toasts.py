"""Toast notifications painted inside a host widget, in theme colors."""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import QEvent, QObject, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from illustrate.gui.kit.themes import colors, current_theme

__all__ = ["ToastKind", "ToastManager"]

ToastKind = Literal["info", "success", "warning", "error"]

_ACCENT_TOKEN: dict[ToastKind, str] = {
    "info": "text-muted",
    "success": "success",
    "warning": "warning",
    "error": "error",
}
_DURATION_MS: dict[ToastKind, int] = {
    "info": 3000,
    "success": 3000,
    "warning": 5000,
    "error": 7000,
}
_MAX_WIDTH = 420
_MIN_WIDTH = 220
_PAD_X = 14
_PAD_Y = 10
_ACCENT_WIDTH = 3
_RADIUS = 6.0
_MARGIN = 14
_SPACING = 8
_MAX_VISIBLE = 4


class _Toast(QWidget):
    """One message card. It ignores the mouse and closes itself on a timer."""

    closed = Signal()

    def __init__(self, text: str, kind: ToastKind, host: QWidget) -> None:
        super().__init__(host)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._text = text
        self._kind: ToastKind = kind
        self._closed = False
        text_left = _ACCENT_WIDTH + _PAD_X
        bounds = QFontMetrics(self.font()).boundingRect(
            QRect(0, 0, _MAX_WIDTH - text_left - _PAD_X, 10_000),
            Qt.TextFlag.TextWordWrap,
            text,
        )
        self.setFixedSize(
            max(_MIN_WIDTH, text_left + bounds.width() + _PAD_X),
            bounds.height() + 2 * _PAD_Y,
        )
        QTimer.singleShot(_DURATION_MS[kind], self, self.dismiss)

    def dismiss(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.hide()
        self.closed.emit()
        self.deleteLater()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        table = colors(current_theme() or "dark")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        card = QPainterPath()
        card.addRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), _RADIUS, _RADIUS
        )
        painter.fillPath(card, QColor(table["surface-raised"]))
        painter.setClipPath(card)
        painter.fillRect(
            QRect(0, 0, _ACCENT_WIDTH, self.height()),
            QColor(table[_ACCENT_TOKEN[self._kind]]),
        )
        painter.setClipping(False)
        painter.setPen(QPen(QColor(table["border-default"]), 1))
        painter.drawPath(card)
        painter.setPen(QColor(table["text-primary"]))
        text_left = _ACCENT_WIDTH + _PAD_X
        painter.drawText(
            QRect(
                text_left,
                _PAD_Y,
                self.width() - text_left - _PAD_X,
                self.height() - 2 * _PAD_Y,
            ),
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
            | Qt.TextFlag.TextWordWrap,
            self._text,
        )


class ToastManager(QObject):
    """Stack toasts in the bottom-right corner of ``host``, newest at the bottom.

    At most four stay visible; older or overflowing toasts close early. Call
    ``shutdown`` before the host is destroyed.
    """

    def __init__(self, host: QWidget) -> None:
        super().__init__(host)
        self._host = host
        self._toasts: list[_Toast] = []
        self._shut_down = False
        host.installEventFilter(self)

    def show_toast(self, text: str, kind: ToastKind = "info") -> None:
        if self._shut_down:
            return
        toast = _Toast(text, kind, self._host)
        toast.closed.connect(lambda: self._forget(toast))
        self._toasts.append(toast)
        while len(self._toasts) > _MAX_VISIBLE:
            self._toasts[0].dismiss()
        toast.show()
        toast.raise_()
        self._layout()

    def visible_count(self) -> int:
        return len(self._toasts)

    def shutdown(self) -> None:
        self._shut_down = True
        self._host.removeEventFilter(self)
        for toast in list(self._toasts):
            toast.dismiss()

    def _forget(self, toast: _Toast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
            self._layout()

    def _layout(self) -> None:
        right = self._host.width() - _MARGIN
        bottom = self._host.height() - _MARGIN
        overflow: list[_Toast] = []
        for toast in reversed(self._toasts):
            top = bottom - toast.height()
            if top < _MARGIN:
                overflow.append(toast)
                continue
            toast.move(right - toast.width(), top)
            bottom = top - _SPACING
        for toast in overflow:
            toast.dismiss()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._host and event.type() == QEvent.Type.Resize:
            self._layout()
        return super().eventFilter(watched, event)
