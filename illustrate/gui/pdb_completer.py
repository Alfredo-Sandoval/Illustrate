from __future__ import annotations

from collections import OrderedDict

from PySide6.QtCore import (
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QRect,
    QSize,
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QCompleter,
    QLineEdit,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)
from illustrate.gui.pdb_suggestions import PdbSuggestion, SuggestionModel
from illustrate.gui.workers.pdb_autocomplete import PdbAutocompleteWorker as _FetchWorker

# The popup is at least this wide so titles stay readable under a narrow field.
_POPUP_MIN_WIDTH = 360


def _title_font(font: QFont) -> QFont:
    title_font = QFont(font)
    if title_font.pixelSize() > 0:
        title_font.setPixelSize(title_font.pixelSize() - 1)
    elif title_font.pointSizeF() > 0:
        title_font.setPointSizeF(title_font.pointSizeF() - 1)
    return title_font


class _SuggestionDelegate(QStyledItemDelegate):
    """Draw each row as bold PDB ID plus a smaller title."""

    _PAD = 6
    _GAP = 3

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        self.initStyleOption(option, index)
        painter.save()

        style = option.widget.style() if option.widget else None
        if style:
            style.drawPrimitive(
                QStyle.PrimitiveElement.PE_PanelItemViewItem,
                option,
                painter,
                option.widget,
            )

        pdb_id = index.data(Qt.ItemDataRole.EditRole) or ""
        title = index.data(Qt.ItemDataRole.UserRole) or ""
        rect = option.rect.adjusted(self._PAD, self._PAD, -self._PAD, -self._PAD)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)

        id_font = QFont(option.font)
        id_font.setBold(True)
        id_fm = QFontMetrics(id_font)
        painter.setFont(id_font)
        text_color = (
            option.palette.highlightedText().color()
            if selected
            else option.palette.text().color()
        )
        painter.setPen(QPen(text_color))
        id_rect = QRect(rect.x(), rect.y(), rect.width(), id_fm.height())
        painter.drawText(
            id_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, pdb_id
        )

        title_font = _title_font(option.font)
        title_fm = QFontMetrics(title_font)
        painter.setFont(title_font)
        if not selected:
            painter.setPen(QPen(option.palette.placeholderText().color()))
        title_rect = QRect(
            rect.x(), id_rect.bottom() + self._GAP, rect.width(), title_fm.height()
        )
        elided = title_fm.elidedText(
            title, Qt.TextElideMode.ElideRight, title_rect.width()
        )
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            elided,
        )

        painter.restore()

    def sizeHint(
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> QSize:
        id_font = QFont(option.font)
        id_font.setBold(True)
        title_font = _title_font(option.font)
        height = (
            self._PAD
            + QFontMetrics(id_font).height()
            + self._GAP
            + QFontMetrics(title_font).height()
            + self._PAD
        )
        return QSize(option.rect.width(), height)


class PdbCompleter(QObject):
    """Debounce RCSB lookups for a line edit and emit the selected PDB ID."""

    activated = Signal(str)  # the PDB ID of the suggestion the user picked
    _fetch_requested = Signal(str)

    def __init__(self, line_edit: QLineEdit, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._line_edit = line_edit
        self._stopped = False
        self._last_query = ""
        self._result_cache: OrderedDict[str, list[PdbSuggestion]] = OrderedDict()

        self._model = SuggestionModel(self)
        self._completer = QCompleter(self)
        self._completer.setModel(self._model)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setMaxVisibleItems(10)
        line_edit.setCompleter(self._completer)

        popup = self._completer.popup()
        assert popup is not None
        popup.setObjectName("pdbSuggestions")  # styled in illustrate/gui/themes/*.qss
        delegate = _SuggestionDelegate(popup)
        popup.setItemDelegate(delegate)

        self._completer.activated.connect(self.activated.emit)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._on_timer)

        line_edit.textEdited.connect(self._on_text_edited)

        self._thread = QThread(self)
        self._worker = _FetchWorker()
        self._worker.moveToThread(self._thread)
        self._fetch_requested.connect(
            self._worker.fetch, Qt.ConnectionType.QueuedConnection
        )
        self._worker.finished.connect(self._on_results)
        self._thread.finished.connect(self._worker.close_connections)
        self._thread.finished.connect(self._worker.deleteLater)

    def _on_text_edited(self, text: str) -> None:
        self._last_query = text.strip()
        if len(self._last_query) < 2:
            self._model.set_suggestions([])
            popup = self._completer.popup()
            if popup is not None:
                popup.hide()
            return
        self._timer.start()

    def _on_timer(self) -> None:
        if self._stopped:
            return
        query = self._last_query
        if query:
            cached = self._result_cache.get(query.lower())
            if cached is not None:
                self._result_cache.move_to_end(query.lower())
                self._show_results(query, list(cached))
                return
            # Lookups still queued for text typed earlier are skipped.
            self._worker.set_latest_query(query)
            if not self._thread.isRunning():
                self._thread.start()
            self._fetch_requested.emit(query)

    def _on_results(self, query_text: str, suggestions: list[PdbSuggestion]) -> None:
        if self._stopped:
            return
        if self._line_edit.text().strip().lower() != query_text.strip().lower():
            return
        self._store_results(query_text, suggestions)
        self._show_results(query_text, suggestions)

    def _show_results(self, query_text: str, suggestions: list[PdbSuggestion]) -> None:
        self._model.set_suggestions(suggestions)
        self._completer.setCompletionPrefix(query_text)
        # Anchor under the field; widen past a narrow field so titles stay readable.
        rect = self._line_edit.rect()
        rect.setWidth(max(rect.width(), _POPUP_MIN_WIDTH))
        self._completer.complete(rect)

    def _store_results(self, query_text: str, suggestions: list[PdbSuggestion]) -> None:
        cache_key = query_text.lower()
        self._result_cache[cache_key] = list(suggestions)
        self._result_cache.move_to_end(cache_key)
        while len(self._result_cache) > 32:
            self._result_cache.popitem(last=False)

    def stop(self) -> None:
        """Stop issuing lookups and request thread exit after the current lookup."""
        if self._stopped:
            return
        self._stopped = True
        self._timer.stop()
        self._thread.quit()

    def is_running(self) -> bool:
        return self._thread.isRunning()
