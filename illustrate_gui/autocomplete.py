"""PDB autocomplete — async search-as-you-type with RCSB API."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QRect,
    QSize,
    Slot,
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

# ── Data ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PdbSuggestion:
    pdb_id: str
    title: str


_STRIP_EM = re.compile(r"</?em>")


# ── Model ────────────────────────────────────────────────────────────


class SuggestionModel(QAbstractListModel):
    """Backing model for the QCompleter popup."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[PdbSuggestion] = []

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        if parent is not None and parent.isValid():
            return 0
        return len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:  # type: ignore[override]
        if not index.isValid():
            return None
        item = self._items[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return item.pdb_id
        if role == Qt.ItemDataRole.UserRole:
            return item.title
        return None

    def set_suggestions(self, items: list[PdbSuggestion]) -> None:
        self.beginResetModel()
        self._items = items
        self.endResetModel()


# ── Network worker (runs on QThread) ─────────────────────────────────


class _FetchWorker(QObject):
    finished = Signal(str, list)  # (query_text, [PdbSuggestion, ...])

    @Slot(str)
    def fetch(self, query: str) -> None:
        suggestions: list[PdbSuggestion] = []
        try:
            pdb_ids = self._suggest_ids(query)
            if pdb_ids:
                suggestions = self._fetch_titles(pdb_ids)
        except Exception:  # noqa: BLE001 — best-effort autocomplete
            pass
        self.finished.emit(query, suggestions)

    # -- Step 1: RCSB suggest API --

    def _suggest_ids(self, query: str) -> list[str]:
        # Try term suggest (prefix match on PDB IDs) and basic suggest
        # (free-text on titles/keywords), merge results.
        ids: list[str] = []

        # Term suggest — good for "2hh" → "2HHB"
        term_payload = json.dumps({
            "type": "term",
            "suggest": {
                "text": query,
                "completion": [{"attribute": "rcsb_entry_container_identifiers.entry_id"}],
                "size": 10,
            },
            "results_content_type": ["experimental"],
        })
        term_url = f"https://search.rcsb.org/rcsbsearch/v2/suggest?json={quote(term_payload)}"
        try:
            with urlopen(term_url, timeout=5) as resp:  # noqa: S310
                data = json.loads(resp.read())
            for entry in data.get("suggestions", {}).get(
                "rcsb_entry_container_identifiers.entry_id", []
            ):
                pid = _STRIP_EM.sub("", entry.get("text", "")).strip().upper()
                if pid and pid not in ids:
                    ids.append(pid)
        except Exception:  # noqa: BLE001
            pass

        # If we got enough term results, skip basic suggest
        if len(ids) >= 5:
            return ids[:10]

        # Basic suggest — good for "hemoglobin" → titles containing it
        basic_payload = json.dumps({
            "type": "basic",
            "suggest": {"text": query, "size": 10},
            "results_content_type": ["experimental"],
        })
        basic_url = f"https://search.rcsb.org/rcsbsearch/v2/suggest?json={quote(basic_payload)}"
        try:
            with urlopen(basic_url, timeout=5) as resp:  # noqa: S310
                data = json.loads(resp.read())
            # Basic suggest returns titles grouped by field — we need to
            # resolve them to PDB IDs via a search query instead.
            # Fall back to full-text search for name-based queries.
        except Exception:  # noqa: BLE001
            pass

        # If term suggest found nothing, try full-text search
        if not ids:
            ids = self._fulltext_search(query)

        return ids[:10]

    def _fulltext_search(self, query: str) -> list[str]:
        body = json.dumps({
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": query},
            },
            "return_type": "entry",
            "request_options": {
                "results_content_type": ["experimental"],
                "paginate": {"start": 0, "rows": 10},
            },
        }).encode()
        req = Request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())
        return [hit["identifier"] for hit in data.get("result_set", [])]

    # -- Step 2: GraphQL for titles --

    def _fetch_titles(self, pdb_ids: list[str]) -> list[PdbSuggestion]:
        ids_str = ", ".join(f'"{pid}"' for pid in pdb_ids)
        gql = f'{{ entries(entry_ids: [{ids_str}]) {{ rcsb_id struct {{ title }} }} }}'
        req = Request(
            "https://data.rcsb.org/graphql",
            data=json.dumps({"query": gql}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read())

        results: list[PdbSuggestion] = []
        for entry in data.get("data", {}).get("entries", []):
            if entry is None:
                continue
            pid = entry.get("rcsb_id", "")
            title = (entry.get("struct") or {}).get("title", "")
            results.append(PdbSuggestion(pid, title.title()))
        return results


# ── Custom delegate (two-line rows) ──────────────────────────────────


class _SuggestionDelegate(QStyledItemDelegate):
    """Draws each row as bold PDB ID + smaller grey title."""

    _PAD = 6
    _GAP = 3

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:  # type: ignore[override]
        self.initStyleOption(option, index)
        painter.save()

        style = option.widget.style() if option.widget else None
        if style:
            style.drawPrimitive(QStyle.PrimitiveElement.PE_PanelItemViewItem, option, painter, option.widget)

        pdb_id = index.data(Qt.ItemDataRole.EditRole) or ""
        title = index.data(Qt.ItemDataRole.UserRole) or ""
        rect = option.rect.adjusted(self._PAD, self._PAD, -self._PAD, -self._PAD)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)

        # PDB ID — bold
        id_font = QFont(option.font)
        id_font.setBold(True)
        id_fm = QFontMetrics(id_font)
        painter.setFont(id_font)
        text_color = option.palette.highlightedText().color() if selected else option.palette.text().color()
        painter.setPen(QPen(text_color))
        id_rect = QRect(rect.x(), rect.y(), rect.width(), id_fm.height())
        painter.drawText(id_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, pdb_id)

        # Title — smaller, grey
        t_font = QFont(option.font)
        # Qt can provide a style font with an undefined point size (-1).
        if t_font.pointSize() > 0:
            t_font.setPointSize(t_font.pointSize())
        t_fm = QFontMetrics(t_font)
        painter.setFont(t_font)
        if not selected:
            painter.setPen(QPen(Qt.GlobalColor.gray))
        t_rect = QRect(rect.x(), id_rect.bottom() + self._GAP, rect.width(), t_fm.height())
        elided = t_fm.elidedText(title, Qt.TextElideMode.ElideRight, t_rect.width())
        painter.drawText(t_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:  # type: ignore[override]
        id_font = QFont(option.font)
        id_font.setBold(True)
        t_font = QFont(option.font)
        if t_font.pointSize() > 0:
            t_font.setPointSize(t_font.pointSize())
        h = self._PAD + QFontMetrics(id_font).height() + self._GAP + QFontMetrics(t_font).height() + self._PAD
        return QSize(option.rect.width(), h)


# ── Public completer class ───────────────────────────────────────────


class PdbCompleter(QObject):
    """Async PDB autocomplete for a QLineEdit.

    Usage::

        line_edit = QLineEdit()
        completer = PdbCompleter(line_edit)
        completer.activated.connect(on_pdb_selected)
    """

    activated = Signal(str)  # emitted with PDB ID when user picks a suggestion
    _fetch_requested = Signal(str)

    def __init__(self, line_edit: QLineEdit, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._line_edit = line_edit
        self._last_query = ""

        # Model + completer
        self._model = SuggestionModel(self)
        self._completer = QCompleter(self)
        self._completer.setModel(self._model)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setMaxVisibleItems(10)
        line_edit.setCompleter(self._completer)

        # Custom delegate for two-line rows
        popup = self._completer.popup()
        assert popup is not None
        delegate = _SuggestionDelegate(popup)
        popup.setItemDelegate(delegate)
        popup.setMinimumWidth(500)
        # Style the popup to match dark theme
        popup.setStyleSheet(
            "QListView { background-color: #252526; color: #cccccc;"
            " border: 1px solid #444444; }"
            "QListView::item { padding: 4px; }"
            "QListView::item:selected { background-color: #094771; }"
        )

        # Forward activated
        self._completer.activated.connect(self.activated.emit)

        # Debounce timer
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._on_timer)

        line_edit.textEdited.connect(self._on_text_edited)

        # Worker thread
        self._thread = QThread(self)
        self._worker = _FetchWorker()
        self._worker.moveToThread(self._thread)
        self._fetch_requested.connect(self._worker.fetch, Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(self._on_results)
        self._thread.start()

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
        query = self._last_query
        if query:
            self._fetch_requested.emit(query)

    def _on_results(self, query_text: str, suggestions: list[PdbSuggestion]) -> None:
        if self._line_edit.text().strip() != query_text:
            return
        self._model.set_suggestions(suggestions)
        self._completer.setCompletionPrefix(query_text)
        # Show popup at full width so titles are readable
        rect = self._line_edit.rect()
        rect.setWidth(500)
        self._completer.complete(rect)

    def cleanup(self) -> None:
        self._thread.quit()
        self._thread.wait(7000)
