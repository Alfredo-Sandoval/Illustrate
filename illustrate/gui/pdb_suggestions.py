from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QAbstractListModel, QModelIndex, QObject, QPersistentModelIndex, Qt


@dataclass(slots=True)
class PdbSuggestion:
    pdb_id: str
    title: str


class SuggestionModel(QAbstractListModel):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[PdbSuggestion] = []

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._items)

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid():
            return None
        item = self._items[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return item.pdb_id
        if role in (Qt.ItemDataRole.UserRole, Qt.ItemDataRole.ToolTipRole):
            return item.title
        return None

    def set_suggestions(self, items: list[PdbSuggestion]) -> None:
        self.beginResetModel()
        self._items = items
        self.endResetModel()
