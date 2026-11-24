from __future__ import annotations

from collections import OrderedDict
from http.client import HTTPException, HTTPSConnection
import json
import re
from typing import cast
from urllib.parse import quote

from PySide6.QtCore import QObject, Signal, Slot

from illustrate.gui.pdb_suggestions import PdbSuggestion


_STRIP_EM = re.compile(r"</?em>")
# A classic PDB ID, or the start of one: a digit, then up to three letters or digits.
_PDB_ID_PREFIX = re.compile(r"\d[A-Za-z0-9]{0,3}")
_SEARCH_HOST = "search.rcsb.org"
_DATA_HOST = "data.rcsb.org"
_TIMEOUT_S = 5
# OSError covers refused, reset, and timed-out connections.
_LOOKUP_ERRORS = (HTTPException, OSError, json.JSONDecodeError, KeyError, TypeError)


def _json_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object in the RCSB response.")
    return cast(dict[str, object], value)


class PdbAutocompleteWorker(QObject):
    """Fetch suggestions using cached query results and structure titles."""

    finished = Signal(str, list)  # (query_text, [PdbSuggestion, ...])

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._query_cache: OrderedDict[str, list[PdbSuggestion]] = OrderedDict()
        self._title_cache: OrderedDict[str, PdbSuggestion] = OrderedDict()
        # One kept-alive connection per host: a new TLS handshake for every
        # request costs more than the request itself.
        self._connections: dict[str, HTTPSConnection] = {}
        self._latest_query: str | None = None

    def set_latest_query(self, query: str) -> None:
        """Mark the current query so the worker can skip superseded lookups.

        Call from the GUI thread before queuing the new request.
        """
        self._latest_query = query

    @Slot()
    def close_connections(self) -> None:
        """Release the kept-alive sockets on the lookup thread."""
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()

    def _is_superseded(self, query: str) -> bool:
        latest = self._latest_query
        return latest is not None and latest != query

    def _request_json(
        self, host: str, method: str, path: str, body: bytes | None = None
    ) -> dict[str, object]:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        # A kept-alive connection the server has closed fails once; retry on a new one.
        for attempt in range(2):
            connection = self._connections.get(host)
            if connection is None:
                connection = HTTPSConnection(host, timeout=_TIMEOUT_S)
                self._connections[host] = connection
            try:
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                payload = response.read()
            except (HTTPException, OSError):
                connection.close()
                del self._connections[host]
                if attempt:
                    raise
                continue
            if response.status == 204:  # the search service's "no results"
                return {}
            if response.status != 200:
                raise HTTPException(f"{host} answered {response.status}")
            return _json_object(json.loads(payload))
        raise HTTPException(f"{host} did not answer")

    @Slot(str)
    def fetch(self, query: str) -> None:
        query_text = str(query).strip()
        suggestions: list[PdbSuggestion] = []
        cache_key = query_text.lower()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._query_cache.move_to_end(cache_key)
            self.finished.emit(query, list(cached))
            return

        if self._is_superseded(query):
            return
        try:
            pdb_ids = self._suggest_ids(query_text)
        except _LOOKUP_ERRORS:
            pdb_ids = []
        if pdb_ids and not self._is_superseded(query):
            suggestions = [
                self._title_cache.get(pid, PdbSuggestion(pid, "")) for pid in pdb_ids
            ]
            if any(not suggestion.title for suggestion in suggestions):
                # Show the IDs now; the titles follow in a second result.
                self.finished.emit(query, list(suggestions))
                try:
                    suggestions = self._fetch_titles(pdb_ids)
                except _LOOKUP_ERRORS:
                    pass
        if self._is_superseded(query):
            return
        self._store_query(cache_key, suggestions)
        self.finished.emit(query, suggestions)

    def _suggest_ids(self, query: str) -> list[str]:
        # Only something shaped like a PDB ID can complete to one; a name goes
        # straight to the text search instead of paying for an empty ID lookup.
        if _PDB_ID_PREFIX.fullmatch(query):
            try:
                ids = self._complete_ids(query)
            except _LOOKUP_ERRORS:
                ids = []
            if ids:
                return ids[:10]
        return self._fulltext_search(query)[:10]

    def _complete_ids(self, query: str) -> list[str]:
        attribute = "rcsb_entry_container_identifiers.entry_id"
        payload = json.dumps(
            {
                "type": "term",
                "suggest": {
                    "text": query,
                    "completion": [{"attribute": attribute}],
                    "size": 10,
                },
                "results_content_type": ["experimental"],
            }
        )
        data = self._request_json(
            _SEARCH_HOST, "GET", f"/rcsbsearch/v2/suggest?json={quote(payload)}"
        )
        entries = _json_object(data.get("suggestions", {})).get(attribute, [])
        if not isinstance(entries, list):
            raise TypeError("Expected a list of suggestions in the RCSB response.")
        ids: list[str] = []
        for entry in cast(list[object], entries):
            pid = _STRIP_EM.sub("", str(_json_object(entry).get("text", ""))).strip().upper()
            if pid and pid not in ids:
                ids.append(pid)
        return ids

    def _fulltext_search(self, query: str) -> list[str]:
        body = json.dumps(
            {
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
            }
        ).encode()
        data = self._request_json(_SEARCH_HOST, "POST", "/rcsbsearch/v2/query", body)
        hits = data.get("result_set", [])
        if not isinstance(hits, list):
            raise TypeError("Expected a list of hits in the RCSB response.")
        return [str(_json_object(hit)["identifier"]) for hit in cast(list[object], hits)]

    def _fetch_titles(self, pdb_ids: list[str]) -> list[PdbSuggestion]:
        normalized_ids = [pid.strip().upper() for pid in pdb_ids if pid.strip()]
        missing = [pid for pid in normalized_ids if pid not in self._title_cache]
        if missing:
            ids_str = ", ".join(f'"{pid}"' for pid in missing)
            gql = f"{{ entries(entry_ids: [{ids_str}]) {{ rcsb_id struct {{ title }} }} }}"
            data = self._request_json(
                _DATA_HOST, "POST", "/graphql", json.dumps({"query": gql}).encode()
            )
            entries = _json_object(data.get("data", {})).get("entries", [])
            if not isinstance(entries, list):
                raise TypeError("Expected a list of entries in the RCSB response.")
            for entry in cast(list[object], entries):
                if entry is None:
                    continue
                model = _json_object(entry)
                pid = str(model.get("rcsb_id", "")).strip().upper()
                title = str(_json_object(model.get("struct") or {}).get("title", ""))
                if pid:
                    self._store_title(pid, PdbSuggestion(pid, title.title()))

        return [
            self._title_cache.get(pid, PdbSuggestion(pid, "")) for pid in normalized_ids
        ]

    def _store_query(self, cache_key: str, suggestions: list[PdbSuggestion]) -> None:
        self._query_cache[cache_key] = list(suggestions)
        self._query_cache.move_to_end(cache_key)
        while len(self._query_cache) > 32:
            self._query_cache.popitem(last=False)

    def _store_title(self, pdb_id: str, suggestion: PdbSuggestion) -> None:
        self._title_cache[pdb_id] = suggestion
        self._title_cache.move_to_end(pdb_id)
        while len(self._title_cache) > 128:
            self._title_cache.popitem(last=False)


__all__ = ["PdbAutocompleteWorker"]
