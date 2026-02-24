"""PDB suggestion proxy route.

Client-side requests to RCSB suggest/search endpoints can be blocked by origin
policies. This route performs lookups server-side and returns a compact list of
suggestions for the web UI.
"""

from __future__ import annotations

import asyncio
import json
import re
from urllib import parse, request

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api")
_EM_TAG_RE = re.compile(r"</?em>", re.IGNORECASE)


def _http_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, object] | None = None,
) -> dict[str, object]:
    headers: dict[str, str] = {}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, data=data, method=method, headers=headers)
    with request.urlopen(req, timeout=7) as resp:  # nosec B310 - fixed HTTPS host
        payload = resp.read().decode("utf-8")
    parsed = json.loads(payload)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _normalize_id(text: object) -> str:
    value = _EM_TAG_RE.sub("", str(text)).strip().upper()
    return value


def _suggest_ids(query: str) -> list[str]:
    ids: list[str] = []
    term_payload = {
        "type": "term",
        "suggest": {
            "text": query,
            "completion": [{"attribute": "rcsb_entry_container_identifiers.entry_id"}],
            "size": 10,
        },
        "results_content_type": ["experimental"],
    }
    try:
        term_url = "https://search.rcsb.org/rcsbsearch/v2/suggest?json=" + parse.quote(json.dumps(term_payload))
        term_data = _http_json(term_url)
        term_list = (
            term_data.get("suggestions", {})
            .get("rcsb_entry_container_identifiers.entry_id", [])
            if isinstance(term_data.get("suggestions"), dict)
            else []
        )
        if isinstance(term_list, list):
            for entry in term_list:
                if not isinstance(entry, dict):
                    continue
                candidate = _normalize_id(entry.get("text", ""))
                if candidate and candidate not in ids:
                    ids.append(candidate)
    except Exception:
        pass

    if len(ids) >= 5:
        return ids[:10]

    search_payload = {
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
    try:
        search_data = _http_json(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            method="POST",
            body=search_payload,
        )
        result_set = search_data.get("result_set", [])
        if isinstance(result_set, list):
            for hit in result_set:
                if not isinstance(hit, dict):
                    continue
                candidate = _normalize_id(hit.get("identifier", ""))
                if candidate and candidate not in ids:
                    ids.append(candidate)
    except Exception:
        pass
    return ids[:10]


def _lookup_titles(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    ids_str = ", ".join(f'"{pid}"' for pid in ids)
    query = f'{{ entries(entry_ids: [{ids_str}]) {{ rcsb_id struct {{ title }} }} }}'
    try:
        gql_data = _http_json(
            "https://data.rcsb.org/graphql",
            method="POST",
            body={"query": query},
        )
    except Exception:
        return {}

    data_node = gql_data.get("data", {})
    if not isinstance(data_node, dict):
        return {}
    entries = data_node.get("entries", [])
    if not isinstance(entries, list):
        return {}

    titles: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pdb_id = _normalize_id(entry.get("rcsb_id", ""))
        if not pdb_id:
            continue
        struct = entry.get("struct", {})
        title = ""
        if isinstance(struct, dict):
            title = str(struct.get("title", "")).strip()
        titles[pdb_id] = title
    return titles


def _suggest_pdb(query: str) -> list[dict[str, str]]:
    ids = _suggest_ids(query)
    titles = _lookup_titles(ids)
    return [{"pdb_id": pid, "title": titles.get(pid, "")} for pid in ids]


@router.get("/pdb-suggest")
async def suggest_pdb(
    q: str = Query(min_length=2, max_length=64),
) -> list[dict[str, str]]:
    query = q.strip()
    if len(query) < 2:
        return []
    return await asyncio.to_thread(_suggest_pdb, query)
