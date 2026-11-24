from __future__ import annotations

import json
from pathlib import Path


from illustrate.core.types import RenderParams
from illustrate.io.preset_store import (
    format_custom_preset_load_issues,
    load_custom_presets,
    write_custom_presets,
)
from illustrate.io.params_json import PARAMS_JSON_VERSION, params_to_json


def _params_payload() -> dict[str, object]:
    return json.loads(params_to_json(RenderParams(pdb_path="", rules=[])))


def test_load_custom_presets_with_issues_keeps_valid_entries_and_reports_skips(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "presets.json"
    valid_payload = _params_payload()
    store_path.write_text(
        json.dumps(
            {
                "presets": [
                    {"name": "Good", "params": valid_payload},
                    {
                        "name": "Bad Boolean",
                        "params": {
                            "pdb_path": "",
                            "rules": [],
                            "world": {"shadows": "false"},
                        },
                    },
                    {"name": "", "params": valid_payload},
                    7,
                ]
            }
        ),
        encoding="utf-8",
    )

    result = load_custom_presets(store_path)

    assert [(name, params.pdb_path) for name, params in result.presets] == [
        ("Good", "")
    ]
    assert [(issue.location, issue.message) for issue in result.issues] == [
        ("entry 2 (Bad Boolean)", "world.shadows must be a boolean"),
        ("entry 3", "name and params must be present"),
        ("entry 4", "must be an object"),
    ]
    assert "entry 2 (Bad Boolean): world.shadows must be a boolean" in (
        format_custom_preset_load_issues(result.issues)
    )


def test_load_custom_presets_reports_invalid_json(tmp_path: Path) -> None:
    store_path = tmp_path / "presets.json"
    store_path.write_text("{bad", encoding="utf-8")

    result = load_custom_presets(store_path)

    assert result.presets == []
    assert result.issues[0].location == "store"
    assert "invalid JSON at line 1, column 2" in result.issues[0].message


def test_write_custom_presets_records_canonical_schema_version(tmp_path: Path) -> None:
    store_path = tmp_path / "presets.json"

    write_custom_presets([("Good", RenderParams(pdb_path="", rules=[]))], store_path)

    payload = json.loads(store_path.read_text(encoding="utf-8"))
    assert payload["presets"][0]["params"]["schema_version"] == PARAMS_JSON_VERSION
