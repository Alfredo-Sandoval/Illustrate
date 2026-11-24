"""Built-in preset listing and JSON storage for user presets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import cast

from illustrate.presets import PRESET_NAMES, default_rules, preset_library
from illustrate.io.params_json import params_from_json, params_to_json
from illustrate.core.types import RenderParams


@dataclass(frozen=True, slots=True)
class CustomPresetLoadIssue:
    location: str
    message: str


@dataclass(frozen=True, slots=True)
class CustomPresetLoadResult:
    presets: list[tuple[str, RenderParams]]
    issues: list[CustomPresetLoadIssue]


def builtin_preset_items() -> list[tuple[str, RenderParams]]:
    presets = preset_library(default_rules())
    return [(name, preset) for name, preset in zip(PRESET_NAMES, presets)]


def _custom_preset_store_path() -> Path:
    return Path.home() / ".illustrate_presets.json"


def _format_issue(issue: CustomPresetLoadIssue) -> str:
    return f"{issue.location}: {issue.message}"


def format_custom_preset_load_issues(issues: list[CustomPresetLoadIssue]) -> str:
    return "; ".join(_format_issue(issue) for issue in issues)


def load_custom_presets(
    path: str | Path | None = None,
) -> CustomPresetLoadResult:
    """Read custom presets, returning valid entries alongside any load issues."""
    store_path = _custom_preset_store_path() if path is None else Path(path)
    issues: list[CustomPresetLoadIssue] = []

    if not store_path.exists():
        return CustomPresetLoadResult(presets=[], issues=[])
    try:
        text = store_path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(
            CustomPresetLoadIssue("store", f"could not read custom preset store: {exc}")
        )
        return CustomPresetLoadResult(presets=[], issues=issues)

    try:
        payload_raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        issues.append(
            CustomPresetLoadIssue(
                "store",
                f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            )
        )
        return CustomPresetLoadResult(presets=[], issues=issues)

    if not isinstance(payload_raw, dict):
        issues.append(CustomPresetLoadIssue("store", "preset store must be an object"))
        return CustomPresetLoadResult(presets=[], issues=issues)
    payload = cast(dict[str, object], payload_raw)

    items: list[tuple[str, RenderParams]] = []
    presets_payload_raw = payload.get("presets", [])
    if not isinstance(presets_payload_raw, list):
        issues.append(CustomPresetLoadIssue("store.presets", "presets must be a list"))
        return CustomPresetLoadResult(presets=[], issues=issues)

    for entry_number, raw_entry in enumerate(cast(list[object], presets_payload_raw), start=1):
        location = f"entry {entry_number}"
        if not isinstance(raw_entry, dict):
            issues.append(CustomPresetLoadIssue(location, "must be an object"))
            continue
        entry = cast(dict[str, object], raw_entry)
        name_payload = entry.get("name")
        name = name_payload.strip() if isinstance(name_payload, str) else ""
        params_payload = entry.get("params")
        if not name or not isinstance(params_payload, dict):
            issues.append(
                CustomPresetLoadIssue(location, "name and params must be present")
            )
            continue
        try:
            params = params_from_json(json.dumps(params_payload))
        except (TypeError, ValueError) as exc:
            issues.append(CustomPresetLoadIssue(f"{location} ({name})", str(exc)))
            continue
        items.append((name, params))
    return CustomPresetLoadResult(presets=items, issues=issues)


def write_custom_presets(
    presets: list[tuple[str, RenderParams]],
    path: str | Path | None = None,
) -> None:
    payload = {
        "version": 1,
        "presets": [
            {"name": name, "params": json.loads(params_to_json(params))}
            for name, params in presets
        ],
    }
    store_path = _custom_preset_store_path() if path is None else Path(path)
    store_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
