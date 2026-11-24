from __future__ import annotations

import re
from pathlib import Path

import illustrate


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reference_python_symbols_are_root_importable() -> None:
    reference = (REPO_ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
    headings = set(
        re.findall(r"^### `([A-Za-z_][A-Za-z0-9_]*)", reference, flags=re.MULTILINE)
    )

    assert {"render", "RenderParams", "export_png"} <= headings
    assert headings - set(illustrate.__all__) == set()
