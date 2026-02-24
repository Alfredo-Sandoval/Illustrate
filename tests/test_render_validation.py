from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from illustrate.render import params_from_json, render
from illustrate.types import RenderParams, SelectionRule, Transform


def _write_minimal_pdb(path: Path) -> None:
    line = (
        f"{'ATOM':<6}{1:5d} {'CA':<4}{' ':1}{'GLY':>3} {'A':1}{1:4d}    "
        f"{0.000:8.3f}{0.000:8.3f}{-5.000:8.3f}"
    )
    path.write_text(line + "\nEND\n", encoding="utf-8")


def test_render_rejects_rule_color_with_too_few_components(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)

    params = RenderParams(
        pdb_path=str(pdb_path),
        rules=[
            SelectionRule(
                record_name="ATOM  ",
                descriptor="----------",
                res_low=0,
                res_high=9999,
                color=cast(tuple[float, float, float], (1.0, 0.0)),
                radius=1.5,
            )
        ],
    )

    with pytest.raises(ValueError, match="color must contain at least 3 components"):
        render(params)


def test_render_rejects_translate_with_too_few_components(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)

    params = RenderParams(
        pdb_path=str(pdb_path),
        rules=[
            SelectionRule(
                record_name="ATOM  ",
                descriptor="----------",
                res_low=0,
                res_high=9999,
                color=(1.0, 1.0, 1.0),
                radius=1.5,
            )
        ],
        transform=Transform(
            scale=12.0,
            translate=cast(tuple[float, float, float], (1.0,)),
            rotations=[],
            autocenter="auto",
        ),
    )

    with pytest.raises(ValueError, match="transform translate must contain at least 3 components"):
        render(params)


def test_params_from_json_rejects_non_object_top_level_payload() -> None:
    with pytest.raises(TypeError, match="top-level JSON payload must be an object"):
        params_from_json("[]")


def test_params_from_json_allows_null_sections_with_defaults() -> None:
    params = params_from_json('{"pdb_path":"mini.pdb","rules":[],"transform":null,"world":null,"outlines":null}')
    assert params.pdb_path == "mini.pdb"
    assert params.transform.autocenter == "auto"
    assert params.world.width == 0
    assert params.outlines.kernel == 4
