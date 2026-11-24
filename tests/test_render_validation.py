from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from illustrate.io.params_json import (
    PARAMS_JSON_VERSION,
    params_from_json,
    params_to_json,
)
from illustrate.rendering.api import render
from illustrate.core.types import (
    OutlineParams,
    RenderParams,
    SelectionRule,
    Transform,
    WorldParams,
)


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


@pytest.mark.parametrize("kernel", [0, 5])
def test_render_rejects_unknown_outline_kernel(tmp_path: Path, kernel: int) -> None:
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
                color=(1.0, 0.0, 0.0),
                radius=1.5,
            )
        ],
        world=WorldParams(width=40, height=40),
        outlines=OutlineParams(enabled=True, kernel=kernel),
    )

    with pytest.raises(ValueError, match="outlines.kernel must be 1, 2, 3, or 4"):
        render(params)


def test_render_rejects_more_selection_rules_than_the_rule_table_holds(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    rule = SelectionRule(
        record_name="ATOM  ",
        descriptor="----------",
        res_low=0,
        res_high=9999,
        color=(1.0, 0.0, 0.0),
        radius=1.5,
    )
    params = RenderParams(
        pdb_path=str(pdb_path),
        rules=[rule] * 1001,
        world=WorldParams(width=40, height=40),
    )

    with pytest.raises(ValueError, match="at most 1000 selection rules are supported, got 1001"):
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

    with pytest.raises(
        ValueError, match="transform translate must contain at least 3 components"
    ):
        render(params)


def test_params_from_json_rejects_non_object_top_level_payload() -> None:
    with pytest.raises(TypeError, match="top-level JSON payload must be an object"):
        params_from_json("[]")


def test_params_from_json_allows_null_sections_with_defaults() -> None:
    params = params_from_json(
        '{"pdb_path":"mini.pdb","rules":[],"transform":null,"world":null,"outlines":null}'
    )
    assert params.pdb_path == "mini.pdb"
    assert params.transform == Transform()
    assert params.world == WorldParams()
    assert params.outlines == OutlineParams()


def test_params_from_json_keeps_dataclass_defaults_for_partial_outlines() -> None:
    params = params_from_json(
        '{"pdb_path":"mini.pdb","rules":[],"outlines":{"enabled":true}}'
    )
    assert params.outlines.enabled is True
    assert params.outlines.contour_low == 1.0
    assert params.outlines.z_diff_min == 1.0
    assert params.outlines.z_diff_max == 50.0


def test_params_json_records_schema_version() -> None:
    params = RenderParams(pdb_path="mini.pdb", rules=[])
    payload = json.loads(params_to_json(params))
    assert payload["schema_version"] == PARAMS_JSON_VERSION


def test_params_json_round_trip_preserves_nested_render_contract() -> None:
    params = RenderParams(
        pdb_path="mini.pdb",
        rules=[
            SelectionRule(
                record_name="ATOM  ",
                descriptor="----------",
                res_low=1,
                res_high=99,
                color=(0.25, 0.5, 0.75),
                radius=1.25,
            )
        ],
        transform=Transform(
            scale=9.5,
            translate=(1.0, 2.0, 3.0),
            rotations=[("x", 45.0), ("z", -12.5)],
            autocenter="none",
        ),
        world=WorldParams(
            background=(0.1, 0.2, 0.3),
            fog_color=(0.9, 0.8, 0.7),
            fog_front=0.2,
            fog_back=1.4,
            shadows=True,
            shadow_strength=0.004,
            shadow_angle=1.5,
            shadow_min_z=0.25,
            shadow_max_dark=0.4,
            width=321,
            height=123,
        ),
        outlines=OutlineParams(
            enabled=True,
            contour_low=2.0,
            contour_high=11.0,
            kernel=6,
            z_diff_min=0.5,
            z_diff_max=12.5,
            subunit_low=4.0,
            subunit_high=14.0,
            residue_low=5.0,
            residue_high=15.0,
            residue_diff=7000.0,
        ),
    )

    payload = params_to_json(params)
    restored = params_from_json(payload)

    assert restored == params


def test_params_from_json_rejects_unsupported_schema_version() -> None:
    with pytest.raises(
        ValueError, match="unsupported RenderParams schema_version: 999"
    ):
        params_from_json('{"schema_version":999,"pdb_path":"mini.pdb","rules":[]}')


@pytest.mark.parametrize(
    ("payload", "error_type", "match"),
    [
        (
            '{"schema_version":"bad","pdb_path":"mini.pdb","rules":[]}',
            TypeError,
            "schema_version must be an integer",
        ),
        (
            '{"schema_version":1.0,"pdb_path":"mini.pdb","rules":[]}',
            TypeError,
            "schema_version must be an integer",
        ),
        ('{"rules":[]}', TypeError, "pdb_path must be a string"),
        ('{"pdb_path":"mini.pdb","rules":{}}', TypeError, "rules must be a list"),
        (
            '{"pdb_path":"mini.pdb","rules":[1]}',
            TypeError,
            r"rules\[0\] must be an object",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[{"record_name":"ATOM  ","descriptor":"----------","res_low":0,"res_high":1,"radius":1.0,"color":[1,0]}]}',
            ValueError,
            r"rules\[0\]\.color must contain at least 3 components",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[{"record_name":"ATOM  ","descriptor":"----------","res_low":0.5,"res_high":1,"radius":1.0}]}',
            ValueError,
            r"rules\[0\]\.res_low must be an integer-compatible value",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[],"transform":{"translate":[1,2]}}',
            ValueError,
            "transform.translate must contain at least 3 components",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[],"transform":{"rotations":"x"}}',
            TypeError,
            "transform.rotations must be a list",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[],"transform":{"rotations":[["q",10]]}}',
            ValueError,
            r"transform\.rotations\[0\]\[0\] must be one of: x, y, z",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[],"transform":{"autocenter":"sideways"}}',
            ValueError,
            "unsupported autocenter mode",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[],"world":[]}',
            TypeError,
            "world must be an object",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[],"world":{"shadows":"false"}}',
            TypeError,
            "world.shadows must be a boolean",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[],"world":{"fog_front":NaN}}',
            ValueError,
            "world.fog_front must be finite",
        ),
        (
            '{"pdb_path":"mini.pdb","rules":[],"outlines":{"enabled":1}}',
            TypeError,
            "outlines.enabled must be a boolean",
        ),
    ],
)
def test_params_from_json_rejects_malformed_contract_sections(
    payload: str,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        params_from_json(payload)
