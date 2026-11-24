from __future__ import annotations

import math
from pathlib import Path
import re

import pytest
from hypothesis import given, settings, strategies as st

from illustrate.formats.command_stream import (
    ParseError,
    parse_command_file,
    parse_command_stream,
)


def test_parse_invalid_numeric_is_tolerated_in_default_mode() -> None:
    stream = "\n".join(
        [
            "read",
            "mini.pdb",
            "END",
            "translate",
            "not-a-number",
            "scale",
            "2.0",
            "calculate",
            "out.pnm",
        ],
    )

    program = parse_command_stream(stream)

    assert math.isclose(program.transform.xtran, 0.0)
    assert math.isclose(program.transform.ytran, 0.0)
    assert math.isclose(program.transform.ztran, 0.0)
    assert math.isclose(program.transform.scale, 2.0)
    assert program.warnings == ["invalid numeric line for translate: 'not-a-number'"]


def test_parse_short_numeric_line_warns_in_default_mode() -> None:
    stream = "\n".join(
        [
            "read",
            "mini.pdb",
            "END",
            "translate",
            "1.0,2.0",
            "calculate",
            "out.pnm",
        ],
    )

    program = parse_command_stream(stream)

    assert math.isclose(program.transform.xtran, 0.0)
    assert math.isclose(program.transform.ytran, 0.0)
    assert math.isclose(program.transform.ztran, 0.0)
    assert len(program.warnings) == 1
    assert "short numeric line for translate" in program.warnings[0]
    assert "expected at least 3 values, got 2" in program.warnings[0]


def test_parse_invalid_numeric_fails_in_strict_mode() -> None:
    stream = "\n".join(
        [
            "read",
            "mini.pdb",
            "END",
            "translate",
            "not-a-number",
            "calculate",
            "out.pnm",
        ],
    )

    with pytest.raises(ParseError, match=r"invalid numeric line for translate"):
        parse_command_stream(stream, strict_input=True)


def test_parse_strict_mode_rejects_missing_read() -> None:
    stream = "\n".join(
        [
            "scale",
            "2.0",
            "calculate",
            "out.pnm",
        ],
    )

    with pytest.raises(ParseError, match=re.escape("missing read card")):
        parse_command_stream(stream, strict_input=True)


def test_parse_strict_mode_rejects_missing_calculate() -> None:
    stream = "\n".join(
        [
            "read",
            "mini.pdb",
            "END",
        ],
    )

    with pytest.raises(ParseError, match=re.escape("missing calculate card")):
        parse_command_stream(stream, strict_input=True)


def test_parse_command_file_reads_path(tmp_path: Path) -> None:
    command_path = tmp_path / "mini.inp"
    command_path.write_text(
        "\n".join(
            [
                "read",
                "mini.pdb",
                "END",
                "scale",
                "2.0",
                "calculate",
                "out.pnm",
            ]
        ),
        encoding="utf-8",
    )

    program = parse_command_file(command_path)

    assert program.pdb_file == Path("mini.pdb")
    assert program.output_file == Path("out.pnm")
    assert math.isclose(program.transform.scale, 2.0)


def test_parse_command_file_fails_on_invalid_utf8(tmp_path: Path) -> None:
    command_path = tmp_path / "bad.inp"
    command_path.write_bytes(b"read\nmini.pdb\n\xff\n")

    with pytest.raises(UnicodeDecodeError):
        parse_command_file(command_path)


@settings(derandomize=True, max_examples=100)
@given(
    card=st.sampled_from(["translate", "scale", "xrot", "world", "illustrate"]),
    lines=st.lists(st.text(alphabet="0123456789,+-.e infNaN", max_size=60), max_size=3),
    strict=st.booleans(),
)
def test_numeric_command_payloads_parse_or_raise_parse_error(
    card: str, lines: list[str], strict: bool,
) -> None:
    stream = "\n".join(["read", "mini.pdb", "END", card, *lines])
    try:
        program = parse_command_stream(stream, strict_input=strict)
    except ParseError as exc:
        assert str(exc)
        return
    assert math.isfinite(program.transform.scale)
    assert all(math.isfinite(value) for value in (
        program.transform.xtran, program.transform.ytran, program.transform.ztran,
        program.world.fog_front, program.world.fog_back,
    ))


@settings(derandomize=True, max_examples=100)
@given(
    translation=st.tuples(st.integers(-1000, 1000), st.integers(-1000, 1000), st.integers(-1000, 1000)),
    scale=st.integers(1, 50),
)
def test_numeric_transform_cards_round_trip(
    translation: tuple[int, int, int], scale: int,
) -> None:
    stream = f"read\nmini.pdb\nEND\ntranslate\n{translation[0]},{translation[1]},{translation[2]}\nscale\n{scale}\ncalculate\nout.png\n"
    program = parse_command_stream(stream, strict_input=True)
    actual = program.transform
    assert (actual.xtran, actual.ytran, actual.ztran) == translation
    assert actual.scale == scale
    rebuilt = f"read\nmini.pdb\nEND\ntranslate\n{actual.xtran},{actual.ytran},{actual.ztran}\nscale\n{actual.scale}\ncalculate\nout.png\n"
    again = parse_command_stream(rebuilt, strict_input=True).transform
    assert (again.xtran, again.ytran, again.ztran, again.scale) == (*translation, scale)
