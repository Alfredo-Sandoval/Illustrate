from __future__ import annotations

import math
import re

import pytest

from illustrate.parser import ParseError, parse_command_stream


def test_parse_invalid_numeric_is_tolerated_in_legacy_mode() -> None:
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
    assert program.warnings == []


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
