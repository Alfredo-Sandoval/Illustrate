from __future__ import annotations

import math

from illustrate.math3d import rotate_xyz
from illustrate.parser import parse_command_stream


def test_parser_three_char_commands_and_cumulative_state() -> None:
    stream = "\n".join(
        [
            "read",
            "fake.pdb",
            "END",
            "translate",
            "1.0,2.0,3.0",
            "trans",
            "4.0,5.0,6.0",
            "scale",
            "2.0",
            "sca",
            "0.5",
            "calculate",
            "out.pnm",
        ],
    )

    program = parse_command_stream(stream)
    assert math.isclose(program.transform.xtran, 5.0)
    assert math.isclose(program.transform.ytran, 7.0)
    assert math.isclose(program.transform.ztran, 9.0)
    assert math.isclose(program.transform.scale, 1.0)


def test_unknown_command_is_non_fatal_in_legacy_mode() -> None:
    stream = "\n".join(
        [
            "read",
            "fake.pdb",
            "END",
            "foo",
            "calculate",
            "out.pnm",
        ],
    )

    program = parse_command_stream(stream)
    assert program.output_file is not None
    assert any("unknown command card" in warning for warning in program.warnings)


def test_rotation_concatenation_matches_fortran_order() -> None:
    stream = "\n".join(
        [
            "read",
            "fake.pdb",
            "END",
            "zrot",
            "90.0",
            "zrot",
            "90.0",
            "calculate",
            "out.pnm",
        ],
    )

    program = parse_command_stream(stream)
    x, y, z = rotate_xyz(1.0, 0.0, 0.0, program.transform.rm)

    assert abs(float(x) + 1.0) < 1e-5
    assert abs(float(y) - 0.0) < 1e-5
    assert abs(float(z) - 0.0) < 1e-5
