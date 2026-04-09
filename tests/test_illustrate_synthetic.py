from __future__ import annotations

from pathlib import Path

import numpy as np

from illustrate.parser import parse_command_stream
from illustrate.pdb import read_and_classify_atoms
from illustrate.render import _render_program as render


def _write_minimal_pdb(path: Path) -> None:
    line = (
        f"{'ATOM':<6}{1:5d} {'CA':<4}{' ':1}{'GLY':>3} {'A':1}{1:4d}    "
        f"{0.000:8.3f}{0.000:8.3f}{-5.000:8.3f}"
    )
    path.write_text(line + "\nEND\n", encoding="utf-8")


def test_render_empty_after_radius_zero_filter(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)

    command_stream = "\n".join(
        [
            "read",
            "mini.pdb",
            "ATOM  ---------- 0,9999, 1.0,1.0,1.0, 0.0",
            "END",
            "center",
            "cen",
            "wor",
            "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            "0,0.0023,2.0,1.0,0.2",
            "40,40",
            "calculate",
            "out.pnm",
        ],
    )

    program = parse_command_stream(command_stream)
    assert program.pdb_file is not None
    atoms = read_and_classify_atoms(tmp_path / program.pdb_file, program.selection_rules)
    result = render(program, atoms)

    assert result.width == 40
    assert result.height == 40
    assert result.rgb.shape == (40, 40, 3)
    assert np.all(result.opacity == 0)


def test_autosize_with_negative_frame_values(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)

    command_stream = "\n".join(
        [
            "read",
            "mini.pdb",
            "ATOM  ---------- 0,9999, 1.0,0.5,0.5, 1.5",
            "END",
            "center",
            "aut",
            "scale",
            "10.0",
            "wor",
            "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            "0,0.0023,2.0,1.0,0.2",
            "-10,-10",
            "calculate",
            "out.pnm",
        ],
    )

    program = parse_command_stream(command_stream)
    assert program.pdb_file is not None
    atoms = read_and_classify_atoms(tmp_path / program.pdb_file, program.selection_rules)
    result = render(program, atoms)

    assert result.width > 0
    assert result.height > 0
    assert result.width % 2 == 0
    assert result.height % 2 == 0
