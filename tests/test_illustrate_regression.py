from __future__ import annotations

from pathlib import Path

import pytest

from illustrate.formats.command_stream import ParseError, parse_command_stream
from illustrate.formats.pdb import read_and_classify_atoms
from illustrate.rendering.api import render_from_command_file


def _pdb_atom_line(serial: int, chain: str, x: float, y: float, z: float) -> str:
    return (
        f"{'ATOM':<6}{serial:5d} {'CA':<4}{' ':1}{'GLY':>3} {chain:1}{1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
    )


def test_biomt_chain_filtering_keeps_only_whitelisted_chain(tmp_path: Path) -> None:
    pdb_lines = [
        "REMARK 350 BIOMOLECULE: 1",
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A",
        "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000",
        "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000",
        "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000",
        "END",
        _pdb_atom_line(1, "A", 0.0, 0.0, -5.0),
        _pdb_atom_line(2, "B", 1.0, 1.0, -5.0),
        "END",
    ]
    pdb_path = tmp_path / "biomt.pdb"
    pdb_path.write_text("\n".join(pdb_lines) + "\n", encoding="utf-8")

    stream = "\n".join(
        [
            "read",
            "biomt.pdb",
            "ATOM  ---------- 0,9999, 1.0,1.0,1.0, 1.5",
            "END",
            "calculate",
            "out.pnm",
        ],
    )
    program = parse_command_stream(stream)
    assert program.pdb_file is not None
    atoms = read_and_classify_atoms(tmp_path / program.pdb_file, program.selection_rules)

    assert atoms.n == 1


def test_strict_mode_rejects_unknown_command() -> None:
    stream = "\n".join(["foo", "bar"])
    with pytest.raises(ParseError):
        parse_command_stream(stream, strict_input=True)


def test_render_from_command_file_propagates_strict_input_to_pdb_parse(tmp_path: Path) -> None:
    bad_atom = (
        f"{'ATOM':<6}{1:5d} {'CA':<4}{' ':1}{'GLY':>3} {'A':1}{1:4d}    "
        f"{'abc':>8}{0.000:8.3f}{-5.000:8.3f}"
    )
    pdb_path = tmp_path / "bad_coords.pdb"
    pdb_path.write_text(bad_atom + "\nEND\n", encoding="utf-8")

    stream = "\n".join(
        [
            "read",
            "bad_coords.pdb",
            "ATOM  ---------- 0,9999, 1.0,1.0,1.0, 1.5",
            "END",
            "wor",
            "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            "0.0,0.0023,2.0,1.0,0.2",
            "20,20",
            "calculate",
            "out.pnm",
        ],
    )

    with pytest.raises(ValueError, match="could not convert string to float"):
        render_from_command_file(stream, strict_input=True, base_dir=tmp_path)


def test_render_from_command_file_base_dir_resolves_relative_pdb_path(
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "mini.pdb"
    pdb_path.write_text(_pdb_atom_line(1, "A", 0.0, 0.0, -5.0) + "\nEND\n", encoding="utf-8")

    stream = "\n".join(
        [
            "read",
            "mini.pdb",
            "ATOM  ---------- 0,9999, 1.0,1.0,1.0, 1.5",
            "END",
            "calculate",
            "out.pnm",
        ],
    )

    with pytest.raises(FileNotFoundError):
        render_from_command_file(stream)

    result = render_from_command_file(stream, base_dir=tmp_path)
    assert result.rgb.shape[2] == 3
