from __future__ import annotations

from pathlib import Path

from illustrate.pdb import read_and_classify_atoms
from illustrate.types import SelectionRule


def _write_pdb(path: Path, atom_line: str) -> None:
    path.write_text(atom_line + "\nEND\n", encoding="utf-8")


def _minimal_atom_line(x: str = "0.000", y: str = "0.000", z: str = "-5.000") -> str:
    return (
        f"{'ATOM':<6}{1:5d} {'CA':<4}{' ':1}{'GLY':>3} {'A':1}{1:4d}    "
        f"{x:>8}{y:>8}{z:>8}"
    )


def test_invalid_fixed_float_defaults_to_zero_in_non_strict_mode(tmp_path: Path) -> None:
    bad = _minimal_atom_line(x="abc")
    _write_pdb(tmp_path / "bad.pdb", bad)

    rules = [
        SelectionRule(
            record_name="ATOM",
            descriptor="----------",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.0, 0.0),
            radius=1.5,
        )
    ]

    atoms = read_and_classify_atoms(tmp_path / "bad.pdb", rules, strict_input=False)

    assert atoms.n == 1
    assert atoms.coord[1, 0] == 0.0


def test_invalid_fixed_float_fails_in_strict_mode(tmp_path: Path) -> None:
    bad = _minimal_atom_line(y="not-a-number")
    _write_pdb(tmp_path / "bad.pdb", bad)

    rules = [
        SelectionRule(
            record_name="ATOM",
            descriptor="----------",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.0, 0.0),
            radius=1.5,
        )
    ]

    try:
        read_and_classify_atoms(tmp_path / "bad.pdb", rules, strict_input=True)
    except ValueError as exc:
        assert "could not convert string to float" in str(exc)
    else:
        raise AssertionError("Expected strict parser to fail on invalid fixed-width float")


def test_invalid_biomt_values_are_ignored_in_non_strict_mode(tmp_path: Path) -> None:
    atom = _minimal_atom_line()
    pdb_lines = [
        "REMARK 350 BIOMOLECULE: 1",
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A",
        "REMARK 350   BIOMT1   1  BAD BAD BAD BAD",
        "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000",
        "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000",
        "END",
        atom,
        "END",
    ]
    pdb_path = tmp_path / "bad_biomt.pdb"
    pdb_path.write_text("\n".join(pdb_lines) + "\n", encoding="utf-8")

    rules = [
        SelectionRule(
            record_name="ATOM",
            descriptor="----------",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.0, 0.0),
            radius=1.5,
        )
    ]

    atoms = read_and_classify_atoms(pdb_path, rules, strict_input=False)
    assert atoms.n == 1


def test_invalid_biomt_values_fail_in_strict_mode(tmp_path: Path) -> None:
    atom = _minimal_atom_line()
    pdb_lines = [
        "REMARK 350 BIOMOLECULE: 1",
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A",
        "REMARK 350   BIOMT1   1  BAD BAD BAD BAD",
        "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000",
        "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000",
        "END",
        atom,
        "END",
    ]
    pdb_path = tmp_path / "bad_biomt.pdb"
    pdb_path.write_text("\n".join(pdb_lines) + "\n", encoding="utf-8")

    rules = [
        SelectionRule(
            record_name="ATOM",
            descriptor="----------",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.0, 0.0),
            radius=1.5,
        )
    ]

    try:
        read_and_classify_atoms(pdb_path, rules, strict_input=True)
    except ValueError as exc:
        assert "could not convert string to float" in str(exc)
    else:
        raise AssertionError("Expected strict parser to fail on malformed BIOMT values")


def test_biomt_overflow_is_capped_in_non_strict_mode(tmp_path: Path) -> None:
    atom = _minimal_atom_line()
    lines = [
        "REMARK 350 BIOMOLECULE: 1",
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A",
    ]
    for i in range(1, 502):
        lines.append(f"REMARK 350   BIOMT1 {i:>3d}  1.000000  0.000000  0.000000        0.00000")
        lines.append(f"REMARK 350   BIOMT2 {i:>3d}  0.000000  1.000000  0.000000        0.00000")
        lines.append(f"REMARK 350   BIOMT3 {i:>3d}  0.000000  0.000000  1.000000        0.00000")
    lines.extend(["END", atom, "END"])
    pdb_path = tmp_path / "biomt_overflow.pdb"
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rules = [
        SelectionRule(
            record_name="ATOM",
            descriptor="----------",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.0, 0.0),
            radius=1.5,
        )
    ]

    atoms = read_and_classify_atoms(pdb_path, rules, strict_input=False)
    assert atoms.n == 1
    assert atoms.nbiomat == 500


def test_biomt_overflow_fails_in_strict_mode(tmp_path: Path) -> None:
    atom = _minimal_atom_line()
    lines = [
        "REMARK 350 BIOMOLECULE: 1",
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A",
    ]
    for i in range(1, 502):
        lines.append(f"REMARK 350   BIOMT1 {i:>3d}  1.000000  0.000000  0.000000        0.00000")
        lines.append(f"REMARK 350   BIOMT2 {i:>3d}  0.000000  1.000000  0.000000        0.00000")
        lines.append(f"REMARK 350   BIOMT3 {i:>3d}  0.000000  0.000000  1.000000        0.00000")
    lines.extend(["END", atom, "END"])
    pdb_path = tmp_path / "biomt_overflow.pdb"
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rules = [
        SelectionRule(
            record_name="ATOM",
            descriptor="----------",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.0, 0.0),
            radius=1.5,
        )
    ]

    try:
        read_and_classify_atoms(pdb_path, rules, strict_input=True)
    except ValueError as exc:
        assert "too many BIOMT transforms" in str(exc)
    else:
        raise AssertionError("Expected strict parser to fail when BIOMT count exceeds limit")
