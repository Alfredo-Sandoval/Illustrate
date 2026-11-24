from __future__ import annotations

from pathlib import Path

import pytest

from illustrate.formats.pdb import classify_atoms, read_and_classify_atoms, read_pdb_records
from illustrate.core.types import SelectionRule


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

    with pytest.raises(ValueError, match="could not convert string to float"):
        read_and_classify_atoms(tmp_path / "bad.pdb", rules, strict_input=True)


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

    with pytest.raises(ValueError, match="could not convert string to float"):
        read_and_classify_atoms(pdb_path, rules, strict_input=True)


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

    with pytest.raises(ValueError, match="too many BIOMT transforms"):
        read_and_classify_atoms(pdb_path, rules, strict_input=True)


def test_atoms_directly_after_biomt_block_are_kept(tmp_path: Path) -> None:
    # Trimmed headers can omit the blank REMARK line that closes the assembly
    # block; the ATOM records that follow must still be read.
    pdb_lines = [
        "REMARK 350 BIOMOLECULE: 1",
        "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A",
        "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000",
        "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000",
        "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000",
        _minimal_atom_line(x="1.000"),
        _minimal_atom_line(x="2.000"),
        "END",
    ]
    pdb_path = tmp_path / "trimmed_header.pdb"
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

    atoms = read_and_classify_atoms(pdb_path, rules, strict_input=True)
    assert atoms.n == 2
    assert atoms.nbiomat == 1
    assert atoms.coord[1:, 0].tolist() == [1.0, 2.0]


def test_truncated_atom_line_fails_in_strict_mode(tmp_path: Path) -> None:
    truncated = _minimal_atom_line()[:30]  # ends before the coordinate columns
    _write_pdb(tmp_path / "truncated.pdb", truncated)

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

    with pytest.raises(ValueError, match="missing required fixed-column float field"):
        read_and_classify_atoms(tmp_path / "truncated.pdb", rules, strict_input=True)


def test_one_read_classifies_like_a_fresh_read_for_each_rule_list(tmp_path: Path) -> None:
    def atom(record: str, serial: int, name: str, residue: str, chain: str, number: int) -> str:
        return (
            f"{record:<6}{serial:5d} {name:<4} {residue:>3} {chain}{number:4d}    "
            f"{serial:8.3f}{0.0:8.3f}{-5.0:8.3f}"
        )

    pdb_path = tmp_path / "chains.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                atom("ATOM", 1, "CA", "GLY", "A", 1),
                atom("ATOM", 2, "H", "GLY", "A", 1),
                atom("ATOM", 3, "CA", "ALA", "B", 2),
                "MODEL        2",
                atom("HETATM", 4, "O", "HOH", "B", 3),
                atom("ATOM", 5, "CA", "GLY", "B", 40),
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def rule(record: str, descriptor: str, radius: float, res_high: int = 9999) -> SelectionRule:
        return SelectionRule(record, descriptor, 0, res_high, (1.0, 0.0, 0.0), radius)

    rule_lists = [
        [rule("ATOM", "----------", 1.5)],
        # Hidden hydrogens, chain B only up to residue 9, then water.
        [rule("ATOM", "-H--------", 0.0), rule("ATOM", "---------B", 1.5, 9), rule("HETATM", "-----HOH--", 1.0)],
        [],
    ]
    records = read_pdb_records(pdb_path)
    for rules in rule_lists:
        expected = read_and_classify_atoms(pdb_path, rules)
        actual = classify_atoms(records, rules)
        assert actual.n == expected.n
        for field in ("coord", "type_idx", "res", "su"):
            assert getattr(actual, field).tolist() == getattr(expected, field).tolist()

    kept = classify_atoms(records, rule_lists[1])
    assert kept.type_idx.tolist() == [0, 2, 3]
    assert kept.res.tolist() == [9999, 2, 3]
    # The MODEL record starts a new subunit even though the chain is unchanged.
    assert kept.su.tolist() == [9999, 1, 2]
