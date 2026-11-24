from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from illustrate.core.types import AtomTable, SelectionRule



def _fixed_80_columns(line: str) -> str:
    return line.rstrip("\n\r")[:80].ljust(80)


def _parse_free_floats(text: str) -> list[float]:
    cleaned = text.replace(",", " ")
    return [float(tok) for tok in cleaned.split()]


def _parse_free_floats_lenient(text: str, strict_input: bool) -> list[float]:
    try:
        return _parse_free_floats(text)
    except ValueError:
        if strict_input:
            raise
        return []


def _parse_fixed_int(text: str, strict_input: bool, default: int = 0) -> int:
    stripped = text.strip()
    if not stripped:
        if strict_input:
            raise ValueError("missing required fixed-column integer field")
        return default
    try:
        return int(stripped)
    except ValueError:
        if strict_input:
            raise
        return default


def _parse_fixed_float(text: str, strict_input: bool, default: float = 0.0) -> float:
    stripped = text.strip()
    if not stripped:
        if strict_input:
            raise ValueError("missing required fixed-column float field")
        return default
    try:
        return float(stripped)
    except ValueError:
        if strict_input:
            raise
        return default


def _lenient_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class PdbRecords:
    """Atom records and assembly transforms before rule matching.

    Atom arrays have one row per ATOM or HETATM line in file order. Keep these
    records to apply a different rule list without rereading the file.
    """

    record: NDArray[np.uint32]  # (n, 6) code points of columns 1-6
    descriptor: NDArray[np.uint32]  # (n, 10) code points of columns 13-22
    res: NDArray[np.int32]
    coord: NDArray[np.float32]  # (n, 3); unreadable fields are 0.0
    # MODEL records seen before each atom; each one starts a new subunit.
    models_before: NDArray[np.int32]
    # False when a biological assembly is defined and the atom's chain is not in it.
    in_assembly: NDArray[np.bool_]
    # Atom row -> its line, for rows with an unreadable coordinate field.
    bad_coord_lines: dict[int, str]
    biomat: NDArray[np.float32]
    nbiomat: int
    strict_input: bool


def _code_points(text: str, width: int) -> NDArray[np.uint32]:
    flat = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    return np.reshape(flat, (flat.shape[0] // width, width))


def read_pdb_records(pdb_path: str | Path, strict_input: bool = False) -> PdbRecords:
    """Read fixed-column atoms and the BIOMT records for biomolecule 1.

    Retain up to 500 assembly transforms. Strict mode rejects additional
    transforms. Unreadable coordinates become zero and are checked for
    retained atoms during strict classification.
    """
    path = Path(pdb_path)
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [_fixed_80_columns(line) for line in raw_lines]

    biomat = np.zeros((501, 3, 4), dtype=np.float32)
    biomat[:, 0, 0] = 1.0
    biomat[:, 1, 1] = 1.0
    biomat[:, 2, 2] = 1.0
    max_biomat = biomat.shape[0] - 1

    biochain: set[str] = set()
    nbiomat = 0
    active_biomat = 0
    models = 0

    records: list[str] = []
    descriptors: list[str] = []
    res_list: list[int] = []
    coord_list: list[tuple[float, float, float]] = []
    models_before: list[int] = []
    in_assembly: list[bool] = []
    bad_coord_lines: dict[int, str] = {}

    idx = 0
    while idx < len(lines):
        instring = lines[idx]

        if instring[:5] == "MODEL":
            models += 1

        if instring[11:25] == "BIOMOLECULE: 1":
            idx += 1
            while idx < len(lines):
                instring = lines[idx]

                # RCSB files close the assembly block with a blank REMARK line.
                # Any other record also closes it and is parsed normally below,
                # so atoms that directly follow a trimmed header are kept.
                if instring[:6] != "REMARK":
                    break

                if instring[:10] == "REMARK 350" and instring[34:40] == "CHAINS":
                    ich = 42
                    while ich < 80 and instring[ich] != " ":
                        biochain.add(instring[ich])
                        ich += 3

                if instring[13:19] == "BIOMT1":
                    if nbiomat >= max_biomat:
                        if strict_input:
                            raise ValueError(f"too many BIOMT transforms; maximum supported is {max_biomat}")
                        active_biomat = 0
                    else:
                        nbiomat += 1
                        active_biomat = nbiomat
                        values = _parse_free_floats_lenient(instring[19:80], strict_input=strict_input)
                        if len(values) >= 5:
                            biomat[active_biomat, 0, :4] = values[1:5]
                if instring[13:19] == "BIOMT2":
                    if active_biomat != 0:
                        values = _parse_free_floats_lenient(instring[19:80], strict_input=strict_input)
                        if len(values) >= 5:
                            biomat[active_biomat, 1, :4] = values[1:5]
                if instring[13:19] == "BIOMT3":
                    if active_biomat != 0:
                        values = _parse_free_floats_lenient(instring[19:80], strict_input=strict_input)
                        if len(values) >= 5:
                            biomat[active_biomat, 2, :4] = values[1:5]

                idx += 1
                if instring[13:19] == "      ":
                    break
            continue

        idx += 1
        if instring[:4] != "ATOM" and instring[:6] != "HETATM":
            continue

        res_list.append(_parse_fixed_int(instring[22:26], strict_input=strict_input, default=0))
        x = _lenient_float(instring[30:38])
        y = _lenient_float(instring[38:46])
        z = _lenient_float(instring[46:54])
        if x is None or y is None or z is None:
            # Only an atom a rule keeps may fail on this, so classify_atoms decides.
            bad_coord_lines[len(records)] = instring
        coord_list.append((x or 0.0, y or 0.0, z or 0.0))
        records.append(instring[:6])
        descriptors.append(instring[12:22])
        models_before.append(models)
        # Chains seen so far: an atom ahead of the assembly remarks is kept.
        in_assembly.append(not biochain or instring[21] in biochain)

    coord: NDArray[np.float32] = np.zeros((len(coord_list), 3), dtype=np.float32)
    if coord_list:
        coord[:] = coord_list
    return PdbRecords(
        record=_code_points("".join(records), 6),
        descriptor=_code_points("".join(descriptors), 10),
        res=np.asarray(res_list, dtype=np.int32),
        coord=coord,
        models_before=np.asarray(models_before, dtype=np.int32),
        in_assembly=np.asarray(in_assembly, dtype=np.bool_),
        bad_coord_lines=bad_coord_lines,
        biomat=biomat,
        nbiomat=nbiomat,
        strict_input=strict_input,
    )


def classify_atoms(records: PdbRecords, rules: list[SelectionRule]) -> AtomTable:
    """Keep atoms whose first matching rule has a nonzero radius.

    Also exclude chains outside the recorded assembly. In strict mode,
    unreadable coordinates raise ``ValueError`` only for retained atoms.
    """
    count = int(records.res.shape[0])
    # 0: no rule matched yet. A matched atom takes its rule's 1-based index.
    rule_of = np.zeros(count, dtype=np.int32)
    keep = np.zeros(count, dtype=np.bool_)
    for rule_index, rule in enumerate(rules, start=1):
        open_rows = rule_of == 0
        if not open_rows.any():
            break
        record_name = _code_points(f"{rule.record_name:<6}"[:6], 6)[0]
        pattern = _code_points(f"{rule.descriptor:-<10}"[:10], 10)[0]
        fixed = pattern != ord("-")
        matched = (
            open_rows
            & (records.record == record_name).all(axis=1)
            & (records.descriptor[:, fixed] == pattern[fixed]).all(axis=1)
            & (records.res >= rule.res_low)
            & (records.res <= rule.res_high)
        )
        rule_of[matched] = rule_index
        if rule.radius != 0.0:  # a zero radius hides the atoms it matches
            keep |= matched
    keep &= records.in_assembly
    kept = np.flatnonzero(keep)

    if records.strict_input:
        for row in kept.tolist():
            line = records.bad_coord_lines.get(row)
            if line is not None:
                for start in (30, 38, 46):
                    _parse_fixed_float(line[start : start + 8], strict_input=True)

    # A new subunit starts at every MODEL record and wherever the chain of
    # consecutive kept atoms changes.
    chain = records.descriptor[kept, 9]
    chain_starts = np.ones(kept.shape[0], dtype=np.int32)
    chain_starts[1:] = chain[1:] != chain[:-1]
    subunit = records.models_before[kept] + np.cumsum(chain_starts, dtype=np.int32)

    # Row 0 is the unused sentinel the 1-based renderer expects.
    n = int(kept.shape[0])
    coord = np.zeros((n + 1, 3), dtype=np.float32)
    coord[1:] = records.coord[kept]
    type_idx = np.zeros(n + 1, dtype=np.int32)
    type_idx[1:] = rule_of[kept]
    res = np.full(n + 1, 9999, dtype=np.int32)
    res[1:] = records.res[kept]
    su = np.full(n + 1, 9999, dtype=np.int32)
    su[1:] = subunit
    return AtomTable(
        coord=coord,
        type_idx=type_idx,
        res=res,
        su=su,
        biomat=np.array(records.biomat, dtype=np.float32),
        n=n,
        nbiomat=records.nbiomat,
    )


def read_and_classify_atoms(
    pdb_path: str | Path,
    rules: list[SelectionRule],
    strict_input: bool = False,
) -> AtomTable:
    """Read a fixed-column PDB and apply its assembly filter and ordered rules.

    Unreadable coordinates default to zero. With ``strict_input=True``,
    unreadable coordinates on retained atoms raise ``ValueError``.
    """
    return classify_atoms(read_pdb_records(pdb_path, strict_input=strict_input), rules)


def load_pdb(path: str | Path, rules: list[SelectionRule], strict_input: bool = False) -> AtomTable:
    """Read and classify a fixed-column PDB with ``read_and_classify_atoms``."""
    return read_and_classify_atoms(path, rules, strict_input=strict_input)
