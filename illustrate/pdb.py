from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from illustrate.types import AtomTable, SelectionRule



def _a80(line: str) -> str:
    return line.rstrip("\n\r")[:80].ljust(80)


def _parse_free_floats(text: str) -> list[float]:
    cleaned = text.replace(",", " ")
    return [float(tok) for tok in cleaned.split()]


def _parse_free_floats_safe(text: str, strict_input: bool) -> list[float]:
    try:
        return _parse_free_floats(text)
    except ValueError:
        if strict_input:
            raise
        return []


def _parse_fixed_int(text: str, strict_input: bool, default: int = 0) -> int:
    stripped = text.strip()
    if not stripped:
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
        return default
    try:
        return float(stripped)
    except ValueError:
        if strict_input:
            raise
        return default


def read_and_classify_atoms(
    pdb_path: str | Path,
    rules: list[SelectionRule],
    strict_input: bool = False,
) -> AtomTable:
    """Read PDB and classify atoms using fixed-column behavior from the original port."""
    path = Path(pdb_path)
    raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = [_a80(line) for line in raw_lines]

    from illustrate.types import AtomTable

    biomat = np.zeros((501, 3, 4), dtype=np.float32)
    biomat[:, 0, 0] = 1.0
    biomat[:, 1, 1] = 1.0
    biomat[:, 2, 2] = 1.0
    max_biomat = biomat.shape[0] - 1

    biochain: list[str] = [" "]
    nbiochain = 0
    nbiomat = 0
    active_biomat = 0

    coord_list: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    type_list: list[int] = [0]
    res_list: list[int] = [9999]
    su_list: list[int] = [9999]

    nsu = 0
    chain_last: str | None = None

    idx = 0
    while idx < len(lines):
        instring = lines[idx]

        if instring[:5] == "MODEL":
            nsu += 1

        if instring[11:25] == "BIOMOLECULE: 1":
            idx += 1
            while idx < len(lines):
                instring = lines[idx]

                if instring[:10] == "REMARK 350" and instring[34:40] == "CHAINS":
                    ich = 42
                    while ich < 80 and instring[ich] != " ":
                        nbiochain += 1
                        biochain.append(instring[ich])
                        ich += 3

                if instring[13:19] == "BIOMT1":
                    if nbiomat >= max_biomat:
                        if strict_input:
                            raise ValueError(f"too many BIOMT transforms; maximum supported is {max_biomat}")
                        active_biomat = 0
                    else:
                        nbiomat += 1
                        active_biomat = nbiomat
                        values = _parse_free_floats_safe(instring[19:80], strict_input=strict_input)
                        if len(values) >= 5:
                            biomat[active_biomat, 0, :4] = values[1:5]
                if instring[13:19] == "BIOMT2":
                    if active_biomat != 0:
                        values = _parse_free_floats_safe(instring[19:80], strict_input=strict_input)
                        if len(values) >= 5:
                            biomat[active_biomat, 1, :4] = values[1:5]
                if instring[13:19] == "BIOMT3":
                    if active_biomat != 0:
                        values = _parse_free_floats_safe(instring[19:80], strict_input=strict_input)
                        if len(values) >= 5:
                            biomat[active_biomat, 2, :4] = values[1:5]

                if instring[13:19] == "      ":
                    break
                idx += 1

            idx += 1
            continue

        if instring[:4] != "ATOM" and instring[:6] != "HETATM":
            idx += 1
            continue

        ires = _parse_fixed_int(instring[22:26], strict_input=strict_input, default=0)
        atom_saved = False
        skip_atom = False

        for rule_index, rule in enumerate(rules, start=1):
            rule_record_name = f"{rule.record_name:<6}"[:6]
            if instring[:6] != rule_record_name:
                continue

            rule_descriptor = f"{rule.descriptor:-<10}"[:10]
            descriptor_ok = True
            for ia in range(10):
                rule_char = rule_descriptor[ia]
                if rule_char == "-":
                    continue
                if instring[12 + ia] != rule_char:
                    descriptor_ok = False
                    break
            if not descriptor_ok:
                continue

            if ires < rule.res_low or ires > rule.res_high:
                continue

            if rule.radius == 0.0:
                skip_atom = True
                break

            if nbiochain != 0:
                ibioflag = 0
                for ich in range(1, nbiochain + 1):
                    if instring[21:22] == biochain[ich]:
                        ibioflag = 1
                        break
                if ibioflag == 0:
                    skip_atom = True
                    break

            x = _parse_fixed_float(instring[30:38], strict_input=strict_input, default=0.0)
            y = _parse_fixed_float(instring[38:46], strict_input=strict_input, default=0.0)
            z = _parse_fixed_float(instring[46:54], strict_input=strict_input, default=0.0)

            coord_list.append((x, y, z))
            type_list.append(rule_index)

            chain = instring[21:22]
            if chain_last is None or chain != chain_last:
                nsu += 1
                chain_last = chain
            su_list.append(nsu)
            res_list.append(ires)

            atom_saved = True
            break

        if skip_atom or atom_saved:
            idx += 1
            continue

        idx += 1

    coord = np.asarray(coord_list, dtype=np.float32)
    type_idx = np.asarray(type_list, dtype=np.int32)
    res = np.asarray(res_list, dtype=np.int32)
    su = np.asarray(su_list, dtype=np.int32)

    n = len(coord_list) - 1
    return AtomTable(
        coord=coord,
        type_idx=type_idx,
        res=res,
        su=su,
        biomat=biomat,
        n=n,
        nbiomat=nbiomat,
    )


def load_pdb(path: str | Path, rules: list[SelectionRule], strict_input: bool = False) -> AtomTable:
    return read_and_classify_atoms(path, rules, strict_input=strict_input)
