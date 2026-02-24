from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from illustrate.math3d import catenate, rotation_x, rotation_y, rotation_z
from illustrate.types import CommandProgram, OutlineParams, SelectionRule, WorldParams


class ParseError(ValueError):
    pass


@dataclass
class _LineReader:
    lines: list[str]
    index: int = 0

    def next_line(self) -> str:
        if self.index >= len(self.lines):
            raise EOFError
        raw = self.lines[self.index]
        self.index += 1
        line = raw.rstrip("\n\r")
        return line[:80].ljust(80)


def _split_numeric(text: str) -> list[float]:
    cleaned = text.replace(",", " ")
    values: list[float] = []
    for token in cleaned.split():
        values.append(float(token))
    return values


def _next_numeric(reader: _LineReader, expected_at_least: int, strict_input: bool, context: str) -> list[float]:
    line = reader.next_line()
    try:
        values = _split_numeric(line)
    except ValueError as exc:
        if strict_input:
            raise ParseError(f"invalid numeric line for {context}: {line!r}") from exc
        values = []
    if strict_input and len(values) < expected_at_least:
        raise ParseError(f"expected at least {expected_at_least} values for {context}, got {len(values)}")
    return values


def _parse_selection_rule(line: str, strict_input: bool) -> SelectionRule | None:
    record_name = line[:6]
    descriptor = line[6:16]
    try:
        fields = _split_numeric(line[17:80])
    except ValueError:
        if strict_input:
            raise ParseError(f"invalid selection rule payload: {line!r}")
        return None

    if len(fields) < 6:
        if strict_input:
            raise ParseError(f"selection rule must contain 6 numeric values: {line!r}")
        return None

    res_low = int(fields[0])
    res_high = int(fields[1])
    rr, rg, rb = (float(fields[2]), float(fields[3]), float(fields[4]))
    rad = float(fields[5])
    return SelectionRule(
        record_name=record_name,
        descriptor=descriptor,
        res_low=res_low,
        res_high=res_high,
        color=(rr, rg, rb),
        radius=rad,
    )


def parse_command_stream(text: str, strict_input: bool = False) -> CommandProgram:
    """Parse Illustrate command stream using Fortran card semantics."""
    reader = _LineReader(text.splitlines())
    program = CommandProgram()

    while True:
        try:
            command_line = reader.next_line()
        except EOFError:
            break

        command = command_line[:3]

        if command == "rea":
            try:
                pdb_line = reader.next_line()
            except EOFError as exc:
                raise ParseError("missing PDB filename after read card") from exc
            program.pdb_file = Path(pdb_line.strip())

            rules: list[SelectionRule] = []
            while True:
                try:
                    instring = reader.next_line()
                except EOFError as exc:
                    raise ParseError("unexpected EOF while parsing selection rules") from exc
                if instring[:3] == "END":
                    break
                rule = _parse_selection_rule(instring, strict_input=strict_input)
                if rule is not None:
                    rules.append(rule)
                elif strict_input:
                    raise ParseError(f"invalid selection rule line: {instring!r}")
                else:
                    program.warnings.append(f"skipping invalid selection rule line: {instring.rstrip()}")
            program.selection_rules = rules
            continue

        if command == "tra":
            vals = _next_numeric(reader, expected_at_least=3, strict_input=strict_input, context="translate")
            if len(vals) >= 3:
                program.transform.xtran += float(vals[0])
                program.transform.ytran += float(vals[1])
                program.transform.ztran += float(vals[2])
            continue

        if command == "xro":
            vals = _next_numeric(reader, expected_at_least=1, strict_input=strict_input, context="xrot")
            if vals:
                matrixin = rotation_x(float(vals[0]))
                program.transform.rm = catenate(program.transform.rm, matrixin)
            continue

        if command == "yro":
            vals = _next_numeric(reader, expected_at_least=1, strict_input=strict_input, context="yrot")
            if vals:
                matrixin = rotation_y(float(vals[0]))
                program.transform.rm = catenate(program.transform.rm, matrixin)
            continue

        if command == "zro":
            vals = _next_numeric(reader, expected_at_least=1, strict_input=strict_input, context="zrot")
            if vals:
                matrixin = rotation_z(float(vals[0]))
                program.transform.rm = catenate(program.transform.rm, matrixin)
            continue

        if command == "sca":
            vals = _next_numeric(reader, expected_at_least=1, strict_input=strict_input, context="scale")
            if vals:
                program.transform.scale *= float(vals[0])
            continue

        if command == "cen":
            try:
                cent_line = reader.next_line()
            except EOFError as exc:
                raise ParseError("missing centering mode after center card") from exc
            cent = cent_line[:3]
            program.transform.autocenter = 0
            if cent == "aut":
                program.transform.autocenter = 1
            if cent == "cen":
                program.transform.autocenter = 2
            continue

        if command == "wor":
            world = WorldParams()
            vals1 = _next_numeric(reader, expected_at_least=8, strict_input=strict_input, context="world colors")
            if len(vals1) >= 8:
                world.background = (float(vals1[0]), float(vals1[1]), float(vals1[2]))
                world.fog_color = (float(vals1[3]), float(vals1[4]), float(vals1[5]))
                world.fog_front = float(vals1[6])
                world.fog_back = float(vals1[7])

            vals2 = _next_numeric(reader, expected_at_least=5, strict_input=strict_input, context="world shadows")
            if len(vals2) >= 5:
                world.shadows = bool(int(vals2[0]))
                world.shadow_strength = float(vals2[1])
                world.shadow_angle = float(vals2[2])
                world.shadow_min_z = float(vals2[3])
                world.shadow_max_dark = float(vals2[4])

            vals3 = _next_numeric(reader, expected_at_least=2, strict_input=strict_input, context="world size")
            if len(vals3) >= 2:
                world.width = int(vals3[0])
                world.height = int(vals3[1])
            program.world = world
            continue

        if command == "ill":
            ol = OutlineParams(enabled=True)
            vals1 = _next_numeric(reader, expected_at_least=5, strict_input=strict_input, context="illustrate contour")
            if len(vals1) >= 5:
                ol.contour_low = float(vals1[0])
                ol.contour_high = float(vals1[1])
                ol.kernel = int(vals1[2])
                ol.z_diff_min = float(vals1[3])
                ol.z_diff_max = float(vals1[4])

            vals2 = _next_numeric(reader, expected_at_least=2, strict_input=strict_input, context="illustrate subunit")
            if len(vals2) >= 2:
                ol.subunit_low = float(vals2[0])
                ol.subunit_high = float(vals2[1])

            vals3 = _next_numeric(reader, expected_at_least=3, strict_input=strict_input, context="illustrate residue")
            if len(vals3) >= 3:
                ol.residue_low = float(vals3[0])
                ol.residue_high = float(vals3[1])
                ol.residue_diff = float(vals3[2])

            program.outlines = ol
            continue

        if command == "cal":
            try:
                output_line = reader.next_line()
            except EOFError as exc:
                raise ParseError("missing output filename after calculate card") from exc
            program.output_file = Path(output_line.strip())
            break

        if strict_input:
            raise ParseError(f"unknown command card: {command!r}")
        program.warnings.append(f"unknown command card skipped: {command!r}")

    if strict_input:
        if program.pdb_file is None:
            raise ParseError("missing read card")
        if program.output_file is None:
            raise ParseError("missing calculate card")

    return program


def parse_command_file(text: str, strict_input: bool = False) -> CommandProgram:
    return parse_command_stream(Path(text).read_text(encoding="utf-8", errors="ignore"), strict_input=strict_input)
