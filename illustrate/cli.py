from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from illustrate.io import write_optional_png
from illustrate.parser import ParseError, parse_command_stream
from illustrate.pdb import read_and_classify_atoms
from illustrate.render import render


_NETPBM_SUFFIXES = {".pnm", ".ppm", ".pgm", ".pbm"}


def _normalize_png_output_path(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix in _NETPBM_SUFFIXES:
        return path.with_suffix(".png")
    if suffix == "":
        return path.with_suffix(".png")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="illustrate-py",
        description="Python port of Illustrate Fortran renderer",
    )
    parser.add_argument(
        "--strict-input",
        action="store_true",
        help="enable stricter validation for command parsing and PDB parsing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stream_text = sys.stdin.read()
    if not stream_text.strip():
        print("no command stream on stdin", file=sys.stderr)
        return 2

    try:
        program = parse_command_stream(stream_text, strict_input=args.strict_input)
    except ParseError as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        return 2

    for warning in program.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if program.pdb_file is None:
        print("missing read command / PDB filename", file=sys.stderr)
        return 2
    if program.output_file is None:
        print("missing calculate command / output filename", file=sys.stderr)
        return 2

    try:
        atoms = read_and_classify_atoms(
            program.pdb_file,
            program.selection_rules,
            strict_input=args.strict_input,
        )
    except Exception as exc:
        print(f"PDB read/classification error: {exc}", file=sys.stderr)
        return 2

    try:
        result = render(program, atoms)
    except Exception as exc:
        print(f"render error: {exc}", file=sys.stderr)
        return 1

    output_path = _normalize_png_output_path(Path(program.output_file))
    if output_path != Path(program.output_file):
        print(
            f"note: output filename normalized to PNG: {output_path.name}",
            file=sys.stderr,
        )

    write_optional_png(output_path, result.rgb)

    opacity_rgb = np.repeat(result.opacity[:, :, np.newaxis], 3, axis=2)
    write_optional_png(output_path.with_name("opacity.png"), opacity_rgb)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
