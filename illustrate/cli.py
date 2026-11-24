from __future__ import annotations

import copy
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

from illustrate.core.types import AtomTable, RenderParams, RenderResult
from illustrate.formats.command_stream import parse_command_stream
from illustrate.formats.pdb import read_and_classify_atoms
from illustrate.io.image_export import export_png, export_svg
from illustrate.io.params_json import params_from_json, params_to_json
from illustrate.io.rcsb import fetch_pdb
from illustrate.presets import PRESET_NAMES, render_params_from_preset
from illustrate.rendering.api import (
    estimate_render_size,
    max_render_scale,
    render,
    render_from_atoms,
    resolve_command_pdb_path,
)
from illustrate.rendering.pipeline import MAX_IMAGE_DIMENSION

# Blank margin, in pixels, that an auto-sized image keeps around the structure.
_AUTO_PAD = 30
_MIN_IMAGE_DIMENSION = 100
_PDB_ID = re.compile(r"[A-Za-z0-9]{4}")
_ROTATION = re.compile(r"([xyzXYZ])=([-+]?(?:\d+\.?\d*|\.\d+))")
_FORMATS = (".png", ".svg")

app = typer.Typer(
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode=None,
)


class _CliError(Exception):
    """A problem to report to the user without a traceback."""


def _show_version(value: bool) -> None:
    if value:
        typer.echo(metadata.version("illustrate-py"))
        raise typer.Exit()


def _show_presets(value: bool) -> None:
    if value:
        for name in PRESET_NAMES:
            typer.echo(name)
        raise typer.Exit()


def _preset_name(name: str) -> str:
    for known in PRESET_NAMES:
        if known.casefold() == name.strip().casefold():
            return known
    raise _CliError(f"unknown preset {name!r}; choose one of: {', '.join(PRESET_NAMES)}")


def _parse_rotations(values: list[str]) -> list[tuple[str, float]]:
    rotations: list[tuple[str, float]] = []
    for value in values:
        for part in value.split(","):
            match = _ROTATION.fullmatch(part.strip())
            if match is None:
                raise _CliError(f"cannot read rotation {part!r}; write it as x=30, y=-45 or z=90")
            rotations.append((match.group(1).lower(), float(match.group(2))))
    return rotations


def _structure_path(structure: str) -> Path:
    """Use an existing local file before treating the argument as an RCSB ID."""
    path = Path(structure).expanduser()
    if path.is_file():
        return path
    if _PDB_ID.fullmatch(structure) is None:
        raise _CliError(f"{structure}: no such file, and not a four-character PDB ID")
    try:
        return fetch_pdb(structure)
    except (FileNotFoundError, ConnectionError) as exc:
        raise _CliError(f"{structure}: {exc}") from exc


def _fit_to_size(
    atoms: AtomTable, params: RenderParams, width: int | None, height: int | None
) -> None:
    """Set the scale for the supplied pixel bounds, allowing for auto-frame padding."""
    params.world.width = params.world.height = 0
    limit = max_render_scale(params.rules)
    params.transform.scale = min(params.transform.scale, limit)
    frame = 2 * _AUTO_PAD
    # Recompute after resizing because the renderer rounds dimensions to even pixels.
    for _ in range(3):
        bare_width, bare_height = estimate_render_size(atoms, params)
        if bare_width <= 0 or bare_height <= 0:
            raise _CliError("the selection rules leave no atoms to fit")
        factors = [
            (target - frame) / bare
            for target, bare in ((width, bare_width), (height, bare_height))
            if target is not None
        ]
        params.transform.scale = min(params.transform.scale * min(factors), limit)
        if params.transform.scale >= limit:
            break
    params.world.width = params.world.height = -_AUTO_PAD


def _write_image(path: Path, result: RenderResult, *, transparent: bool) -> None:
    opacity = result.opacity if transparent else None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".svg":
            export_svg(path, result.rgb, opacity)
        else:
            export_png(path, result.rgb, opacity)
    except OSError as exc:
        raise _CliError(f"cannot write {path}: {exc}") from exc


def _image_path(path: Path) -> Path:
    if path.suffix == "":
        return path.with_suffix(".png")
    if path.suffix.lower() not in _FORMATS:
        raise _CliError(f"{path}: output must be a .png or .svg file")
    return path


def _render_command_file(
    command_file: Path, output: Path | None, *, backend: str | None, strict: bool, transparent: bool
) -> None:
    """Render an original Illustrate command file (``-`` reads it from standard input)."""
    from_stdin = str(command_file) == "-"
    try:
        text = sys.stdin.read() if from_stdin else command_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _CliError(f"cannot read command file {command_file}: {exc}") from exc
    try:
        program = parse_command_stream(text, strict_input=strict)
        if program.pdb_file is None:
            raise _CliError("the command file has no read card naming a PDB file")
        target = output if output is not None else program.output_file
        if target is None:
            raise _CliError("the command file has no calculate card; name the image with --output")
        pdb_path = resolve_command_pdb_path(
            program.pdb_file, base_dir=None if from_stdin else command_file.parent
        )
        if not pdb_path.is_file():
            raise _CliError(f"{pdb_path}: PDB file named by the command file was not found")
        # The original program wrote PNM; this renderer writes PNG in its place.
        if output is None:
            target = target.with_suffix(".png")
        target = _image_path(target)
        atoms = read_and_classify_atoms(pdb_path, program.selection_rules, strict_input=strict)
        result = render(program, atoms, backend=backend)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        raise _CliError(f"{command_file}: {exc}") from exc
    for warning in program.warnings:
        typer.echo(f"warning: {warning}", err=True)
    _write_image(target, result, transparent=transparent)
    typer.echo(f"{target}  {result.width}x{result.height}")


@app.command(
    epilog="STRUCTURES are PDB files or four-character PDB IDs, which are downloaded from RCSB."
)
def render_structures(
    structures: Annotated[
        list[str] | None,
        typer.Argument(metavar="STRUCTURES...", show_default=False, help="PDB files or PDB IDs."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Image to write (.png or .svg), for a single structure."),
    ] = None,
    outdir: Annotated[
        Path | None, typer.Option("--outdir", help="Directory for the images. Default: here.")
    ] = None,
    image_format: Annotated[
        str, typer.Option("--format", help="Image format when --output is not given: png or svg.")
    ] = "png",
    preset: Annotated[
        str | None, typer.Option("--preset", help="Built-in look; see --list-presets.")
    ] = None,
    settings: Annotated[
        Path | None,
        typer.Option("--settings", help="Settings JSON saved from the GUI or by --save-settings."),
    ] = None,
    rotate: Annotated[
        list[str] | None,
        typer.Option(
            "--rotate",
            help="Rotations in degrees, such as x=30,y=-45. The last one listed turns the structure first. May be repeated.",
        ),
    ] = None,
    scale: Annotated[
        float | None, typer.Option("--scale", min=0.01, help="Pixels per ångström.")
    ] = None,
    width: Annotated[
        int | None,
        typer.Option(
            "--width",
            min=_MIN_IMAGE_DIMENSION,
            max=MAX_IMAGE_DIMENSION,
            help="Scale the structure so the image is this many pixels wide.",
        ),
    ] = None,
    height: Annotated[
        int | None,
        typer.Option(
            "--height",
            min=_MIN_IMAGE_DIMENSION,
            max=MAX_IMAGE_DIMENSION,
            help="Scale the structure so the image is this many pixels tall.",
        ),
    ] = None,
    transparent: Annotated[
        bool, typer.Option("--transparent", help="Make the background transparent.")
    ] = False,
    save_settings: Annotated[
        bool,
        typer.Option("--save-settings", help="Also write the settings used, as a .json beside each image."),
    ] = False,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Renderer backend: numba, numpy or mlx. Default: best available."),
    ] = None,
    command_file: Annotated[
        Path | None,
        typer.Option(
            "--command-file",
            help="Render an original Illustrate command file instead (- for standard input).",
        ),
    ] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Reject malformed PDB or command-file lines.")
    ] = False,
    list_presets: Annotated[
        bool,
        typer.Option("--list-presets", callback=_show_presets, is_eager=True, help="List the presets and exit."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Print the version and exit."),
    ] = False,
) -> None:
    """Render PDB structures as Illustrate-style figures."""
    del list_presets, version
    try:
        failures = _render_all(
            structures or [],
            output=output,
            outdir=outdir,
            image_format=image_format,
            preset=preset,
            settings=settings,
            rotate=rotate or [],
            scale=scale,
            width=width,
            height=height,
            transparent=transparent,
            save_settings=save_settings,
            backend=backend,
            command_file=command_file,
            strict=strict,
        )
    except _CliError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    if failures:
        raise typer.Exit(1)


def _render_all(
    structures: list[str],
    *,
    output: Path | None,
    outdir: Path | None,
    image_format: str,
    preset: str | None,
    settings: Path | None,
    rotate: list[str],
    scale: float | None,
    width: int | None,
    height: int | None,
    transparent: bool,
    save_settings: bool,
    backend: str | None,
    command_file: Path | None,
    strict: bool,
) -> int:
    """Render the requested inputs and count failures in a multi-structure batch.

    Invalid requests and single-input failures are raised to the CLI handler.
    """
    fit = width is not None or height is not None
    if command_file is not None:
        look = structures or preset or settings or rotate or scale is not None or fit
        if look or outdir is not None or save_settings:
            raise _CliError(
                "--command-file takes its structure and look from the file; "
                "only --output, --transparent, --backend and --strict apply"
            )
        _render_command_file(
            command_file, output, backend=backend, strict=strict, transparent=transparent
        )
        return 0

    if preset is not None and settings is not None:
        raise _CliError("use either --preset or --settings, not both")
    if scale is not None and fit:
        raise _CliError("use either --scale or --width/--height, not both")
    suffix = f".{image_format.lower().lstrip('.')}"
    if suffix not in _FORMATS:
        raise _CliError(f"--format must be png or svg, not {image_format!r}")
    rotations = _parse_rotations(rotate)

    if settings is not None:
        try:
            base = params_from_json(settings.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError) as exc:
            raise _CliError(f"cannot read settings {settings}: {exc}") from exc
        if not structures and base.pdb_path:
            structures = [base.pdb_path]
    else:
        base = render_params_from_preset(_preset_name(preset or "Default"), "")
    if not structures:
        raise _CliError("name at least one PDB file or PDB ID (see --help)")

    if output is not None:
        if len(structures) > 1 or outdir is not None:
            raise _CliError("--output names one image; use --outdir for several structures")
        targets = [_image_path(output)]
    else:
        targets = [(outdir or Path()) / f"{Path(item).stem}{suffix}" for item in structures]
        if len(set(targets)) < len(targets):
            raise _CliError("two structures would write the same image; render them separately")

    failures = 0
    for structure, target in zip(structures, targets):
        try:
            path = _structure_path(structure)
            params = copy.deepcopy(base)
            params.pdb_path = str(path)
            params.transform.rotations.extend(rotations)
            if scale is not None:
                params.transform.scale = scale
            try:
                atoms = read_and_classify_atoms(path, params.rules, strict_input=strict)
                if fit:
                    _fit_to_size(atoms, params, width, height)
                result = render_from_atoms(atoms, params, backend=backend)
            except (RuntimeError, TypeError, ValueError, OSError) as exc:
                raise _CliError(f"{structure}: {exc}") from exc
            _write_image(target, result, transparent=transparent)
            if save_settings:
                target.with_suffix(".json").write_text(params_to_json(params), encoding="utf-8")
        except _CliError as exc:
            if len(structures) == 1:
                raise
            typer.echo(f"error: {exc}", err=True)
            failures += 1
            continue
        typer.echo(f"{target}  {result.width}x{result.height}")
    return failures


def main(argv: list[str] | None = None) -> int:
    try:
        app(args=argv, prog_name="illustrate-py")
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else int(exc.code is not None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
