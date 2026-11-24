from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from illustrate import render_from_atoms
from illustrate.formats.command_stream import parse_command_stream
from illustrate.formats.pdb import read_and_classify_atoms
from illustrate.io.preset_store import builtin_preset_items
from illustrate.rendering.api import render_program as render
from illustrate.rendering.kernels import backend_available, supported_backends


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

    assert (result.width, result.height) == (50, 50)
    assert result.rgb.shape == (50, 50, 3)
    assert result.opacity.shape == (50, 50)


@pytest.mark.parametrize("backend", [name for name in supported_backends() if backend_available(name)])
def test_outlines_leave_the_image_frame_empty(tmp_path: Path, backend: str) -> None:
    # Off-image pixels once counted as a different subunit, so every outlined
    # render carried a faint rectangle one pixel inside its frame.
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = next(preset for _, preset in builtin_preset_items() if preset.outlines.enabled)
    params.pdb_path = str(pdb_path)
    params.world.width = 40
    params.world.height = 40
    params.transform.scale = 4.0
    atoms = read_and_classify_atoms(pdb_path, params.rules)

    result = render_from_atoms(atoms, params, backend=backend)

    opacity = np.asarray(result.opacity)
    assert opacity[18:22, 18:22].max() > 0
    frame = np.ones(opacity.shape, dtype=bool)
    frame[3:-3, 3:-3] = False
    assert opacity[frame].max() == 0


def test_scale_too_high_for_atom_radii_names_the_largest_usable_scale(tmp_path: Path) -> None:
    _write_minimal_pdb(tmp_path / "mini.pdb")
    command_stream = "\n".join(
        [
            "read",
            "mini.pdb",
            "ATOM  ---------- 0,9999, 1.0,1.0,1.0, 2.0",
            "END",
            "center",
            "cen",
            "scale",
            "60.0",
            "wor",
            "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0",
            "0,0.0023,2.0,1.0,0.2",
            "-30,-30",
            "calculate",
            "out.pnm",
        ],
    )
    program = parse_command_stream(command_stream)
    assert program.pdb_file is not None
    atoms = read_and_classify_atoms(tmp_path / program.pdb_file, program.selection_rules)

    # 2 Å atoms at scale 60 are 120 px in radius; the limit of 100 px allows scale 50.
    with pytest.raises(ValueError, match=r"Scale is too high .* at most 50\.00"):
        render(program, atoms)
