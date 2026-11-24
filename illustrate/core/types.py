from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class SelectionRule:
    """Atom style for a fixed-column PDB pattern and inclusive residue range.

    The descriptor covers PDB columns 13–22, with a hyphen matching any
    character in its column. The first match supplies RGB values and a radius
    in ångströms. A zero radius hides the atom.
    """

    record_name: str
    descriptor: str
    res_low: int
    res_high: int
    color: tuple[float, float, float]
    radius: float


@dataclass(slots=True)
class WorldParams:
    """Canvas, background, fog, and shadow settings for a render.

    Two positive dimensions fix the canvas size in pixels, capped at 3000 and
    rounded down to even values. With auto-centering and at least one atom,
    a nonpositive dimension triggers autosizing on both axes. Zero fits the
    structure bounds. A negative value adds padding per edge and a positive
    value subtracts it in autosizing mode.
    """

    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fog_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fog_front: float = 1.0
    fog_back: float = 1.0
    shadows: bool = False
    shadow_strength: float = 0.0023
    shadow_angle: float = 2.0
    shadow_min_z: float = 1.0
    shadow_max_dark: float = 0.2
    width: int = 0
    height: int = 0


@dataclass(slots=True)
class OutlineParams:
    """Thresholds for depth, subunit, and residue outlines.

    When enabled, kernels 1 and 2 use a discrete Laplacian of depth for contours.
    Kernels 3 and 4 use neighboring depth differences.
    """

    enabled: bool = False
    contour_low: float = 1.0
    contour_high: float = 10.0
    kernel: int = 4
    z_diff_min: float = 1.0
    z_diff_max: float = 50.0
    subunit_low: float = 3.0
    subunit_high: float = 10.0
    residue_low: float = 3.0
    residue_high: float = 8.0
    residue_diff: float = 6000.0


@dataclass(slots=True)
class TransformState:
    scale: float = 1.0
    xtran: float = 0.0
    ytran: float = 0.0
    ztran: float = 0.0
    autocenter: int = 0
    rm: NDArray[np.float32] = field(
        default_factory=lambda: np.eye(4, dtype=np.float32),
    )


@dataclass(slots=True)
class CommandProgram:
    pdb_file: Path | None = None
    selection_rules: list[SelectionRule] = field(default_factory=list[SelectionRule])
    transform: TransformState = field(default_factory=TransformState)
    world: WorldParams = field(default_factory=WorldParams)
    outlines: OutlineParams = field(default_factory=OutlineParams)
    output_file: Path | None = None
    warnings: list[str] = field(default_factory=list[str])


@dataclass(slots=True)
class AtomTable:
    """Classified coordinates and biological assembly transforms for rendering.

    ``coord`` holds XYZ coordinates in ångströms, with shape ``(n + 1, 3)``.
    Rows 1 through ``n`` align with ``type_idx``, ``res``, and ``su``. Row 0
    is reserved, and ``type_idx`` holds 1-based indices into the original rule
    list. ``biomat`` stores 3×4 assembly transforms at indices 1 through
    ``nbiomat``. A count of zero renders one copy in the source coordinates.
    """

    coord: NDArray[np.float32]
    type_idx: NDArray[np.int32]
    res: NDArray[np.int32]
    su: NDArray[np.int32]
    biomat: NDArray[np.float32]
    n: int
    nbiomat: int


@dataclass(slots=True)
class RenderResult:
    """Rendered uint8 RGB pixels and their separate uint8 opacity plane.

    RGB has shape ``(height, width, 3)`` and opacity has shape
    ``(height, width)``. Rows follow the renderer's X axis and columns follow
    its Y axis. Opacity ranges from 0 (transparent) to 255 (opaque).
    """

    rgb: NDArray[np.uint8]
    opacity: NDArray[np.uint8]
    width: int
    height: int


@dataclass(slots=True)
class Transform:
    """View transform with scale in pixels per ångström.

    Translation is in view coordinates and in ångströms, before scaling.
    Rotations pair an axis name with an angle in degrees. The last listed
    rotation acts first. Both ``auto`` and ``center`` center the structure in
    X and Y. In Z, ``auto`` places the front near the origin and ``center``
    centers the depth bounds. ``none`` skips automatic translation.
    """

    scale: float = 12.0
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotations: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])
    autocenter: str = "auto"


@dataclass(slots=True)
class RenderParams:
    """Everything one render needs: a PDB path, ordered selection rules, and view settings.

    ``transform`` places the structure, ``world`` sets the canvas, fog and
    shadows, and ``outlines`` sets the ink lines.
    """

    pdb_path: str
    rules: list[SelectionRule]
    transform: Transform = field(default_factory=Transform)
    world: WorldParams = field(default_factory=WorldParams)
    outlines: OutlineParams = field(default_factory=OutlineParams)
