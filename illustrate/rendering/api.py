from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
import os
import platform
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
from numpy.typing import NDArray

from illustrate.core.geometry import catenate, rotation_x, rotation_y, rotation_z
from illustrate.rendering.kernels import backend_available, backend_importable, supported_backends
from illustrate.rendering.pipeline import (
    MAX_SCALED_ATOM_RADIUS,
    RenderCancellationToken,
    estimate_program_size,
    render_program as _render_pipeline_program,
)
from illustrate.core.types import (
    AtomTable,
    CommandProgram,
    OutlineParams,
    RenderParams,
    RenderResult,
    SelectionRule,
    Transform,
    TransformState,
    WorldParams,
)

_SPHERE_CACHE_RESOLUTION = 1000.0
_SPHERE_CACHE_MAX_ENTRIES = 256
_SPHERE_CACHE: OrderedDict[int, NDArray[np.float32]] = OrderedDict()
_SPHERE_CACHE_LOCK = threading.Lock()
_MLX_RENDER_LOCK = threading.RLock()

_BACKEND_LABELS = {
    "numpy": "NumPy",
    "numba": "Numba",
    "mlx": "MLX",
}


def host_backend_candidates(
    *,
    platform_name: str | None = None,
    machine_name: str | None = None,
) -> tuple[str, ...]:
    platform_value = sys.platform if platform_name is None else str(platform_name)
    machine_value = platform.machine() if machine_name is None else str(machine_name)

    normalized_platform = platform_value.strip().lower()
    normalized_machine = machine_value.strip().lower()

    # Compiled CPU kernels come before NumPy, which stays as the fallback.
    if normalized_platform == "darwin" and normalized_machine in {"arm64", "aarch64"}:
        return ("mlx", "numba", "numpy")
    return ("numba", "numpy")


def _default_render_backend() -> str:
    supported = tuple(supported_backends())
    candidates = host_backend_candidates()
    for candidate in candidates:
        # Confirm the import before choosing, so a broken install falls back.
        if candidate in supported and backend_available(candidate) and backend_importable(candidate):
            return candidate

    candidate_text = ", ".join(candidates)
    host_text = f"{sys.platform}/{platform.machine()}"
    raise RuntimeError(
        f"No available render backend found for {host_text}. Tried: {candidate_text}"
    )


def resolve_render_backend(backend: str | None = None) -> str:
    """Choose a backend from the argument, then the environment, then the host.

    The environment override is ``ILLUSTRATE_RENDER_BACKEND``. Unsupported
    names raise ``ValueError``, and unavailable requested backends raise
    ``RuntimeError``.
    """
    supported = tuple(supported_backends())

    explicit_raw: str | None = None
    if backend is not None:
        explicit_raw = str(backend)
    else:
        env_raw = os.environ.get("ILLUSTRATE_RENDER_BACKEND")
        if env_raw is not None and env_raw.strip() != "":
            explicit_raw = env_raw

    if explicit_raw is not None:
        normalized = explicit_raw.strip().lower()
        if normalized not in supported:
            raise ValueError(f"Unsupported render backend: {explicit_raw!r}")
        if not backend_available(normalized):
            label = _BACKEND_LABELS.get(normalized, normalized)
            raise RuntimeError(f"{label} backend requested but {label} is unavailable")
        return normalized

    return _default_render_backend()


def max_render_scale(rules: Sequence[SelectionRule]) -> float:
    """Return the scale limit in pixels per ångström based on the largest rule radius.

    Even rules that match no atoms count toward the limit. With no positive
    radii, return infinity.
    """
    largest_radius = max((float(rule.radius) for rule in rules), default=0.0)
    if largest_radius <= 0.0:
        return float("inf")
    return MAX_SCALED_ATOM_RADIUS / largest_radius


def precompute_sphere(scaled_radius: float) -> NDArray[np.float32]:
    """Return read-only ``(sample, XYZ)`` offsets for the positive-Z hemisphere.

    Offsets are in pixels, with X and Y sampled on an integer grid.
    Radii that round to the same thousandth of a pixel share a cache entry.
    """
    irlim = int(scaled_radius)
    if irlim > MAX_SCALED_ATOM_RADIUS:
        raise ValueError(
            f"Scaled atom radius {scaled_radius:.0f} px exceeds the "
            f"{MAX_SCALED_ATOM_RADIUS} px limit; lower the scale or the atom radii."
        )
    cache_key = int(round(float(scaled_radius) * _SPHERE_CACHE_RESOLUTION))
    with _SPHERE_CACHE_LOCK:
        cached = _SPHERE_CACHE.get(cache_key)
        if cached is not None:
            _SPHERE_CACHE.move_to_end(cache_key)
            return cached

    axis = np.arange(-irlim - 1, irlim + 2, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    radius2 = np.float32(float(scaled_radius) * float(scaled_radius))
    dist2 = xx * xx + yy * yy
    mask = dist2 <= radius2
    if np.any(mask):
        zz = np.sqrt(np.maximum(np.subtract(radius2, dist2[mask]), np.float32(0.0))).astype(
            np.float32, copy=False
        )
        sphere = np.column_stack((xx[mask], yy[mask], zz)).astype(
            np.float32, copy=False
        )
    else:
        sphere = np.zeros((0, 3), dtype=np.float32)
    sphere.setflags(write=False)

    with _SPHERE_CACHE_LOCK:
        cached = _SPHERE_CACHE.get(cache_key)
        if cached is not None:
            _SPHERE_CACHE.move_to_end(cache_key)
            return cached

        _SPHERE_CACHE[cache_key] = sphere
        _SPHERE_CACHE.move_to_end(cache_key)
        while len(_SPHERE_CACHE) > _SPHERE_CACHE_MAX_ENTRIES:
            _SPHERE_CACHE.popitem(last=False)
    return sphere


def render_program(
    program: CommandProgram,
    atoms: AtomTable,
    *,
    backend: str = "numpy",
    cancellation_token: RenderCancellationToken | None = None,
) -> RenderResult:
    backend_name = resolve_render_backend(backend)
    # Concurrent render API calls can share MLX's default stream. Guard
    # evaluation through readback so they cannot encode into it concurrently.
    with _MLX_RENDER_LOCK if backend_name == "mlx" else nullcontext():
        return _render_pipeline_program(
            program,
            atoms,
            backend_name=backend_name,
            sphere_lookup=precompute_sphere,
            cancellation_token=cancellation_token,
        )


def prepare_numba_backend(cancellation_token: RenderCancellationToken | None = None) -> None:
    """Warm the Numba raster, outlines, shadows, and byte output for a desktop session."""
    program = CommandProgram(
        selection_rules=[SelectionRule("ATOM  ", "----------", 0, 1, (1.0, 1.0, 1.0), 1.5)],
        transform=TransformState(scale=3.0),
        world=WorldParams(width=24, height=24, shadows=True),
        outlines=OutlineParams(enabled=True),
    )
    indices = np.array([0, 1], dtype=np.int32)
    atoms = AtomTable(
        coord=np.array([[0, 0, 0], [0, 0, -5]], dtype=np.float32),
        type_idx=indices,
        res=indices,
        su=indices,
        biomat=np.zeros((1, 3, 4), dtype=np.float32),
        n=1,
        nbiomat=0,
    )
    render_program(program, atoms, backend="numba", cancellation_token=cancellation_token)


def rotation_matrix(rotations: list[tuple[str, float]]) -> NDArray[np.float32]:
    """Compose X/Y/Z rotations in degrees for XYZ row vectors.

    The last listed rotation acts first on the coordinates.
    """
    matrix = np.eye(4, dtype=np.float32)
    for axis, angle in rotations:
        axis_code = axis.lower()
        if axis_code == "x":
            next_matrix = rotation_x(float(angle))
        elif axis_code == "y":
            next_matrix = rotation_y(float(angle))
        elif axis_code == "z":
            next_matrix = rotation_z(float(angle))
        else:
            raise ValueError(f"unsupported rotation axis: {axis!r}")
        matrix = catenate(matrix, next_matrix)
    return matrix.astype(np.float32)


def autocenter_mode_code(code: str) -> int:
    value = code.strip().lower()
    if value in {"auto", "aut"}:
        return 1
    if value in {"center", "cen"}:
        return 2
    if value in {"none", "off", "0"}:
        return 0
    raise ValueError(f"unsupported autocenter mode: {code!r}")


def _clamp_color(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) < 3:
        raise ValueError("color must contain at least 3 components")
    return (
        float(min(max(value[0], 0.0), 1.0)),
        float(min(max(value[1], 0.0), 1.0)),
        float(min(max(value[2], 0.0), 1.0)),
    )


def _translation_xyz(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) < 3:
        raise ValueError("transform translate must contain at least 3 components")
    return (float(value[0]), float(value[1]), float(value[2]))


def _normalize_rule(rule: SelectionRule) -> SelectionRule:
    return SelectionRule(
        record_name=f"{rule.record_name:<6}"[:6],
        descriptor=f"{rule.descriptor:-<10}"[:10],
        res_low=int(rule.res_low),
        res_high=int(rule.res_high),
        color=_clamp_color(rule.color),
        radius=float(rule.radius),
    )


def _transform_state(transform: Transform) -> TransformState:
    xtran, ytran, ztran = _translation_xyz(transform.translate)
    return TransformState(
        scale=float(transform.scale),
        xtran=xtran,
        ytran=ytran,
        ztran=ztran,
        autocenter=autocenter_mode_code(transform.autocenter),
        rm=rotation_matrix(transform.rotations),
    )


def program_from_params(params: RenderParams) -> CommandProgram:
    """Convert Python settings to a program with padded rules and a rotation matrix."""
    normalized_rules = [_normalize_rule(rule) for rule in params.rules]
    return CommandProgram(
        pdb_file=Path(params.pdb_path),
        selection_rules=normalized_rules,
        transform=_transform_state(params.transform),
        world=WorldParams(
            background=_clamp_color(params.world.background),
            fog_color=_clamp_color(params.world.fog_color),
            fog_front=float(min(max(params.world.fog_front, 0.0), 1.0)),
            fog_back=float(min(max(params.world.fog_back, 0.0), 1.0)),
            shadows=params.world.shadows,
            shadow_strength=float(params.world.shadow_strength),
            shadow_angle=float(params.world.shadow_angle),
            shadow_min_z=float(params.world.shadow_min_z),
            shadow_max_dark=float(params.world.shadow_max_dark),
            width=int(params.world.width),
            height=int(params.world.height),
        ),
        outlines=OutlineParams(
            enabled=bool(params.outlines.enabled),
            contour_low=float(params.outlines.contour_low),
            contour_high=float(params.outlines.contour_high),
            kernel=int(params.outlines.kernel),
            z_diff_min=float(params.outlines.z_diff_min),
            z_diff_max=float(params.outlines.z_diff_max),
            subunit_low=float(params.outlines.subunit_low),
            subunit_high=float(params.outlines.subunit_high),
            residue_low=float(params.outlines.residue_low),
            residue_high=float(params.outlines.residue_high),
            residue_diff=float(params.outlines.residue_diff),
        ),
    )


def load_atoms(params: RenderParams) -> AtomTable:
    from illustrate.formats.pdb import read_and_classify_atoms

    return read_and_classify_atoms(params.pdb_path, params.rules)


@dataclass(frozen=True, slots=True)
class RenderSession:
    """Reuse one classified atom table and backend across renders.

    The session holds the atom table by reference. Keep the same rule indices
    across renders and leave the atom arrays intact while the session uses them.
    """

    atoms: AtomTable
    backend: str

    def render(
        self,
        params: RenderParams,
        *,
        cancellation_token: RenderCancellationToken | None = None,
    ) -> RenderResult:
        return render_from_atoms(
            self.atoms,
            params,
            backend=self.backend,
            cancellation_token=cancellation_token,
        )


def prepare_render_session(
    atoms: AtomTable, *, backend: str | None = None
) -> RenderSession:
    """Pin the selected backend and classified atoms for repeated renders."""
    return RenderSession(atoms=atoms, backend=resolve_render_backend(backend))


def render_from_atoms(
    atoms: AtomTable,
    params: RenderParams,
    *,
    backend: str | None = None,
    cancellation_token: RenderCancellationToken | None = None,
) -> RenderResult:
    """Render from classified atoms without reading the structure file.

    Each atom's stored rule index selects its color and radius. Reclassify
    after changing the rule order, matching criteria, or hidden atoms.
    """
    program = program_from_params(params)
    return render_program(
        program,
        atoms,
        backend=resolve_render_backend(backend),
        cancellation_token=cancellation_token,
    )


def estimate_render_size(atoms: AtomTable, params: RenderParams) -> tuple[int, int]:
    """Return the planned image size as ``(width, height)`` in pixels."""
    program = program_from_params(params)
    return estimate_program_size(program, atoms)


@overload
def render(
    program_or_params: RenderParams,
    atoms: None = None,
    *,
    backend: str | None = None,
) -> RenderResult: ...


@overload
def render(
    program_or_params: CommandProgram,
    atoms: AtomTable,
    *,
    backend: str | None = None,
) -> RenderResult: ...


def render(
    program_or_params: CommandProgram | RenderParams,
    atoms: AtomTable | None = None,
    *,
    backend: str | None = None,
) -> RenderResult:
    """Render from a ``RenderParams`` object or a program and classified atoms.

    With ``RenderParams``, read its PDB file and classify atoms using its
    rules. With a ``CommandProgram``, the supplied atoms must have been
    classified in the order of ``program.selection_rules``.
    """
    if atoms is not None:
        if not isinstance(program_or_params, CommandProgram):
            raise TypeError(
                "render(CommandProgram, AtomTable) requires a CommandProgram as first argument"
            )
        return render_program(
            program_or_params,
            atoms,
            backend=resolve_render_backend(backend),
        )

    if not isinstance(program_or_params, RenderParams):
        raise TypeError("render(RenderParams) requires a RenderParams instance")
    loaded = load_atoms(program_or_params)
    return render_from_atoms(loaded, program_or_params, backend=backend)


def resolve_command_pdb_path(
    pdb_path: str | os.PathLike[str], *, base_dir: str | os.PathLike[str] | None
) -> Path:
    """Look for a relative READ path in the working directory, then ``base_dir``.

    Use ``base_dir`` only when it yields an existing file. Absolute and
    unresolved paths pass through unchanged.
    """
    candidate = Path(pdb_path)
    if candidate.is_absolute() or base_dir is None or candidate.is_file():
        return candidate
    based = Path(base_dir) / candidate
    if based.is_file():
        return based
    return candidate


def render_from_command_file(
    text: str,
    strict_input: bool = False,
    *,
    backend: str | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> RenderResult:
    """Render command-stream text using the structure named by its READ card.

    Relative PDB paths are tried in the working directory, then in ``base_dir``
    when supplied. ``strict_input`` enables strict parsing of both inputs.
    """
    from illustrate.formats.command_stream import parse_command_stream
    from illustrate.formats.pdb import read_and_classify_atoms

    program = parse_command_stream(text, strict_input=strict_input)
    if program.pdb_file is None:
        raise ValueError("command file missing READ card / PDB path")
    pdb_path = resolve_command_pdb_path(program.pdb_file, base_dir=base_dir)
    atoms = read_and_classify_atoms(
        pdb_path,
        program.selection_rules,
        strict_input=strict_input,
    )
    return render_program(
        program, atoms, backend=resolve_render_backend(backend)
    )

