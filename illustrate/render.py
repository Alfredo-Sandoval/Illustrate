from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import json
import math
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from illustrate.math3d import catenate, rotation_x, rotation_y, rotation_z
from illustrate.raster_kernel import backend_available, supported_backends
from illustrate.render_pipeline import estimate_program_size, render_program as _render_pipeline_program
from illustrate.types import (
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

PI = 3.141592
ZBUF_BG = -10000.0
_SPHERE_CACHE_RESOLUTION = 1000.0
_SPHERE_CACHE_MAX_ENTRIES = 256
_SPHERE_CACHE: OrderedDict[int, np.ndarray] = OrderedDict()
_SPHERE_CACHE_LOCK = threading.Lock()


def _resolve_render_backend(backend: str | None) -> str:
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
        return normalized

    for candidate in ("mlx", "cupy", "numpy"):
        if candidate in supported and backend_available(candidate):
            return candidate

    raise RuntimeError("No available render backend found")


def _build_rule_arrays(program: CommandProgram) -> tuple[np.ndarray, np.ndarray]:
    colortype = np.full((1001, 3), 0.5, dtype=np.float32)
    radtype = np.zeros(1001, dtype=np.float32)
    for idx, rule in enumerate(program.selection_rules, start=1):
        if idx >= 1001:
            break
        colortype[idx, 0] = np.float32(rule.color[0])
        colortype[idx, 1] = np.float32(rule.color[1])
        colortype[idx, 2] = np.float32(rule.color[2])
        radtype[idx] = np.float32(rule.radius)
    return colortype, radtype


def _shifted_array(arr: np.ndarray, di: int, dj: int, fill_value: float | int = 0) -> np.ndarray:
    """Return array where result[x,y] = arr[x+di, y+dj] with out-of-bounds fill."""
    result = np.full_like(arr, fill_value)
    h, w = arr.shape[:2]

    sr0, sr1 = max(0, di), min(h, h + di)
    sc0, sc1 = max(0, dj), min(w, w + dj)
    dr0, dr1 = max(0, -di), min(h, h - di)
    dc0, dc1 = max(0, -dj), min(w, w - dj)

    if sr1 > sr0 and sc1 > sc0:
        result[dr0:dr1, dc0:dc1] = arr[sr0:sr1, sc0:sc1]
    return result


def _padded_shift_view(padded: np.ndarray, di: int, dj: int, h: int, w: int, pad: int) -> np.ndarray:
    """Return a shifted view from a pre-padded array without allocating full output arrays."""
    i0 = pad + di
    j0 = pad + dj
    return padded[i0 : i0 + h, j0 : j0 + w]


def _precompute_sphere(scaled_radius: float) -> np.ndarray:
    irlim = int(scaled_radius)
    if irlim > 100:
        raise ValueError("atoms radius * scale > 100")
    cache_key = int(round(float(scaled_radius) * _SPHERE_CACHE_RESOLUTION))
    with _SPHERE_CACHE_LOCK:
        cached = _SPHERE_CACHE.get(cache_key)
        if cached is not None:
            _SPHERE_CACHE.move_to_end(cache_key)
            return cached

    voxels: list[list[float]] = []
    for ix in range(-irlim - 1, irlim + 2):
        for iy in range(-irlim - 1, irlim + 2):
            x = float(ix)
            y = float(iy)
            d = math.sqrt(x * x + y * y)
            if d > scaled_radius:
                continue
            z = math.sqrt(scaled_radius * scaled_radius - d * d)
            voxels.append([x, y, z])

    if voxels:
        sphere = np.array(voxels, dtype=np.float32)
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


def _to_u8(arr: np.ndarray) -> np.ndarray:
    out = np.clip(arr.astype(np.int32), 0, 255)
    return out.astype(np.uint8)


def _render_program(program: CommandProgram, atoms: AtomTable, *, backend: str = "numpy") -> RenderResult:
    return _render_pipeline_program(
        program,
        atoms,
        backend_name=_resolve_render_backend(backend),
        sphere_lookup=_precompute_sphere,
    )


# ---------------------------------------------------------------------------
# Public API (absorbed from illustrate_core)
# ---------------------------------------------------------------------------


def _rotation_matrix(rotations: list[tuple[str, float]]) -> np.ndarray:
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


def _to_autocenter(code: str) -> int:
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


def _translate3(value: Sequence[float]) -> tuple[float, float, float]:
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
    xtran, ytran, ztran = _translate3(transform.translate)
    return TransformState(
        scale=float(transform.scale),
        xtran=xtran,
        ytran=ytran,
        ztran=ztran,
        autocenter=_to_autocenter(transform.autocenter),
        rm=_rotation_matrix(transform.rotations),
    )


def _program_from_params(params: RenderParams) -> CommandProgram:
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
    from illustrate.pdb import read_and_classify_atoms
    return read_and_classify_atoms(params.pdb_path, params.rules)


def render_from_atoms(atoms: AtomTable, params: RenderParams, *, backend: str | None = None) -> RenderResult:
    program = _program_from_params(params)
    return _render_program(program, atoms, backend=_resolve_render_backend(backend))


def estimate_render_size(atoms: AtomTable, params: RenderParams) -> tuple[int, int]:
    """Estimate output width/height without rasterizing spheres."""
    program = _program_from_params(params)
    return estimate_program_size(program, atoms)


def render(
    program_or_params: CommandProgram | RenderParams,
    atoms: AtomTable | None = None,
    *,
    backend: str | None = None,
) -> RenderResult:
    """Render helper for both legacy and modern APIs."""
    if atoms is not None:
        if not isinstance(program_or_params, CommandProgram):
            raise TypeError("render(CommandProgram, AtomTable) requires a CommandProgram as first argument")
        return _render_program(program_or_params, atoms, backend=_resolve_render_backend(backend))

    if not isinstance(program_or_params, RenderParams):
        raise TypeError("render(RenderParams) requires a RenderParams instance")
    loaded = load_atoms(program_or_params)
    return render_from_atoms(loaded, program_or_params, backend=backend)


def render_from_command_file(text: str, strict_input: bool = False, *, backend: str | None = None) -> RenderResult:
    from illustrate.parser import parse_command_stream
    from illustrate.pdb import read_and_classify_atoms
    program = parse_command_stream(text, strict_input=strict_input)
    if program.pdb_file is None:
        raise ValueError("command file missing READ card / PDB path")
    atoms = read_and_classify_atoms(
        program.pdb_file,
        program.selection_rules,
        strict_input=strict_input,
    )
    return _render_program(program, atoms, backend=_resolve_render_backend(backend))


def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


def params_to_json(params: RenderParams) -> str:
    return json.dumps(_to_dict(params), sort_keys=True)


def _coerce_color(payload: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    values: Any = default if not isinstance(payload, (list, tuple)) else payload
    if len(values) < 3:
        return default
    return (float(values[0]), float(values[1]), float(values[2]))


def _coerce_translate(payload: Any) -> tuple[float, float, float]:
    values: Any = payload if isinstance(payload, (list, tuple)) else (0.0, 0.0, 0.0)
    if len(values) < 3:
        return (0.0, 0.0, 0.0)
    return (float(values[0]), float(values[1]), float(values[2]))


def _coerce_object(payload: Any, field_name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items()}
    raise TypeError(f"{field_name} must be an object")


def _coerce_rotations(payload: Any) -> list[tuple[str, float]]:
    if not isinstance(payload, list):
        return []

    rotations: list[tuple[str, float]] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        rotations.append((str(item[0]), float(item[1])))
    return rotations


def params_from_json(json_str: str) -> RenderParams:
    payload_raw = json.loads(json_str)
    if not isinstance(payload_raw, dict):
        raise TypeError("top-level JSON payload must be an object")
    payload = {str(key): value for key, value in payload_raw.items()}

    rules_payload = payload.get("rules", [])
    if not isinstance(rules_payload, list):
        raise TypeError("rules must be a list")

    transform_payload = _coerce_object(payload.get("transform", {}), "transform")
    world_payload = _coerce_object(payload.get("world", {}), "world")
    outlines_payload = _coerce_object(payload.get("outlines", {}), "outlines")

    normalized_rules: list[SelectionRule] = []
    for idx, rule in enumerate(rules_payload):
        if not isinstance(rule, dict):
            raise TypeError(f"rules[{idx}] must be an object")
        normalized_rules.append(
            SelectionRule(
                record_name=str(rule["record_name"]),
                descriptor=str(rule["descriptor"]),
                res_low=int(rule["res_low"]),
                res_high=int(rule["res_high"]),
                color=_coerce_color(rule.get("color"), (1.0, 1.0, 1.0)),
                radius=float(rule["radius"]),
            )
        )

    return RenderParams(
        pdb_path=str(payload["pdb_path"]),
        rules=normalized_rules,
        transform=Transform(
            scale=float(transform_payload.get("scale", 12.0)),
            translate=_coerce_translate(transform_payload.get("translate", (0.0, 0.0, 0.0))),
            rotations=_coerce_rotations(transform_payload.get("rotations", [])),
            autocenter=str(transform_payload.get("autocenter", "auto")),
        ),
        world=WorldParams(
            background=_coerce_color(
                world_payload.get("background", (1.0, 1.0, 1.0)),
                (1.0, 1.0, 1.0),
            ),
            fog_color=_coerce_color(
                world_payload.get("fog_color", (1.0, 1.0, 1.0)),
                (1.0, 1.0, 1.0),
            ),
            fog_front=float(world_payload.get("fog_front", 1.0)),
            fog_back=float(world_payload.get("fog_back", 1.0)),
            shadows=bool(world_payload.get("shadows", False)),
            shadow_strength=float(world_payload.get("shadow_strength", 0.0023)),
            shadow_angle=float(world_payload.get("shadow_angle", 2.0)),
            shadow_min_z=float(world_payload.get("shadow_min_z", 1.0)),
            shadow_max_dark=float(world_payload.get("shadow_max_dark", 0.2)),
            width=int(world_payload.get("width", 0)),
            height=int(world_payload.get("height", 0)),
        ),
        outlines=OutlineParams(
            enabled=bool(outlines_payload.get("enabled", True)),
            contour_low=float(outlines_payload.get("contour_low", 3.0)),
            contour_high=float(outlines_payload.get("contour_high", 10.0)),
            kernel=int(outlines_payload.get("kernel", 4)),
            z_diff_min=float(outlines_payload.get("z_diff_min", 0.0)),
            z_diff_max=float(outlines_payload.get("z_diff_max", 5.0)),
            subunit_low=float(outlines_payload.get("subunit_low", 3.0)),
            subunit_high=float(outlines_payload.get("subunit_high", 10.0)),
            residue_low=float(outlines_payload.get("residue_low", 3.0)),
            residue_high=float(outlines_payload.get("residue_high", 8.0)),
            residue_diff=float(outlines_payload.get("residue_diff", 6000.0)),
        ),
    )
