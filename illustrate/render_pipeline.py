from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import weakref

import numpy as np

from illustrate.raster_kernel import (
    _require_cupy,
    _require_mlx,
    run_composite_kernel,
    run_kernel,
    run_outline12_kernel,
    run_outline34_kernel,
    run_shadow_kernel,
)
from illustrate.types import AtomTable, CommandProgram, OutlineParams, RenderResult, WorldParams

ZBUF_BG = -10000.0
MAX_RULE_TYPES = 1000
RULE_TABLE_SIZE = MAX_RULE_TYPES + 1
MAX_IMAGE_DIMENSION = 3000
_CHUNK_ATOMS = 512
_BACKEND_ARRAY_CACHE_LIMIT = 32

_BackendArrayCacheKey = tuple[str, object, str, tuple[int, ...]]
_BackendArrayCacheValue = tuple[weakref.ReferenceType[np.ndarray] | None, Any]
_BACKEND_ARRAY_CACHE: OrderedDict[_BackendArrayCacheKey, _BackendArrayCacheValue] = OrderedDict()


@dataclass(slots=True)
class SceneLayout:
    width: int
    height: int
    n: int
    nbiomat: int
    biomats: np.ndarray
    coords_arr: np.ndarray
    rm: np.ndarray
    rscale: float
    xtran: float
    ytran: float
    ztran: float
    xtranc: float
    ytranc: float
    ztranc: float


@dataclass(slots=True)
class PreparedScene:
    layout: SceneLayout
    world: WorldParams
    outlines: OutlineParams
    background: np.ndarray
    fog_color: np.ndarray
    fog_front: float
    fog_back: float
    colortype: np.ndarray
    radtype: np.ndarray
    ndes: int


@dataclass(slots=True)
class BackendBuffers:
    backend_name: str
    zpix: Any
    atom_buf: Any
    bio_buf: Any
    cupy_mod: Any | None = None
    mlx_mod: Any | None = None
    shadow_bounds: tuple[int, int, int, int] | None = None


def _cached_backend_array(
    buffers: BackendBuffers,
    values: np.ndarray,
    *,
    cpu_dtype: np.dtype[Any] | type[np.generic],
    backend_dtype_attr: str,
    cache: bool = False,
    cache_key: object | None = None,
) -> Any:
    source = np.asarray(values, dtype=cpu_dtype)
    if buffers.cupy_mod is not None:
        backend_mod = buffers.cupy_mod
        converter_name = "asarray"
    elif buffers.mlx_mod is not None:
        backend_mod = buffers.mlx_mod
        converter_name = "array"
    else:
        return source

    backend_dtype = getattr(backend_mod, backend_dtype_attr)
    if not cache and cache_key is None:
        return getattr(backend_mod, converter_name)(source, dtype=backend_dtype)

    key_token: object = cache_key if cache_key is not None else id(source)
    key: _BackendArrayCacheKey = (buffers.backend_name, key_token, source.dtype.str, tuple(source.shape))
    cached = _BACKEND_ARRAY_CACHE.get(key)
    if cached is not None:
        source_ref, device_array = cached
        if source_ref is None or source_ref() is source:
            _BACKEND_ARRAY_CACHE.move_to_end(key)
            return device_array
        del _BACKEND_ARRAY_CACHE[key]

    device_array = getattr(backend_mod, converter_name)(source, dtype=backend_dtype)
    source_ref = None if cache_key is not None else weakref.ref(source)
    _BACKEND_ARRAY_CACHE[key] = (source_ref, device_array)
    while len(_BACKEND_ARRAY_CACHE) > _BACKEND_ARRAY_CACHE_LIMIT:
        _BACKEND_ARRAY_CACHE.popitem(last=False)
    return device_array


def _backend_float_array(
    buffers: BackendBuffers,
    values: np.ndarray,
    *,
    cache: bool = False,
    cache_key: object | None = None,
) -> Any:
    return _cached_backend_array(
        buffers,
        values,
        cpu_dtype=np.float32,
        backend_dtype_attr="float32",
        cache=cache,
        cache_key=cache_key,
    )


def _backend_int_array(
    buffers: BackendBuffers,
    values: np.ndarray,
    *,
    cache: bool = False,
    cache_key: object | None = None,
) -> Any:
    return _cached_backend_array(
        buffers,
        values,
        cpu_dtype=np.int32,
        backend_dtype_attr="int32",
        cache=cache,
        cache_key=cache_key,
    )


def _build_rule_arrays(program: CommandProgram) -> tuple[np.ndarray, np.ndarray]:
    colortype = np.full((RULE_TABLE_SIZE, 3), 0.5, dtype=np.float32)
    radtype = np.zeros(RULE_TABLE_SIZE, dtype=np.float32)
    for idx, rule in enumerate(program.selection_rules, start=1):
        if idx >= RULE_TABLE_SIZE:
            break
        colortype[idx, 0] = np.float32(rule.color[0])
        colortype[idx, 1] = np.float32(rule.color[1])
        colortype[idx, 2] = np.float32(rule.color[2])
        radtype[idx] = np.float32(rule.radius)
    return colortype, radtype


def _merge_shadow_bounds(
    current: tuple[int, int, int, int] | None,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> tuple[int, int, int, int] | None:
    if x0 >= x1 or y0 >= y1:
        return current
    if current is None:
        return (x0, x1, y0, y1)
    cx0, cx1, cy0, cy1 = current
    return (min(cx0, x0), max(cx1, x1), min(cy0, y0), max(cy1, y1))


def _build_biomats(atoms: AtomTable) -> tuple[int, np.ndarray]:
    nbiomat = max(int(atoms.nbiomat), 1)
    if atoms.nbiomat == 0:
        biomats = np.zeros((2, 4, 4), dtype=np.float32)
        biomats[1] = np.eye(4, dtype=np.float32)
        return nbiomat, biomats

    biomats = np.zeros((nbiomat + 1, 4, 4), dtype=np.float32)
    for ibio in range(1, nbiomat + 1):
        biomats[ibio] = np.eye(4, dtype=np.float32)
        biomats[ibio, :3, :4] = atoms.biomat[ibio]
    return nbiomat, biomats


def _scale_rule_radii(radtype: np.ndarray, scale: float, ndes: int) -> float:
    radius_max = 0.0
    for idx in range(1, ndes + 1):
        radtype[idx] = np.float32(radtype[idx] * scale)
        radius_max = max(radius_max, float(radtype[idx]))
    return radius_max


def _prepare_layout(program: CommandProgram, atoms: AtomTable, radtype: np.ndarray, ndes: int) -> SceneLayout:
    transform = program.transform
    world = program.world

    ixsize = min(int(world.width), MAX_IMAGE_DIMENSION)
    iysize = min(int(world.height), MAX_IMAGE_DIMENSION)

    nbiomat, biomats = _build_biomats(atoms)
    n = int(atoms.n)
    rm = transform.rm.astype(np.float32)
    rscale = float(transform.scale)
    xtran = float(transform.xtran)
    ytran = float(transform.ytran)
    ztran = float(transform.ztran)
    radius_max = _scale_rule_radii(radtype, rscale, ndes)
    coords_arr = atoms.coord[1 : n + 1].astype(np.float32) if n > 0 else np.zeros((0, 3), dtype=np.float32)

    xtranc = 0.0
    ytranc = 0.0
    ztranc = 0.0
    if transform.autocenter > 0 and n > 0:
        xmin = ymin = zmin = 10000.0
        xmax = ymax = zmax = -10000.0

        for ibio in range(1, nbiomat + 1):
            bm = biomats[ibio]
            rotated_bio = coords_arr @ bm[:3, :3].T + bm[:3, 3]
            rotated = rotated_bio @ rm[:3, :3]

            xmin = min(xmin, float(rotated[:, 0].min()))
            xmax = max(xmax, float(rotated[:, 0].max()))
            ymin = min(ymin, float(rotated[:, 1].min()))
            ymax = max(ymax, float(rotated[:, 1].max()))
            zmin = min(zmin, float(rotated[:, 2].min()))
            zmax = max(zmax, float(rotated[:, 2].max()))

        xtranc = -xmin - (xmax - xmin) / 2.0
        ytranc = -ymin - (ymax - ymin) / 2.0

        if transform.autocenter == 1:
            ztranc = -zmax - radius_max - 1.0
        elif transform.autocenter == 2:
            ztranc = -zmin - (zmax - zmin) / 2.0

        if ixsize <= 0 or iysize <= 0:
            ixsize = int(-2.0 * ixsize + 2.0 * radius_max + (xmax - xmin) * rscale)
            iysize = int(-2.0 * iysize + 2.0 * radius_max + (ymax - ymin) * rscale)

    ixsize = min(ixsize, MAX_IMAGE_DIMENSION)
    iysize = min(iysize, MAX_IMAGE_DIMENSION)
    ixsize = (ixsize // 2) * 2
    iysize = (iysize // 2) * 2

    return SceneLayout(
        width=ixsize,
        height=iysize,
        n=n,
        nbiomat=nbiomat,
        biomats=biomats,
        coords_arr=coords_arr,
        rm=rm,
        rscale=rscale,
        xtran=xtran,
        ytran=ytran,
        ztran=ztran,
        xtranc=xtranc,
        ytranc=ytranc,
        ztranc=ztranc,
    )


def prepare_scene(program: CommandProgram, atoms: AtomTable) -> PreparedScene:
    colortype, radtype = _build_rule_arrays(program)
    ndes = min(len(program.selection_rules), MAX_RULE_TYPES)

    world = program.world
    background = np.minimum(np.array(world.background, dtype=np.float32), 1.0)
    fog_color = np.minimum(np.array(world.fog_color, dtype=np.float32), 1.0)
    colortype[0] = background

    layout = _prepare_layout(program, atoms, radtype, ndes)
    return PreparedScene(
        layout=layout,
        world=world,
        outlines=program.outlines,
        background=background,
        fog_color=fog_color,
        fog_front=float(min(world.fog_front, 1.0)),
        fog_back=float(min(abs(world.fog_back), 1.0)),
        colortype=colortype,
        radtype=radtype,
        ndes=ndes,
    )


def estimate_program_size(program: CommandProgram, atoms: AtomTable) -> tuple[int, int]:
    colortype, radtype = _build_rule_arrays(program)
    del colortype
    ndes = min(len(program.selection_rules), MAX_RULE_TYPES)
    layout = _prepare_layout(program, atoms, radtype, ndes)
    return (layout.width, layout.height)


def _initialize_backend_buffers(backend_name: str, width: int, height: int) -> BackendBuffers:
    if backend_name == "cupy":
        cp = _require_cupy()
        return BackendBuffers(
            backend_name=backend_name,
            zpix=cp.full((width, height), ZBUF_BG, dtype=cp.float32),
            atom_buf=cp.zeros((width, height), dtype=cp.int32),
            bio_buf=cp.ones((width, height), dtype=cp.int32),
            cupy_mod=cp,
        )

    if backend_name == "mlx":
        mx = _require_mlx()
        return BackendBuffers(
            backend_name=backend_name,
            zpix=mx.full((width, height), ZBUF_BG, dtype=mx.float32),
            atom_buf=mx.zeros((width, height), dtype=mx.int32),
            bio_buf=mx.ones((width, height), dtype=mx.int32),
            mlx_mod=mx,
        )

    return BackendBuffers(
        backend_name=backend_name,
        zpix=np.full((width, height), ZBUF_BG, dtype=np.float32),
        atom_buf=np.zeros((width, height), dtype=np.int32),
        bio_buf=np.ones((width, height), dtype=np.int32),
    )


def _rasterize_atoms(
    scene: PreparedScene,
    atoms: AtomTable,
    buffers: BackendBuffers,
    sphere_lookup: Callable[[float], np.ndarray],
) -> None:
    layout = scene.layout
    if layout.n <= 0:
        return

    atom_types = atoms.type_idx[1 : layout.n + 1]
    half_ix = float(layout.width) / 2.0
    half_iy = float(layout.height) / 2.0
    fix = float(layout.width)
    fiy = float(layout.height)

    for irad in range(1, scene.ndes + 1):
        sphere = sphere_lookup(float(scene.radtype[irad]))
        if len(sphere) == 0:
            continue

        matching = np.where(atom_types == irad)[0]
        if len(matching) == 0:
            continue

        nv = len(sphere)
        sx_min = float(np.min(sphere[:, 0]))
        sx_max = float(np.max(sphere[:, 0]))
        sy_min = float(np.min(sphere[:, 1]))
        sy_max = float(np.max(sphere[:, 1]))
        sphere_key = ("sphere", id(sphere))
        sx = _backend_float_array(buffers, sphere[:, 0], cache_key=(sphere_key, 0))
        sy = _backend_float_array(buffers, sphere[:, 1], cache_key=(sphere_key, 1))
        sz = _backend_float_array(buffers, sphere[:, 2], cache_key=(sphere_key, 2))

        for ibio in range(1, layout.nbiomat + 1):
            bm = layout.biomats[ibio]
            centers_bio = layout.coords_arr[matching] @ bm[:3, :3].T + bm[:3, 3]
            centers_rot = centers_bio @ layout.rm[:3, :3]

            cx_all = (centers_rot[:, 0] + layout.xtranc + layout.xtran) * layout.rscale
            cy_all = (centers_rot[:, 1] + layout.ytranc + layout.ytran) * layout.rscale
            cz_all = (centers_rot[:, 2] + layout.ztranc + layout.ztran) * layout.rscale

            visible = cz_all < 0.0
            if not np.any(visible):
                continue

            vis_idx = np.where(visible)[0]
            cx_visible = cx_all[vis_idx]
            cy_visible = cy_all[vis_idx]
            x0 = max(0, min(layout.width, int(np.floor(float(np.min(cx_visible)) + half_ix + sx_min)) - 1))
            x1 = max(0, min(layout.width, int(np.ceil(float(np.max(cx_visible)) + half_ix + sx_max))))
            y0 = max(0, min(layout.height, int(np.floor(float(np.min(cy_visible)) + half_iy + sy_min)) - 1))
            y1 = max(0, min(layout.height, int(np.ceil(float(np.max(cy_visible)) + half_iy + sy_max))))
            buffers.shadow_bounds = _merge_shadow_bounds(
                buffers.shadow_bounds,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
            )
            cx_vis = _backend_float_array(buffers, cx_all[vis_idx])
            cy_vis = _backend_float_array(buffers, cy_all[vis_idx])
            cz_vis = _backend_float_array(buffers, cz_all[vis_idx])
            ia_vis = _backend_int_array(buffers, matching[vis_idx] + 1)

            for chunk_start in range(0, len(vis_idx), _CHUNK_ATOMS):
                chunk_end = min(chunk_start + _CHUNK_ATOMS, len(vis_idx))
                c_cx = cx_vis[chunk_start:chunk_end]
                c_cy = cy_vis[chunk_start:chunk_end]
                c_cz = cz_vis[chunk_start:chunk_end]
                c_ia = ia_vis[chunk_start:chunk_end]
                buffers.zpix, buffers.atom_buf, buffers.bio_buf = run_kernel(
                    backend=buffers.backend_name,
                    sx=sx,
                    sy=sy,
                    sz=sz,
                    c_cx=c_cx,
                    c_cy=c_cy,
                    c_cz=c_cz,
                    c_ia=c_ia,
                    half_ix=half_ix,
                    half_iy=half_iy,
                    fix=fix,
                    fiy=fiy,
                    nv=nv,
                    ibio=ibio,
                    zpix=buffers.zpix,
                    atom_buf=buffers.atom_buf,
                    bio_buf=buffers.bio_buf,
                )


def _outline_input(zpix: Any, buffers: BackendBuffers) -> Any:
    if buffers.cupy_mod is not None:
        return buffers.cupy_mod.minimum(zpix, 0.0)
    if buffers.mlx_mod is not None:
        return buffers.mlx_mod.minimum(zpix, 0.0)
    return np.minimum(zpix, 0.0)


def _precompute_outline(scene: PreparedScene, atoms: AtomTable, buffers: BackendBuffers) -> Any | None:
    outlines = scene.outlines
    if not outlines.enabled:
        return None

    su_lookup = _backend_int_array(buffers, atoms.su, cache=True)
    res_lookup = _backend_int_array(buffers, atoms.res, cache=True)

    if outlines.kernel in (3, 4):
        return run_outline34_kernel(
            backend=buffers.backend_name,
            zpix=_outline_input(buffers.zpix, buffers),
            atom_buf=buffers.atom_buf,
            bio_buf=buffers.bio_buf,
            su_lookup=su_lookup,
            res_lookup=res_lookup,
            residue_diff=float(outlines.residue_diff),
            residue_low=float(outlines.residue_low),
            residue_high=float(outlines.residue_high),
            subunit_low=float(outlines.subunit_low),
            subunit_high=float(outlines.subunit_high),
            z_diff_min=float(outlines.z_diff_min * scene.layout.rscale),
            z_diff_max=float(outlines.z_diff_max * scene.layout.rscale),
            contour_low=float(outlines.contour_low),
            contour_high=float(outlines.contour_high),
            kernel=int(outlines.kernel),
        )

    if outlines.kernel in (1, 2):
        return run_outline12_kernel(
            backend=buffers.backend_name,
            zpix=_outline_input(buffers.zpix, buffers),
            atom_buf=buffers.atom_buf,
            bio_buf=buffers.bio_buf,
            su_lookup=su_lookup,
            res_lookup=res_lookup,
            residue_diff=float(outlines.residue_diff),
            residue_low=float(outlines.residue_low),
            residue_high=float(outlines.residue_high),
            subunit_low=float(outlines.subunit_low),
            subunit_high=float(outlines.subunit_high),
            contour_low=float(outlines.contour_low),
            contour_high=float(outlines.contour_high),
            kernel=int(outlines.kernel),
        )

    return None


def _shadow_mask(scene: PreparedScene, buffers: BackendBuffers) -> Any:
    width = scene.layout.width
    height = scene.layout.height
    if scene.world.shadows:
        bounds = buffers.shadow_bounds
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            if x0 >= x1 or y0 >= y1:
                bounds = None
        if bounds is None:
            if buffers.cupy_mod is not None:
                return buffers.cupy_mod.ones((width, height), dtype=buffers.cupy_mod.float32)
            if buffers.mlx_mod is not None:
                return buffers.mlx_mod.ones((width, height), dtype=buffers.mlx_mod.float32)
            return np.ones((width, height), dtype=np.float32)

        x0, x1, y0, y1 = bounds
        if x0 == 0 and x1 == width and y0 == 0 and y1 == height:
            return run_shadow_kernel(
                backend=buffers.backend_name,
                zpix=buffers.zpix,
                atom_buf=buffers.atom_buf,
                shadow_strength=float(scene.world.shadow_strength),
                shadow_angle=float(scene.world.shadow_angle),
                shadow_min_z=float(scene.world.shadow_min_z),
                shadow_max_dark=float(scene.world.shadow_max_dark),
            )

        shadow_crop = run_shadow_kernel(
            backend=buffers.backend_name,
            zpix=buffers.zpix[x0:x1, y0:y1],
            atom_buf=buffers.atom_buf[x0:x1, y0:y1],
            shadow_strength=float(scene.world.shadow_strength),
            shadow_angle=float(scene.world.shadow_angle),
            shadow_min_z=float(scene.world.shadow_min_z),
            shadow_max_dark=float(scene.world.shadow_max_dark),
        )
        if buffers.cupy_mod is not None:
            full_shadow = buffers.cupy_mod.ones((width, height), dtype=buffers.cupy_mod.float32)
            full_shadow[x0:x1, y0:y1] = shadow_crop
            return full_shadow
        if buffers.mlx_mod is not None:
            mx = buffers.mlx_mod
            full_shadow = mx.ones((width, height), dtype=mx.float32)
            crop_delta = mx.ones((x1 - x0, y1 - y0), dtype=mx.float32) - shadow_crop
            return full_shadow.at[x0:x1, y0:y1].subtract(crop_delta)
        full_shadow = np.ones((width, height), dtype=np.float32)
        full_shadow[x0:x1, y0:y1] = np.asarray(shadow_crop, dtype=np.float32)
        return full_shadow

    if buffers.cupy_mod is not None:
        return buffers.cupy_mod.ones((width, height), dtype=buffers.cupy_mod.float32)
    if buffers.mlx_mod is not None:
        return buffers.mlx_mod.ones((width, height), dtype=buffers.mlx_mod.float32)
    return np.ones((width, height), dtype=np.float32)


def _to_u8(arr: np.ndarray) -> np.ndarray:
    out = np.clip(arr.astype(np.int32), 0, 255)
    return out.astype(np.uint8)


def _render_precomputed_outline(
    scene: PreparedScene,
    atoms: AtomTable,
    buffers: BackendBuffers,
    pconetot: Any,
    precomputed_outline: Any | None,
) -> RenderResult | None:
    outline_opacity = precomputed_outline
    if outline_opacity is None:
        if scene.outlines.enabled:
            return None
        width = scene.layout.width
        height = scene.layout.height
        if buffers.cupy_mod is not None:
            outline_opacity = buffers.cupy_mod.zeros((width, height), dtype=buffers.cupy_mod.float32)
        elif buffers.mlx_mod is not None:
            outline_opacity = buffers.mlx_mod.zeros((width, height), dtype=buffers.mlx_mod.float32)
        else:
            return None

    type_lookup = _backend_int_array(buffers, atoms.type_idx, cache=True)
    color_lut = _backend_float_array(buffers, scene.colortype)
    fog_color = _backend_float_array(buffers, scene.fog_color.astype(np.float32, copy=False))

    rgb_linear, alpha_linear = run_composite_kernel(
        backend=buffers.backend_name,
        zpix=buffers.zpix,
        atom_buf=buffers.atom_buf,
        pconetot=pconetot,
        l_opacity=outline_opacity,
        type_lookup=type_lookup,
        colortype=color_lut,
        fog_color=fog_color,
        fog_front=float(scene.fog_front),
        fog_back=float(scene.fog_back),
        zbuf_bg=float(ZBUF_BG),
    )
    if buffers.cupy_mod is not None:
        rgb_linear = buffers.cupy_mod.asnumpy(rgb_linear)
        alpha_linear = buffers.cupy_mod.asnumpy(alpha_linear)
    elif buffers.mlx_mod is not None:
        mx = buffers.mlx_mod
        scale = mx.array(np.float32(255.0), dtype=mx.float32)
        zero = mx.array(np.float32(0.0), dtype=mx.float32)
        hi = mx.array(np.float32(255.0), dtype=mx.float32)
        rgb_u8 = mx.clip(rgb_linear * scale, zero, hi).astype(mx.uint8)
        alpha_u8 = mx.clip(alpha_linear * scale, zero, hi).astype(mx.uint8)
        rgb_u8 = mx.swapaxes(rgb_u8, 0, 1)
        alpha_u8 = mx.swapaxes(alpha_u8, 0, 1)
        mx.eval(rgb_u8, alpha_u8)
        rgb = np.array(rgb_u8, dtype=np.uint8)
        opacity = np.array(alpha_u8, dtype=np.uint8)
        return RenderResult(rgb=rgb, opacity=opacity, width=scene.layout.width, height=scene.layout.height)
    else:
        rgb_linear = np.asarray(rgb_linear, dtype=np.float32)
        alpha_linear = np.asarray(alpha_linear, dtype=np.float32)

    rgb = np.swapaxes(_to_u8(rgb_linear * 255.0), 0, 1)
    opacity = np.swapaxes(_to_u8(alpha_linear * 255.0), 0, 1)
    return RenderResult(rgb=rgb, opacity=opacity, width=scene.layout.width, height=scene.layout.height)


def _materialize_numpy(
    buffers: BackendBuffers,
    pconetot: Any,
    precomputed_outline: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if buffers.cupy_mod is not None:
        zpix = buffers.cupy_mod.asnumpy(buffers.zpix)
        atom_buf = buffers.cupy_mod.asnumpy(buffers.atom_buf)
        bio_buf = buffers.cupy_mod.asnumpy(buffers.bio_buf)
        pconetot_np = buffers.cupy_mod.asnumpy(pconetot)
        outline_np = None if precomputed_outline is None else buffers.cupy_mod.asnumpy(precomputed_outline)
        return zpix, atom_buf, bio_buf, pconetot_np, outline_np

    if buffers.mlx_mod is not None:
        if precomputed_outline is None:
            buffers.mlx_mod.eval(buffers.zpix, buffers.atom_buf, buffers.bio_buf, pconetot)
        else:
            buffers.mlx_mod.eval(buffers.zpix, buffers.atom_buf, buffers.bio_buf, pconetot, precomputed_outline)
        zpix = np.array(buffers.zpix, dtype=np.float32)
        atom_buf = np.array(buffers.atom_buf, dtype=np.int32)
        bio_buf = np.array(buffers.bio_buf, dtype=np.int32)
        pconetot_np = np.array(pconetot, dtype=np.float32)
        outline_np = None if precomputed_outline is None else np.array(precomputed_outline, dtype=np.float32)
        return zpix, atom_buf, bio_buf, pconetot_np, outline_np

    return (
        np.asarray(buffers.zpix, dtype=np.float32),
        np.asarray(buffers.atom_buf, dtype=np.int32),
        np.asarray(buffers.bio_buf, dtype=np.int32),
        np.asarray(pconetot, dtype=np.float32),
        None if precomputed_outline is None else np.asarray(precomputed_outline, dtype=np.float32),
    )


def _padded_shift_view(padded: np.ndarray, di: int, dj: int, h: int, w: int, pad: int) -> np.ndarray:
    i0 = pad + di
    j0 = pad + dj
    return padded[i0 : i0 + h, j0 : j0 + w]


def _group_outline_opacity(
    outlines: OutlineParams,
    atoms: AtomTable,
    atom_buf: np.ndarray,
    bio_buf: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    su_map = atoms.su[atom_buf]
    res_map = atoms.res[atom_buf]
    res_map_f = res_map.astype(np.float32)

    r_count = np.zeros((width, height), dtype=np.float32)
    g_count = np.zeros((width, height), dtype=np.float32)
    rg_pad = 2
    su_padded = np.pad(su_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    bio_padded = np.pad(bio_buf, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=0)
    res_padded = np.pad(res_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            if abs(di * dj) == 4:
                continue
            shifted_su = _padded_shift_view(su_padded, di, dj, width, height, rg_pad)
            shifted_bio = _padded_shift_view(bio_padded, di, dj, width, height, rg_pad)
            shifted_res = _padded_shift_view(res_padded, di, dj, width, height, rg_pad)

            r_count += ((su_map != shifted_su) | (bio_buf != shifted_bio)).astype(np.float32)
            g_count += (np.abs(res_map_f - shifted_res.astype(np.float32)) > outlines.residue_diff).astype(np.float32)

    if outlines.residue_high != outlines.residue_low:
        g_opacity = np.clip(
            (g_count - outlines.residue_low) / (outlines.residue_high - outlines.residue_low),
            0.0,
            1.0,
        )
    else:
        g_opacity = np.zeros((width, height), dtype=np.float32)
    if outlines.subunit_high != outlines.subunit_low:
        r_opacity = np.clip(
            (r_count - outlines.subunit_low) / (outlines.subunit_high - outlines.subunit_low),
            0.0,
            1.0,
        )
    else:
        r_opacity = np.zeros_like(g_opacity)
    g_opacity = np.maximum(g_opacity, r_opacity)

    g_opacity[0, :] = 0.0
    g_opacity[-1, :] = 0.0
    g_opacity[:, 0] = 0.0
    g_opacity[:, -1] = 0.0
    return g_opacity


def _depth_outline_opacity(outlines: OutlineParams, zpix: np.ndarray, rscale: float, width: int, height: int) -> np.ndarray:
    kernel = outlines.kernel
    contour_low = outlines.contour_low
    contour_high = outlines.contour_high

    if kernel in (3, 4):
        l_total = np.zeros((width, height), dtype=np.float32)
        if kernel == 3:
            offsets = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
        else:
            offsets = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if abs(i * j) != 4]

        zpad = 2
        zpix_padded = np.pad(zpix, ((zpad, zpad), (zpad, zpad)), mode="constant", constant_values=0.0)
        z_diff_min = outlines.z_diff_min * rscale
        z_diff_max = outlines.z_diff_max * rscale
        denom = z_diff_max - z_diff_min
        for di, dj in offsets:
            shifted_z = _padded_shift_view(zpix_padded, di, dj, width, height, zpad)
            rd = np.abs(zpix - shifted_z)
            if denom != 0.0:
                rd_norm = np.where(rd > z_diff_min, np.minimum((rd - z_diff_min) / denom, 1.0), 0.0)
            else:
                rd_norm = np.zeros_like(rd)
            l_total += rd_norm

        if contour_high != contour_low:
            l_val = np.clip((l_total - contour_low) / (contour_high - contour_low), 0.0, 1.0)
        else:
            l_val = np.zeros_like(l_total)
        return np.where(l_val > 0, np.minimum(1.5 * l_val, 1.0), 0.0)

    if kernel not in (1, 2):
        return np.zeros((width, height), dtype=np.float32)

    if kernel == 1:
        weights = {
            (-1, -1): -0.8,
            (-1, 0): -1.0,
            (-1, 1): -0.8,
            (0, -1): -1.0,
            (0, 0): 7.2,
            (0, 1): -1.0,
            (1, -1): -0.8,
            (1, 0): -1.0,
            (1, 1): -0.8,
        }
    else:
        weights = {
            (-1, -1): -0.8,
            (-1, 0): -1.0,
            (-1, 1): -0.8,
            (0, -1): -1.0,
            (0, 0): 8.8,
            (0, 1): -1.0,
            (1, -1): -0.8,
            (1, 0): -1.0,
            (1, 1): -0.8,
            (2, -1): -0.1,
            (2, 0): -0.2,
            (2, 1): -0.1,
            (-2, -1): -0.1,
            (-2, 0): -0.2,
            (-2, 1): -0.1,
            (-1, 2): -0.1,
            (0, 2): -0.2,
            (1, 2): -0.1,
            (-1, -2): -0.1,
            (0, -2): -0.2,
            (1, -2): -0.1,
        }

    lap = np.zeros((width, height), dtype=np.float32)
    zpad = 2
    zpix_padded = np.pad(zpix, ((zpad, zpad), (zpad, zpad)), mode="constant", constant_values=0.0)
    for (di, dj), weight in weights.items():
        lap += weight * _padded_shift_view(zpix_padded, di, dj, width, height, zpad)
    lap = np.abs(lap / 3.0)

    rl = np.zeros((width, height), dtype=np.float32)
    l_opacity_ave = np.zeros((width, height), dtype=np.float32)
    l_center = np.zeros((width, height), dtype=np.float32)

    lap_pad = 1
    lap_padded = np.pad(lap, ((lap_pad, lap_pad), (lap_pad, lap_pad)), mode="constant", constant_values=0.0)
    for ixl in range(-1, 2):
        for iyl in range(-1, 2):
            l_shifted = _padded_shift_view(lap_padded, ixl, iyl, width, height, lap_pad)
            if contour_high != contour_low:
                l_val = np.clip((l_shifted - contour_low) / (contour_high - contour_low), 0.0, 1.0)
            else:
                l_val = np.zeros_like(l_shifted)
            rl += (l_val > 0).astype(np.float32)
            l_opacity_ave += l_val
            if ixl == 0 and iyl == 0:
                l_center = l_val.copy()

    l_opacity = np.where(rl >= 6.0, l_opacity_ave / 6.0, l_center)
    return np.clip(l_opacity, 0.0, 1.0)


def _outline_opacity(
    scene: PreparedScene,
    atoms: AtomTable,
    zpix: np.ndarray,
    atom_buf: np.ndarray,
    bio_buf: np.ndarray,
    precomputed_outline: np.ndarray | None,
) -> np.ndarray:
    width = scene.layout.width
    height = scene.layout.height
    if precomputed_outline is not None:
        return np.asarray(precomputed_outline, dtype=np.float32)
    if not scene.outlines.enabled:
        return np.zeros((width, height), dtype=np.float32)

    g_opacity = _group_outline_opacity(scene.outlines, atoms, atom_buf, bio_buf, width, height)
    l_opacity = _depth_outline_opacity(scene.outlines, zpix, scene.layout.rscale, width, height)
    l_opacity[:2, :] = 0.0
    l_opacity[-2:, :] = 0.0
    l_opacity[:, :2] = 0.0
    l_opacity[:, -2:] = 0.0
    return np.maximum(l_opacity, g_opacity)


def _fog_factor(scene: PreparedScene, zpix: np.ndarray) -> np.ndarray:
    zpix_max = float(zpix.max())
    mol_mask = zpix != ZBUF_BG
    zpix_min = float(zpix[mol_mask].min()) if np.any(mol_mask) else 100000.0
    zpix_max = min(zpix_max, 0.0)
    zpix_spread = zpix_max - zpix_min
    zpix[:] = np.minimum(zpix, 0.0)

    pfogdiff = scene.fog_front - scene.fog_back
    if zpix_spread != 0.0:
        pfh = scene.fog_front - (zpix_max - zpix) / zpix_spread * pfogdiff
    else:
        pfh = np.full_like(zpix, scene.fog_front)
    pfh = np.where(zpix < zpix_min, 1.0, pfh)
    return pfh


def _compose_numpy(
    scene: PreparedScene,
    atoms: AtomTable,
    zpix: np.ndarray,
    atom_buf: np.ndarray,
    bio_buf: np.ndarray,
    pconetot: np.ndarray,
    precomputed_outline: np.ndarray | None,
) -> RenderResult:
    width = scene.layout.width
    height = scene.layout.height
    pix = np.zeros((width, height, 4), dtype=np.float32)
    pfh = _fog_factor(scene, zpix)
    l_opacity = _outline_opacity(scene, atoms, zpix, atom_buf, bio_buf, precomputed_outline)
    pixel_types = atoms.type_idx[atom_buf]

    for ic in range(3):
        atom_colors = scene.colortype[pixel_types, ic]
        rcolor = pfh * (pconetot * atom_colors) + (1.0 - pfh) * scene.fog_color[ic]
        pix[:, :, ic] = (1.0 - l_opacity) * rcolor

    has_atom = (pixel_types != 0).astype(np.float32)
    pix[:, :, 3] = np.maximum(has_atom, l_opacity)

    rgb = np.swapaxes(_to_u8(pix[:, :, :3] * 255.0), 0, 1)
    opacity = np.swapaxes(_to_u8(pix[:, :, 3] * 255.0), 0, 1)
    return RenderResult(rgb=rgb, opacity=opacity, width=width, height=height)


def render_program(
    program: CommandProgram,
    atoms: AtomTable,
    *,
    backend_name: str,
    sphere_lookup: Callable[[float], np.ndarray],
) -> RenderResult:
    scene = prepare_scene(program, atoms)
    width = scene.layout.width
    height = scene.layout.height
    if width <= 0 or height <= 0:
        rgb_empty = np.zeros((0, 0, 3), dtype=np.uint8)
        op_empty = np.zeros((0, 0), dtype=np.uint8)
        return RenderResult(rgb=rgb_empty, opacity=op_empty, width=width, height=height)

    buffers = _initialize_backend_buffers(backend_name, width, height)
    _rasterize_atoms(scene, atoms, buffers, sphere_lookup)
    precomputed_outline = _precompute_outline(scene, atoms, buffers)
    pconetot = _shadow_mask(scene, buffers)

    fast_result = _render_precomputed_outline(scene, atoms, buffers, pconetot, precomputed_outline)
    if fast_result is not None:
        return fast_result

    zpix, atom_buf, bio_buf, pconetot_np, outline_np = _materialize_numpy(buffers, pconetot, precomputed_outline)
    return _compose_numpy(scene, atoms, zpix, atom_buf, bio_buf, pconetot_np, outline_np)
