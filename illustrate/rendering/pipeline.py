from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import threading
from typing import Literal, Protocol, cast
import weakref

import numpy as np
from numpy.typing import NDArray

from illustrate.rendering.kernels import (
    ArrayBuffer,
    DeviceArray,
    MlxArray,
    require_mlx,
    run_composite_kernel,
    run_composite_u8_kernel,
    run_raster_kernel,
    run_outline12_kernel,
    run_outline34_kernel,
    run_shadow_kernel,
)
from illustrate.core.types import AtomTable, CommandProgram, OutlineParams, RenderResult, WorldParams

_BACKGROUND_DEPTH = -10000.0
_MAX_RULE_TYPES = 1000
_RULE_TABLE_SIZE = _MAX_RULE_TYPES + 1
MAX_IMAGE_DIMENSION = 3000
# Largest atom radius in pixels (rule radius x scale) the sphere stamps support.
MAX_SCALED_ATOM_RADIUS = 100
_NUMPY_CHUNK_ATOMS = 256
_NUMPY_RASTER_ELEMS_BUDGET = 524_288
_GPU_CHUNK_ATOMS = 512
_GPU_CHUNK_MIN_ATOMS = 128
_GPU_CHUNK_MAX_ATOMS = 4096
_GPU_RASTER_ELEMS_BUDGET = 8_000_000
_BACKEND_ARRAY_CACHE_LIMIT = 32

_BackendArrayCacheKey = tuple[str, int, object, np.dtype[np.generic], tuple[int, ...]]
_BackendArrayCacheValue = tuple[weakref.ReferenceType[NDArray[np.generic]], DeviceArray]


class _DeviceBufferOps(Protocol):
    @property
    def float32(self) -> object: ...
    @property
    def int32(self) -> object: ...

    def full(self, shape: tuple[int, int], value: float, /, *, dtype: object) -> DeviceArray: ...
    def zeros(self, shape: tuple[int, int], /, *, dtype: object) -> DeviceArray: ...
    def ones(self, shape: tuple[int, int], /, *, dtype: object) -> DeviceArray: ...
    def minimum(self, array: ArrayBuffer, limit: float, /) -> DeviceArray: ...


class _MlxBufferOps(_DeviceBufferOps, Protocol):
    @property
    def uint8(self) -> object: ...

    def array(self, values: NDArray[np.generic] | np.float32, /, *, dtype: object) -> DeviceArray: ...
    def ones(self, shape: tuple[int, int], /, *, dtype: object) -> MlxArray: ...
    def clip(self, array: ArrayBuffer, low: DeviceArray, high: DeviceArray, /) -> DeviceArray: ...
    def eval(self, *arrays: ArrayBuffer) -> None: ...
_BACKEND_ARRAY_CACHE: OrderedDict[_BackendArrayCacheKey, _BackendArrayCacheValue] = OrderedDict()
_BACKEND_ARRAY_CACHE_LOCK = threading.Lock()


class RenderCancellationToken(Protocol):
    def is_cancelled(self) -> bool:
        ...


class RenderCancelled(RuntimeError):
    """Raised when cooperative render cancellation stops stale work."""


def _check_cancelled(cancellation_token: RenderCancellationToken | None) -> None:
    if cancellation_token is not None and cancellation_token.is_cancelled():
        raise RenderCancelled("render cancelled")


def _gpu_chunk_atom_count(sample_count: int) -> int:
    if sample_count <= 0:
        return _GPU_CHUNK_ATOMS
    budget_count = _GPU_RASTER_ELEMS_BUDGET // int(sample_count)
    return max(_GPU_CHUNK_MIN_ATOMS, min(_GPU_CHUNK_MAX_ATOMS, int(budget_count)))


@dataclass(slots=True)
class SceneLayout:
    # Sizes of the [x, y] raster buffers, Fortran's ixsize and iysize. As in
    # the original Illustrate, x runs down the output image and y across it,
    # so ``width`` is the image height in rows and ``height`` its width.
    width: int
    height: int
    n: int
    nbiomat: int
    biomats: NDArray[np.float32]
    coords_arr: NDArray[np.float32]
    rm: NDArray[np.float32]
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
    background: NDArray[np.float32]
    fog_color: NDArray[np.float32]
    fog_front: float
    fog_back: float
    colortype: NDArray[np.float32]
    radtype: NDArray[np.float32]
    ndes: int


@dataclass(slots=True)
class BackendBuffers:
    backend_name: str
    zpix: ArrayBuffer
    atom_buf: ArrayBuffer
    bio_buf: ArrayBuffer
    mlx_mod: _MlxBufferOps | None = None
    shadow_bounds: tuple[int, int, int, int] | None = None


def _cached_backend_array(
    buffers: BackendBuffers,
    values: NDArray[np.generic],
    *,
    cpu_dtype: np.dtype[np.generic] | type[np.generic],
    backend_dtype_attr: Literal["float32", "int32"],
    cache: bool = False,
    cache_owner: NDArray[np.generic] | None = None,
    cache_part: object = None,
) -> ArrayBuffer:
    """Convert values to the active backend, optionally reusing a cached upload.

    Supply ``cache_owner`` and ``cache_part`` to identify a view of a
    longer-lived array, or set ``cache=True`` to key by the converted NumPy
    source. Cache hits check the owner's identity through a weak reference.
    Cached data must stay unchanged since the key tracks identity, not contents.
    """
    source = np.asarray(values, dtype=cpu_dtype)
    if buffers.mlx_mod is not None:
        backend_mod = buffers.mlx_mod
        upload = backend_mod.array
    else:
        return source

    backend_dtype = backend_mod.float32 if backend_dtype_attr == "float32" else backend_mod.int32
    owner = cache_owner if cache_owner is not None else (source if cache else None)
    if owner is None:
        return upload(source, dtype=backend_dtype)

    key: _BackendArrayCacheKey = (
        buffers.backend_name,
        id(owner),
        cache_part,
        source.dtype,
        tuple(source.shape),
    )
    with _BACKEND_ARRAY_CACHE_LOCK:
        cached = _BACKEND_ARRAY_CACHE.get(key)
        if cached is not None:
            owner_ref, device_array = cached
            if owner_ref() is owner:
                _BACKEND_ARRAY_CACHE.move_to_end(key)
                return device_array
            del _BACKEND_ARRAY_CACHE[key]

    device_array = upload(source, dtype=backend_dtype)
    with _BACKEND_ARRAY_CACHE_LOCK:
        cached = _BACKEND_ARRAY_CACHE.get(key)
        if cached is not None:
            owner_ref, existing_array = cached
            if owner_ref() is owner:
                _BACKEND_ARRAY_CACHE.move_to_end(key)
                return existing_array
        _BACKEND_ARRAY_CACHE[key] = (weakref.ref(owner), device_array)
        while len(_BACKEND_ARRAY_CACHE) > _BACKEND_ARRAY_CACHE_LIMIT:
            _BACKEND_ARRAY_CACHE.popitem(last=False)
    return device_array


def _backend_float_array(
    buffers: BackendBuffers,
    values: NDArray[np.generic],
    *,
    cache: bool = False,
    cache_owner: NDArray[np.generic] | None = None,
    cache_part: object = None,
) -> ArrayBuffer:
    return _cached_backend_array(
        buffers,
        values,
        cpu_dtype=np.float32,
        backend_dtype_attr="float32",
        cache=cache,
        cache_owner=cache_owner,
        cache_part=cache_part,
    )


def _backend_int_array(
    buffers: BackendBuffers,
    values: NDArray[np.generic],
    *,
    cache: bool = False,
    cache_owner: NDArray[np.generic] | None = None,
    cache_part: object = None,
) -> ArrayBuffer:
    return _cached_backend_array(
        buffers,
        values,
        cpu_dtype=np.int32,
        backend_dtype_attr="int32",
        cache=cache,
        cache_owner=cache_owner,
        cache_part=cache_part,
    )


def _build_rule_arrays(program: CommandProgram) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    # Atoms matched by rules past the table would never be drawn; reject them.
    if len(program.selection_rules) > _MAX_RULE_TYPES:
        raise ValueError(
            f"at most {_MAX_RULE_TYPES} selection rules are supported, "
            f"got {len(program.selection_rules)}"
        )
    colortype = np.full((_RULE_TABLE_SIZE, 3), 0.5, dtype=np.float32)
    radtype = np.zeros(_RULE_TABLE_SIZE, dtype=np.float32)
    for idx, rule in enumerate(program.selection_rules, start=1):
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


def _build_biomats(atoms: AtomTable) -> tuple[int, NDArray[np.float32]]:
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


def _scale_rule_radii(radtype: NDArray[np.float32], scale: float, ndes: int) -> float:
    if ndes <= 0:
        return 0.0
    scaled = radtype[1 : ndes + 1]
    scaled *= np.float32(scale)
    return float(scaled.max()) if scaled.size else 0.0


def _prepare_layout(program: CommandProgram, atoms: AtomTable, radtype: NDArray[np.float32], ndes: int) -> SceneLayout:
    transform = program.transform
    world = program.world

    # x spans the image rows and y its columns (see SceneLayout).
    ixsize = min(int(world.height), MAX_IMAGE_DIMENSION)
    iysize = min(int(world.width), MAX_IMAGE_DIMENSION)

    nbiomat, biomats = _build_biomats(atoms)
    n = int(atoms.n)
    rm = transform.rm.astype(np.float32)
    rscale = float(transform.scale)
    xtran = float(transform.xtran)
    ytran = float(transform.ytran)
    ztran = float(transform.ztran)
    radius_max = _scale_rule_radii(radtype, rscale, ndes)
    if int(radius_max) > MAX_SCALED_ATOM_RADIUS:
        largest_radius = radius_max / rscale
        raise ValueError(
            "Scale is too high for these atom radii. "
            f"At scale {rscale:g}, {largest_radius:g} Å atoms are {radius_max:.0f} px in radius, "
            f"and the limit is {MAX_SCALED_ATOM_RADIUS} px: use a scale of at most "
            f"{MAX_SCALED_ATOM_RADIUS / largest_radius:.2f} or smaller radii."
        )
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
    # Unknown kernels would silently drop the depth outline; reject them instead.
    if program.outlines.enabled and program.outlines.kernel not in (1, 2, 3, 4):
        raise ValueError(f"outlines.kernel must be 1, 2, 3, or 4, got {program.outlines.kernel}")
    colortype, radtype = _build_rule_arrays(program)
    ndes = min(len(program.selection_rules), _MAX_RULE_TYPES)

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
    ndes = min(len(program.selection_rules), _MAX_RULE_TYPES)
    layout = _prepare_layout(program, atoms, radtype, ndes)
    # (image width, image height); see SceneLayout.
    return (layout.height, layout.width)


def _initialize_backend_buffers(backend_name: str, width: int, height: int) -> BackendBuffers:
    if backend_name == "mlx":
        mx = cast(_MlxBufferOps, require_mlx())
        return BackendBuffers(
            backend_name=backend_name,
            zpix=mx.full((width, height), _BACKGROUND_DEPTH, dtype=mx.float32),
            atom_buf=mx.zeros((width, height), dtype=mx.int32),
            bio_buf=mx.ones((width, height), dtype=mx.int32),
            mlx_mod=mx,
        )

    return BackendBuffers(
        backend_name=backend_name,
        zpix=np.full((width, height), _BACKGROUND_DEPTH, dtype=np.float32),
        atom_buf=np.zeros((width, height), dtype=np.int32),
        bio_buf=np.ones((width, height), dtype=np.int32),
    )


def _rasterize_atoms(
    scene: PreparedScene,
    atoms: AtomTable,
    buffers: BackendBuffers,
    sphere_lookup: Callable[[float], NDArray[np.float32]],
    cancellation_token: RenderCancellationToken | None = None,
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
        _check_cancelled(cancellation_token)
        sphere = sphere_lookup(float(scene.radtype[irad]))
        if len(sphere) == 0:
            continue

        matching = np.where(atom_types == irad)[0]
        if len(matching) == 0:
            continue

        nv = len(sphere)
        if buffers.backend_name == "numpy":
            # Large sphere stamps need smaller batches to bound scratch space
            # and keep cancellation checkpoints close together.
            chunk_atoms = min(_NUMPY_CHUNK_ATOMS, max(1, _NUMPY_RASTER_ELEMS_BUDGET // nv))
        else:
            chunk_atoms = _gpu_chunk_atom_count(nv)
        sx_min = float(np.min(sphere[:, 0]))
        sx_max = float(np.max(sphere[:, 0]))
        sy_min = float(np.min(sphere[:, 1]))
        sy_max = float(np.max(sphere[:, 1]))
        sx = _backend_float_array(buffers, sphere[:, 0], cache_owner=sphere, cache_part=0)
        sy = _backend_float_array(buffers, sphere[:, 1], cache_owner=sphere, cache_part=1)
        sz = _backend_float_array(buffers, sphere[:, 2], cache_owner=sphere, cache_part=2)

        for ibio in range(1, layout.nbiomat + 1):
            _check_cancelled(cancellation_token)
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

            for chunk_start in range(0, len(vis_idx), chunk_atoms):
                _check_cancelled(cancellation_token)
                chunk_end = min(chunk_start + chunk_atoms, len(vis_idx))
                c_cx = cx_vis[chunk_start:chunk_end]
                c_cy = cy_vis[chunk_start:chunk_end]
                c_cz = cz_vis[chunk_start:chunk_end]
                c_ia = ia_vis[chunk_start:chunk_end]
                buffers.zpix, buffers.atom_buf, buffers.bio_buf = run_raster_kernel(
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


def _outline_input(zpix: ArrayBuffer, buffers: BackendBuffers) -> ArrayBuffer:
    if buffers.mlx_mod is not None:
        return buffers.mlx_mod.minimum(zpix, 0.0)
    return np.minimum(np.asarray(zpix, dtype=np.float32), 0.0)


def _precompute_outline(scene: PreparedScene, atoms: AtomTable, buffers: BackendBuffers) -> ArrayBuffer | None:
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


def _shadow_mask(
    scene: PreparedScene, buffers: BackendBuffers, cancellation_token: RenderCancellationToken | None = None
) -> ArrayBuffer:
    width = scene.layout.width
    height = scene.layout.height
    if scene.world.shadows:
        bounds = buffers.shadow_bounds
        if bounds is not None:
            x0, x1, y0, y1 = bounds
            if x0 >= x1 or y0 >= y1:
                bounds = None
        if bounds is None:
            if buffers.mlx_mod is not None:
                return buffers.mlx_mod.ones((width, height), dtype=buffers.mlx_mod.float32)
            return np.ones((width, height), dtype=np.float32)

        x0, x1, y0, y1 = bounds
        render_shadow = run_shadow_kernel
        if cancellation_token is not None:
            render_shadow = partial(run_shadow_kernel, partial(_check_cancelled, cancellation_token))
        if x0 == 0 and x1 == width and y0 == 0 and y1 == height:
            return render_shadow(
                backend=buffers.backend_name,
                zpix=buffers.zpix,
                atom_buf=buffers.atom_buf,
                shadow_strength=float(scene.world.shadow_strength),
                shadow_angle=float(scene.world.shadow_angle),
                shadow_min_z=float(scene.world.shadow_min_z),
                shadow_max_dark=float(scene.world.shadow_max_dark),
            )

        shadow_crop = render_shadow(
            backend=buffers.backend_name,
            zpix=buffers.zpix[x0:x1, y0:y1],
            atom_buf=buffers.atom_buf[x0:x1, y0:y1],
            shadow_strength=float(scene.world.shadow_strength),
            shadow_angle=float(scene.world.shadow_angle),
            shadow_min_z=float(scene.world.shadow_min_z),
            shadow_max_dark=float(scene.world.shadow_max_dark),
        )
        if buffers.mlx_mod is not None:
            mx = buffers.mlx_mod
            full_shadow = mx.ones((width, height), dtype=mx.float32)
            crop_delta = mx.ones((x1 - x0, y1 - y0), dtype=mx.float32) - shadow_crop
            return full_shadow.at[x0:x1, y0:y1].subtract(crop_delta)
        full_shadow = np.ones((width, height), dtype=np.float32)
        full_shadow[x0:x1, y0:y1] = np.asarray(shadow_crop, dtype=np.float32)
        return full_shadow

    if buffers.mlx_mod is not None:
        return buffers.mlx_mod.ones((width, height), dtype=buffers.mlx_mod.float32)
    return np.ones((width, height), dtype=np.float32)


def _image_result(rgb: NDArray[np.uint8], opacity: NDArray[np.uint8], layout: SceneLayout) -> RenderResult:
    """Wrap ``[x, y]`` buffers as the output image, without transposing.

    Rows follow x and columns follow y, as in the original Illustrate. Swapping
    the axes here would mirror the structure and flip its handedness.
    """
    return RenderResult(rgb=rgb, opacity=opacity, width=layout.height, height=layout.width)


def _render_precomputed_outline(
    scene: PreparedScene,
    atoms: AtomTable,
    buffers: BackendBuffers,
    pconetot: ArrayBuffer,
    precomputed_outline: ArrayBuffer | None,
) -> RenderResult:
    outline_opacity = precomputed_outline
    if outline_opacity is None:
        # prepare_scene rejects outlines.enabled with an unsupported kernel, so
        # _precompute_outline always returns an array once outlines are on.
        if scene.outlines.enabled:
            raise RuntimeError("outlines are enabled but no outline buffer was computed")
        width = scene.layout.width
        height = scene.layout.height
        if buffers.mlx_mod is not None:
            outline_opacity = buffers.mlx_mod.zeros((width, height), dtype=buffers.mlx_mod.float32)
        else:
            outline_opacity = np.zeros((width, height), dtype=np.float32)

    type_lookup = _backend_int_array(buffers, atoms.type_idx, cache=True)
    color_lut = _backend_float_array(buffers, scene.colortype)
    fog_color = _backend_float_array(buffers, scene.fog_color.astype(np.float32, copy=False))

    if buffers.mlx_mod is None:
        rgb, opacity = run_composite_u8_kernel(
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
            zbuf_bg=float(_BACKGROUND_DEPTH),
        )
        return _image_result(rgb, opacity, scene.layout)

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
        zbuf_bg=float(_BACKGROUND_DEPTH),
    )
    mx = buffers.mlx_mod
    scale = mx.array(np.float32(255.0), dtype=mx.float32)
    zero = mx.array(np.float32(0.0), dtype=mx.float32)
    hi = mx.array(np.float32(255.0), dtype=mx.float32)
    rgb_u8 = mx.clip(cast(DeviceArray, rgb_linear) * scale, zero, hi).astype(mx.uint8)
    alpha_u8 = mx.clip(cast(DeviceArray, alpha_linear) * scale, zero, hi).astype(mx.uint8)
    mx.eval(rgb_u8, alpha_u8)
    rgb = np.array(rgb_u8, dtype=np.uint8)
    opacity = np.array(alpha_u8, dtype=np.uint8)
    return _image_result(rgb, opacity, scene.layout)


def render_program(
    program: CommandProgram,
    atoms: AtomTable,
    *,
    backend_name: str,
    sphere_lookup: Callable[[float], NDArray[np.float32]],
    cancellation_token: RenderCancellationToken | None = None,
) -> RenderResult:
    scene = prepare_scene(program, atoms)
    _check_cancelled(cancellation_token)
    width = scene.layout.width
    height = scene.layout.height
    if width <= 0 or height <= 0:
        rgb_empty = np.zeros((0, 0, 3), dtype=np.uint8)
        op_empty = np.zeros((0, 0), dtype=np.uint8)
        return _image_result(rgb_empty, op_empty, scene.layout)

    buffers = _initialize_backend_buffers(backend_name, width, height)
    _rasterize_atoms(scene, atoms, buffers, sphere_lookup, cancellation_token)
    _check_cancelled(cancellation_token)
    precomputed_outline = _precompute_outline(scene, atoms, buffers)
    _check_cancelled(cancellation_token)
    pconetot = _shadow_mask(scene, buffers, cancellation_token)
    _check_cancelled(cancellation_token)

    result = _render_precomputed_outline(scene, atoms, buffers, pconetot, precomputed_outline)
    _check_cancelled(cancellation_token)
    return result
