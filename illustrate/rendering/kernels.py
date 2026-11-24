from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import importlib
import importlib.util
import os
from threading import Lock
from types import ModuleType
from typing import Final, Protocol, TypedDict, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Unpack


class DeviceArray(Protocol):
    """Array operations the renderer uses on MLX device arrays."""

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def dtype(self) -> object: ...

    def __getitem__(self, index: object, /) -> DeviceArray: ...
    def __setitem__(self, index: object, value: object, /) -> None: ...
    def __mul__(self, other: object, /) -> DeviceArray: ...
    def __sub__(self, other: object, /) -> DeviceArray: ...
    def astype(self, dtype: object, /) -> DeviceArray: ...
    def __add__(self, other: object, /) -> DeviceArray: ...
    def __rsub__(self, other: object, /) -> DeviceArray: ...
    def __truediv__(self, other: object, /) -> DeviceArray: ...
    def __and__(self, other: object, /) -> DeviceArray: ...
    def __or__(self, other: object, /) -> DeviceArray: ...
    def __lt__(self, other: object, /) -> DeviceArray: ...
    def __le__(self, other: object, /) -> DeviceArray: ...
    def __gt__(self, other: object, /) -> DeviceArray: ...
    def __ge__(self, other: object, /) -> DeviceArray: ...
    def reshape(self, shape: tuple[int, ...], /) -> DeviceArray: ...


class _MlxIndexedUpdate(Protocol):
    def maximum(self, value: DeviceArray, /) -> MlxArray: ...
    def subtract(self, value: DeviceArray, /) -> MlxArray: ...


class _MlxArrayAt(Protocol):
    def __getitem__(self, index: object, /) -> _MlxIndexedUpdate: ...


class MlxArray(DeviceArray, Protocol):
    @property
    def at(self) -> _MlxArrayAt: ...

    def reshape(self, shape: tuple[int, ...], /) -> MlxArray: ...


ArrayBuffer = NDArray[np.generic] | DeviceArray


class _RasterInputs(TypedDict):
    sx: ArrayBuffer
    sy: ArrayBuffer
    sz: ArrayBuffer
    c_cx: ArrayBuffer
    c_cy: ArrayBuffer
    c_cz: ArrayBuffer
    c_ia: ArrayBuffer
    half_ix: float
    half_iy: float
    fix: float
    fiy: float
    nv: int
    ibio: int
    zpix: ArrayBuffer
    atom_buf: ArrayBuffer
    bio_buf: ArrayBuffer


class _ShadowInputs(TypedDict):
    zpix: ArrayBuffer
    atom_buf: ArrayBuffer
    shadow_strength: float
    shadow_angle: float
    shadow_min_z: float
    shadow_max_dark: float


class _Outline12Inputs(TypedDict):
    zpix: ArrayBuffer
    atom_buf: ArrayBuffer
    bio_buf: ArrayBuffer
    su_lookup: ArrayBuffer
    res_lookup: ArrayBuffer
    residue_diff: float
    residue_low: float
    residue_high: float
    subunit_low: float
    subunit_high: float
    contour_low: float
    contour_high: float
    kernel: int


class _Outline34Inputs(_Outline12Inputs):
    z_diff_min: float
    z_diff_max: float


class _CompositeInputs(TypedDict):
    zpix: ArrayBuffer
    atom_buf: ArrayBuffer
    pconetot: ArrayBuffer
    l_opacity: ArrayBuffer
    type_lookup: ArrayBuffer
    colortype: ArrayBuffer
    fog_color: ArrayBuffer
    fog_front: float
    fog_back: float
    zbuf_bg: float


# Constants and static kernels

# Depth of padding and rejected samples: behind the image background, so they never win.
_ZBUF_OUTSIDE_IMAGE: Final[np.float32] = np.float32(-100000.0)
_ZBUF_FAR: Final[np.float32] = np.float32(100000.0)

_SHADOW_RADIUS_PIXELS: Final[int] = 50
_SHADOW_STEP_PIXELS: Final[int] = 5
_SHADOW_RADIUS_SQUARED: Final[int] = _SHADOW_RADIUS_PIXELS * _SHADOW_RADIUS_PIXELS

_MLX_SHADOW_MAX_CHUNK: Final[int] = 32
_MLX_SHADOW_ELEMS_BUDGET: Final[int] = 18_000_000
_NUMPY_SHADOW_TILE_PIXELS: Final[int] = 65_536
_NUMPY_SHADOW_WORKERS: Final[int] = min(4, os.cpu_count() or 1)
_NUMPY_SHADOW_PARALLEL = Lock()
_NUMPY_OUTLINE_TILE_PIXELS: Final[int] = 32_768
_NUMPY_CONTOUR_TILE_PIXELS: Final[int] = 131_072
_NUMPY_OUTLINE_ELEMS_BUDGET: Final[int] = 12_000_000

_LAPLACIAN_SCALE: Final[np.float32] = np.float32(3.0)
_LAPLACE_AVERAGE_THRESHOLD: Final[np.float32] = np.float32(6.0)


_SHADOW_OFFSETS = tuple(
    (di, dj, float(np.sqrt(float(di * di + dj * dj))))
    for di in range(-_SHADOW_RADIUS_PIXELS, _SHADOW_RADIUS_PIXELS + 1, _SHADOW_STEP_PIXELS)
    for dj in range(-_SHADOW_RADIUS_PIXELS, _SHADOW_RADIUS_PIXELS + 1, _SHADOW_STEP_PIXELS)
    if not (di == 0 and dj == 0) and (di * di + dj * dj) <= _SHADOW_RADIUS_SQUARED
)


def _shadow_offsets_by_radius() -> dict[np.float32, list[tuple[int, int]]]:
    offsets: dict[np.float32, list[tuple[int, int]]] = defaultdict(list)
    for di, dj, radius in _SHADOW_OFFSETS:
        offsets[np.float32(radius)].append((di, dj))
    return offsets


_SHADOW_OFFSETS_BY_RADIUS = _shadow_offsets_by_radius()

_RG_OFFSETS = tuple(
    (di, dj)
    for di in range(-2, 3)
    for dj in range(-2, 3)
    if abs(di * dj) != 4
)

_LAPLACE_NEIGHBOR_OFFSETS = tuple((di, dj) for di in range(-1, 2) for dj in range(-1, 2))

_KERNEL3_OFFSETS = tuple((di, dj) for di in range(-1, 2) for dj in range(-1, 2))

_KERNEL4_OFFSETS = tuple(
    (di, dj)
    for di in range(-2, 3)
    for dj in range(-2, 3)
    if abs(di * dj) != 4
)

_KERNEL1_LAP_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

_KERNEL1_LAP_WEIGHTS = np.array(
    (-0.8, -1.0, -0.8, -1.0, 7.2, -1.0, -0.8, -1.0, -0.8),
    dtype=np.float32,
)

_KERNEL2_LAP_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
    (2, -1),
    (2, 0),
    (2, 1),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-1, 2),
    (0, 2),
    (1, 2),
    (-1, -2),
    (0, -2),
    (1, -2),
)

_KERNEL2_LAP_WEIGHTS = np.array(
    (
        -0.8,
        -1.0,
        -0.8,
        -1.0,
        8.8,
        -1.0,
        -0.8,
        -1.0,
        -0.8,
        -0.1,
        -0.2,
        -0.1,
        -0.1,
        -0.2,
        -0.1,
        -0.1,
        -0.2,
        -0.1,
        -0.1,
        -0.2,
        -0.1,
    ),
    dtype=np.float32,
)


# Optional backend loading

_OPTIONAL_IMPORT_ERRORS = (ImportError, OSError)


def _module_spec_exists(module_name: str) -> bool:
    parent_name, _, _child_name = module_name.rpartition(".")
    try:
        if parent_name and importlib.util.find_spec(parent_name) is None:
            return False
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=None)
def _import_optional_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


@lru_cache(maxsize=None)
def _module_import_succeeds(module_name: str) -> bool:
    if not _module_spec_exists(module_name):
        return False
    try:
        _import_optional_module(module_name)
    except _OPTIONAL_IMPORT_ERRORS:
        return False
    return True


def clear_optional_module_probe_cache() -> None:
    """Discard cached backend probes and kernels when tests replace GPU modules."""
    _import_optional_module.cache_clear()
    _module_import_succeeds.cache_clear()
    _mlx_array_type.cache_clear()
    _mlx_shadow_chunk_radii.cache_clear()
    _mlx_shadow_offsets.cache_clear()
    _mlx_shadow_metal_kernel.cache_clear()
    _mlx_outline12_lap_weights.cache_clear()
    _mlx_outline34_metal_kernel.cache_clear()


def _require_optional_module(module_name: str, error_message: str) -> ModuleType:
    try:
        return _import_optional_module(module_name)
    except _OPTIONAL_IMPORT_ERRORS as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(error_message) from exc


def require_mlx() -> ModuleType:
    return _require_optional_module("mlx.core", "MLX backend requested but MLX is unavailable")


def require_numba_kernels() -> ModuleType:
    """Import the compiled CPU kernels, which import Numba on first use."""
    return _require_optional_module(
        "illustrate.rendering.numba_kernels", "Numba backend requested but Numba is unavailable"
    )


@lru_cache(maxsize=1)
def _mlx_array_type() -> type[MlxArray]:
    mx = require_mlx()
    return type(cast(MlxArray, mx.array(np.zeros((0,), dtype=np.float32))))


def _as_mlx_array(value: ArrayBuffer, *, dtype: object | None = None) -> MlxArray:
    mx = require_mlx()
    array_type = _mlx_array_type()
    if isinstance(value, array_type):
        if dtype is not None and getattr(value, "dtype", None) != dtype:
            return cast(MlxArray, value.astype(dtype))
        return value
    if dtype is not None:
        return mx.array(value, dtype=dtype)
    return mx.array(value)


# Shared validation and small helpers


def _validate_raster_bounds(shape: tuple[int, ...], fix: float, fiy: float) -> tuple[float, float]:
    if len(shape) != 2:
        raise ValueError(f"raster buffers must be 2D, got shape={shape!r}")

    fix_f = float(fix)
    fiy_f = float(fiy)
    if not np.isfinite(fix_f) or not np.isfinite(fiy_f):
        raise ValueError(f"raster bounds must be finite, got fix={fix}, fiy={fiy}")
    if fix_f <= 0.0 or fiy_f <= 0.0:
        raise ValueError(f"raster bounds must be positive, got fix={fix}, fiy={fiy}")
    if fix_f > float(shape[0]) or fiy_f > float(shape[1]):
        raise ValueError(
            f"raster bounds exceed buffer shape: fix={fix_f}, fiy={fiy_f}, shape={shape}"
        )
    return fix_f, fiy_f


def _validate_outline12_kernel(kernel: int) -> None:
    if kernel not in (1, 2):
        raise ValueError(f"outline12 kernel expects 1 or 2, got {kernel}")


def _validate_outline34_kernel(kernel: int) -> None:
    if kernel not in (3, 4):
        raise ValueError(f"outline34 kernel expects 3 or 4, got {kernel}")


def _scaled_opacity_numpy(x: NDArray[np.float32], low: float, high: float) -> NDArray[np.float32]:
    if high == low:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - low) / (high - low)).clip(0.0, 1.0).astype(np.float32)


def _zero_edges_numpy(x: NDArray[np.float32], border: int) -> NDArray[np.float32]:
    if border <= 0:
        return x
    x[:border, :] = 0.0
    x[-border:, :] = 0.0
    x[:, :border] = 0.0
    x[:, -border:] = 0.0
    return x


def _scaled_opacity_mlx(x: DeviceArray, low: float, high: float) -> DeviceArray:
    mx = require_mlx()
    if high == low:
        return mx.zeros(x.shape, dtype=mx.float32)
    return mx.clip((x - low) / (high - low), 0.0, 1.0).astype(mx.float32)


def _zero_edges_mlx(x: DeviceArray, border: int) -> DeviceArray:
    mx = require_mlx()
    if border <= 0:
        return x
    height, width = x.shape
    rows = mx.arange(height)[:, None]
    cols = mx.arange(width)[None, :]
    mask = (rows < border) | (rows >= (height - border)) | (cols < border) | (cols >= (width - border))
    return mx.where(mask, np.float32(0.0), x)


@lru_cache(maxsize=8)
def _shadow_offset_chunks(chunk_size: int) -> tuple[tuple[tuple[int, int, float], ...], ...]:
    if chunk_size <= 0:
        raise ValueError(f"shadow chunk size must be positive, got {chunk_size}")
    return tuple(
        tuple(_SHADOW_OFFSETS[start : start + chunk_size])
        for start in range(0, len(_SHADOW_OFFSETS), chunk_size)
    )


@lru_cache(maxsize=8)
def _mlx_shadow_chunk_radii(chunk_size: int) -> tuple[DeviceArray, ...]:
    mx = require_mlx()
    return tuple(
        _as_mlx_array(
            np.array([radius for _di, _dj, radius in chunk], dtype=np.float32),
            dtype=mx.float32,
        )[:, None, None]
        for chunk in _shadow_offset_chunks(chunk_size)
    )


@lru_cache(maxsize=2)
def _mlx_outline12_lap_weights(kernel: int) -> DeviceArray:
    mx = require_mlx()
    if kernel == 1:
        return _as_mlx_array(_KERNEL1_LAP_WEIGHTS, dtype=mx.float32)
    if kernel == 2:
        return _as_mlx_array(_KERNEL2_LAP_WEIGHTS, dtype=mx.float32)
    raise ValueError(f"outline12 kernel expects 1 or 2, got {kernel}")


def _raster_winners_numpy(
    zpix: NDArray[np.float32],
    rows: NDArray[np.int32],
    cols: NDArray[np.int32],
    depth: NDArray[np.float32],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Update depth in place and return flat pixel indices and winning sample indices."""
    width = zpix.shape[1]
    if zpix.flags.c_contiguous:
        linear = np.multiply(rows, width, dtype=np.intp)
        linear += cols
        flat_depth = np.reshape(zpix, -1)
        np.maximum.at(flat_depth, linear, depth)
        winners = np.flatnonzero(flat_depth[linear] == depth)
        winning_pixels = linear[winners]
    else:
        np.maximum.at(zpix, (rows, cols), depth)
        winners = np.flatnonzero(zpix[rows, cols] == depth)
        winning_pixels = np.multiply(rows[winners], width, dtype=np.intp)
        winning_pixels += cols[winners]

    # Equal depths use the last sample in raster order. Deduplicate only
    # the winners so we never need a second image-sized scatter buffer.
    pixels, last = np.unique(winning_pixels[::-1], return_index=True)
    return pixels, winners[winners.size - 1 - last]


# Raster kernels


def _raster_chunk_numpy(
    *,
    sx: NDArray[np.float32],
    sy: NDArray[np.float32],
    sz: NDArray[np.float32],
    c_cx: NDArray[np.float32],
    c_cy: NDArray[np.float32],
    c_cz: NDArray[np.float32],
    c_ia: NDArray[np.int32],
    half_ix: float,
    half_iy: float,
    fix: float,
    fiy: float,
    nv: int,
    ibio: int,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    bio_buf: NDArray[np.int32],
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
    del nv

    zpix_np = np.asarray(zpix)
    atom_np = np.asarray(atom_buf)
    bio_np = np.asarray(bio_buf)
    fix_f, fiy_f = _validate_raster_bounds(zpix_np.shape, fix, fiy)

    sx_np = np.asarray(sx)
    sy_np = np.asarray(sy)
    sz_np = np.asarray(sz)
    cx_np = np.asarray(c_cx)
    cy_np = np.asarray(c_cy)
    cz_np = np.asarray(c_cz)
    ia_np = np.asarray(c_ia, dtype=atom_np.dtype)

    px = sx_np[np.newaxis, :] + cx_np[:, np.newaxis]
    py = sy_np[np.newaxis, :] + cy_np[:, np.newaxis]
    px += half_ix
    py += half_iy
    if not px.size:
        return zpix_np, atom_np, bio_np

    valid = None
    if not (px.min() >= 1.0 and px.max() <= fix_f and py.min() >= 1.0 and py.max() <= fiy_f):
        valid = (px >= 1.0) & (px <= fix_f) & (py >= 1.0) & (py <= fiy_f)
        if not valid.any():
            return zpix_np, atom_np, bio_np

    pz = sz_np[np.newaxis, :] + cz_np[:, np.newaxis]
    if valid is not None:
        px, py, pz = px[valid], py[valid], pz[valid]
    rows = px.astype(np.int32).ravel()
    cols = py.astype(np.int32).ravel()
    rows -= 1
    cols -= 1
    pixels, samples = _raster_winners_numpy(
        zpix_np, rows, cols, pz.astype(zpix_np.dtype, copy=False).ravel()
    )
    if not samples.size:
        return zpix_np, atom_np, bio_np
    if valid is not None:
        samples = np.flatnonzero(valid)[samples]
    winning_rows, winning_cols = np.divmod(pixels, zpix_np.shape[1])
    atom_np[winning_rows, winning_cols] = ia_np[samples // sx_np.size]
    bio_np[winning_rows, winning_cols] = ibio
    return zpix_np, atom_np, bio_np


def _raster_chunk_mlx(
    *,
    sx: ArrayBuffer,
    sy: ArrayBuffer,
    sz: ArrayBuffer,
    c_cx: ArrayBuffer,
    c_cy: ArrayBuffer,
    c_cz: ArrayBuffer,
    c_ia: ArrayBuffer,
    half_ix: float,
    half_iy: float,
    fix: float,
    fiy: float,
    nv: int,
    ibio: int,
    zpix: ArrayBuffer,
    atom_buf: ArrayBuffer,
    bio_buf: ArrayBuffer,
) -> tuple[MlxArray, MlxArray, MlxArray]:
    mx = require_mlx()

    zpix_mx = _as_mlx_array(zpix, dtype=mx.float32)
    atom_buf_mx = _as_mlx_array(atom_buf, dtype=mx.int32)
    bio_buf_mx = _as_mlx_array(bio_buf, dtype=mx.int32)
    fix_f, fiy_f = _validate_raster_bounds(zpix_mx.shape, fix, fiy)

    sx_mx = _as_mlx_array(sx, dtype=mx.float32)
    sy_mx = _as_mlx_array(sy, dtype=mx.float32)
    sz_mx = _as_mlx_array(sz, dtype=mx.float32)
    cx_mx = _as_mlx_array(c_cx, dtype=mx.float32)
    cy_mx = _as_mlx_array(c_cy, dtype=mx.float32)
    cz_mx = _as_mlx_array(c_cz, dtype=mx.float32)
    ia_mx = _as_mlx_array(c_ia, dtype=mx.int32)

    all_px = (mx.expand_dims(sx_mx, 0) + mx.expand_dims(cx_mx, 1) + half_ix).reshape((-1,))
    all_py = (mx.expand_dims(sy_mx, 0) + mx.expand_dims(cy_mx, 1) + half_iy).reshape((-1,))
    all_pz = (mx.expand_dims(sz_mx, 0) + mx.expand_dims(cz_mx, 1)).reshape((-1,))

    # Keep the scatter arrays fixed-size by routing rejected samples to pixel
    # (0, 0) with a depth behind the background.
    valid = (all_px >= 1.0) & (all_px <= fix_f) & (all_py >= 1.0) & (all_py <= fiy_f)
    zero_i = mx.zeros(valid.shape, dtype=mx.int32)
    invalid_depth = mx.full(all_pz.shape, _ZBUF_OUTSIDE_IMAGE, dtype=mx.float32)
    ipx_v = mx.where(valid, all_px.astype(mx.int32) - 1, zero_i)
    ipy_v = mx.where(valid, all_py.astype(mx.int32) - 1, zero_i)
    pz_v = mx.where(valid, all_pz, invalid_depth)
    ia_v = mx.repeat(ia_mx, nv).astype(mx.int32)

    width_y = int(zpix_mx.shape[1])
    linear_v = ipx_v * width_y + ipy_v

    zpix_flat = zpix_mx.reshape((-1,))
    zpix_flat = zpix_flat.at[linear_v].maximum(pz_v)

    winners = valid & mx.equal(pz_v, zpix_flat[linear_v])
    rank = mx.where(winners, mx.arange(linear_v.shape[0], dtype=mx.int32) + 1, zero_i)

    atom_flat = atom_buf_mx.reshape((-1,))
    rank_buf = mx.zeros(atom_flat.shape, dtype=mx.int32)
    rank_buf = rank_buf.at[linear_v].maximum(rank)
    touched = rank_buf > 0

    atom_lookup = mx.concatenate([mx.zeros((1,), dtype=mx.int32), ia_v], axis=0)
    atom_updates = mx.take(atom_lookup, rank_buf)
    atom_flat = mx.where(touched, atom_updates, atom_flat)

    bio_flat = bio_buf_mx.reshape((-1,))
    bio_updates = mx.full(rank_buf.shape, ibio, dtype=bio_flat.dtype)
    bio_flat = mx.where(touched, bio_updates, bio_flat)

    zpix_mx = zpix_flat.reshape(zpix_mx.shape)
    atom_buf_mx = atom_flat.reshape(atom_buf_mx.shape)
    bio_buf_mx = bio_flat.reshape(bio_buf_mx.shape)
    return zpix_mx, atom_buf_mx, bio_buf_mx


# Shadow kernels


def _shadow_counts_sparse_numpy(
    depth: NDArray[np.float32],
    has_atom: NDArray[np.bool_],
    padded: NDArray[np.float32],
    min_z: np.float32,
    angle: np.float32,
    checkpoint: Callable[[], None] | None,
) -> NDArray[np.int16]:
    rows, cols = np.nonzero(has_atom)
    pad = _SHADOW_RADIUS_PIXELS
    stride = padded.shape[1]
    centers = (rows + pad) * stride + cols + pad
    neighbors = np.empty_like(centers)
    flat_depth = np.reshape(padded, -1)
    atom_depth = depth[rows, cols]
    min_threshold = atom_depth + min_z
    radius_threshold_base = atom_depth - min_z
    count = np.zeros(atom_depth.shape, dtype=np.int16)
    threshold = np.empty_like(atom_depth)
    hits = np.empty(atom_depth.shape, dtype=np.bool_)
    for radius, offsets in _SHADOW_OFFSETS_BY_RADIUS.items():
        if checkpoint is not None:
            checkpoint()
        np.add(radius_threshold_base, np.multiply(radius, angle), out=threshold)
        np.maximum(threshold, min_threshold, out=threshold)
        for di, dj in offsets:
            # Translating flat indices avoids scanning the full mask for every offset.
            np.add(centers, di * stride + dj, out=neighbors)
            np.greater(flat_depth[neighbors], threshold, out=hits)
            np.add(count, hits, out=count)
    return count


def _shadow_counts_tiled_numpy(
    zpix: NDArray[np.float32],
    has_atom: NDArray[np.bool_],
    padded: NDArray[np.float32],
    min_z: np.float32,
    angle: np.float32,
    checkpoint: Callable[[], None] | None,
) -> NDArray[np.int16]:
    """Count occluders in row strips and return counts for occupied pixels."""
    height, width = zpix.shape
    pad = _SHADOW_RADIUS_PIXELS
    tile_pixels = 2 * _NUMPY_SHADOW_TILE_PIXELS if zpix.size >= 1_000_000 else _NUMPY_SHADOW_TILE_PIXELS
    tile_rows = min(height, max(1, tile_pixels // width))
    counts = np.zeros(zpix.shape, dtype=np.int16)
    radius_deltas = tuple(
        (np.multiply(radius, angle), offsets) for radius, offsets in _SHADOW_OFFSETS_BY_RADIUS.items()
    )
    errors = np.geterr()

    def count_strip(start: int) -> None:
        end = min(start + tile_rows, height)
        columns = np.flatnonzero(has_atom[start:end].any(axis=0))
        if columns.size == 0:
            return
        left, right = int(columns[0]), int(columns[-1]) + 1
        depth = zpix[start:end, left:right]
        threshold = np.empty_like(depth)
        hits = np.empty(depth.shape, dtype=np.bool_)
        count = np.zeros(depth.shape, dtype=np.int16)
        # NumPy's floating-point error mode is local to the calling thread.
        with np.errstate(**errors):
            min_threshold = depth + min_z
            radius_threshold_base = depth - min_z
            for delta, offsets in radius_deltas:
                np.add(radius_threshold_base, delta, out=threshold)
                np.maximum(threshold, min_threshold, out=threshold)
                for di, dj in offsets:
                    neighbor = padded[start + pad + di : end + pad + di, left + pad + dj : right + pad + dj]
                    np.greater(neighbor, threshold, out=hits)
                    np.add(count, hits, out=count)
        counts[start:end, left:right] = count

    starts = range(0, height, tile_rows)
    can_parallelize = len(starts) >= 3 and _NUMPY_SHADOW_WORKERS > 1 and np.geterrcall() is None
    if can_parallelize and _NUMPY_SHADOW_PARALLEL.acquire(blocking=False):
        try:
            # Only one render gets an extra thread team at a time. Join each
            # batch before polling cancellation on the original render thread.
            with ThreadPoolExecutor(
                max_workers=_NUMPY_SHADOW_WORKERS, thread_name_prefix="illustrate-shadow"
            ) as pool:
                for i in range(0, len(starts), _NUMPY_SHADOW_WORKERS):
                    if checkpoint is not None:
                        checkpoint()
                    for _ in pool.map(count_strip, starts[i : i + _NUMPY_SHADOW_WORKERS]):
                        pass
        finally:
            _NUMPY_SHADOW_PARALLEL.release()
    else:
        for start in starts:
            if checkpoint is not None:
                checkpoint()
            count_strip(start)
    return counts[has_atom]


def _shadow_cone_numpy(
    *,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    shadow_strength: float,
    shadow_angle: float,
    shadow_min_z: float,
    shadow_max_dark: float,
    checkpoint: Callable[[], None] | None = None,
) -> NDArray[np.float32]:
    zpix_np = np.asarray(zpix, dtype=np.float32)
    atom_np = np.asarray(atom_buf)
    has_atom = atom_np != 0
    occupied = np.count_nonzero(has_atom)
    pconetot = np.ones(zpix_np.shape, dtype=np.float32)
    if occupied == 0:
        return pconetot

    shadow_pad = _SHADOW_RADIUS_PIXELS
    zpix_shadow_padded = np.pad(
        zpix_np,
        ((shadow_pad, shadow_pad), (shadow_pad, shadow_pad)),
        mode="constant",
        constant_values=_ZBUF_OUTSIDE_IMAGE,
    )
    strength = np.float32(shadow_strength)
    min_z = np.float32(shadow_min_z)
    angle = np.float32(shadow_angle)
    max_dark = np.float32(shadow_max_dark)
    # Parallel strips win at a lower density on large images.
    sparse_divisor = 6 if has_atom.size >= 1_000_000 and _NUMPY_SHADOW_WORKERS > 1 else 4
    if occupied * sparse_divisor < has_atom.size:
        counts = _shadow_counts_sparse_numpy(zpix_np, has_atom, zpix_shadow_padded, min_z, angle, checkpoint)
    else:
        counts = _shadow_counts_tiled_numpy(zpix_np, has_atom, zpix_shadow_padded, min_z, angle, checkpoint)
    shadowed = np.subtract(np.float32(1.0), counts.astype(np.float32) * strength)
    pconetot[has_atom] = np.maximum(shadowed, max_dark)
    return pconetot


@lru_cache(maxsize=1)
def _mlx_shadow_offsets() -> tuple[MlxArray, MlxArray, MlxArray]:
    mx = require_mlx()
    di, dj, radius = zip(*_SHADOW_OFFSETS)
    return (
        mx.array(di, dtype=mx.int32),
        mx.array(dj, dtype=mx.int32),
        mx.array(radius, dtype=mx.float32),
    )


@lru_cache(maxsize=1)
def _mlx_shadow_metal_kernel() -> _MetalOutlineKernel:
    """Build a Metal kernel that samples shadows directly from the depth buffer."""
    mx = require_mlx()
    return mx.fast.metal_kernel(
        name="illustrate_shadow_cone",
        input_names=["zpix", "atom_buf", "di", "dj", "radius", "params"],
        output_names=["visibility"],
        # The pipeline passes cropped views. The kernel indexes contiguous rows.
        ensure_row_contiguous=True,
        source="""
        #pragma clang fp contract(off)
        const uint index = thread_position_in_grid.x;
        if (atom_buf[index] == 0) {
            visibility[index] = 1.0f;
            return;
        }

        const int rows = zpix_shape[0];
        const int cols = zpix_shape[1];
        const int row = index / cols;
        const int col = index % cols;
        const float center = zpix[index];
        const float strength = params[0];
        const float angle = params[1];
        const float min_z = params[2];
        const float max_dark = params[3];
        const float background = params[4];
        int count = 0;
        for (int i = 0; i < di_shape[0]; ++i) {
            const int r = row + di[i];
            const int c = col + dj[i];
            const float neighbor = (r >= 0 && r < rows && c >= 0 && c < cols)
                ? zpix[r * cols + c]
                : background;
            const float depth_diff = neighbor - center;
            count += depth_diff > min_z && radius[i] * angle < depth_diff + min_z;
        }
        visibility[index] = metal::max(1.0f - float(count) * strength, max_dark);
        """,
    )


def _shadow_cone_mlx(
    *,
    zpix: ArrayBuffer,
    atom_buf: ArrayBuffer,
    shadow_strength: float,
    shadow_angle: float,
    shadow_min_z: float,
    shadow_max_dark: float,
) -> DeviceArray:
    mx = require_mlx()

    zpix_mx = _as_mlx_array(zpix, dtype=mx.float32)
    atom_mx = _as_mlx_array(atom_buf, dtype=mx.int32)
    if zpix_mx.ndim != 2 or zpix_mx.shape != atom_mx.shape:
        raise ValueError("MLX shadows require matching 2D depth and atom buffers")
    if zpix_mx.size == 0:
        return zpix_mx

    if mx.default_device().type == mx.gpu and mx.metal.is_available():
        params = mx.array(
            [shadow_strength, shadow_angle, shadow_min_z, shadow_max_dark, float(_ZBUF_OUTSIDE_IMAGE)],
            dtype=mx.float32,
        )
        kernel = _mlx_shadow_metal_kernel()
        return kernel(
            inputs=[zpix_mx, atom_mx, *_mlx_shadow_offsets(), params],
            grid=(zpix_mx.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[zpix_mx.shape],
            output_dtypes=[mx.float32],
        )[0]

    return _shadow_cone_mlx_array_ops(
        zpix_mx, atom_mx, shadow_strength, shadow_angle, shadow_min_z, shadow_max_dark,
    )


def _shadow_cone_mlx_array_ops(
    zpix_mx: MlxArray,
    atom_mx: MlxArray,
    shadow_strength: float,
    shadow_angle: float,
    shadow_min_z: float,
    shadow_max_dark: float,
) -> DeviceArray:
    """Compute shadows on an MLX device without Metal, including the CPU."""
    mx = require_mlx()
    has_atom = mx.not_equal(atom_mx, 0)
    shadow_pad = _SHADOW_RADIUS_PIXELS
    zpix_shadow_padded = mx.pad(
        zpix_mx,
        [(shadow_pad, shadow_pad), (shadow_pad, shadow_pad)],
        mode="constant",
        constant_values=float(_ZBUF_OUTSIDE_IMAGE),
    )
    height, width = zpix_mx.shape
    strength = np.float32(shadow_strength)
    min_z = np.float32(shadow_min_z)
    angle = np.float32(shadow_angle)
    max_dark = np.float32(shadow_max_dark)
    pix_count = max(1, height * width)
    chunk_size = max(1, min(_MLX_SHADOW_MAX_CHUNK, _MLX_SHADOW_ELEMS_BUDGET // pix_count))
    zpix_plane = zpix_mx[None, :, :]
    has_atom_plane = has_atom[None, :, :]
    count = mx.zeros(zpix_mx.shape, dtype=mx.int32)

    for chunk, radii in zip(_shadow_offset_chunks(chunk_size), _mlx_shadow_chunk_radii(chunk_size)):
        shifted_z = mx.stack(
            [
                zpix_shadow_padded[
                    shadow_pad + di : shadow_pad + di + height,
                    shadow_pad + dj : shadow_pad + dj + width,
                ]
                for di, dj, _radius in chunk
            ],
            axis=0,
        )
        rzdiff = shifted_z - zpix_plane
        shadow_mask = has_atom_plane & (rzdiff > min_z) & ((radii * angle) < (rzdiff + min_z))
        count = count + mx.sum(shadow_mask, axis=0).astype(mx.int32)

    pconetot = 1.0 - count.astype(mx.float32) * strength
    pconetot = mx.maximum(pconetot, max_dark)
    pconetot = mx.where(has_atom, pconetot, np.float32(1.0))
    return pconetot


# Residue/subunit outline helper shared by outline12 and outline34


def _residue_subunit_opacity_numpy(
    *,
    atom_buf: NDArray[np.int32],
    bio_buf: NDArray[np.int32],
    su_lookup: NDArray[np.int32],
    res_lookup: NDArray[np.int32],
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
) -> NDArray[np.float32]:
    atom_np = np.asarray(atom_buf, dtype=np.int32)
    bio_np = np.asarray(bio_buf, dtype=np.int32)
    su_lookup_np = np.asarray(su_lookup, dtype=np.int32)
    res_lookup_np = np.asarray(res_lookup, dtype=np.int32)

    height, width = atom_np.shape
    su_map = su_lookup_np[atom_np]
    res_map = res_lookup_np[atom_np].astype(np.float32)
    residue_diff_f = np.float32(residue_diff)

    rg_pad = 2
    # Off-image neighbours copy the border pixel, so the frame is not an outline.
    su_padded = np.pad(su_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="edge")
    bio_padded = np.pad(bio_np, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="edge")
    res_padded = np.pad(res_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="edge")

    tile_rows = min(height, max(1, _NUMPY_OUTLINE_TILE_PIXELS // width))
    tile_shape = (tile_rows, width)
    subunit_counts = np.empty(tile_shape, dtype=np.uint8)
    residue_counts = np.empty(tile_shape, dtype=np.uint8)
    subunit_hits = np.empty(tile_shape, dtype=np.bool_)
    hits_buf = np.empty(tile_shape, dtype=np.bool_)
    diff_buf = np.empty(tile_shape, dtype=np.float32)
    opacity = np.empty((height, width), dtype=np.float32)
    # Scan neighboring slices with strip-sized scratch space. Each count
    # fits in a byte because the stencil contains only 21 offsets.
    for start in range(0, height, tile_rows):
        end = min(start + tile_rows, height)
        rows = end - start
        r_count, g_count = subunit_counts[:rows], residue_counts[:rows]
        changed, hits, diff = subunit_hits[:rows], hits_buf[:rows], diff_buf[:rows]
        r_count.fill(0)
        g_count.fill(0)
        for di, dj in _RG_OFFSETS:
            i0, j0 = rg_pad + di, rg_pad + dj
            neighbor = (slice(start + i0, end + i0), slice(j0, j0 + width))
            np.not_equal(su_map[start:end], su_padded[neighbor], out=changed)
            np.not_equal(bio_np[start:end], bio_padded[neighbor], out=hits)
            np.logical_or(changed, hits, out=hits)
            np.add(r_count, hits, out=r_count)
            np.subtract(res_map[start:end], res_padded[neighbor], out=diff)
            np.abs(diff, out=diff)
            np.greater(diff, residue_diff_f, out=hits)
            np.add(g_count, hits, out=g_count)
        g_opacity = _scaled_opacity_numpy(g_count.astype(np.float32), residue_low, residue_high)
        r_opacity = _scaled_opacity_numpy(r_count.astype(np.float32), subunit_low, subunit_high)
        np.maximum(g_opacity, r_opacity, out=opacity[start:end])
    return _zero_edges_numpy(opacity, border=1)


def _residue_subunit_opacity_mlx(
    *,
    atom_buf: ArrayBuffer,
    bio_buf: ArrayBuffer,
    su_lookup: ArrayBuffer,
    res_lookup: ArrayBuffer,
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
) -> DeviceArray:
    mx = require_mlx()

    atom_mx = _as_mlx_array(atom_buf, dtype=mx.int32)
    bio_mx = _as_mlx_array(bio_buf, dtype=mx.int32)
    su_lookup_mx = _as_mlx_array(su_lookup, dtype=mx.int32)
    res_lookup_mx = _as_mlx_array(res_lookup, dtype=mx.int32)

    height, width = atom_mx.shape
    su_map = mx.take(su_lookup_mx, atom_mx)
    res_map = mx.take(res_lookup_mx, atom_mx)
    res_map_f = res_map.astype(mx.float32)
    residue_diff_f = np.float32(residue_diff)

    r_count = mx.zeros((height, width), dtype=mx.float32)
    g_count = mx.zeros((height, width), dtype=mx.float32)
    rg_pad = 2
    # Off-image neighbours copy the border pixel, so the frame is not an outline.
    su_padded = mx.pad(su_map, [(rg_pad, rg_pad), (rg_pad, rg_pad)], mode="edge")
    bio_padded = mx.pad(bio_mx, [(rg_pad, rg_pad), (rg_pad, rg_pad)], mode="edge")
    res_padded = mx.pad(res_map, [(rg_pad, rg_pad), (rg_pad, rg_pad)], mode="edge")
    res_padded_f = res_padded.astype(mx.float32)

    for di, dj in _RG_OFFSETS:
        i0 = rg_pad + di
        j0 = rg_pad + dj
        shifted_su = su_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_bio = bio_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_res = res_padded_f[i0 : i0 + height, j0 : j0 + width]
        r_mask = mx.not_equal(su_map, shifted_su) | mx.not_equal(bio_mx, shifted_bio)
        r_count = r_count + r_mask.astype(mx.float32)
        g_count = g_count + (mx.abs(res_map_f - shifted_res) > residue_diff_f).astype(mx.float32)

    g_opacity = _scaled_opacity_mlx(g_count, residue_low, residue_high)
    r_opacity = _scaled_opacity_mlx(r_count, subunit_low, subunit_high)
    g_opacity = mx.maximum(g_opacity, r_opacity)
    return _zero_edges_mlx(g_opacity, border=1)


# Outline kernels: kernel 3 / kernel 4


def _add_depth_contours_numpy(
    depth: NDArray[np.float32],
    opacity: NDArray[np.float32],
    offsets: tuple[tuple[int, int], ...],
    z_diff_min: float,
    z_diff_max: float,
    contour_low: float,
    contour_high: float,
) -> None:
    """Merge the depth contours into the interior of the outline opacity plane."""
    height, width = depth.shape
    denom = float(z_diff_max - z_diff_min)
    if min(height, width) <= 4:
        return

    floor = np.float32(z_diff_min)
    limits = np.finfo(np.float32)
    positive_scale = float(limits.tiny) <= denom <= float(limits.max) and np.isfinite(floor)
    if denom == 0.0:
        offsets = ()
    elif floor >= 0.0:
        offsets = tuple(offset for offset in offsets if offset != (0, 0))
    # The two-pixel border has zero depth opacity. Limiting work to the
    # interior also guarantees that every neighbor is already in bounds.
    tile_rows = min(height - 4, max(1, _NUMPY_CONTOUR_TILE_PIXELS // (width - 4)))
    diff_buf = np.empty((tile_rows, width - 4), dtype=np.float32)
    sum_buf = np.empty_like(diff_buf)
    for start in range(2, height - 2, tile_rows):
        end = min(start + tile_rows, height - 2)
        center = depth[start:end, 2:-2]
        diff, total = diff_buf[: end - start], sum_buf[: end - start]
        total.fill(0.0)
        for di, dj in offsets:
            neighbor = depth[start + di : end + di, 2 + dj : width - 2 + dj]
            np.subtract(center, neighbor, out=diff)
            np.abs(diff, out=diff)
            if positive_scale:
                # Rejected differences, including NaN, contribute zero.
                np.subtract(diff, floor, out=diff)
                np.fmax(diff, np.float32(0.0), out=diff)
                np.divide(diff, denom, out=diff)
                np.minimum(diff, np.float32(1.0), out=diff)
                np.add(total, diff, out=total)
            else:
                total += np.where(diff > floor, np.minimum((diff - floor) / denom, 1.0), 0.0).astype(np.float32)
        level = _scaled_opacity_numpy(total, contour_low, contour_high)
        contour = np.where(level > 0.0, np.minimum(level * np.float32(1.5), np.float32(1.0)), np.float32(0.0))
        target = opacity[start:end, 2:-2]
        np.maximum(contour, target, out=target)


def _outline_kernel34_numpy(
    *,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    bio_buf: NDArray[np.int32],
    su_lookup: NDArray[np.int32],
    res_lookup: NDArray[np.int32],
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
    z_diff_min: float,
    z_diff_max: float,
    contour_low: float,
    contour_high: float,
    kernel: int,
) -> NDArray[np.float32]:
    _validate_outline34_kernel(kernel)

    zpix_np = np.asarray(zpix, dtype=np.float32)
    g_opacity = _residue_subunit_opacity_numpy(
        atom_buf=atom_buf,
        bio_buf=bio_buf,
        su_lookup=su_lookup,
        res_lookup=res_lookup,
        residue_diff=residue_diff,
        residue_low=residue_low,
        residue_high=residue_high,
        subunit_low=subunit_low,
        subunit_high=subunit_high,
    )

    offsets_k = _KERNEL3_OFFSETS if kernel == 3 else _KERNEL4_OFFSETS
    _add_depth_contours_numpy(zpix_np, g_opacity, offsets_k, z_diff_min, z_diff_max, contour_low, contour_high)
    return g_opacity


class _MetalOutlineKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: list[DeviceArray],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[object],
    ) -> list[DeviceArray]: ...


@lru_cache(maxsize=1)
def _mlx_outline34_metal_kernel() -> _MetalOutlineKernel:
    """Build a Metal kernel that combines group borders and contours per pixel."""
    mx = require_mlx()
    return cast(_MetalOutlineKernel, mx.fast.metal_kernel(
        name="illustrate_outline34",
        input_names=["depth", "assembly", "subunit", "residue", "params"],
        output_names=["opacity"],
        ensure_row_contiguous=True,
        header="""
        enum OutlineParam {
            residue_diff, residue_low, residue_range, subunit_low, subunit_range,
            depth_min, depth_range, contour_low, contour_range, contour_radius
        };

        float outline_opacity(float count, float low, float range) {
            return range == 0.0f ? 0.0f : clamp(metal::precise::divide(count - low, range), 0.0f, 1.0f);
        }
        """,
        source="""
        #pragma clang fp contract(off)
        const uint index = thread_position_in_grid.x;
        const int rows = depth_shape[0];
        const int cols = depth_shape[1];
        if (index >= uint(rows * cols)) return;
        const int row = index / cols;
        const int col = index % cols;
        // Group borders skip one pixel at the edge. The depth contour skips two.
        if (row == 0 || col == 0 || row == rows - 1 || col == cols - 1) {
            opacity[index] = 0.0f;
            return;
        }
        const bool depth_interior = row >= 2 && col >= 2 && row < rows - 2 && col < cols - 2;
        const int radius = int(params[contour_radius]);
        const int center_subunit = subunit[index];
        const int center_assembly = assembly[index];
        const float center_residue = float(residue[index]);
        const float center_depth = depth[index];
        int subunit_count = 0;
        int residue_count = 0;
        float depth_count = 0.0f;
        for (int di = -2; di <= 2; ++di) {
            for (int dj = -2; dj <= 2; ++dj) {
                // Exclude the four corners of the 5x5 group stencil.
                if (abs(di * dj) == 4) continue;
                // Match edge padding for the group stencil.
                const int r = clamp(row + di, 0, rows - 1);
                const int c = clamp(col + dj, 0, cols - 1);
                const int neighbor = r * cols + c;
                subunit_count += center_subunit != subunit[neighbor] || center_assembly != assembly[neighbor];
                residue_count += abs(center_residue - float(residue[neighbor])) > params[residue_diff];
                if (depth_interior && params[depth_range] != 0.0f && abs(di) <= radius && abs(dj) <= radius) {
                    const float diff = abs(center_depth - depth[neighbor]);
                    if (diff > params[depth_min]) {
                        depth_count += min(metal::precise::divide(diff - params[depth_min], params[depth_range]), 1.0f);
                    }
                }
            }
        }
        const float groups = max(
            outline_opacity(float(subunit_count), params[subunit_low], params[subunit_range]),
            outline_opacity(float(residue_count), params[residue_low], params[residue_range])
        );
        const float contour = depth_interior
            ? min(1.5f * outline_opacity(depth_count, params[contour_low], params[contour_range]), 1.0f)
            : 0.0f;
        opacity[index] = max(groups, contour);
        """,
    ))


def _outline_kernel34_mlx(
    *,
    zpix: ArrayBuffer,
    atom_buf: ArrayBuffer,
    bio_buf: ArrayBuffer,
    su_lookup: ArrayBuffer,
    res_lookup: ArrayBuffer,
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
    z_diff_min: float,
    z_diff_max: float,
    contour_low: float,
    contour_high: float,
    kernel: int,
) -> DeviceArray:
    _validate_outline34_kernel(kernel)
    mx = require_mlx()

    zpix_mx = _as_mlx_array(zpix, dtype=mx.float32)
    height, width = zpix_mx.shape
    atom_mx = _as_mlx_array(atom_buf, dtype=mx.int32)
    bio_mx = _as_mlx_array(bio_buf, dtype=mx.int32)
    if atom_mx.shape != zpix_mx.shape or bio_mx.shape != zpix_mx.shape:
        raise ValueError("MLX outlines require matching depth, atom, and assembly buffer shapes")
    if zpix_mx.size == 0:
        return mx.zeros((height, width), dtype=mx.float32)

    denom = float(z_diff_max - z_diff_min)
    if mx.default_device().type == mx.gpu and mx.metal.is_available():
        # Gather with MLX before local Metal reads to preserve the backend's
        # index semantics. The Metal kernel only indexes matching image planes.
        subunit = mx.take(_as_mlx_array(su_lookup, dtype=mx.int32), atom_mx)
        residue = mx.take(_as_mlx_array(res_lookup, dtype=mx.int32), atom_mx)
        # Match the OutlineParam enum in the Metal kernel.
        params = mx.array([
            residue_diff, residue_low, residue_high - residue_low,
            subunit_low, subunit_high - subunit_low, z_diff_min, denom,
            contour_low, contour_high - contour_low, 1 if kernel == 3 else 2,
        ], dtype=mx.float32)
        return _mlx_outline34_metal_kernel()(
            inputs=[zpix_mx, bio_mx, subunit, residue, params],
            grid=(height * width, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(height, width)],
            output_dtypes=[mx.float32],
        )[0]

    g_opacity = _residue_subunit_opacity_mlx(
        atom_buf=atom_mx,
        bio_buf=bio_mx,
        su_lookup=su_lookup,
        res_lookup=res_lookup,
        residue_diff=residue_diff,
        residue_low=residue_low,
        residue_high=residue_high,
        subunit_low=subunit_low,
        subunit_high=subunit_high,
    )

    offsets_k = _KERNEL3_OFFSETS if kernel == 3 else _KERNEL4_OFFSETS
    zpad = 2
    zpix_padded = mx.pad(zpix_mx, [(zpad, zpad), (zpad, zpad)], mode="constant", constant_values=0.0)
    l_total = mx.zeros((height, width), dtype=mx.float32)
    z_diff_min_f = np.float32(z_diff_min)

    if denom != 0.0:
        for di, dj in offsets_k:
            i0 = zpad + di
            j0 = zpad + dj
            shifted_z = zpix_padded[i0 : i0 + height, j0 : j0 + width]
            rd = mx.abs(zpix_mx - shifted_z)
            rd_norm = mx.where(rd > z_diff_min_f, mx.minimum((rd - z_diff_min_f) / denom, 1.0), 0.0)
            l_total = l_total + rd_norm.astype(mx.float32)

    l_val = _scaled_opacity_mlx(l_total, contour_low, contour_high)
    # Keep the MLX array on the left. A NumPy scalar on the left materializes
    # the entire opacity plane on the host before MLX can finish the render.
    l_opacity = mx.where(l_val > 0.0, mx.minimum(l_val * 1.5, 1.0), 0.0)
    l_opacity = _zero_edges_mlx(l_opacity.astype(mx.float32), border=2)
    return mx.maximum(l_opacity, g_opacity)


# Outline kernels: kernel 1 / kernel 2


def _outline_kernel12_numpy(
    *,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    bio_buf: NDArray[np.int32],
    su_lookup: NDArray[np.int32],
    res_lookup: NDArray[np.int32],
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
    contour_low: float,
    contour_high: float,
    kernel: int,
) -> NDArray[np.float32]:
    _validate_outline12_kernel(kernel)

    zpix_np = np.asarray(zpix, dtype=np.float32)
    height, width = zpix_np.shape
    pix_count = max(1, height * width)
    g_opacity = _residue_subunit_opacity_numpy(
        atom_buf=atom_buf,
        bio_buf=bio_buf,
        su_lookup=su_lookup,
        res_lookup=res_lookup,
        residue_diff=residue_diff,
        residue_low=residue_low,
        residue_high=residue_high,
        subunit_low=subunit_low,
        subunit_high=subunit_high,
    )

    if kernel == 1:
        lap_offsets = _KERNEL1_LAP_OFFSETS
        lap_weights = _KERNEL1_LAP_WEIGHTS
    else:
        lap_offsets = _KERNEL2_LAP_OFFSETS
        lap_weights = _KERNEL2_LAP_WEIGHTS

    zpad = 2
    zpix_padded = np.pad(zpix_np, ((zpad, zpad), (zpad, zpad)), mode="constant", constant_values=0.0)
    lap = np.zeros((height, width), dtype=np.float32)
    lap_chunk = max(1, min(len(lap_offsets), _NUMPY_OUTLINE_ELEMS_BUDGET // max(1, pix_count)))

    for start in range(0, len(lap_offsets), lap_chunk):
        offsets = lap_offsets[start : start + lap_chunk]
        weights = lap_weights[start : start + lap_chunk]
        shifted_z = np.stack(
            [zpix_padded[zpad + di : zpad + di + height, zpad + dj : zpad + dj + width] for di, dj in offsets],
            axis=0,
        ).astype(np.float32)
        lap += np.sum(shifted_z * weights[:, None, None], axis=0, dtype=np.float32)

    lap = np.abs(lap / _LAPLACIAN_SCALE)

    rl = np.zeros((height, width), dtype=np.float32)
    l_opacity_ave = np.zeros((height, width), dtype=np.float32)
    l_center = np.zeros((height, width), dtype=np.float32)
    lap_pad = 1
    lap_padded = np.pad(lap, ((lap_pad, lap_pad), (lap_pad, lap_pad)), mode="constant", constant_values=0.0)
    denom = float(contour_high - contour_low)

    for di, dj in _LAPLACE_NEIGHBOR_OFFSETS:
        shifted = lap_padded[lap_pad + di : lap_pad + di + height, lap_pad + dj : lap_pad + dj + width]
        if denom != 0.0:
            l_v = ((shifted - contour_low) / denom).clip(0.0, 1.0).astype(np.float32)
        else:
            l_v = np.zeros((height, width), dtype=np.float32)
        rl += (l_v > 0).astype(np.float32)
        l_opacity_ave += l_v
        if di == 0 and dj == 0:
            l_center = l_v

    l_opacity = np.where(rl >= _LAPLACE_AVERAGE_THRESHOLD, l_opacity_ave / _LAPLACE_AVERAGE_THRESHOLD, l_center)
    l_opacity = l_opacity.clip(0.0, 1.0).astype(np.float32)
    l_opacity = _zero_edges_numpy(l_opacity, border=2)
    return np.maximum(l_opacity, g_opacity)


def _outline_kernel12_mlx(
    *,
    zpix: ArrayBuffer,
    atom_buf: ArrayBuffer,
    bio_buf: ArrayBuffer,
    su_lookup: ArrayBuffer,
    res_lookup: ArrayBuffer,
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
    contour_low: float,
    contour_high: float,
    kernel: int,
) -> DeviceArray:
    _validate_outline12_kernel(kernel)
    mx = require_mlx()

    zpix_mx = _as_mlx_array(zpix, dtype=mx.float32)
    height, width = zpix_mx.shape
    g_opacity = _residue_subunit_opacity_mlx(
        atom_buf=atom_buf,
        bio_buf=bio_buf,
        su_lookup=su_lookup,
        res_lookup=res_lookup,
        residue_diff=residue_diff,
        residue_low=residue_low,
        residue_high=residue_high,
        subunit_low=subunit_low,
        subunit_high=subunit_high,
    )

    if kernel == 1:
        lap_offsets = _KERNEL1_LAP_OFFSETS
        lap_weights = _mlx_outline12_lap_weights(1)
    else:
        lap_offsets = _KERNEL2_LAP_OFFSETS
        lap_weights = _mlx_outline12_lap_weights(2)

    zpad = 2
    zpix_padded = mx.pad(zpix_mx, [(zpad, zpad), (zpad, zpad)], mode="constant", constant_values=0.0)
    lap = mx.zeros((height, width), dtype=mx.float32)
    for idx, (di, dj) in enumerate(lap_offsets):
        shifted_z = zpix_padded[zpad + di : zpad + di + height, zpad + dj : zpad + dj + width]
        lap = lap + shifted_z * lap_weights[idx]
    lap = mx.abs(lap / np.float32(_LAPLACIAN_SCALE))

    denom = float(contour_high - contour_low)
    if denom != 0.0:
        rl = mx.zeros((height, width), dtype=mx.float32)
        l_opacity_ave = mx.zeros((height, width), dtype=mx.float32)
        l_center = mx.zeros((height, width), dtype=mx.float32)
        lap_pad = 1
        lap_padded = mx.pad(lap, [(lap_pad, lap_pad), (lap_pad, lap_pad)], mode="constant", constant_values=0.0)

        for di, dj in _LAPLACE_NEIGHBOR_OFFSETS:
            shifted = lap_padded[lap_pad + di : lap_pad + di + height, lap_pad + dj : lap_pad + dj + width]
            l_v = mx.clip((shifted - contour_low) / denom, 0.0, 1.0).astype(mx.float32)
            rl = rl + (l_v > 0).astype(mx.float32)
            l_opacity_ave = l_opacity_ave + l_v
            if di == 0 and dj == 0:
                l_center = l_v

        l_opacity = mx.where(
            rl >= np.float32(_LAPLACE_AVERAGE_THRESHOLD),
            l_opacity_ave / np.float32(_LAPLACE_AVERAGE_THRESHOLD),
            l_center,
        )
        l_opacity = mx.clip(l_opacity, 0.0, 1.0).astype(mx.float32)
    else:
        l_opacity = mx.zeros((height, width), dtype=mx.float32)

    l_opacity = _zero_edges_mlx(l_opacity, border=2)
    return mx.maximum(l_opacity, g_opacity)


# Composite kernels


def _composite_numpy(
    *,
    zpix: ArrayBuffer,
    atom_buf: ArrayBuffer,
    pconetot: ArrayBuffer,
    l_opacity: ArrayBuffer,
    type_lookup: ArrayBuffer,
    colortype: ArrayBuffer,
    fog_color: ArrayBuffer,
    fog_front: float,
    fog_back: float,
    zbuf_bg: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    zpix_np = np.asarray(zpix, dtype=np.float32)
    atom_np = np.asarray(atom_buf, dtype=np.int32)
    cone_np = np.asarray(pconetot, dtype=np.float32)
    l_opacity_np = np.asarray(l_opacity, dtype=np.float32)
    type_lookup_np = np.asarray(type_lookup, dtype=np.int32)
    color_lut = np.asarray(colortype, dtype=np.float32)
    fog_color_np = np.asarray(fog_color, dtype=np.float32)

    zpix_max = min(float(zpix_np.max()), 0.0)
    mol_mask = zpix_np != np.float32(zbuf_bg)
    zpix_min = float(zpix_np[mol_mask].min()) if np.any(mol_mask) else float(_ZBUF_FAR)
    zpix_clamped = np.minimum(zpix_np, np.float32(0.0))
    zpix_spread = zpix_max - zpix_min
    pfogdiff = float(fog_front - fog_back)

    if zpix_spread != 0.0:
        pfh = float(fog_front) - (zpix_max - zpix_clamped) / zpix_spread * pfogdiff
    else:
        pfh = np.full_like(zpix_clamped, np.float32(fog_front))
    pfh = np.where(zpix_clamped < zpix_min, 1.0, pfh).astype(np.float32)

    pixel_types = type_lookup_np[atom_np]
    # The color lookup allocates the output plane, so it can also hold
    # the intermediate colors without touching the palette.
    rgb = color_lut[pixel_types]
    rgb *= cone_np[:, :, None]
    rgb *= pfh[:, :, None]
    rgb += (1.0 - pfh[:, :, None]) * fog_color_np[None, None, :]
    rgb *= 1.0 - l_opacity_np[:, :, None]
    alpha = np.maximum(pixel_types != 0, l_opacity_np)
    return rgb, alpha


def _composite_mlx(
    *,
    zpix: ArrayBuffer,
    atom_buf: ArrayBuffer,
    pconetot: ArrayBuffer,
    l_opacity: ArrayBuffer,
    type_lookup: ArrayBuffer,
    colortype: ArrayBuffer,
    fog_color: ArrayBuffer,
    fog_front: float,
    fog_back: float,
    zbuf_bg: float,
) -> tuple[DeviceArray, DeviceArray]:
    mx = require_mlx()

    zpix_mx = _as_mlx_array(zpix, dtype=mx.float32)
    atom_mx = _as_mlx_array(atom_buf, dtype=mx.int32)
    cone_mx = _as_mlx_array(pconetot, dtype=mx.float32)
    l_opacity_mx = _as_mlx_array(l_opacity, dtype=mx.float32)
    type_lookup_mx = _as_mlx_array(type_lookup, dtype=mx.int32)
    color_lut = _as_mlx_array(colortype, dtype=mx.float32)
    fog_color_mx = _as_mlx_array(fog_color, dtype=mx.float32)
    zero_f = mx.array(np.float32(0.0), dtype=mx.float32)
    one_f = mx.array(np.float32(1.0), dtype=mx.float32)
    zbuf_bg_f = mx.array(np.float32(zbuf_bg), dtype=mx.float32)
    zfar_f = mx.array(_ZBUF_FAR, dtype=mx.float32)
    fog_front_f = mx.array(np.float32(fog_front), dtype=mx.float32)
    pfogdiff_f = mx.array(np.float32(fog_front - fog_back), dtype=mx.float32)

    zpix_max = mx.minimum(mx.max(zpix_mx), zero_f)
    mol_mask = mx.not_equal(zpix_mx, zbuf_bg_f)
    zpix_min = mx.min(mx.where(mol_mask, zpix_mx, zfar_f))
    zpix_clamped = mx.minimum(zpix_mx, zero_f)
    zpix_spread = zpix_max - zpix_min

    safe_spread = mx.where(mx.not_equal(zpix_spread, zero_f), zpix_spread, one_f)
    pfh_raw = fog_front_f - (zpix_max - zpix_clamped) / safe_spread * pfogdiff_f
    pfh = mx.where(
        mx.not_equal(zpix_spread, zero_f),
        pfh_raw,
        mx.full(zpix_mx.shape, np.float32(fog_front), dtype=mx.float32),
    )
    pfh = mx.where(zpix_clamped < zpix_min, one_f, pfh).astype(mx.float32)

    atom_idx = atom_mx.astype(mx.int32)
    pixel_types = mx.take(type_lookup_mx, atom_idx, axis=0).astype(mx.int32)
    atom_colors = mx.take(color_lut, pixel_types, axis=0)
    height, width = zpix_mx.shape
    pfh_3 = pfh.reshape((height, width, 1))
    cone_3 = cone_mx.reshape((height, width, 1))
    l_opacity_3 = l_opacity_mx.astype(mx.float32).reshape((height, width, 1))
    fog_color_3 = fog_color_mx.reshape((1, 1, 3))
    rcolor = pfh_3 * (cone_3 * atom_colors) + (one_f - pfh_3) * fog_color_3
    rgb = (one_f - l_opacity_3) * rcolor
    alpha = mx.maximum(mx.not_equal(pixel_types, 0).astype(mx.float32), l_opacity_mx.astype(mx.float32))
    return rgb, alpha


# Numba kernels load lazily so NumPy-only and MLX renders never import Numba.


# These wrappers validate inputs and pass the shared stencils and sentinel depths.


def _raster_chunk_numba(
    *,
    sx: ArrayBuffer,
    sy: ArrayBuffer,
    sz: ArrayBuffer,
    c_cx: ArrayBuffer,
    c_cy: ArrayBuffer,
    c_cz: ArrayBuffer,
    c_ia: ArrayBuffer,
    half_ix: float,
    half_iy: float,
    fix: float,
    fiy: float,
    nv: int,
    ibio: int,
    zpix: ArrayBuffer,
    atom_buf: ArrayBuffer,
    bio_buf: ArrayBuffer,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
    del nv
    zpix_np = np.asarray(zpix, dtype=np.float32)
    atom_np = np.asarray(atom_buf, dtype=np.int32)
    bio_np = np.asarray(bio_buf, dtype=np.int32)
    fix_f, fiy_f = _validate_raster_bounds(zpix_np.shape, fix, fiy)
    require_numba_kernels().raster_chunk(
        sx=sx,
        sy=sy,
        sz=sz,
        c_cx=c_cx,
        c_cy=c_cy,
        c_cz=c_cz,
        c_ia=c_ia,
        half_ix=half_ix,
        half_iy=half_iy,
        fix=fix_f,
        fiy=fiy_f,
        ibio=ibio,
        zpix=zpix_np,
        atom_buf=atom_np,
        bio_buf=bio_np,
    )
    return zpix_np, atom_np, bio_np


def _shadow_cone_numba(**kwargs: Unpack[_ShadowInputs]) -> ArrayBuffer:
    return require_numba_kernels().shadow_cone(
        **kwargs,
        offsets_by_radius=_SHADOW_OFFSETS_BY_RADIUS,
        pad=_SHADOW_RADIUS_PIXELS,
        outside_depth=float(_ZBUF_OUTSIDE_IMAGE),
    )


def _outline_kernel34_numba(
    *,
    zpix: ArrayBuffer,
    atom_buf: ArrayBuffer,
    bio_buf: ArrayBuffer,
    su_lookup: ArrayBuffer,
    res_lookup: ArrayBuffer,
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
    z_diff_min: float,
    z_diff_max: float,
    contour_low: float,
    contour_high: float,
    kernel: int,
) -> ArrayBuffer:
    _validate_outline34_kernel(kernel)
    return require_numba_kernels().outline_kernel34(
        zpix=zpix,
        atom_buf=atom_buf,
        bio_buf=bio_buf,
        su_lookup=su_lookup,
        res_lookup=res_lookup,
        residue_diff=residue_diff,
        residue_low=residue_low,
        residue_high=residue_high,
        subunit_low=subunit_low,
        subunit_high=subunit_high,
        z_diff_min=z_diff_min,
        z_diff_max=z_diff_max,
        contour_low=contour_low,
        contour_high=contour_high,
        group_offsets=_RG_OFFSETS,
        contour_offsets=_KERNEL3_OFFSETS if kernel == 3 else _KERNEL4_OFFSETS,
    )


def _composite_numba(**kwargs: Unpack[_CompositeInputs]) -> tuple[ArrayBuffer, ArrayBuffer]:
    return require_numba_kernels().composite(**kwargs, zbuf_far=float(_ZBUF_FAR))


# Dispatch API


RASTER_DISPATCH: Final[Mapping[str, Callable[..., tuple[ArrayBuffer, ArrayBuffer, ArrayBuffer]]]] = {
    "numpy": _raster_chunk_numpy,
    "numba": _raster_chunk_numba,
    "mlx": _raster_chunk_mlx,
}

SHADOW_DISPATCH: Final[Mapping[str, Callable[..., ArrayBuffer]]] = {
    "numpy": _shadow_cone_numpy,
    "numba": _shadow_cone_numba,
    "mlx": _shadow_cone_mlx,
}

OUTLINE34_DISPATCH: Final[Mapping[str, Callable[..., ArrayBuffer]]] = {
    "numpy": _outline_kernel34_numpy,
    "numba": _outline_kernel34_numba,
    "mlx": _outline_kernel34_mlx,
}

OUTLINE12_DISPATCH: Final[Mapping[str, Callable[..., ArrayBuffer]]] = {
    "numpy": _outline_kernel12_numpy,
    # The Laplacian kernels are not compiled. No preset uses them.
    "numba": _outline_kernel12_numpy,
    "mlx": _outline_kernel12_mlx,
}

COMPOSITE_DISPATCH: Final[Mapping[str, Callable[..., tuple[ArrayBuffer, ArrayBuffer]]]] = {
    "numpy": _composite_numpy,
    "numba": _composite_numba,
    "mlx": _composite_mlx,
}

_BACKEND_MODULES: Final[dict[str, str | None]] = {
    "numpy": None,
    "numba": "numba",
    "mlx": "mlx.core",
}

R = TypeVar("R")


def _normalize_backend(backend: object) -> str:
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")
    return backend.strip().lower()


def _run_backend_dispatch(
    table: Mapping[str, Callable[..., R]],
    *,
    backend: str,
    **kwargs: object,
) -> R:
    backend_name = _normalize_backend(backend)
    try:
        fn = table[backend_name]
    except KeyError:
        valid = ", ".join(sorted(table))
        raise ValueError(f"Unsupported backend {backend!r}; expected one of: {valid}") from None
    return fn(**kwargs)


def run_raster_kernel(
    *, backend: str, **kwargs: Unpack[_RasterInputs]
) -> tuple[ArrayBuffer, ArrayBuffer, ArrayBuffer]:
    return _run_backend_dispatch(RASTER_DISPATCH, backend=backend, **kwargs)


def run_shadow_kernel(
    checkpoint: Callable[[], None] | None = None, /, *, backend: str, **kwargs: Unpack[_ShadowInputs]
) -> ArrayBuffer:
    if checkpoint is not None and _normalize_backend(backend) == "numpy":
        return _run_backend_dispatch(SHADOW_DISPATCH, backend=backend, checkpoint=checkpoint, **kwargs)
    return _run_backend_dispatch(SHADOW_DISPATCH, backend=backend, **kwargs)


def run_outline34_kernel(*, backend: str, **kwargs: Unpack[_Outline34Inputs]) -> ArrayBuffer:
    return _run_backend_dispatch(OUTLINE34_DISPATCH, backend=backend, **kwargs)


def run_outline12_kernel(*, backend: str, **kwargs: Unpack[_Outline12Inputs]) -> ArrayBuffer:
    return _run_backend_dispatch(OUTLINE12_DISPATCH, backend=backend, **kwargs)


def run_composite_kernel(*, backend: str, **kwargs: Unpack[_CompositeInputs]) -> tuple[ArrayBuffer, ArrayBuffer]:
    return _run_backend_dispatch(COMPOSITE_DISPATCH, backend=backend, **kwargs)


def _quantize_u8_numpy(linear: NDArray[np.float32]) -> NDArray[np.uint8]:
    # Truncate to integers, then clamp to the 8-bit range.
    return np.multiply(linear, np.float32(255.0)).astype(np.int32).clip(0, 255).astype(np.uint8)


def run_composite_u8_kernel(
    *, backend: str, **kwargs: Unpack[_CompositeInputs]
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Composite on a CPU backend and return 8-bit RGB and alpha.

    Numba quantizes each pixel as it shades it. Other backends composite,
    then quantize the linear planes with NumPy.
    """
    if _normalize_backend(backend) == "numba":
        rgb, alpha = require_numba_kernels().composite_u8(**kwargs, zbuf_far=float(_ZBUF_FAR))
        return rgb, alpha
    rgb_linear, alpha_linear = run_composite_kernel(backend=backend, **kwargs)
    return (
        _quantize_u8_numpy(np.asarray(rgb_linear, dtype=np.float32)),
        _quantize_u8_numpy(np.asarray(alpha_linear, dtype=np.float32)),
    )


def supported_backends() -> tuple[str, ...]:
    return tuple(RASTER_DISPATCH)


# Importing Numba runs hundreds of modules, so availability only checks that
# it is installed. Status probes then cost nothing on hosts that render with MLX.
_INSTALLED_ONLY_PROBES: Final[frozenset[str]] = frozenset({"numba"})


def backend_available(backend: str) -> bool:
    """Report whether a backend is installed, importing it unless that is slow."""
    name = _normalize_backend(backend)
    module_name = _BACKEND_MODULES.get(name)
    if name not in _BACKEND_MODULES:
        return False
    if module_name is None:
        return True
    if name in _INSTALLED_ONLY_PROBES:
        return _module_spec_exists(module_name)
    return _module_import_succeeds(module_name)


def backend_importable(backend: str) -> bool:
    """Report whether a backend's package imports, even when that is slow."""
    name = _normalize_backend(backend)
    module_name = _BACKEND_MODULES.get(name)
    if name not in _BACKEND_MODULES:
        return False
    return module_name is None or _module_import_succeeds(module_name)


__all__ = [
    "RASTER_DISPATCH",
    "SHADOW_DISPATCH",
    "OUTLINE34_DISPATCH",
    "OUTLINE12_DISPATCH",
    "COMPOSITE_DISPATCH",
    "backend_available",
    "backend_importable",
    "run_composite_kernel",
    "run_composite_u8_kernel",
    "run_raster_kernel",
    "run_outline12_kernel",
    "run_outline34_kernel",
    "run_shadow_kernel",
    "supported_backends",
]
