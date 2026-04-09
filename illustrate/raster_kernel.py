from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import importlib
import importlib.util
import subprocess
import sys
from typing import Any

import numpy as np

_OPTIONAL_IMPORT_PROBE_TIMEOUT_SECONDS = 5.0
_OPTIONAL_IMPORT_PROBE = "import importlib, sys; importlib.import_module(sys.argv[1])"


def _module_spec_exists(module_name: str) -> bool:
    """Best-effort module existence check for optional GPU backends."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        # find_spec("pkg.submodule") raises when parent package is absent.
        return False


@lru_cache(maxsize=None)
def _module_import_succeeds(module_name: str) -> bool:
    """Probe optional modules in a subprocess so broken installs cannot abort us."""
    if not _module_spec_exists(module_name):
        return False

    executable = sys.executable.strip()
    if executable == "":
        return False

    try:
        completed = subprocess.run(
            [executable, "-c", _OPTIONAL_IMPORT_PROBE, module_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_OPTIONAL_IMPORT_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return completed.returncode == 0


def _clear_optional_module_probe_cache() -> None:
    _module_import_succeeds.cache_clear()


def _require_optional_module(module_name: str, error_message: str) -> Any:
    if not _module_import_succeeds(module_name):
        raise RuntimeError(error_message)

    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(error_message) from exc


def _require_cupy() -> Any:
    return _require_optional_module("cupy", "CuPy backend requested but CuPy is unavailable")


def _require_mlx() -> Any:
    return _require_optional_module("mlx.core", "MLX backend requested but MLX is unavailable")


@lru_cache(maxsize=1)
def _mlx_array_type() -> type[Any]:
    mx = _require_mlx()
    return type(mx.array(np.zeros((0,), dtype=np.float32)))


def _as_mlx_array(value: Any, *, dtype: Any | None = None) -> Any:
    mx = _require_mlx()
    array_type = _mlx_array_type()
    if isinstance(value, array_type):
        if dtype is not None and getattr(value, "dtype", None) != dtype:
            return value.astype(dtype)
        return value
    if dtype is not None:
        return mx.array(value, dtype=dtype)
    return mx.array(value)


@lru_cache(maxsize=8)
def _shadow_offset_chunks(chunk_size: int) -> tuple[tuple[tuple[int, int, float], ...], ...]:
    if chunk_size <= 0:
        raise ValueError(f"shadow chunk size must be positive, got {chunk_size}")
    return tuple(
        tuple(_SHADOW_OFFSETS[start : start + chunk_size])
        for start in range(0, len(_SHADOW_OFFSETS), chunk_size)
    )


@lru_cache(maxsize=8)
def _mlx_shadow_chunk_radii(chunk_size: int) -> tuple[Any, ...]:
    mx = _require_mlx()
    return tuple(
        _as_mlx_array(
            np.array([radius for _di, _dj, radius in chunk], dtype=np.float32),
            dtype=mx.float32,
        )[:, None, None]
        for chunk in _shadow_offset_chunks(chunk_size)
    )


@lru_cache(maxsize=2)
def _mlx_outline12_lap_weights(kernel: int) -> Any:
    mx = _require_mlx()
    if kernel == 1:
        return _as_mlx_array(_KERNEL1_LAP_WEIGHTS, dtype=mx.float32)
    if kernel == 2:
        return _as_mlx_array(_KERNEL2_LAP_WEIGHTS, dtype=mx.float32)
    raise ValueError(f"outline12 kernel expects 1 or 2, got {kernel}")


def _raster_chunk_numpy(
    *,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    c_cx: np.ndarray,
    c_cy: np.ndarray,
    c_cz: np.ndarray,
    c_ia: np.ndarray,
    half_ix: float,
    half_iy: float,
    fix: float,
    fiy: float,
    nv: int,
    ibio: int,
    zpix: np.ndarray,
    atom_buf: np.ndarray,
    bio_buf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_px = (sx[np.newaxis, :] + c_cx[:, np.newaxis] + half_ix).ravel()
    all_py = (sy[np.newaxis, :] + c_cy[:, np.newaxis] + half_iy).ravel()
    all_pz = (sz[np.newaxis, :] + c_cz[:, np.newaxis]).ravel()

    valid = (all_px >= 1.0) & (all_px <= fix) & (all_py >= 1.0) & (all_py <= fiy)
    if not np.any(valid):
        return zpix, atom_buf, bio_buf

    ipx_v = all_px[valid].astype(np.int32) - 1
    ipy_v = all_py[valid].astype(np.int32) - 1
    pz_v = all_pz[valid]
    ia_v = np.repeat(c_ia, nv)[valid]

    np.maximum.at(zpix, (ipx_v, ipy_v), pz_v)

    winners = pz_v == zpix[ipx_v, ipy_v]
    if np.any(winners):
        atom_buf[ipx_v[winners], ipy_v[winners]] = ia_v[winners]
        bio_buf[ipx_v[winners], ipy_v[winners]] = ibio
    return zpix, atom_buf, bio_buf


def _raster_chunk_cupy(
    *,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    c_cx: np.ndarray,
    c_cy: np.ndarray,
    c_cz: np.ndarray,
    c_ia: np.ndarray,
    half_ix: float,
    half_iy: float,
    fix: float,
    fiy: float,
    nv: int,
    ibio: int,
    zpix: Any,
    atom_buf: Any,
    bio_buf: Any,
) -> tuple[Any, Any, Any]:
    cp = _require_cupy()

    zpix_gpu = cp.asarray(zpix)
    atom_buf_gpu = cp.asarray(atom_buf)
    bio_buf_gpu = cp.asarray(bio_buf)

    sx_gpu = cp.asarray(sx)
    sy_gpu = cp.asarray(sy)
    sz_gpu = cp.asarray(sz)
    cx_gpu = cp.asarray(c_cx)
    cy_gpu = cp.asarray(c_cy)
    cz_gpu = cp.asarray(c_cz)
    ia_gpu = cp.asarray(c_ia)

    all_px = (sx_gpu[cp.newaxis, :] + cx_gpu[:, cp.newaxis] + half_ix).ravel()
    all_py = (sy_gpu[cp.newaxis, :] + cy_gpu[:, cp.newaxis] + half_iy).ravel()
    all_pz = (sz_gpu[cp.newaxis, :] + cz_gpu[:, cp.newaxis]).ravel()

    valid = (all_px >= 1.0) & (all_px <= fix) & (all_py >= 1.0) & (all_py <= fiy)
    if not bool(cp.any(valid)):
        return zpix_gpu, atom_buf_gpu, bio_buf_gpu

    ipx_v = all_px[valid].astype(cp.int32) - 1
    ipy_v = all_py[valid].astype(cp.int32) - 1
    pz_v = all_pz[valid]
    ia_v = cp.repeat(ia_gpu, nv)[valid]

    cp.maximum.at(zpix_gpu, (ipx_v, ipy_v), pz_v)

    winners = pz_v == zpix_gpu[ipx_v, ipy_v]
    if bool(cp.any(winners)):
        atom_buf_gpu[ipx_v[winners], ipy_v[winners]] = ia_v[winners]
        bio_buf_gpu[ipx_v[winners], ipy_v[winners]] = ibio
    return zpix_gpu, atom_buf_gpu, bio_buf_gpu


def _raster_chunk_mlx(
    *,
    sx: np.ndarray,
    sy: np.ndarray,
    sz: np.ndarray,
    c_cx: np.ndarray,
    c_cy: np.ndarray,
    c_cz: np.ndarray,
    c_ia: np.ndarray,
    half_ix: float,
    half_iy: float,
    fix: float,
    fiy: float,
    nv: int,
    ibio: int,
    zpix: Any,
    atom_buf: Any,
    bio_buf: Any,
) -> tuple[Any, Any, Any]:
    mx = _require_mlx()

    del fix, fiy

    zpix_mx = _as_mlx_array(zpix, dtype=mx.float32)
    atom_buf_mx = _as_mlx_array(atom_buf, dtype=mx.int32)
    bio_buf_mx = _as_mlx_array(bio_buf, dtype=mx.int32)

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

    x_max = float(zpix_mx.shape[0])
    y_max = float(zpix_mx.shape[1])
    valid = (all_px >= 1.0) & (all_px <= x_max) & (all_py >= 1.0) & (all_py <= y_max)
    zero_i = mx.zeros(valid.shape, dtype=mx.int32)
    invalid_depth = mx.full(all_pz.shape, np.float32(-100000.0), dtype=mx.float32)
    ipx_v = mx.where(valid, all_px.astype(mx.int32) - 1, zero_i)
    ipy_v = mx.where(valid, all_py.astype(mx.int32) - 1, zero_i)
    pz_v = mx.where(valid, all_pz, invalid_depth)
    ia_v = mx.repeat(ia_mx, nv).astype(mx.int32)
    width_y = int(zpix_mx.shape[1])
    linear_v = ipx_v * width_y + ipy_v

    zpix_flat = zpix_mx.reshape((-1,))
    zpix_flat = zpix_flat.at[linear_v].maximum(pz_v)

    winners = valid & (pz_v == zpix_flat[linear_v])
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

    atom_buf_mx = atom_flat.reshape(atom_buf_mx.shape)
    bio_buf_mx = bio_flat.reshape(bio_buf_mx.shape)

    zpix_mx = zpix_flat.reshape(zpix_mx.shape)
    return zpix_mx, atom_buf_mx, bio_buf_mx


def _shadow_cone_numpy(
    *,
    zpix: np.ndarray,
    atom_buf: np.ndarray,
    shadow_strength: float,
    shadow_angle: float,
    shadow_min_z: float,
    shadow_max_dark: float,
) -> np.ndarray:
    has_atom = atom_buf != 0
    shadow_pad = 50
    zpix_shadow_padded = np.pad(
        zpix,
        ((shadow_pad, shadow_pad), (shadow_pad, shadow_pad)),
        mode="constant",
        constant_values=-100000.0,
    )
    height, width = zpix.shape
    strength = np.float32(shadow_strength)
    min_z = np.float32(shadow_min_z)
    angle = np.float32(shadow_angle)
    max_dark = np.float32(shadow_max_dark)
    pix_count = max(1, height * width)
    chunk = max(1, min(_NUMPY_SHADOW_MAX_CHUNK, _NUMPY_SHADOW_ELEMS_BUDGET // pix_count))
    count = np.zeros((height, width), dtype=np.int16)

    for start in range(0, len(_SHADOW_OFFSETS), chunk):
        offsets = _SHADOW_OFFSETS[start : start + chunk]
        shifted_z = np.stack(
            [
                zpix_shadow_padded[shadow_pad + di : shadow_pad + di + height, shadow_pad + dj : shadow_pad + dj + width]
                for di, dj, _radius in offsets
            ],
            axis=0,
        )
        rzdiff = shifted_z - zpix[None, :, :]
        radii = np.array([radius for _di, _dj, radius in offsets], dtype=np.float32)[:, None, None]
        shadow_mask = has_atom[None, :, :] & (rzdiff > min_z) & ((radii * angle) < (rzdiff + min_z))
        count += shadow_mask.sum(axis=0, dtype=np.int16)

    pconetot = np.float32(1.0) - count.astype(np.float32) * strength
    pconetot = np.maximum(pconetot, max_dark)
    pconetot[~has_atom] = 1.0
    return pconetot


def _shadow_cone_cupy(
    *,
    zpix: Any,
    atom_buf: Any,
    shadow_strength: float,
    shadow_angle: float,
    shadow_min_z: float,
    shadow_max_dark: float,
):
    cp = _require_cupy()

    zpix_gpu = cp.asarray(zpix)
    atom_gpu = cp.asarray(atom_buf)
    has_atom = atom_gpu != 0
    pconetot = cp.ones(zpix_gpu.shape, dtype=cp.float32)
    shadow_pad = 50
    zpix_shadow_padded = cp.pad(
        zpix_gpu,
        ((shadow_pad, shadow_pad), (shadow_pad, shadow_pad)),
        mode="constant",
        constant_values=-100000.0,
    )
    height, width = zpix_gpu.shape
    strength = cp.float32(shadow_strength)
    min_z = cp.float32(shadow_min_z)
    angle = cp.float32(shadow_angle)
    max_dark = cp.float32(shadow_max_dark)

    for di, dj, radius in _SHADOW_OFFSETS:
        i0 = shadow_pad + di
        j0 = shadow_pad + dj
        shifted_z = zpix_shadow_padded[i0 : i0 + height, j0 : j0 + width]
        rzdiff = shifted_z - zpix_gpu
        shadow_mask = has_atom & (rzdiff > min_z) & ((radius * angle) < (rzdiff + min_z))
        pconetot = pconetot - strength * shadow_mask.astype(cp.float32)

    pconetot = cp.maximum(pconetot, max_dark)
    pconetot = cp.where(has_atom, pconetot, cp.float32(1.0))
    return pconetot


def _shadow_cone_mlx(
    *,
    zpix: Any,
    atom_buf: Any,
    shadow_strength: float,
    shadow_angle: float,
    shadow_min_z: float,
    shadow_max_dark: float,
):
    mx = _require_mlx()

    zpix_mx = _as_mlx_array(zpix, dtype=mx.float32)
    atom_mx = _as_mlx_array(atom_buf, dtype=mx.int32)
    has_atom: Any = atom_mx != 0
    pconetot = mx.ones(zpix_mx.shape, dtype=mx.float32)
    shadow_pad = 50
    zpix_shadow_padded = mx.pad(
        zpix_mx,
        [(shadow_pad, shadow_pad), (shadow_pad, shadow_pad)],
        mode="constant",
        constant_values=-100000.0,
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
                zpix_shadow_padded[shadow_pad + di : shadow_pad + di + height, shadow_pad + dj : shadow_pad + dj + width]
                for di, dj, _radius in chunk
            ],
            axis=0,
        )
        rzdiff = shifted_z - zpix_plane
        shadow_mask: Any = has_atom_plane & (rzdiff > min_z) & ((radii * angle) < (rzdiff + min_z))
        count = count + mx.sum(shadow_mask, axis=0).astype(mx.int32)

    pconetot = np.float32(1.0) - count.astype(mx.float32) * strength
    pconetot = mx.maximum(pconetot, max_dark)
    pconetot = mx.where(has_atom, pconetot, np.float32(1.0))
    return pconetot


def _outline_kernel34_numpy(
    *,
    zpix: np.ndarray,
    atom_buf: np.ndarray,
    bio_buf: np.ndarray,
    su_lookup: np.ndarray,
    res_lookup: np.ndarray,
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
) -> np.ndarray:
    height, width = zpix.shape
    su_map = su_lookup[atom_buf]
    res_map = res_lookup[atom_buf]

    r_count = np.zeros((height, width), dtype=np.float32)
    g_count = np.zeros((height, width), dtype=np.float32)
    rg_pad = 2
    su_padded = np.pad(su_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    bio_padded = np.pad(bio_buf, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=0)
    res_padded = np.pad(res_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    res_map_f = res_map.astype(np.float32)
    residue_diff_f = np.float32(residue_diff)

    for di, dj in _RG_OFFSETS:
        i0 = rg_pad + di
        j0 = rg_pad + dj
        shifted_su = su_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_bio = bio_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_res = res_padded[i0 : i0 + height, j0 : j0 + width]
        r_count += ((su_map != shifted_su) | (bio_buf != shifted_bio)).astype(np.float32)
        g_count += (np.abs(res_map_f - shifted_res.astype(np.float32)) > residue_diff_f).astype(np.float32)

    if residue_high != residue_low:
        g_opacity = np.clip((g_count - residue_low) / (residue_high - residue_low), 0.0, 1.0)
    else:
        g_opacity = np.zeros((height, width), dtype=np.float32)
    if subunit_high != subunit_low:
        r_opacity = np.clip((r_count - subunit_low) / (subunit_high - subunit_low), 0.0, 1.0)
    else:
        r_opacity = np.zeros((height, width), dtype=np.float32)
    g_opacity = np.maximum(g_opacity, r_opacity)
    g_opacity[0, :] = 0.0
    g_opacity[-1, :] = 0.0
    g_opacity[:, 0] = 0.0
    g_opacity[:, -1] = 0.0

    offsets_k = _KERNEL3_OFFSETS if kernel == 3 else _KERNEL4_OFFSETS
    zpad = 2
    zpix_padded = np.pad(zpix, ((zpad, zpad), (zpad, zpad)), mode="constant", constant_values=0.0)
    l_total = np.zeros((height, width), dtype=np.float32)
    denom = float(z_diff_max - z_diff_min)
    z_diff_min_f = np.float32(z_diff_min)

    for di, dj in offsets_k:
        i0 = zpad + di
        j0 = zpad + dj
        shifted_z = zpix_padded[i0 : i0 + height, j0 : j0 + width]
        rd = np.abs(zpix - shifted_z)
        if denom != 0.0:
            rd_norm = np.where(rd > z_diff_min_f, np.minimum((rd - z_diff_min_f) / denom, 1.0), 0.0)
        else:
            rd_norm = np.zeros_like(rd)
        l_total += rd_norm

    if contour_high != contour_low:
        l_val = np.clip((l_total - contour_low) / (contour_high - contour_low), 0.0, 1.0)
    else:
        l_val = np.zeros((height, width), dtype=np.float32)
    l_opacity = np.where(l_val > 0.0, np.minimum(1.5 * l_val, 1.0), 0.0)
    l_opacity[:2, :] = 0.0
    l_opacity[-2:, :] = 0.0
    l_opacity[:, :2] = 0.0
    l_opacity[:, -2:] = 0.0
    return np.maximum(l_opacity, g_opacity)


def _outline_kernel34_cupy(
    *,
    zpix: Any,
    atom_buf: Any,
    bio_buf: Any,
    su_lookup: np.ndarray,
    res_lookup: np.ndarray,
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
):
    cp = _require_cupy()

    zpix_gpu = cp.asarray(zpix)
    atom_gpu = cp.asarray(atom_buf)
    bio_gpu = cp.asarray(bio_buf)
    su_lookup_gpu = cp.asarray(su_lookup)
    res_lookup_gpu = cp.asarray(res_lookup)

    height, width = zpix_gpu.shape
    su_map = su_lookup_gpu[atom_gpu]
    res_map = res_lookup_gpu[atom_gpu]

    r_count = cp.zeros((height, width), dtype=cp.float32)
    g_count = cp.zeros((height, width), dtype=cp.float32)
    rg_pad = 2
    su_padded = cp.pad(su_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    bio_padded = cp.pad(bio_gpu, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=0)
    res_padded = cp.pad(res_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    res_map_f = res_map.astype(cp.float32)
    residue_diff_f = cp.float32(residue_diff)

    for di, dj in _RG_OFFSETS:
        i0 = rg_pad + di
        j0 = rg_pad + dj
        shifted_su = su_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_bio = bio_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_res = res_padded[i0 : i0 + height, j0 : j0 + width]
        r_count = r_count + ((su_map != shifted_su) | (bio_gpu != shifted_bio)).astype(cp.float32)
        g_count = g_count + (cp.abs(res_map_f - shifted_res.astype(cp.float32)) > residue_diff_f).astype(cp.float32)

    if residue_high != residue_low:
        g_opacity = cp.clip((g_count - residue_low) / (residue_high - residue_low), 0.0, 1.0)
    else:
        g_opacity = cp.zeros((height, width), dtype=cp.float32)
    if subunit_high != subunit_low:
        r_opacity = cp.clip((r_count - subunit_low) / (subunit_high - subunit_low), 0.0, 1.0)
    else:
        r_opacity = cp.zeros((height, width), dtype=cp.float32)
    g_opacity = cp.maximum(g_opacity, r_opacity)
    g_opacity[0, :] = 0.0
    g_opacity[-1, :] = 0.0
    g_opacity[:, 0] = 0.0
    g_opacity[:, -1] = 0.0

    offsets_k = _KERNEL3_OFFSETS if kernel == 3 else _KERNEL4_OFFSETS
    zpad = 2
    zpix_padded = cp.pad(zpix_gpu, ((zpad, zpad), (zpad, zpad)), mode="constant", constant_values=0.0)
    l_total = cp.zeros((height, width), dtype=cp.float32)
    denom = float(z_diff_max - z_diff_min)
    z_diff_min_f = cp.float32(z_diff_min)

    for di, dj in offsets_k:
        i0 = zpad + di
        j0 = zpad + dj
        shifted_z = zpix_padded[i0 : i0 + height, j0 : j0 + width]
        rd = cp.abs(zpix_gpu - shifted_z)
        if denom != 0.0:
            rd_norm = cp.where(rd > z_diff_min_f, cp.minimum((rd - z_diff_min_f) / denom, 1.0), 0.0)
        else:
            rd_norm = cp.zeros_like(rd)
        l_total = l_total + rd_norm

    if contour_high != contour_low:
        l_val = cp.clip((l_total - contour_low) / (contour_high - contour_low), 0.0, 1.0)
    else:
        l_val = cp.zeros((height, width), dtype=cp.float32)
    l_opacity = cp.where(l_val > 0.0, cp.minimum(1.5 * l_val, 1.0), 0.0)
    l_opacity[:2, :] = 0.0
    l_opacity[-2:, :] = 0.0
    l_opacity[:, :2] = 0.0
    l_opacity[:, -2:] = 0.0
    return cp.maximum(l_opacity, g_opacity)


def _outline_kernel34_mlx(
    *,
    zpix: Any,
    atom_buf: Any,
    bio_buf: Any,
    su_lookup: np.ndarray,
    res_lookup: np.ndarray,
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
):
    mx = _require_mlx()

    zpix_mx = _as_mlx_array(zpix, dtype=mx.float32)
    atom_mx = _as_mlx_array(atom_buf, dtype=mx.int32)
    bio_mx = _as_mlx_array(bio_buf, dtype=mx.int32)
    su_lookup_mx = _as_mlx_array(su_lookup, dtype=mx.int32)
    res_lookup_mx = _as_mlx_array(res_lookup, dtype=mx.int32)

    height, width = zpix_mx.shape
    su_map = mx.take(su_lookup_mx, atom_mx)
    res_map = mx.take(res_lookup_mx, atom_mx)

    r_count = mx.zeros((height, width), dtype=mx.float32)
    g_count = mx.zeros((height, width), dtype=mx.float32)
    rg_pad = 2
    su_padded = mx.pad(su_map, [(rg_pad, rg_pad), (rg_pad, rg_pad)], mode="constant", constant_values=9999)
    bio_padded = mx.pad(bio_mx, [(rg_pad, rg_pad), (rg_pad, rg_pad)], mode="constant", constant_values=0)
    res_padded = mx.pad(res_map, [(rg_pad, rg_pad), (rg_pad, rg_pad)], mode="constant", constant_values=9999)
    res_map_f = res_map.astype(mx.float32)
    res_padded_f = res_padded.astype(mx.float32)
    residue_diff_f = np.float32(residue_diff)

    for di, dj in _RG_OFFSETS:
        i0 = rg_pad + di
        j0 = rg_pad + dj
        shifted_su = su_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_bio = bio_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_res = res_padded_f[i0 : i0 + height, j0 : j0 + width]
        r_mask: Any = (su_map != shifted_su) | (bio_mx != shifted_bio)
        r_count = r_count + r_mask.astype(mx.float32)
        g_count = g_count + (mx.abs(res_map_f - shifted_res) > residue_diff_f).astype(mx.float32)

    if residue_high != residue_low:
        g_opacity = mx.clip((g_count - residue_low) / (residue_high - residue_low), 0.0, 1.0)
    else:
        g_opacity = mx.zeros((height, width), dtype=mx.float32)
    if subunit_high != subunit_low:
        r_opacity = mx.clip((r_count - subunit_low) / (subunit_high - subunit_low), 0.0, 1.0)
    else:
        r_opacity = mx.zeros((height, width), dtype=mx.float32)
    g_opacity = mx.maximum(g_opacity, r_opacity)
    rows = mx.arange(height)[:, None]
    cols = mx.arange(width)[None, :]
    g_edge_mask: Any = (rows == 0) | (rows == (height - 1)) | (cols == 0) | (cols == (width - 1))
    g_opacity = mx.where(g_edge_mask, np.float32(0.0), g_opacity)

    offsets_k = _KERNEL3_OFFSETS if kernel == 3 else _KERNEL4_OFFSETS
    zpad = 2
    zpix_padded = mx.pad(zpix_mx, [(zpad, zpad), (zpad, zpad)], mode="constant", constant_values=0.0)
    l_total = mx.zeros((height, width), dtype=mx.float32)
    denom = float(z_diff_max - z_diff_min)
    z_diff_min_f = np.float32(z_diff_min)

    if denom != 0.0:
        for di, dj in offsets_k:
            i0 = zpad + di
            j0 = zpad + dj
            shifted_z = zpix_padded[i0 : i0 + height, j0 : j0 + width]
            rd = mx.abs(zpix_mx - shifted_z)
            rd_norm = mx.where(rd > z_diff_min_f, mx.minimum((rd - z_diff_min_f) / denom, 1.0), 0.0)
            l_total = l_total + rd_norm

    if contour_high != contour_low:
        l_val = mx.clip((l_total - contour_low) / (contour_high - contour_low), 0.0, 1.0)
    else:
        l_val = mx.zeros((height, width), dtype=mx.float32)
    l_opacity = mx.where(l_val > 0.0, mx.minimum(1.5 * l_val, 1.0), 0.0)
    l_edge_mask = (rows < 2) | (rows >= (height - 2)) | (cols < 2) | (cols >= (width - 2))
    l_opacity = mx.where(l_edge_mask, np.float32(0.0), l_opacity)
    return mx.maximum(l_opacity, g_opacity)


def _outline_kernel12_numpy(
    *,
    zpix: np.ndarray,
    atom_buf: np.ndarray,
    bio_buf: np.ndarray,
    su_lookup: np.ndarray,
    res_lookup: np.ndarray,
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
    contour_low: float,
    contour_high: float,
    kernel: int,
) -> np.ndarray:
    if kernel not in (1, 2):
        raise ValueError(f"outline12 kernel expects 1 or 2, got {kernel}")

    zpix_np = np.asarray(zpix, dtype=np.float32)
    atom_np = np.asarray(atom_buf, dtype=np.int32)
    bio_np = np.asarray(bio_buf, dtype=np.int32)
    su_lookup_np = np.asarray(su_lookup, dtype=np.int32)
    res_lookup_np = np.asarray(res_lookup, dtype=np.int32)

    height, width = zpix_np.shape
    pix_count = max(1, height * width)

    su_map = su_lookup_np[atom_np]
    res_map = res_lookup_np[atom_np]
    res_map_f = res_map.astype(np.float32)
    residue_diff_f = np.float32(residue_diff)

    r_count = np.zeros((height, width), dtype=np.float32)
    g_count = np.zeros((height, width), dtype=np.float32)
    rg_pad = 2
    su_padded = np.pad(su_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    bio_padded = np.pad(bio_np, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=0)
    res_padded = np.pad(res_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)

    rg_chunk = max(1, min(len(_RG_OFFSETS), _NUMPY_OUTLINE_ELEMS_BUDGET // max(1, 3 * pix_count)))
    for start in range(0, len(_RG_OFFSETS), rg_chunk):
        offsets = _RG_OFFSETS[start : start + rg_chunk]
        shifted_su = np.stack(
            [su_padded[rg_pad + di : rg_pad + di + height, rg_pad + dj : rg_pad + dj + width] for di, dj in offsets],
            axis=0,
        )
        shifted_bio = np.stack(
            [bio_padded[rg_pad + di : rg_pad + di + height, rg_pad + dj : rg_pad + dj + width] for di, dj in offsets],
            axis=0,
        )
        shifted_res = np.stack(
            [res_padded[rg_pad + di : rg_pad + di + height, rg_pad + dj : rg_pad + dj + width] for di, dj in offsets],
            axis=0,
        )
        r_count += np.sum(
            (su_map[None, :, :] != shifted_su) | (bio_np[None, :, :] != shifted_bio),
            axis=0,
            dtype=np.int16,
        ).astype(np.float32)
        g_count += np.sum(
            np.abs(res_map_f[None, :, :] - shifted_res.astype(np.float32)) > residue_diff_f,
            axis=0,
            dtype=np.int16,
        ).astype(np.float32)

    if residue_high != residue_low:
        g_opacity = np.clip((g_count - residue_low) / (residue_high - residue_low), 0.0, 1.0)
    else:
        g_opacity = np.zeros((height, width), dtype=np.float32)
    if subunit_high != subunit_low:
        r_opacity = np.clip((r_count - subunit_low) / (subunit_high - subunit_low), 0.0, 1.0)
    else:
        r_opacity = np.zeros((height, width), dtype=np.float32)
    g_opacity = np.maximum(g_opacity, r_opacity)
    g_opacity[0, :] = 0.0
    g_opacity[-1, :] = 0.0
    g_opacity[:, 0] = 0.0
    g_opacity[:, -1] = 0.0

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

    lap = np.abs(lap / np.float32(3.0))

    rl = np.zeros((height, width), dtype=np.float32)
    l_opacity_ave = np.zeros((height, width), dtype=np.float32)
    l_center = np.zeros((height, width), dtype=np.float32)
    lap_pad = 1
    lap_padded = np.pad(lap, ((lap_pad, lap_pad), (lap_pad, lap_pad)), mode="constant", constant_values=0.0)
    denom = float(contour_high - contour_low)

    for di, dj in _LAPLACE_NEIGHBOR_OFFSETS:
        shifted = lap_padded[lap_pad + di : lap_pad + di + height, lap_pad + dj : lap_pad + dj + width]
        if denom != 0.0:
            l_v = np.clip((shifted - contour_low) / denom, 0.0, 1.0).astype(np.float32)
        else:
            l_v = np.zeros((height, width), dtype=np.float32)
        rl += (l_v > 0).astype(np.float32)
        l_opacity_ave += l_v
        if di == 0 and dj == 0:
            l_center = l_v.copy()

    l_opacity = np.where(rl >= 6.0, l_opacity_ave / 6.0, l_center)
    l_opacity = np.clip(l_opacity, 0.0, 1.0)
    l_opacity[:2, :] = 0.0
    l_opacity[-2:, :] = 0.0
    l_opacity[:, :2] = 0.0
    l_opacity[:, -2:] = 0.0
    return np.maximum(l_opacity, g_opacity)


def _outline_kernel12_cupy(
    *,
    zpix: Any,
    atom_buf: Any,
    bio_buf: Any,
    su_lookup: np.ndarray,
    res_lookup: np.ndarray,
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
    contour_low: float,
    contour_high: float,
    kernel: int,
):
    if kernel not in (1, 2):
        raise ValueError(f"outline12 kernel expects 1 or 2, got {kernel}")

    cp = _require_cupy()

    zpix_gpu = cp.asarray(zpix, dtype=cp.float32)
    atom_gpu = cp.asarray(atom_buf, dtype=cp.int32)
    bio_gpu = cp.asarray(bio_buf, dtype=cp.int32)
    su_lookup_gpu = cp.asarray(su_lookup, dtype=cp.int32)
    res_lookup_gpu = cp.asarray(res_lookup, dtype=cp.int32)

    height, width = zpix_gpu.shape
    su_map = su_lookup_gpu[atom_gpu]
    res_map = res_lookup_gpu[atom_gpu]
    res_map_f = res_map.astype(cp.float32)
    residue_diff_f = cp.float32(residue_diff)

    r_count = cp.zeros((height, width), dtype=cp.float32)
    g_count = cp.zeros((height, width), dtype=cp.float32)
    rg_pad = 2
    su_padded = cp.pad(su_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    bio_padded = cp.pad(bio_gpu, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=0)
    res_padded = cp.pad(res_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)

    for di, dj in _RG_OFFSETS:
        i0 = rg_pad + di
        j0 = rg_pad + dj
        shifted_su = su_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_bio = bio_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_res = res_padded[i0 : i0 + height, j0 : j0 + width]
        r_count = r_count + ((su_map != shifted_su) | (bio_gpu != shifted_bio)).astype(cp.float32)
        g_count = g_count + (cp.abs(res_map_f - shifted_res.astype(cp.float32)) > residue_diff_f).astype(cp.float32)

    if residue_high != residue_low:
        g_opacity = cp.clip((g_count - residue_low) / (residue_high - residue_low), 0.0, 1.0)
    else:
        g_opacity = cp.zeros((height, width), dtype=cp.float32)
    if subunit_high != subunit_low:
        r_opacity = cp.clip((r_count - subunit_low) / (subunit_high - subunit_low), 0.0, 1.0)
    else:
        r_opacity = cp.zeros((height, width), dtype=cp.float32)
    g_opacity = cp.maximum(g_opacity, r_opacity)
    g_opacity[0, :] = 0.0
    g_opacity[-1, :] = 0.0
    g_opacity[:, 0] = 0.0
    g_opacity[:, -1] = 0.0

    if kernel == 1:
        lap_offsets = _KERNEL1_LAP_OFFSETS
        lap_weights = cp.asarray(_KERNEL1_LAP_WEIGHTS, dtype=cp.float32)
    else:
        lap_offsets = _KERNEL2_LAP_OFFSETS
        lap_weights = cp.asarray(_KERNEL2_LAP_WEIGHTS, dtype=cp.float32)

    zpad = 2
    zpix_padded = cp.pad(zpix_gpu, ((zpad, zpad), (zpad, zpad)), mode="constant", constant_values=0.0)
    shifted_z = cp.stack(
        [zpix_padded[zpad + di : zpad + di + height, zpad + dj : zpad + dj + width] for di, dj in lap_offsets],
        axis=0,
    ).astype(cp.float32)
    lap = cp.sum(shifted_z * lap_weights[:, None, None], axis=0, dtype=cp.float32)
    lap = cp.abs(lap / cp.float32(3.0))

    rl = cp.zeros((height, width), dtype=cp.float32)
    l_opacity_ave = cp.zeros((height, width), dtype=cp.float32)
    l_center = cp.zeros((height, width), dtype=cp.float32)
    lap_pad = 1
    lap_padded = cp.pad(lap, ((lap_pad, lap_pad), (lap_pad, lap_pad)), mode="constant", constant_values=0.0)
    denom = float(contour_high - contour_low)

    for di, dj in _LAPLACE_NEIGHBOR_OFFSETS:
        shifted = lap_padded[lap_pad + di : lap_pad + di + height, lap_pad + dj : lap_pad + dj + width]
        if denom != 0.0:
            l_v = cp.clip((shifted - contour_low) / denom, 0.0, 1.0).astype(cp.float32)
        else:
            l_v = cp.zeros((height, width), dtype=cp.float32)
        rl = rl + (l_v > 0).astype(cp.float32)
        l_opacity_ave = l_opacity_ave + l_v
        if di == 0 and dj == 0:
            l_center = l_v.copy()

    l_opacity = cp.where(rl >= cp.float32(6.0), l_opacity_ave / cp.float32(6.0), l_center)
    l_opacity = cp.clip(l_opacity, 0.0, 1.0)
    l_opacity[:2, :] = 0.0
    l_opacity[-2:, :] = 0.0
    l_opacity[:, :2] = 0.0
    l_opacity[:, -2:] = 0.0
    return cp.maximum(l_opacity, g_opacity)


def _outline_kernel12_mlx(
    *,
    zpix: Any,
    atom_buf: Any,
    bio_buf: Any,
    su_lookup: np.ndarray,
    res_lookup: np.ndarray,
    residue_diff: float,
    residue_low: float,
    residue_high: float,
    subunit_low: float,
    subunit_high: float,
    contour_low: float,
    contour_high: float,
    kernel: int,
):
    if kernel not in (1, 2):
        raise ValueError(f"outline12 kernel expects 1 or 2, got {kernel}")

    mx = _require_mlx()

    zpix_mx = _as_mlx_array(zpix, dtype=mx.float32)
    atom_mx = _as_mlx_array(atom_buf, dtype=mx.int32)
    bio_mx = _as_mlx_array(bio_buf, dtype=mx.int32)
    su_lookup_mx = _as_mlx_array(su_lookup, dtype=mx.int32)
    res_lookup_mx = _as_mlx_array(res_lookup, dtype=mx.int32)

    height, width = zpix_mx.shape
    su_map = mx.take(su_lookup_mx, atom_mx)
    res_map = mx.take(res_lookup_mx, atom_mx)
    res_map_f = res_map.astype(mx.float32)
    residue_diff_f = np.float32(residue_diff)

    r_count = mx.zeros((height, width), dtype=mx.float32)
    g_count = mx.zeros((height, width), dtype=mx.float32)
    rg_pad = 2
    su_padded = mx.pad(su_map, [(rg_pad, rg_pad), (rg_pad, rg_pad)], mode="constant", constant_values=9999)
    bio_padded = mx.pad(bio_mx, [(rg_pad, rg_pad), (rg_pad, rg_pad)], mode="constant", constant_values=0)
    res_padded = mx.pad(res_map, [(rg_pad, rg_pad), (rg_pad, rg_pad)], mode="constant", constant_values=9999)
    res_padded_f = res_padded.astype(mx.float32)

    for di, dj in _RG_OFFSETS:
        i0 = rg_pad + di
        j0 = rg_pad + dj
        shifted_su = su_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_bio = bio_padded[i0 : i0 + height, j0 : j0 + width]
        shifted_res = res_padded_f[i0 : i0 + height, j0 : j0 + width]
        r_count = r_count + ((su_map != shifted_su) | (bio_mx != shifted_bio)).astype(mx.float32)
        g_count = g_count + (mx.abs(res_map_f - shifted_res) > residue_diff_f).astype(mx.float32)

    if residue_high != residue_low:
        g_opacity = mx.clip((g_count - residue_low) / (residue_high - residue_low), 0.0, 1.0)
    else:
        g_opacity = mx.zeros((height, width), dtype=mx.float32)
    if subunit_high != subunit_low:
        r_opacity = mx.clip((r_count - subunit_low) / (subunit_high - subunit_low), 0.0, 1.0)
    else:
        r_opacity = mx.zeros((height, width), dtype=mx.float32)
    g_opacity = mx.maximum(g_opacity, r_opacity)
    rows = mx.arange(height)[:, None]
    cols = mx.arange(width)[None, :]
    g_edge_mask: Any = (rows == 0) | (rows == (height - 1)) | (cols == 0) | (cols == (width - 1))
    g_opacity = mx.where(g_edge_mask, np.float32(0.0), g_opacity)

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
    lap = mx.abs(lap / np.float32(3.0))

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

        l_opacity = mx.where(rl >= np.float32(6.0), l_opacity_ave / np.float32(6.0), l_center)
        l_opacity = mx.clip(l_opacity, 0.0, 1.0)
    else:
        l_opacity = mx.zeros((height, width), dtype=mx.float32)
    l_edge_mask: Any = (rows < 2) | (rows >= (height - 2)) | (cols < 2) | (cols >= (width - 2))
    l_opacity = mx.where(l_edge_mask, np.float32(0.0), l_opacity)
    return mx.maximum(l_opacity, g_opacity)


def _composite_numpy(
    *,
    zpix: np.ndarray,
    atom_buf: np.ndarray,
    pconetot: np.ndarray,
    l_opacity: np.ndarray,
    type_lookup: np.ndarray,
    colortype: np.ndarray,
    fog_color: np.ndarray,
    fog_front: float,
    fog_back: float,
    zbuf_bg: float,
) -> tuple[np.ndarray, np.ndarray]:
    zpix_np = np.asarray(zpix, dtype=np.float32)
    atom_np = np.asarray(atom_buf, dtype=np.int32)
    cone_np = np.asarray(pconetot, dtype=np.float32)
    l_opacity_np = np.asarray(l_opacity, dtype=np.float32)
    type_lookup_np = np.asarray(type_lookup, dtype=np.int32)
    color_lut = np.asarray(colortype, dtype=np.float32)
    fog_color_np = np.asarray(fog_color, dtype=np.float32)

    zpix_max = min(float(zpix_np.max()), 0.0)
    mol_mask = zpix_np != np.float32(zbuf_bg)
    zpix_min = float(zpix_np[mol_mask].min()) if np.any(mol_mask) else 100000.0
    zpix_clamped = np.minimum(zpix_np, 0.0)
    zpix_spread = zpix_max - zpix_min
    pfogdiff = float(fog_front - fog_back)

    if zpix_spread != 0.0:
        pfh = float(fog_front) - (zpix_max - zpix_clamped) / zpix_spread * pfogdiff
    else:
        pfh = np.full_like(zpix_clamped, np.float32(fog_front))
    pfh = np.where(zpix_clamped < zpix_min, 1.0, pfh).astype(np.float32)

    pixel_types = type_lookup_np[atom_np]
    atom_colors = color_lut[pixel_types]
    rcolor = pfh[:, :, None] * (cone_np[:, :, None] * atom_colors) + (1.0 - pfh[:, :, None]) * fog_color_np[None, None, :]
    rgb = (1.0 - l_opacity_np[:, :, None]) * rcolor
    alpha = np.maximum((pixel_types != 0).astype(np.float32), l_opacity_np)
    return rgb.astype(np.float32), alpha.astype(np.float32)


def _composite_cupy(
    *,
    zpix: Any,
    atom_buf: Any,
    pconetot: Any,
    l_opacity: Any,
    type_lookup: np.ndarray,
    colortype: np.ndarray,
    fog_color: np.ndarray,
    fog_front: float,
    fog_back: float,
    zbuf_bg: float,
):
    cp = _require_cupy()

    zpix_gpu = cp.asarray(zpix)
    atom_gpu = cp.asarray(atom_buf, dtype=cp.int32)
    cone_gpu = cp.asarray(pconetot, dtype=cp.float32)
    l_opacity_gpu = cp.asarray(l_opacity, dtype=cp.float32)
    type_lookup_gpu = cp.asarray(type_lookup, dtype=cp.int32)
    color_lut = cp.asarray(colortype, dtype=cp.float32)
    fog_color_gpu = cp.asarray(fog_color, dtype=cp.float32)

    zpix_max = cp.minimum(cp.max(zpix_gpu), cp.float32(0.0))
    mol_mask = zpix_gpu != cp.float32(zbuf_bg)
    zpix_min = cp.min(cp.where(mol_mask, zpix_gpu, cp.float32(100000.0)))
    zpix_clamped = cp.minimum(zpix_gpu, cp.float32(0.0))
    zpix_spread = zpix_max - zpix_min
    pfogdiff = cp.float32(fog_front - fog_back)

    safe_spread = cp.where(zpix_spread != cp.float32(0.0), zpix_spread, cp.float32(1.0))
    pfh_raw = cp.float32(fog_front) - (zpix_max - zpix_clamped) / safe_spread * pfogdiff
    pfh = cp.where(
        zpix_spread != cp.float32(0.0),
        pfh_raw,
        cp.full(zpix_gpu.shape, cp.float32(fog_front), dtype=cp.float32),
    )
    pfh = cp.where(zpix_clamped < zpix_min, cp.float32(1.0), pfh).astype(cp.float32)

    pixel_types = type_lookup_gpu[atom_gpu]
    atom_colors = color_lut[pixel_types]
    rcolor = pfh[:, :, None] * (cone_gpu[:, :, None] * atom_colors) + (cp.float32(1.0) - pfh[:, :, None]) * fog_color_gpu[None, None, :]
    rgb = (cp.float32(1.0) - l_opacity_gpu[:, :, None]) * rcolor
    alpha = cp.maximum((pixel_types != 0).astype(cp.float32), l_opacity_gpu)
    return rgb.astype(cp.float32), alpha.astype(cp.float32)


def _composite_mlx(
    *,
    zpix: Any,
    atom_buf: Any,
    pconetot: Any,
    l_opacity: Any,
    type_lookup: np.ndarray,
    colortype: np.ndarray,
    fog_color: np.ndarray,
    fog_front: float,
    fog_back: float,
    zbuf_bg: float,
):
    mx = _require_mlx()

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
    zfar_f = mx.array(np.float32(100000.0), dtype=mx.float32)
    fog_front_f = mx.array(np.float32(fog_front), dtype=mx.float32)
    pfogdiff_f = mx.array(np.float32(fog_front - fog_back), dtype=mx.float32)

    zpix_max = mx.minimum(mx.max(zpix_mx), zero_f)
    mol_mask: Any = zpix_mx != zbuf_bg_f
    zpix_min = mx.min(mx.where(mol_mask, zpix_mx, zfar_f))
    zpix_clamped = mx.minimum(zpix_mx, zero_f)
    zpix_spread = zpix_max - zpix_min

    safe_spread = mx.where(zpix_spread != zero_f, zpix_spread, one_f)
    pfh_raw = fog_front_f - (zpix_max - zpix_clamped) / safe_spread * pfogdiff_f
    pfh = mx.where(
        zpix_spread != zero_f,
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
    alpha = mx.maximum((pixel_types != 0).astype(mx.float32), l_opacity_mx.astype(mx.float32))
    return rgb, alpha


KERNEL_DISPATCH = {
    "numpy": _raster_chunk_numpy,
    "cupy": _raster_chunk_cupy,
    "mlx": _raster_chunk_mlx,
}

SHADOW_DISPATCH = {
    "numpy": _shadow_cone_numpy,
    "cupy": _shadow_cone_cupy,
    "mlx": _shadow_cone_mlx,
}

OUTLINE34_DISPATCH = {
    "numpy": _outline_kernel34_numpy,
    "cupy": _outline_kernel34_cupy,
    "mlx": _outline_kernel34_mlx,
}

OUTLINE12_DISPATCH = {
    "numpy": _outline_kernel12_numpy,
    "cupy": _outline_kernel12_cupy,
    "mlx": _outline_kernel12_mlx,
}

COMPOSITE_DISPATCH = {
    "numpy": _composite_numpy,
    "cupy": _composite_cupy,
    "mlx": _composite_mlx,
}

_SHADOW_OFFSETS = tuple(
    (di, dj, float(np.sqrt(float(di * di + dj * dj))))
    for di in range(-50, 51, 5)
    for dj in range(-50, 51, 5)
    if not (di == 0 and dj == 0) and (di * di + dj * dj) <= 2500
)
_NUMPY_SHADOW_MAX_CHUNK = 16
_NUMPY_SHADOW_ELEMS_BUDGET = 18_000_000
_MLX_SHADOW_MAX_CHUNK = 32
_MLX_SHADOW_ELEMS_BUDGET = 18_000_000
_NUMPY_OUTLINE_ELEMS_BUDGET = 12_000_000
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


def run_kernel(*, backend: str, **kwargs):
    backend_name = backend.strip().lower()
    if backend_name not in KERNEL_DISPATCH:
        raise ValueError(f"Unsupported backend: {backend}")
    return KERNEL_DISPATCH[backend_name](**kwargs)


def run_shadow_kernel(*, backend: str, **kwargs):
    backend_name = backend.strip().lower()
    if backend_name not in SHADOW_DISPATCH:
        raise ValueError(f"Unsupported backend: {backend}")
    return SHADOW_DISPATCH[backend_name](**kwargs)


def run_outline34_kernel(*, backend: str, **kwargs):
    backend_name = backend.strip().lower()
    if backend_name not in OUTLINE34_DISPATCH:
        raise ValueError(f"Unsupported backend: {backend}")
    return OUTLINE34_DISPATCH[backend_name](**kwargs)


def run_outline12_kernel(*, backend: str, **kwargs):
    backend_name = backend.strip().lower()
    if backend_name not in OUTLINE12_DISPATCH:
        raise ValueError(f"Unsupported backend: {backend}")
    return OUTLINE12_DISPATCH[backend_name](**kwargs)


def run_composite_kernel(*, backend: str, **kwargs):
    backend_name = backend.strip().lower()
    if backend_name not in COMPOSITE_DISPATCH:
        raise ValueError(f"Unsupported backend: {backend}")
    return COMPOSITE_DISPATCH[backend_name](**kwargs)


def supported_backends() -> Sequence[str]:
    return tuple(KERNEL_DISPATCH.keys())


def backend_available(backend: str) -> bool:
    name = backend.strip().lower()
    if name == "numpy":
        return True
    if name == "cupy":
        return _module_import_succeeds("cupy")
    if name == "mlx":
        return _module_import_succeeds("mlx.core")
    return False
