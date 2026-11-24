# Scalar reads inside the compiled kernels are untyped in NumPy's stubs, so
# Pyright cannot follow the Numba dialect. Only the unknown-type rules are off.
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
"""Compiled CPU kernels for the ``numba`` render backend.

The NumPy kernels in :mod:`illustrate.rendering.kernels` are the reference,
and these produce identical bytes. Their callers there pass the stencils and
sentinel depths, so both backends share one definition. Every kernel repeats
the reference's float32 operations in the same order, and ``fastmath`` stays
off so LLVM cannot fuse or reorder them.

Machine code is cached on disk after the first compile. Numba falls back to a
per-user cache directory when the package directory is read-only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import importlib
import threading
from typing import Final, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

_numba = importlib.import_module("numba")

_Kernel = TypeVar("_Kernel", bound=Callable[..., None])


def _compiled(fn: _Kernel) -> _Kernel:
    return cast(
        _Kernel,
        _numba.njit(cache=True, nogil=True, parallel=True, fastmath=False, error_model="numpy")(fn),
    )


_Helper = TypeVar("_Helper", bound=Callable[..., object])


def _compiled_helper(fn: _Helper) -> _Helper:
    """Compile a serial function that the parallel kernels call per pixel."""
    return cast(_Helper, _numba.njit(cache=True, nogil=True, fastmath=False, error_model="numpy")(fn))


_prange = cast(Callable[[int], range], _numba.prange)

# Numba's default workqueue threading layer aborts the process when two
# Python threads launch parallel kernels at once, so launches are serialized.
_LAUNCH_LOCK: Final = threading.Lock()


def _offset_arrays(offsets: Iterable[tuple[int, int]]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    pairs: NDArray[np.int64] = np.array(list(offsets), dtype=np.int64).reshape(-1, 2)
    return np.ascontiguousarray(pairs[:, 0]), np.ascontiguousarray(pairs[:, 1])


# Raster

_BANDS_PER_THREAD: Final = 4


@_compiled
def _raster_bands(
    run_x: NDArray[np.float32],
    run_starts: NDArray[np.int64],
    sy: NDArray[np.float32],
    sz: NDArray[np.float32],
    cx: NDArray[np.float32],
    cy: NDArray[np.float32],
    cz: NDArray[np.float32],
    ia: NDArray[np.int32],
    half_ix: np.float32,
    half_iy: np.float32,
    fix: np.float32,
    fiy: np.float32,
    ibio: np.int32,
    runs_sorted: bool,
    band_count: int,
    thread_count: int,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    bio_buf: NDArray[np.int32],
) -> None:
    # Each band owns a disjoint row range and visits atoms and samples in
    # raster order, keeping a sample when its depth ties or beats the stored
    # one. That reproduces the NumPy rule that equal depths go to the last sample.
    # A run is a stretch of samples that share one sx, and so one image row.
    rows = zpix.shape[0]
    run_count = run_x.shape[0]
    band_rows = (rows + band_count - 1) // band_count
    one = np.float32(1.0)
    # Columns left of this bound always pass the py <= fiy test.
    last_floor = np.int64(np.floor(fiy))
    # prange hands each thread a contiguous block of tasks. Striding tasks
    # across bands spreads the molecule's central rows over every thread.
    per_thread = band_count // thread_count
    for task in _prange(band_count):
        band = (task % per_thread) * thread_count + task // per_thread
        row0 = band * band_rows
        row1 = min(rows, row0 + band_rows)
        first_px = np.float32(row0 + 1)
        end_px = np.float32(row1 + 1)
        for atom in range(cx.shape[0]):
            center_x: np.float32 = cx[atom]
            first_run = 0
            end_run = run_count
            if runs_sorted:
                # Rows grow with sx, so the band holds a contiguous span of runs.
                first_run_px: np.float32 = np.float32(run_x[0] + center_x) + half_ix
                last_run_px: np.float32 = np.float32(run_x[run_count - 1] + center_x) + half_ix
                if last_run_px < first_px or first_run_px >= end_px:
                    continue
                lo = 0
                hi = run_count
                while lo < hi:
                    mid = (lo + hi) // 2
                    if np.float32(run_x[mid] + center_x) + half_ix < first_px:
                        lo = mid + 1
                    else:
                        hi = mid
                first_run = lo
                hi = run_count
                while lo < hi:
                    mid = (lo + hi) // 2
                    if np.float32(run_x[mid] + center_x) + half_ix < end_px:
                        lo = mid + 1
                    else:
                        hi = mid
                end_run = lo
            center_y: np.float32 = cy[atom]
            center_z: np.float32 = cz[atom]
            atom_id = ia[atom]
            for run in range(first_run, end_run):
                px: np.float32 = np.float32(run_x[run] + center_x) + half_ix
                if not (px >= one and px <= fix):
                    continue
                row = np.int32(px) - 1
                if row < row0 or row >= row1:
                    continue
                depth_row = zpix[row]
                atom_row = atom_buf[row]
                bio_row = bio_buf[row]
                start = run_starts[run]
                count = run_starts[run + 1] - start
                run_y = sy[start : start + count]
                run_z = sz[start : start + count]
                # Stamp runs hold consecutive integer sy, which land on
                # consecutive columns unless rounding at a power of two skips
                # one. Check that first, then update the span with selects.
                first_floor = np.int64(np.floor(np.float32(np.float32(run_y[0] + center_y) + half_iy)))
                misses = 0
                for i in range(count):
                    py: np.float32 = np.float32(np.float32(run_y[i] + center_y) + half_iy)
                    misses += np.int64(np.floor(py)) != first_floor + i
                if misses == 0:
                    # py >= 1 exactly when floor(py) >= 1. Only the column at
                    # floor(fiy) can hold samples on both sides of fiy.
                    low = max(0, 1 - first_floor)
                    high = min(count, last_floor - first_floor + 1)
                    if high > low and np.float32(np.float32(run_y[high - 1] + center_y) + half_iy) > fiy:
                        high -= 1
                    if high <= low:
                        continue
                    first_col = first_floor - 1 + low
                    span_depth = depth_row[first_col : first_col + high - low]
                    span_atom = atom_row[first_col : first_col + high - low]
                    span_bio = bio_row[first_col : first_col + high - low]
                    span_z = run_z[low:high]
                    for i in range(high - low):
                        depth: np.float32 = np.float32(span_z[i] + center_z)
                        stored: np.float32 = span_depth[i]
                        wins = depth >= stored
                        # Bit masks rather than ternaries let the integer
                        # updates vectorize. take is all ones when depth wins.
                        take = -np.int32(wins)
                        span_depth[i] = depth if wins else stored
                        span_atom[i] = (atom_id & take) | (span_atom[i] & ~take)
                        span_bio[i] = (ibio & take) | (span_bio[i] & ~take)
                    continue
                for i in range(count):
                    py = np.float32(np.float32(run_y[i] + center_y) + half_iy)
                    if not (py >= one and py <= fiy):
                        continue
                    col = np.int32(py) - 1
                    depth = np.float32(run_z[i] + center_z)
                    if depth >= depth_row[col]:
                        depth_row[col] = depth
                        atom_row[col] = atom_id
                        bio_row[col] = ibio


def raster_chunk(
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
    ibio: int,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    bio_buf: NDArray[np.int32],
) -> None:
    """Rasterize a chunk of atoms in place. ``fix`` and ``fiy`` must be validated."""
    sx_c = np.ascontiguousarray(sx, dtype=np.float32)
    if sx_c.size == 0 or np.size(c_cx) == 0:
        return
    changes = np.flatnonzero(sx_c[1:] != sx_c[:-1]) + 1
    run_starts = np.concatenate(([0], changes, [sx_c.size])).astype(np.int64)
    run_x = sx_c[run_starts[:-1]]
    # The kernel needs C-contiguous buffers. Strided views get updated copies.
    zpix_c = np.ascontiguousarray(zpix, dtype=np.float32)
    atom_c = np.ascontiguousarray(atom_buf, dtype=np.int32)
    bio_c = np.ascontiguousarray(bio_buf, dtype=np.int32)
    # Bands are a multiple of the thread count, so striding visits each once.
    thread_count = max(1, min(int(_numba.get_num_threads()), zpix_c.shape[0]))
    band_count = thread_count * max(1, min(_BANDS_PER_THREAD, zpix_c.shape[0] // thread_count))
    with _LAUNCH_LOCK:
        _raster_bands(
            run_x,
            run_starts,
            np.ascontiguousarray(sy, dtype=np.float32),
            np.ascontiguousarray(sz, dtype=np.float32),
            np.ascontiguousarray(c_cx, dtype=np.float32),
            np.ascontiguousarray(c_cy, dtype=np.float32),
            np.ascontiguousarray(c_cz, dtype=np.float32),
            np.ascontiguousarray(c_ia, dtype=np.int32),
            np.float32(half_ix),
            np.float32(half_iy),
            np.float32(fix),
            np.float32(fiy),
            np.int32(ibio),
            bool(np.all(run_x[1:] > run_x[:-1])),
            band_count,
            thread_count,
            zpix_c,
            atom_c,
            bio_c,
        )
    if zpix_c is not zpix:
        zpix[...] = zpix_c
    if atom_c is not atom_buf:
        atom_buf[...] = atom_c
    if bio_c is not bio_buf:
        bio_buf[...] = bio_c


# Shadows


@_compiled
def _shadow_rows(
    padded: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    pad: int,
    di: NDArray[np.int64],
    dj: NDArray[np.int64],
    radius_angle: NDArray[np.float32],
    group_starts: NDArray[np.int64],
    strength: np.float32,
    min_z: np.float32,
    max_dark: np.float32,
    out: NDArray[np.float32],
) -> None:
    rows, cols = atom_buf.shape
    one = np.float32(1.0)
    for row in _prange(rows):
        for col in range(cols):
            out[row, col] = one
        left = cols
        right = 0
        for col in range(cols):
            if atom_buf[row, col] != 0:
                left = min(left, col)
                right = col + 1
        if left >= right:
            continue
        width = right - left
        center = padded[row + pad, left + pad : right + pad]
        base = np.empty(width, dtype=np.float32)
        floor = np.empty(width, dtype=np.float32)
        threshold = np.empty(width, dtype=np.float32)
        count = np.zeros(width, dtype=np.int32)
        for col in range(width):
            base[col] = center[col] - min_z
            floor[col] = center[col] + min_z
        # Scan each offset along the row so the inner loop reads contiguous
        # memory and vectorizes, rather than gathering every neighbor per pixel.
        # Four offsets share each pass, so thresholds and counts load once.
        for group in range(radius_angle.shape[0]):
            angle_term: np.float32 = radius_angle[group]
            for col in range(width):
                value: np.float32 = np.float32(base[col] + angle_term)
                threshold[col] = value if value >= floor[col] else floor[col]
            offset = group_starts[group]
            end = group_starts[group + 1]
            while offset + 4 <= end:
                first = padded[row + pad + di[offset], left + pad + dj[offset] : right + pad + dj[offset]]
                second = padded[row + pad + di[offset + 1], left + pad + dj[offset + 1] : right + pad + dj[offset + 1]]
                third = padded[row + pad + di[offset + 2], left + pad + dj[offset + 2] : right + pad + dj[offset + 2]]
                fourth = padded[row + pad + di[offset + 3], left + pad + dj[offset + 3] : right + pad + dj[offset + 3]]
                for col in range(width):
                    limit: np.float32 = threshold[col]
                    count[col] += (
                        np.int32(first[col] > limit)
                        + np.int32(second[col] > limit)
                        + np.int32(third[col] > limit)
                        + np.int32(fourth[col] > limit)
                    )
                offset += 4
            while offset < end:
                neighbor = padded[row + pad + di[offset], left + pad + dj[offset] : right + pad + dj[offset]]
                for col in range(width):
                    count[col] += neighbor[col] > threshold[col]
                offset += 1
        for col in range(width):
            if atom_buf[row, left + col] != 0:
                shade: np.float32 = np.float32(one - np.float32(count[col]) * strength)
                out[row, left + col] = shade if shade > max_dark else max_dark


def shadow_cone(
    *,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    shadow_strength: float,
    shadow_angle: float,
    shadow_min_z: float,
    shadow_max_dark: float,
    offsets_by_radius: Mapping[np.float32, Sequence[tuple[int, int]]],
    pad: int,
    outside_depth: float,
) -> NDArray[np.float32]:
    """Shade occupied pixels by counting occluders at ``offsets_by_radius``.

    ``pad`` must cover the largest offset. Off-image neighbors read
    ``outside_depth``. Hit counts do not depend on order, so offsets that share
    a radius share one threshold.
    """
    depth = np.asarray(zpix, dtype=np.float32)
    atom = np.ascontiguousarray(atom_buf, dtype=np.int32)
    out = np.empty(depth.shape, dtype=np.float32)
    if depth.size == 0:
        return out
    padded = np.pad(depth, pad, mode="constant", constant_values=np.float32(outside_depth))
    di, dj = _offset_arrays(offset for offsets in offsets_by_radius.values() for offset in offsets)
    group_sizes = [len(offsets) for offsets in offsets_by_radius.values()]
    group_starts = np.concatenate(([0], np.cumsum(group_sizes))).astype(np.int64)
    # Same float32 product as the NumPy kernel's np.multiply(radius, angle).
    radii = np.array(list(offsets_by_radius), dtype=np.float32)
    radius_angle = np.multiply(radii, np.float32(shadow_angle))
    with _LAUNCH_LOCK:
        _shadow_rows(
            padded,
            atom,
            pad,
            di,
            dj,
            radius_angle,
            group_starts,
            np.float32(shadow_strength),
            np.float32(shadow_min_z),
            np.float32(shadow_max_dark),
            out,
        )
    return out


# Outlines, kernels 3 and 4


@_compiled
def _group_maps(
    atom_buf: NDArray[np.int32],
    su_lookup: NDArray[np.int32],
    res_lookup: NDArray[np.int32],
    subunit: NDArray[np.int32],
    residue: NDArray[np.float32],
) -> None:
    rows, cols = atom_buf.shape
    for row in _prange(rows):
        for col in range(cols):
            atom = atom_buf[row, col]
            subunit[row, col] = su_lookup[atom]
            residue[row, col] = np.float32(res_lookup[atom])


@_compiled_helper
def _border_hit(
    subunit: np.int32, assembly: np.int32, near_subunit: np.int32, near_assembly: np.int32
) -> np.int32:
    return np.int32((near_subunit != subunit) | (near_assembly != assembly))


@_compiled_helper
def _residue_hit(residue: np.float32, near_residue: np.float32, residue_diff: np.float32) -> np.int32:
    return np.int32(abs(np.float32(residue - near_residue)) > residue_diff)


@_compiled_helper
def _contour_add(
    total: np.float32, depth: np.float32, near_depth: np.float32, depth_min: np.float32, depth_range: np.float32
) -> np.float32:
    diff = abs(np.float32(depth - near_depth))
    step = np.float32(np.float32(diff - depth_min) / depth_range)
    step = min(step, np.float32(1.0))
    return np.float32(total + step) if diff > depth_min else total


@_compiled
def _outline_rows(
    depth: NDArray[np.float32],
    subunit: NDArray[np.int32],
    assembly: NDArray[np.int32],
    residue: NDArray[np.float32],
    group_di: NDArray[np.int64],
    group_dj: NDArray[np.int64],
    contour_di: NDArray[np.int64],
    contour_dj: NDArray[np.int64],
    residue_diff: np.float32,
    residue_scaled: bool,
    residue_low: np.float32,
    residue_range: np.float32,
    subunit_scaled: bool,
    subunit_low: np.float32,
    subunit_range: np.float32,
    contour_on: bool,
    depth_min: np.float32,
    depth_range: np.float32,
    contour_scaled: bool,
    contour_low: np.float32,
    contour_range: np.float32,
    out: NDArray[np.float32],
) -> None:
    rows, cols = depth.shape
    zero = np.float32(0.0)
    one = np.float32(1.0)
    group_count = group_di.shape[0]
    contour_count = contour_di.shape[0]
    # Offsets stay within two pixels, so every neighbor of the interior
    # columns is in bounds. Index slices by a loop counter there, which lets
    # Numba drop its negative-index checks and vectorize. Four offsets share a
    # pass, so the center values and running counts load once per pass.
    inner_end = max(cols - 2, 2)
    span = inner_end - 2
    for row in _prange(rows):
        for col in range(cols):
            out[row, col] = zero
        # Group borders skip one pixel at the edge, the depth contour two.
        if row == 0 or row == rows - 1 or cols < 3:
            continue
        subunit_count = np.zeros(cols, dtype=np.int32)
        residue_count = np.zeros(cols, dtype=np.int32)
        here_subunit = subunit[row, 2:inner_end]
        here_assembly = assembly[row, 2:inner_end]
        here_residue = residue[row, 2:inner_end]
        subunit_hits = subunit_count[2:inner_end]
        residue_hits = residue_count[2:inner_end]
        offset = 0
        while offset < group_count:
            # Clamped rows match the NumPy kernel's edge padding.
            r0 = min(max(row + group_di[offset], 0), rows - 1)
            s0 = group_dj[offset] + 2
            if offset + 4 <= group_count:
                r1 = min(max(row + group_di[offset + 1], 0), rows - 1)
                r2 = min(max(row + group_di[offset + 2], 0), rows - 1)
                r3 = min(max(row + group_di[offset + 3], 0), rows - 1)
                s1 = group_dj[offset + 1] + 2
                s2 = group_dj[offset + 2] + 2
                s3 = group_dj[offset + 3] + 2
                sub0 = subunit[r0, s0 : s0 + span]
                sub1 = subunit[r1, s1 : s1 + span]
                sub2 = subunit[r2, s2 : s2 + span]
                sub3 = subunit[r3, s3 : s3 + span]
                asm0 = assembly[r0, s0 : s0 + span]
                asm1 = assembly[r1, s1 : s1 + span]
                asm2 = assembly[r2, s2 : s2 + span]
                asm3 = assembly[r3, s3 : s3 + span]
                res0 = residue[r0, s0 : s0 + span]
                res1 = residue[r1, s1 : s1 + span]
                res2 = residue[r2, s2 : s2 + span]
                res3 = residue[r3, s3 : s3 + span]
                for i in range(span):
                    su = here_subunit[i]
                    asm = here_assembly[i]
                    res = here_residue[i]
                    subunit_hits[i] += (
                        _border_hit(su, asm, sub0[i], asm0[i])
                        + _border_hit(su, asm, sub1[i], asm1[i])
                        + _border_hit(su, asm, sub2[i], asm2[i])
                        + _border_hit(su, asm, sub3[i], asm3[i])
                    )
                    residue_hits[i] += (
                        _residue_hit(res, res0[i], residue_diff)
                        + _residue_hit(res, res1[i], residue_diff)
                        + _residue_hit(res, res2[i], residue_diff)
                        + _residue_hit(res, res3[i], residue_diff)
                    )
                offset += 4
            else:
                sub0 = subunit[r0, s0 : s0 + span]
                asm0 = assembly[r0, s0 : s0 + span]
                res0 = residue[r0, s0 : s0 + span]
                for i in range(span):
                    subunit_hits[i] += _border_hit(here_subunit[i], here_assembly[i], sub0[i], asm0[i])
                    residue_hits[i] += _residue_hit(here_residue[i], res0[i], residue_diff)
                offset += 1
        # Columns 1 and cols - 2 clamp neighbors that fall off the edge.
        for edge in range(2):
            col = 1 if edge == 0 else cols - 2
            if edge == 1 and col < 2:
                continue
            for offset in range(group_count):
                near_row = min(max(row + int(group_di[offset]), 0), rows - 1)
                near = min(max(col + int(group_dj[offset]), 0), cols - 1)
                subunit_count[col] += _border_hit(
                    subunit[row, col], assembly[row, col], subunit[near_row, near], assembly[near_row, near]
                )
                residue_count[col] += _residue_hit(residue[row, col], residue[near_row, near], residue_diff)
        total = np.zeros(cols, dtype=np.float32)
        contour_row = contour_on and row >= 2 and row < rows - 2
        if contour_row:
            # Accumulate in the NumPy kernel's offset order so the float32
            # sum rounds identically.
            here_depth = depth[row, 2:inner_end]
            sums = total[2:inner_end]
            offset = 0
            while offset < contour_count:
                s0 = contour_dj[offset] + 2
                near0 = depth[row + contour_di[offset], s0 : s0 + span]
                if offset + 4 <= contour_count:
                    s1 = contour_dj[offset + 1] + 2
                    s2 = contour_dj[offset + 2] + 2
                    s3 = contour_dj[offset + 3] + 2
                    near1 = depth[row + contour_di[offset + 1], s1 : s1 + span]
                    near2 = depth[row + contour_di[offset + 2], s2 : s2 + span]
                    near3 = depth[row + contour_di[offset + 3], s3 : s3 + span]
                    for i in range(span):
                        here: np.float32 = here_depth[i]
                        running = _contour_add(sums[i], here, near0[i], depth_min, depth_range)
                        running = _contour_add(running, here, near1[i], depth_min, depth_range)
                        running = _contour_add(running, here, near2[i], depth_min, depth_range)
                        sums[i] = _contour_add(running, here, near3[i], depth_min, depth_range)
                    offset += 4
                else:
                    for i in range(span):
                        sums[i] = _contour_add(sums[i], here_depth[i], near0[i], depth_min, depth_range)
                    offset += 1
        for col in range(1, cols - 1):
            group = zero
            if residue_scaled:
                level: np.float32 = np.float32(np.float32(np.float32(residue_count[col]) - residue_low) / residue_range)
                group = min(max(level, zero), one)
            if subunit_scaled:
                level = np.float32(np.float32(np.float32(subunit_count[col]) - subunit_low) / subunit_range)
                group = max(group, min(max(level, zero), one))
            contour = zero
            if contour_row and contour_scaled and col >= 2 and col < cols - 2:
                level = np.float32(np.float32(total[col] - contour_low) / contour_range)
                level = min(max(level, zero), one)
                if level > zero:
                    contour = min(np.float32(level * np.float32(1.5)), one)
            out[row, col] = contour if contour > group else group


def outline_kernel34(
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
    group_offsets: Sequence[tuple[int, int]],
    contour_offsets: Sequence[tuple[int, int]],
) -> NDArray[np.float32]:
    """Combine group borders and depth contours from validated stencils.

    Offsets must stay within two pixels, because the kernel reads the
    neighbors of interior columns without bounds checks.
    """
    if any(max(abs(di), abs(dj)) > 2 for di, dj in (*group_offsets, *contour_offsets)):
        raise ValueError("outline offsets must stay within two pixels")
    depth = np.ascontiguousarray(zpix, dtype=np.float32)
    atom = np.ascontiguousarray(atom_buf, dtype=np.int32)
    out = np.empty(depth.shape, dtype=np.float32)
    if depth.size == 0:
        return out
    subunit = np.empty(atom.shape, dtype=np.int32)
    residue = np.empty(atom.shape, dtype=np.float32)
    group_di, group_dj = _offset_arrays(group_offsets)
    contour_di, contour_dj = _offset_arrays(contour_offsets)
    depth_range = float(z_diff_max - z_diff_min)
    with _LAUNCH_LOCK:
        _group_maps(
            atom,
            np.ascontiguousarray(su_lookup, dtype=np.int32),
            np.ascontiguousarray(res_lookup, dtype=np.int32),
            subunit,
            residue,
        )
        _outline_rows(
            depth,
            subunit,
            np.ascontiguousarray(bio_buf, dtype=np.int32),
            residue,
            group_di,
            group_dj,
            contour_di,
            contour_dj,
            np.float32(residue_diff),
            residue_high != residue_low,
            np.float32(residue_low),
            np.float32(residue_high - residue_low),
            subunit_high != subunit_low,
            np.float32(subunit_low),
            np.float32(subunit_high - subunit_low),
            depth_range != 0.0,
            np.float32(z_diff_min),
            np.float32(depth_range),
            contour_high != contour_low,
            np.float32(contour_low),
            np.float32(contour_high - contour_low),
            out,
        )
    return out


# Composite


@_compiled_helper
def _channel(
    haze: np.float32, shade: np.float32, color: np.float32, fog_weight: np.float32, fog: np.float32, ink: np.float32
) -> np.float32:
    lit = np.float32(haze * np.float32(shade * color))
    return np.float32(ink * np.float32(lit + np.float32(fog_weight * fog)))


@_compiled_helper
def _shade_pixel(
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    cone: NDArray[np.float32],
    outline: NDArray[np.float32],
    type_lookup: NDArray[np.int32],
    colors: NDArray[np.float32],
    fog: NDArray[np.float32],
    use_spread: bool,
    zpix_max: np.float32,
    zpix_spread: np.float32,
    zpix_min: np.float32,
    fog_front: np.float32,
    fog_diff: np.float32,
    row: int,
    col: int,
) -> tuple[np.float32, np.float32, np.float32, np.float32]:
    """Return linear red, green, blue and alpha for one pixel."""
    zero = np.float32(0.0)
    one = np.float32(1.0)
    depth: np.float32 = min(zpix[row, col], zero)
    if depth < zpix_min:
        haze = one
    elif use_spread:
        haze = np.float32(fog_front - np.float32(np.float32(np.float32(zpix_max - depth) / zpix_spread) * fog_diff))
    else:
        haze = fog_front
    pixel_type = int(type_lookup[atom_buf[row, col]])
    shade: np.float32 = cone[row, col]
    line: np.float32 = outline[row, col]
    fog_weight = np.float32(one - haze)
    ink = np.float32(one - line)
    covered = one if pixel_type != 0 else zero
    return (
        _channel(haze, shade, colors[pixel_type, 0], fog_weight, fog[0], ink),
        _channel(haze, shade, colors[pixel_type, 1], fog_weight, fog[1], ink),
        _channel(haze, shade, colors[pixel_type, 2], fog_weight, fog[2], ink),
        covered if covered > line else line,
    )


@_compiled_helper
def _to_u8(value: np.float32) -> np.uint8:
    # Truncate, then clamp, as the NumPy path does with astype(int32).clip(0, 255).
    level = np.int32(np.float32(value * np.float32(255.0)))
    return np.uint8(min(max(level, 0), 255))


@_compiled
def _composite_pixels(
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    cone: NDArray[np.float32],
    outline: NDArray[np.float32],
    type_lookup: NDArray[np.int32],
    colors: NDArray[np.float32],
    fog: NDArray[np.float32],
    use_spread: bool,
    zpix_max: np.float32,
    zpix_spread: np.float32,
    zpix_min: np.float32,
    fog_front: np.float32,
    fog_diff: np.float32,
    rgb: NDArray[np.float32],
    alpha: NDArray[np.float32],
) -> None:
    rows, cols = zpix.shape
    for row in _prange(rows):
        for col in range(cols):
            red, green, blue, cover = _shade_pixel(
                zpix, atom_buf, cone, outline, type_lookup, colors, fog,
                use_spread, zpix_max, zpix_spread, zpix_min, fog_front, fog_diff, row, col,
            )
            rgb[row, col, 0] = red
            rgb[row, col, 1] = green
            rgb[row, col, 2] = blue
            alpha[row, col] = cover


@_compiled
def _composite_pixels_u8(
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    cone: NDArray[np.float32],
    outline: NDArray[np.float32],
    type_lookup: NDArray[np.int32],
    colors: NDArray[np.float32],
    fog: NDArray[np.float32],
    use_spread: bool,
    zpix_max: np.float32,
    zpix_spread: np.float32,
    zpix_min: np.float32,
    fog_front: np.float32,
    fog_diff: np.float32,
    rgb: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
) -> None:
    rows, cols = zpix.shape
    for row in _prange(rows):
        for col in range(cols):
            red, green, blue, cover = _shade_pixel(
                zpix, atom_buf, cone, outline, type_lookup, colors, fog,
                use_spread, zpix_max, zpix_spread, zpix_min, fog_front, fog_diff, row, col,
            )
            rgb[row, col, 0] = _to_u8(red)
            rgb[row, col, 1] = _to_u8(green)
            rgb[row, col, 2] = _to_u8(blue)
            alpha[row, col] = _to_u8(cover)


@_compiled
def _depth_extents(
    depth: NDArray[np.float32],
    background: np.float32,
    row_max: NDArray[np.float32],
    row_min: NDArray[np.float32],
    row_covered: NDArray[np.bool_],
) -> None:
    # Per-row extremes, where NaN wins as it does in ndarray.max and min. The
    # minimum covers only pixels that differ from the background depth.
    rows, cols = depth.shape
    for row in _prange(rows):
        high: np.float32 = depth[row, 0]
        low = np.float32(np.inf)
        covered = False
        for col in range(cols):
            value: np.float32 = depth[row, col]
            if value > high or value != value:
                high = value if high == high else high
            if value != background:
                covered = True
                if value < low or value != value:
                    low = value if low == low else low
        row_max[row] = high
        row_min[row] = low
        row_covered[row] = covered


def _composite(
    kernel: Callable[..., None],
    out_dtype: type[np.float32] | type[np.uint8],
    *,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    pconetot: NDArray[np.float32],
    l_opacity: NDArray[np.float32],
    type_lookup: NDArray[np.int32],
    colortype: NDArray[np.float32],
    fog_color: NDArray[np.float32],
    fog_front: float,
    fog_back: float,
    zbuf_bg: float,
    zbuf_far: float,
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    depth = np.ascontiguousarray(zpix, dtype=np.float32)
    rgb = np.empty((*depth.shape, 3), dtype=out_dtype)
    alpha = np.empty(depth.shape, dtype=out_dtype)
    if depth.size == 0:
        return rgb, alpha
    # Scalars follow the NumPy kernel: Python floats, rounded to float32
    # where they meet the image.
    row_max = np.empty(depth.shape[0], dtype=np.float32)
    row_min = np.empty(depth.shape[0], dtype=np.float32)
    row_covered = np.empty(depth.shape[0], dtype=np.bool_)
    with _LAUNCH_LOCK:
        _depth_extents(depth, np.float32(zbuf_bg), row_max, row_min, row_covered)
    zpix_max = min(float(row_max.max()), 0.0)
    zpix_min = float(row_min[row_covered].min()) if row_covered.any() else float(zbuf_far)
    zpix_spread = zpix_max - zpix_min
    with _LAUNCH_LOCK:
        kernel(
            depth,
            np.ascontiguousarray(atom_buf, dtype=np.int32),
            np.ascontiguousarray(pconetot, dtype=np.float32),
            np.ascontiguousarray(l_opacity, dtype=np.float32),
            np.ascontiguousarray(type_lookup, dtype=np.int32),
            np.ascontiguousarray(colortype, dtype=np.float32),
            np.ascontiguousarray(fog_color, dtype=np.float32),
            zpix_spread != 0.0,
            np.float32(zpix_max),
            np.float32(zpix_spread),
            np.float32(zpix_min),
            np.float32(fog_front),
            np.float32(float(fog_front - fog_back)),
            rgb,
            alpha,
        )
    return rgb, alpha


def composite(
    *,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    pconetot: NDArray[np.float32],
    l_opacity: NDArray[np.float32],
    type_lookup: NDArray[np.int32],
    colortype: NDArray[np.float32],
    fog_color: NDArray[np.float32],
    fog_front: float,
    fog_back: float,
    zbuf_bg: float,
    zbuf_far: float,
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    """Return linear RGB and alpha. ``zbuf_far`` is the fog floor of an empty image."""
    return _composite(
        _composite_pixels, np.float32, zpix=zpix, atom_buf=atom_buf, pconetot=pconetot,
        l_opacity=l_opacity, type_lookup=type_lookup, colortype=colortype, fog_color=fog_color,
        fog_front=fog_front, fog_back=fog_back, zbuf_bg=zbuf_bg, zbuf_far=zbuf_far,
    )


def composite_u8(
    *,
    zpix: NDArray[np.float32],
    atom_buf: NDArray[np.int32],
    pconetot: NDArray[np.float32],
    l_opacity: NDArray[np.float32],
    type_lookup: NDArray[np.int32],
    colortype: NDArray[np.float32],
    fog_color: NDArray[np.float32],
    fog_front: float,
    fog_back: float,
    zbuf_bg: float,
    zbuf_far: float,
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    """Return 8-bit RGB and alpha without materializing the linear planes."""
    return _composite(
        _composite_pixels_u8, np.uint8, zpix=zpix, atom_buf=atom_buf, pconetot=pconetot,
        l_opacity=l_opacity, type_lookup=type_lookup, colortype=colortype, fog_color=fog_color,
        fog_front=fog_front, fog_back=fog_back, zbuf_bg=zbuf_bg, zbuf_far=zbuf_far,
    )


__all__ = ["composite", "composite_u8", "outline_kernel34", "raster_chunk", "shadow_cone"]
