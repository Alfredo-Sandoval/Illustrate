from __future__ import annotations

from collections.abc import Sequence
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from illustrate.math3d import catenate, rotation_x, rotation_y, rotation_z
from illustrate.raster_kernel import (
    backend_available,
    run_composite_kernel,
    run_kernel,
    run_outline12_kernel,
    run_outline34_kernel,
    run_shadow_kernel,
    supported_backends,
)
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
_SPHERE_CACHE: dict[int, np.ndarray] = {}


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
    cached = _SPHERE_CACHE.get(cache_key)
    if cached is not None:
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
    _SPHERE_CACHE[cache_key] = sphere
    return sphere


def _to_u8(arr: np.ndarray) -> np.ndarray:
    out = np.clip(arr.astype(np.int32), 0, 255)
    return out.astype(np.uint8)


def _render_program(program: CommandProgram, atoms: AtomTable, *, backend: str = "numpy") -> RenderResult:
    backend_name = _resolve_render_backend(backend)
    transform = program.transform
    world = program.world
    outlines = program.outlines

    colortype, radtype = _build_rule_arrays(program)
    ndes = min(len(program.selection_rules), 1000)

    # Match Fortran world clamping semantics.
    background = np.array(world.background, dtype=np.float32)
    fog_color = np.array(world.fog_color, dtype=np.float32)
    background = np.minimum(background, 1.0)
    fog_color = np.minimum(fog_color, 1.0)
    colortype[0] = background

    fog_front = float(min(world.fog_front, 1.0))
    fog_back = float(abs(world.fog_back))
    fog_back = min(fog_back, 1.0)

    ixsize = int(world.width)
    iysize = int(world.height)
    ixsize = min(ixsize, 3000)
    iysize = min(iysize, 3000)

    nbiomat = max(int(atoms.nbiomat), 1)
    if atoms.nbiomat == 0:
        biomats = np.zeros((2, 4, 4), dtype=np.float32)
        biomats[1] = np.eye(4, dtype=np.float32)
    else:
        biomats = np.zeros((nbiomat + 1, 4, 4), dtype=np.float32)
        for ibio in range(1, nbiomat + 1):
            biomats[ibio] = np.eye(4, dtype=np.float32)
            biomats[ibio, :3, :4] = atoms.biomat[ibio]

    n = int(atoms.n)
    rm = transform.rm.astype(np.float32)
    rscale = float(transform.scale)
    xtran, ytran, ztran = float(transform.xtran), float(transform.ytran), float(transform.ztran)

    radius_max = 0.0
    for i in range(1, ndes + 1):
        radtype[i] = np.float32(radtype[i] * rscale)
        radius_max = max(radius_max, float(radtype[i]))

    coords_arr = atoms.coord[1 : n + 1].astype(np.float32) if n > 0 else np.zeros((0, 3), dtype=np.float32)

    # Auto-centering / auto-size
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

    ixsize = min(ixsize, 3000)
    iysize = min(iysize, 3000)
    ixsize = (ixsize // 2) * 2
    iysize = (iysize // 2) * 2

    if ixsize <= 0 or iysize <= 0:
        rgb_empty = np.zeros((0, 0, 3), dtype=np.uint8)
        op_empty = np.zeros((0, 0), dtype=np.uint8)
        return RenderResult(rgb=rgb_empty, opacity=op_empty, width=ixsize, height=iysize)

    pix = np.zeros((ixsize, iysize, 4), dtype=np.float32)
    cupy_mod = None
    mlx_mod = None
    if backend_name == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("CuPy backend requested but CuPy is unavailable") from exc
        cupy_mod = cp
        zpix = cp.full((ixsize, iysize), ZBUF_BG, dtype=cp.float32)
        atom_buf = cp.zeros((ixsize, iysize), dtype=cp.int32)
        bio_buf = cp.ones((ixsize, iysize), dtype=cp.int32)
    elif backend_name == "mlx":
        try:
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("MLX backend requested but MLX is unavailable") from exc
        mlx_mod = mx
        zpix = mx.full((ixsize, iysize), ZBUF_BG, dtype=mx.float32)
        atom_buf = mx.zeros((ixsize, iysize), dtype=mx.int32)
        bio_buf = mx.ones((ixsize, iysize), dtype=mx.int32)
    else:
        zpix = np.full((ixsize, iysize), ZBUF_BG, dtype=np.float32)
        atom_buf = np.zeros((ixsize, iysize), dtype=np.int32)
        bio_buf = np.ones((ixsize, iysize), dtype=np.int32)

    # Sphere rasterization (vectorized)
    _CHUNK_ATOMS = 512  # atoms per chunk to bound memory (~20 MB)

    if n > 0:
        atom_types = atoms.type_idx[1 : n + 1]
        half_ix = float(ixsize) / 2.0
        half_iy = float(iysize) / 2.0
        fix = float(ixsize)
        fiy = float(iysize)

        for irad in range(1, ndes + 1):
            sphere = _precompute_sphere(float(radtype[irad]))
            if len(sphere) == 0:
                continue

            matching = np.where(atom_types == irad)[0]
            if len(matching) == 0:
                continue

            nv = len(sphere)  # voxels per sphere
            sx = sphere[:, 0]  # (V,)
            sy = sphere[:, 1]
            sz = sphere[:, 2]

            for ibio in range(1, nbiomat + 1):
                bm = biomats[ibio]

                # Batch biomat transform: (N, 3) @ (3, 3).T + (3,)
                centers_bio = coords_arr[matching] @ bm[:3, :3].T + bm[:3, 3]
                # Batch rotation: (N, 3) @ (3, 3)
                centers_rot = centers_bio @ rm[:3, :3]

                # Add centering + translation, then scale
                cx_all = (centers_rot[:, 0] + xtranc + xtran) * rscale
                cy_all = (centers_rot[:, 1] + ytranc + ytran) * rscale
                cz_all = (centers_rot[:, 2] + ztranc + ztran) * rscale

                # Cull atoms behind camera
                visible = cz_all < 0.0
                if not np.any(visible):
                    continue

                vis_idx = np.where(visible)[0]
                cx_vis = cx_all[vis_idx]
                cy_vis = cy_all[vis_idx]
                cz_vis = cz_all[vis_idx]
                ia_vis = matching[vis_idx] + 1  # 1-based atom IDs

                # Process in chunks to bound memory
                for chunk_start in range(0, len(vis_idx), _CHUNK_ATOMS):
                    chunk_end = min(chunk_start + _CHUNK_ATOMS, len(vis_idx))
                    c_cx = cx_vis[chunk_start:chunk_end]  # (M,)
                    c_cy = cy_vis[chunk_start:chunk_end]
                    c_cz = cz_vis[chunk_start:chunk_end]
                    c_ia = ia_vis[chunk_start:chunk_end]
                    zpix, atom_buf, bio_buf = run_kernel(
                        backend=backend_name,
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
                        zpix=zpix,
                        atom_buf=atom_buf,
                        bio_buf=bio_buf,
                    )

    precomputed_outline = None
    if outlines.enabled and outlines.kernel in (3, 4):
        # Match historical behavior: outline kernels operate on z values clamped at 0.
        if cupy_mod is not None:
            zpix_for_outline = cupy_mod.minimum(zpix, 0.0)
        elif mlx_mod is not None:
            zpix_for_outline = mlx_mod.minimum(zpix, 0.0)
        else:
            zpix_for_outline = np.minimum(zpix, 0.0)
        precomputed_outline = run_outline34_kernel(
            backend=backend_name,
            zpix=zpix_for_outline,
            atom_buf=atom_buf,
            bio_buf=bio_buf,
            su_lookup=atoms.su,
            res_lookup=atoms.res,
            residue_diff=float(outlines.residue_diff),
            residue_low=float(outlines.residue_low),
            residue_high=float(outlines.residue_high),
            subunit_low=float(outlines.subunit_low),
            subunit_high=float(outlines.subunit_high),
            z_diff_min=float(outlines.z_diff_min * rscale),
            z_diff_max=float(outlines.z_diff_max * rscale),
            contour_low=float(outlines.contour_low),
            contour_high=float(outlines.contour_high),
            kernel=int(outlines.kernel),
        )
    elif outlines.enabled and outlines.kernel in (1, 2) and backend_name == "numpy":
        precomputed_outline = run_outline12_kernel(
            backend=backend_name,
            zpix=np.minimum(zpix, 0.0),
            atom_buf=atom_buf,
            bio_buf=bio_buf,
            su_lookup=atoms.su,
            res_lookup=atoms.res,
            residue_diff=float(outlines.residue_diff),
            residue_low=float(outlines.residue_low),
            residue_high=float(outlines.residue_high),
            subunit_low=float(outlines.subunit_low),
            subunit_high=float(outlines.subunit_high),
            contour_low=float(outlines.contour_low),
            contour_high=float(outlines.contour_high),
            kernel=int(outlines.kernel),
        )

    if world.shadows:
        pconetot = run_shadow_kernel(
            backend=backend_name,
            zpix=zpix,
            atom_buf=atom_buf,
            shadow_strength=float(world.shadow_strength),
            shadow_angle=float(world.shadow_angle),
            shadow_min_z=float(world.shadow_min_z),
            shadow_max_dark=float(world.shadow_max_dark),
        )
    else:
        if cupy_mod is not None:
            pconetot = cupy_mod.ones((ixsize, iysize), dtype=cupy_mod.float32)
        elif mlx_mod is not None:
            pconetot = mlx_mod.ones((ixsize, iysize), dtype=mlx_mod.float32)
        else:
            pconetot = np.ones((ixsize, iysize), dtype=np.float32)

    if precomputed_outline is not None:
        rgb_linear, alpha_linear = run_composite_kernel(
            backend=backend_name,
            zpix=zpix,
            atom_buf=atom_buf,
            pconetot=pconetot,
            l_opacity=precomputed_outline,
            type_lookup=atoms.type_idx,
            colortype=colortype,
            fog_color=fog_color.astype(np.float32),
            fog_front=float(fog_front),
            fog_back=float(fog_back),
            zbuf_bg=float(ZBUF_BG),
        )
        if cupy_mod is not None:
            rgb_linear = cupy_mod.asnumpy(rgb_linear)
            alpha_linear = cupy_mod.asnumpy(alpha_linear)
        elif mlx_mod is not None:
            mlx_mod.eval(rgb_linear, alpha_linear)
            rgb_linear = np.array(rgb_linear, dtype=np.float32)
            alpha_linear = np.array(alpha_linear, dtype=np.float32)
        else:
            rgb_linear = np.asarray(rgb_linear, dtype=np.float32)
            alpha_linear = np.asarray(alpha_linear, dtype=np.float32)
        rgb = np.swapaxes(_to_u8(rgb_linear * 255.0), 0, 1)
        opacity = np.swapaxes(_to_u8(alpha_linear * 255.0), 0, 1)
        return RenderResult(
            rgb=rgb,
            opacity=opacity,
            width=ixsize,
            height=iysize,
        )

    if cupy_mod is not None:
        zpix = cupy_mod.asnumpy(zpix)
        atom_buf = cupy_mod.asnumpy(atom_buf)
        bio_buf = cupy_mod.asnumpy(bio_buf)
        pconetot = cupy_mod.asnumpy(pconetot)
        if precomputed_outline is not None:
            precomputed_outline = cupy_mod.asnumpy(precomputed_outline)
    elif mlx_mod is not None:
        if precomputed_outline is None:
            mlx_mod.eval(zpix, atom_buf, bio_buf, pconetot)
        else:
            mlx_mod.eval(zpix, atom_buf, bio_buf, pconetot, precomputed_outline)
        zpix = np.array(zpix, dtype=np.float32)
        atom_buf = np.array(atom_buf, dtype=np.int32)
        bio_buf = np.array(bio_buf, dtype=np.int32)
        pconetot = np.array(pconetot, dtype=np.float32)
        if precomputed_outline is not None:
            precomputed_outline = np.array(precomputed_outline, dtype=np.float32)

    # Z stats and clamping
    zpix_max = float(zpix.max())
    mol_mask = zpix != ZBUF_BG
    zpix_min = float(zpix[mol_mask].min()) if np.any(mol_mask) else 100000.0
    zpix_max = min(zpix_max, 0.0)
    zpix_spread = zpix_max - zpix_min
    zpix = np.minimum(zpix, 0.0)

    z_diff_min = outlines.z_diff_min * rscale
    z_diff_max = outlines.z_diff_max * rscale
    pfogdiff = fog_front - fog_back

    # Fog
    if zpix_spread != 0.0:
        pfh = fog_front - (zpix_max - zpix) / zpix_spread * pfogdiff
    else:
        pfh = np.full_like(zpix, fog_front)
    pfh = np.where(zpix < zpix_min, 1.0, pfh)

    # Outlines
    l_opacity = np.zeros((ixsize, iysize), dtype=np.float32)
    g_opacity = np.zeros((ixsize, iysize), dtype=np.float32)

    if precomputed_outline is not None:
        l_opacity = np.asarray(precomputed_outline, dtype=np.float32)
    elif outlines.enabled:
        su_map = atoms.su[atom_buf]
        res_map = atoms.res[atom_buf]
        res_map_f = res_map.astype(np.float32)

        r_count = np.zeros((ixsize, iysize), dtype=np.float32)
        g_count = np.zeros((ixsize, iysize), dtype=np.float32)
        rg_pad = 2
        su_padded = np.pad(su_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
        bio_padded = np.pad(bio_buf, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=0)
        res_padded = np.pad(res_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if abs(di * dj) == 4:
                    continue
                shifted_su = _padded_shift_view(su_padded, di, dj, ixsize, iysize, rg_pad)
                shifted_bio = _padded_shift_view(bio_padded, di, dj, ixsize, iysize, rg_pad)
                shifted_res = _padded_shift_view(res_padded, di, dj, ixsize, iysize, rg_pad)

                r_count += ((su_map != shifted_su) | (bio_buf != shifted_bio)).astype(np.float32)
                g_count += (np.abs(res_map_f - shifted_res.astype(np.float32)) > outlines.residue_diff).astype(np.float32)

        if outlines.residue_high != outlines.residue_low:
            g_opacity = np.clip((g_count - outlines.residue_low) / (outlines.residue_high - outlines.residue_low), 0.0, 1.0)
        if outlines.subunit_high != outlines.subunit_low:
            r_opacity = np.clip((r_count - outlines.subunit_low) / (outlines.subunit_high - outlines.subunit_low), 0.0, 1.0)
        else:
            r_opacity = np.zeros_like(g_opacity)
        g_opacity = np.maximum(g_opacity, r_opacity)

        g_opacity[0, :] = 0.0
        g_opacity[-1, :] = 0.0
        g_opacity[:, 0] = 0.0
        g_opacity[:, -1] = 0.0

        kernel = outlines.kernel
        contour_low = outlines.contour_low
        contour_high = outlines.contour_high

        if kernel in (3, 4):
            l_total = np.zeros((ixsize, iysize), dtype=np.float32)
            if kernel == 3:
                offsets_k = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
            else:
                offsets_k = [(i, j) for i in range(-2, 3) for j in range(-2, 3) if abs(i * j) != 4]

            zpad = 2
            zpix_padded = np.pad(zpix, ((zpad, zpad), (zpad, zpad)), mode="constant", constant_values=0.0)

            denom = z_diff_max - z_diff_min
            for di, dj in offsets_k:
                shifted_z = _padded_shift_view(zpix_padded, di, dj, ixsize, iysize, zpad)
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
            l_opacity = np.where(l_val > 0, np.minimum(1.5 * l_val, 1.0), 0.0)

        elif kernel in (1, 2):
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

            lap = np.zeros((ixsize, iysize), dtype=np.float32)
            zpad = 2
            zpix_padded = np.pad(zpix, ((zpad, zpad), (zpad, zpad)), mode="constant", constant_values=0.0)
            for (di, dj), w in weights.items():
                lap += w * _padded_shift_view(zpix_padded, di, dj, ixsize, iysize, zpad)
            lap = np.abs(lap / 3.0)

            rl = np.zeros((ixsize, iysize), dtype=np.float32)
            l_opacity_ave = np.zeros((ixsize, iysize), dtype=np.float32)
            l_center = np.zeros((ixsize, iysize), dtype=np.float32)

            lap_pad = 1
            lap_padded = np.pad(lap, ((lap_pad, lap_pad), (lap_pad, lap_pad)), mode="constant", constant_values=0.0)

            for ixl in range(-1, 2):
                for iyl in range(-1, 2):
                    l_shifted = _padded_shift_view(lap_padded, ixl, iyl, ixsize, iysize, lap_pad)
                    if contour_high != contour_low:
                        l_v = np.clip((l_shifted - contour_low) / (contour_high - contour_low), 0.0, 1.0)
                    else:
                        l_v = np.zeros_like(l_shifted)
                    rl += (l_v > 0).astype(np.float32)
                    l_opacity_ave += l_v
                    if ixl == 0 and iyl == 0:
                        l_center = l_v.copy()

            l_opacity = np.where(rl >= 6.0, l_opacity_ave / 6.0, l_center)
            l_opacity = np.clip(l_opacity, 0.0, 1.0)

        l_opacity[:2, :] = 0.0
        l_opacity[-2:, :] = 0.0
        l_opacity[:, :2] = 0.0
        l_opacity[:, -2:] = 0.0
        l_opacity = np.maximum(l_opacity, g_opacity)

    pixel_types = atoms.type_idx[atom_buf]

    for ic in range(3):
        atom_colors = colortype[pixel_types, ic]
        rcolor = pfh * (pconetot * atom_colors) + (1.0 - pfh) * fog_color[ic]
        pix[:, :, ic] = (1.0 - l_opacity) * rcolor

    has_atom = (pixel_types != 0).astype(np.float32)
    pix[:, :, 3] = np.maximum(has_atom, l_opacity)

    # Internal buffers are width-major (x, y). Public image arrays are row-major (y, x)
    # so width/height metadata matches PNG/SVG/HTTP image encoders.
    rgb = np.swapaxes(_to_u8(pix[:, :, :3] * 255.0), 0, 1)
    opacity = np.swapaxes(_to_u8(pix[:, :, 3] * 255.0), 0, 1)

    return RenderResult(
        rgb=rgb,
        opacity=opacity,
        width=ixsize,
        height=iysize,
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
    transform = program.transform
    world = program.world

    _colortype, radtype = _build_rule_arrays(program)
    ndes = min(len(program.selection_rules), 1000)

    ixsize = int(world.width)
    iysize = int(world.height)
    ixsize = min(ixsize, 3000)
    iysize = min(iysize, 3000)

    nbiomat = max(int(atoms.nbiomat), 1)
    if atoms.nbiomat == 0:
        biomats = np.zeros((2, 4, 4), dtype=np.float32)
        biomats[1] = np.eye(4, dtype=np.float32)
    else:
        biomats = np.zeros((nbiomat + 1, 4, 4), dtype=np.float32)
        for ibio in range(1, nbiomat + 1):
            biomats[ibio] = np.eye(4, dtype=np.float32)
            biomats[ibio, :3, :4] = atoms.biomat[ibio]

    n = int(atoms.n)
    rm = transform.rm.astype(np.float32)
    rscale = float(transform.scale)
    radius_max = 0.0
    for i in range(1, ndes + 1):
        radtype[i] = np.float32(radtype[i] * rscale)
        radius_max = max(radius_max, float(radtype[i]))

    coords_arr = atoms.coord[1 : n + 1].astype(np.float32) if n > 0 else np.zeros((0, 3), dtype=np.float32)

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

        if ixsize <= 0 or iysize <= 0:
            ixsize = int(-2.0 * ixsize + 2.0 * radius_max + (xmax - xmin) * rscale)
            iysize = int(-2.0 * iysize + 2.0 * radius_max + (ymax - ymin) * rscale)

    ixsize = min(ixsize, 3000)
    iysize = min(iysize, 3000)
    ixsize = (ixsize // 2) * 2
    iysize = (iysize // 2) * 2
    return (ixsize, iysize)


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


def _to_dict(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj


def params_to_json(params: RenderParams) -> str:
    return json.dumps(_to_dict(params), sort_keys=True)


def _coerce_color(payload: object, default: tuple[float, float, float]) -> tuple[float, float, float]:
    values = default if not isinstance(payload, (list, tuple)) else payload
    if len(values) < 3:
        return default
    return (float(values[0]), float(values[1]), float(values[2]))


def _coerce_translate(payload: object) -> tuple[float, float, float]:
    values = payload if isinstance(payload, (list, tuple)) else (0.0, 0.0, 0.0)
    if len(values) < 3:
        return (0.0, 0.0, 0.0)
    return (float(values[0]), float(values[1]), float(values[2]))


def _coerce_object(payload: object, field_name: str) -> dict[str, object]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"{field_name} must be an object")


def params_from_json(json_str: str) -> RenderParams:
    payload = json.loads(json_str)
    if not isinstance(payload, dict):
        raise TypeError("top-level JSON payload must be an object")

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
            rotations=[(str(axis), float(angle)) for axis, angle in transform_payload.get("rotations", [])],
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
