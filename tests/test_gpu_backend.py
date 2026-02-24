from __future__ import annotations

import importlib.util
import importlib
from pathlib import Path

import numpy as np
import pytest

from illustrate.pdb import load_pdb
from illustrate.raster_kernel import (
    backend_available,
    run_composite_kernel,
    run_kernel,
    run_outline12_kernel,
    run_outline34_kernel,
    run_shadow_kernel,
    supported_backends,
)
from illustrate.render import render, render_from_atoms
from illustrate.types import RenderParams, SelectionRule

render_module = importlib.import_module("illustrate.render")


def _write_minimal_pdb(path: Path) -> None:
    line = (
        f"{'ATOM':<6}{1:5d} {'CA':<4}{' ':1}{'GLY':>3} {'A':1}{1:4d}    "
        f"{0.000:8.3f}{0.000:8.3f}{-5.000:8.3f}"
    )
    path.write_text(line + "\nEND\n", encoding="utf-8")


def _minimal_params(pdb_path: Path) -> RenderParams:
    params = RenderParams(
        pdb_path=str(pdb_path),
        rules=[
            SelectionRule(
                record_name="ATOM  ",
                descriptor="----------",
                res_low=0,
                res_high=9999,
                color=(1.0, 0.5, 0.5),
                radius=1.5,
            )
        ],
    )
    params.world.width = 80
    params.world.height = 80
    return params


def test_raster_kernel_dispatch_contract_keys_present() -> None:
    assert tuple(supported_backends()) == ("numpy", "cupy", "mlx")


def test_raster_kernel_dispatch_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        run_kernel(backend="unknown")


def test_shadow_kernel_dispatch_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        run_shadow_kernel(
            backend="unknown",
            zpix=np.zeros((4, 4), dtype=np.float32),
            atom_buf=np.zeros((4, 4), dtype=np.int32),
            shadow_strength=0.0023,
            shadow_angle=2.0,
            shadow_min_z=1.0,
            shadow_max_dark=0.2,
        )


def test_outline_kernel_dispatch_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        run_outline34_kernel(
            backend="unknown",
            zpix=np.zeros((4, 4), dtype=np.float32),
            atom_buf=np.zeros((4, 4), dtype=np.int32),
            bio_buf=np.ones((4, 4), dtype=np.int32),
            su_lookup=np.zeros((8,), dtype=np.int32),
            res_lookup=np.zeros((8,), dtype=np.int32),
            residue_diff=6000.0,
            residue_low=3.0,
            residue_high=8.0,
            subunit_low=3.0,
            subunit_high=10.0,
            z_diff_min=0.0,
            z_diff_max=5.0,
            contour_low=3.0,
            contour_high=10.0,
            kernel=4,
        )


def test_outline12_kernel_dispatch_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        run_outline12_kernel(
            backend="unknown",
            zpix=np.zeros((4, 4), dtype=np.float32),
            atom_buf=np.zeros((4, 4), dtype=np.int32),
            bio_buf=np.ones((4, 4), dtype=np.int32),
            su_lookup=np.zeros((8,), dtype=np.int32),
            res_lookup=np.zeros((8,), dtype=np.int32),
            residue_diff=6000.0,
            residue_low=3.0,
            residue_high=8.0,
            subunit_low=3.0,
            subunit_high=10.0,
            contour_low=3.0,
            contour_high=10.0,
            kernel=2,
        )


def test_composite_kernel_dispatch_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        run_composite_kernel(
            backend="unknown",
            zpix=np.zeros((4, 4), dtype=np.float32),
            atom_buf=np.zeros((4, 4), dtype=np.int32),
            pconetot=np.ones((4, 4), dtype=np.float32),
            l_opacity=np.zeros((4, 4), dtype=np.float32),
            type_lookup=np.zeros((8,), dtype=np.int32),
            colortype=np.zeros((8, 3), dtype=np.float32),
            fog_color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            fog_front=1.0,
            fog_back=1.0,
            zbuf_bg=-10000.0,
        )


def test_backend_available_handles_missing_parent_package(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find_spec(name: str):
        if name in {"mlx.core", "cupy"}:
            raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
        return object()

    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)
    assert backend_available("numpy") is True
    assert backend_available("cupy") is False
    assert backend_available("mlx") is False


def test_shadow_kernel_contract_numpy_vs_mlx() -> None:
    rng = np.random.default_rng(7)
    zpix = rng.normal(loc=-4.0, scale=2.0, size=(22, 24)).astype(np.float32)
    atom_buf = np.zeros((22, 24), dtype=np.int32)
    atom_buf[3:19, 4:20] = 1

    numpy_out = run_shadow_kernel(
        backend="numpy",
        zpix=zpix,
        atom_buf=atom_buf,
        shadow_strength=0.0023,
        shadow_angle=2.0,
        shadow_min_z=1.0,
        shadow_max_dark=0.2,
    )

    if importlib.util.find_spec("mlx.core") is None:
        with pytest.raises(RuntimeError, match="MLX backend requested but MLX is unavailable"):
            run_shadow_kernel(
                backend="mlx",
                zpix=zpix,
                atom_buf=atom_buf,
                shadow_strength=0.0023,
                shadow_angle=2.0,
                shadow_min_z=1.0,
                shadow_max_dark=0.2,
            )
        return

    mlx_out = run_shadow_kernel(
        backend="mlx",
        zpix=zpix,
        atom_buf=atom_buf,
        shadow_strength=0.0023,
        shadow_angle=2.0,
        shadow_min_z=1.0,
        shadow_max_dark=0.2,
    )
    import mlx.core as mx

    mx.eval(mlx_out)
    mlx_np = np.array(mlx_out, dtype=np.float32)
    assert numpy_out.shape == mlx_np.shape
    assert np.max(np.abs(numpy_out - mlx_np)) <= 1e-5


def _outline12_reference(
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
    h, w = zpix.shape
    su_map = su_lookup[atom_buf]
    res_map = res_lookup[atom_buf]
    res_map_f = res_map.astype(np.float32)

    r_count = np.zeros((h, w), dtype=np.float32)
    g_count = np.zeros((h, w), dtype=np.float32)
    rg_pad = 2
    su_padded = np.pad(su_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    bio_padded = np.pad(bio_buf, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=0)
    res_padded = np.pad(res_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="constant", constant_values=9999)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            if abs(di * dj) == 4:
                continue
            shifted_su = su_padded[rg_pad + di : rg_pad + di + h, rg_pad + dj : rg_pad + dj + w]
            shifted_bio = bio_padded[rg_pad + di : rg_pad + di + h, rg_pad + dj : rg_pad + dj + w]
            shifted_res = res_padded[rg_pad + di : rg_pad + di + h, rg_pad + dj : rg_pad + dj + w]
            r_count += ((su_map != shifted_su) | (bio_buf != shifted_bio)).astype(np.float32)
            g_count += (np.abs(res_map_f - shifted_res.astype(np.float32)) > residue_diff).astype(np.float32)

    if residue_high != residue_low:
        g_opacity = np.clip((g_count - residue_low) / (residue_high - residue_low), 0.0, 1.0)
    else:
        g_opacity = np.zeros((h, w), dtype=np.float32)
    if subunit_high != subunit_low:
        r_opacity = np.clip((r_count - subunit_low) / (subunit_high - subunit_low), 0.0, 1.0)
    else:
        r_opacity = np.zeros_like(g_opacity)
    g_opacity = np.maximum(g_opacity, r_opacity)
    g_opacity[0, :] = 0.0
    g_opacity[-1, :] = 0.0
    g_opacity[:, 0] = 0.0
    g_opacity[:, -1] = 0.0

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

    lap = np.zeros((h, w), dtype=np.float32)
    zpad = 2
    zpix_padded = np.pad(zpix, ((zpad, zpad), (zpad, zpad)), mode="constant", constant_values=0.0)
    for (di, dj), weight in weights.items():
        shifted_z = zpix_padded[zpad + di : zpad + di + h, zpad + dj : zpad + dj + w]
        lap += weight * shifted_z
    lap = np.abs(lap / 3.0)

    rl = np.zeros((h, w), dtype=np.float32)
    l_opacity_ave = np.zeros((h, w), dtype=np.float32)
    l_center = np.zeros((h, w), dtype=np.float32)
    lap_pad = 1
    lap_padded = np.pad(lap, ((lap_pad, lap_pad), (lap_pad, lap_pad)), mode="constant", constant_values=0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            shifted_l = lap_padded[lap_pad + di : lap_pad + di + h, lap_pad + dj : lap_pad + dj + w]
            if contour_high != contour_low:
                l_v = np.clip((shifted_l - contour_low) / (contour_high - contour_low), 0.0, 1.0)
            else:
                l_v = np.zeros_like(shifted_l)
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


def test_outline_kernel12_contract_numpy_reference() -> None:
    rng = np.random.default_rng(17)
    zpix = np.minimum(rng.normal(loc=-2.5, scale=1.7, size=(24, 26)).astype(np.float32), 0.0)
    atom_buf = rng.integers(0, 8, size=(24, 26), dtype=np.int32)
    bio_buf = rng.integers(1, 4, size=(24, 26), dtype=np.int32)
    su_lookup = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    res_lookup = np.array([0, 10, 20, 30, 40, 50, 60, 70], dtype=np.int32)

    for kernel in (1, 2):
        actual = run_outline12_kernel(
            backend="numpy",
            zpix=zpix,
            atom_buf=atom_buf,
            bio_buf=bio_buf,
            su_lookup=su_lookup,
            res_lookup=res_lookup,
            residue_diff=5.0,
            residue_low=3.0,
            residue_high=8.0,
            subunit_low=3.0,
            subunit_high=10.0,
            contour_low=3.0,
            contour_high=10.0,
            kernel=kernel,
        )
        expected = _outline12_reference(
            zpix=zpix,
            atom_buf=atom_buf,
            bio_buf=bio_buf,
            su_lookup=su_lookup,
            res_lookup=res_lookup,
            residue_diff=5.0,
            residue_low=3.0,
            residue_high=8.0,
            subunit_low=3.0,
            subunit_high=10.0,
            contour_low=3.0,
            contour_high=10.0,
            kernel=kernel,
        )
        assert actual.shape == expected.shape
        assert np.max(np.abs(actual - expected)) <= 1e-5


def test_outline_kernel34_contract_numpy_vs_mlx() -> None:
    rng = np.random.default_rng(13)
    zpix = rng.normal(loc=-3.0, scale=1.8, size=(28, 30)).astype(np.float32)
    atom_buf = rng.integers(0, 8, size=(28, 30), dtype=np.int32)
    bio_buf = rng.integers(1, 4, size=(28, 30), dtype=np.int32)
    su_lookup = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    res_lookup = np.array([0, 10, 20, 30, 40, 50, 60, 70], dtype=np.int32)

    numpy_out = run_outline34_kernel(
        backend="numpy",
        zpix=zpix,
        atom_buf=atom_buf,
        bio_buf=bio_buf,
        su_lookup=su_lookup,
        res_lookup=res_lookup,
        residue_diff=5.0,
        residue_low=3.0,
        residue_high=8.0,
        subunit_low=3.0,
        subunit_high=10.0,
        z_diff_min=0.0,
        z_diff_max=5.0,
        contour_low=3.0,
        contour_high=10.0,
        kernel=4,
    )

    if importlib.util.find_spec("mlx.core") is None:
        with pytest.raises(RuntimeError, match="MLX backend requested but MLX is unavailable"):
            run_outline34_kernel(
                backend="mlx",
                zpix=zpix,
                atom_buf=atom_buf,
                bio_buf=bio_buf,
                su_lookup=su_lookup,
                res_lookup=res_lookup,
                residue_diff=5.0,
                residue_low=3.0,
                residue_high=8.0,
                subunit_low=3.0,
                subunit_high=10.0,
                z_diff_min=0.0,
                z_diff_max=5.0,
                contour_low=3.0,
                contour_high=10.0,
                kernel=4,
            )
        return

    mlx_out = run_outline34_kernel(
        backend="mlx",
        zpix=zpix,
        atom_buf=atom_buf,
        bio_buf=bio_buf,
        su_lookup=su_lookup,
        res_lookup=res_lookup,
        residue_diff=5.0,
        residue_low=3.0,
        residue_high=8.0,
        subunit_low=3.0,
        subunit_high=10.0,
        z_diff_min=0.0,
        z_diff_max=5.0,
        contour_low=3.0,
        contour_high=10.0,
        kernel=4,
    )
    import mlx.core as mx

    mx.eval(mlx_out)
    mlx_np = np.array(mlx_out, dtype=np.float32)
    assert numpy_out.shape == mlx_np.shape
    assert np.max(np.abs(numpy_out - mlx_np)) <= 1e-5


def test_composite_kernel_contract_numpy_vs_mlx() -> None:
    rng = np.random.default_rng(21)
    zpix = rng.normal(loc=-2.0, scale=1.5, size=(20, 22)).astype(np.float32)
    atom_buf = rng.integers(0, 7, size=(20, 22), dtype=np.int32)
    pconetot = rng.uniform(0.6, 1.0, size=(20, 22)).astype(np.float32)
    l_opacity = rng.uniform(0.0, 0.9, size=(20, 22)).astype(np.float32)
    type_lookup = np.array([0, 3, 5, 9, 15, 23, 37, 99], dtype=np.int32)
    colortype = rng.uniform(0.0, 1.0, size=(1001, 3)).astype(np.float32)
    fog_color = np.array([0.8, 0.7, 0.9], dtype=np.float32)

    rgb_np, alpha_np = run_composite_kernel(
        backend="numpy",
        zpix=zpix,
        atom_buf=atom_buf,
        pconetot=pconetot,
        l_opacity=l_opacity,
        type_lookup=type_lookup,
        colortype=colortype,
        fog_color=fog_color,
        fog_front=0.9,
        fog_back=0.3,
        zbuf_bg=-10000.0,
    )

    if importlib.util.find_spec("mlx.core") is None:
        with pytest.raises(RuntimeError, match="MLX backend requested but MLX is unavailable"):
            run_composite_kernel(
                backend="mlx",
                zpix=zpix,
                atom_buf=atom_buf,
                pconetot=pconetot,
                l_opacity=l_opacity,
                type_lookup=type_lookup,
                colortype=colortype,
                fog_color=fog_color,
                fog_front=0.9,
                fog_back=0.3,
                zbuf_bg=-10000.0,
            )
        return

    rgb_mlx, alpha_mlx = run_composite_kernel(
        backend="mlx",
        zpix=zpix,
        atom_buf=atom_buf,
        pconetot=pconetot,
        l_opacity=l_opacity,
        type_lookup=type_lookup,
        colortype=colortype,
        fog_color=fog_color,
        fog_front=0.9,
        fog_back=0.3,
        zbuf_bg=-10000.0,
    )
    import mlx.core as mx

    mx.eval(rgb_mlx, alpha_mlx)
    rgb_mlx_np = np.array(rgb_mlx, dtype=np.float32)
    alpha_mlx_np = np.array(alpha_mlx, dtype=np.float32)
    assert rgb_np.shape == rgb_mlx_np.shape
    assert alpha_np.shape == alpha_mlx_np.shape
    assert np.max(np.abs(rgb_np - rgb_mlx_np)) <= 1e-5
    assert np.max(np.abs(alpha_np - alpha_mlx_np)) <= 1e-5


def test_render_rejects_unknown_backend(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    with pytest.raises(ValueError, match="Unsupported render backend"):
        render(params, backend="unknown")


def test_render_env_backend_rejects_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    monkeypatch.setenv("ILLUSTRATE_RENDER_BACKEND", "bad-backend")
    with pytest.raises(ValueError, match="Unsupported render backend"):
        render(params)


def test_resolve_render_backend_auto_prefers_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ILLUSTRATE_RENDER_BACKEND", raising=False)
    monkeypatch.setattr(render_module, "backend_available", lambda name: name == "mlx")
    assert render_module._resolve_render_backend(None) == "mlx"


def test_resolve_render_backend_auto_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ILLUSTRATE_RENDER_BACKEND", raising=False)
    monkeypatch.setattr(render_module, "backend_available", lambda name: name == "numpy")
    assert render_module._resolve_render_backend(None) == "numpy"


def test_resolve_render_backend_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILLUSTRATE_RENDER_BACKEND", "numpy")
    monkeypatch.setattr(render_module, "backend_available", lambda _name: False)
    assert render_module._resolve_render_backend(None) == "numpy"


def test_render_cupy_backend_contract(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    atoms = load_pdb(pdb_path, params.rules)

    if importlib.util.find_spec("cupy") is None:
        with pytest.raises(RuntimeError, match="CuPy backend requested but CuPy is unavailable"):
            render_from_atoms(atoms, params, backend="cupy")
        return

    cpu = render_from_atoms(atoms, params, backend="numpy")
    gpu = render_from_atoms(atoms, params, backend="cupy")

    assert cpu.rgb.shape == gpu.rgb.shape
    assert cpu.opacity.shape == gpu.opacity.shape
    assert int(np.max(np.abs(cpu.rgb.astype(np.int16) - gpu.rgb.astype(np.int16)))) <= 1
    assert int(np.max(np.abs(cpu.opacity.astype(np.int16) - gpu.opacity.astype(np.int16)))) <= 1


def test_render_mlx_backend_contract(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    atoms = load_pdb(pdb_path, params.rules)

    if importlib.util.find_spec("mlx.core") is None:
        with pytest.raises(RuntimeError, match="MLX backend requested but MLX is unavailable"):
            render_from_atoms(atoms, params, backend="mlx")
        return

    cpu = render_from_atoms(atoms, params, backend="numpy")
    mlx = render_from_atoms(atoms, params, backend="mlx")

    assert cpu.rgb.shape == mlx.rgb.shape
    assert cpu.opacity.shape == mlx.opacity.shape
    assert int(np.max(np.abs(cpu.rgb.astype(np.int16) - mlx.rgb.astype(np.int16)))) <= 1
    assert int(np.max(np.abs(cpu.opacity.astype(np.int16) - mlx.opacity.astype(np.int16)))) <= 1
