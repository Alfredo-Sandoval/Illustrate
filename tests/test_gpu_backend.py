from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from illustrate.pdb import load_pdb
import illustrate.raster_kernel as raster_kernel_module
import illustrate.render_pipeline as render_pipeline_module
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

    raster_kernel_module._clear_optional_module_probe_cache()
    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)
    assert backend_available("numpy") is True
    assert backend_available("cupy") is False
    assert backend_available("mlx") is False
    raster_kernel_module._clear_optional_module_probe_cache()


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

    if not backend_available("mlx"):
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


@pytest.mark.parametrize("backend_name", ["cupy", "mlx"])
def test_outline12_kernel_dispatch_supports_gpu_backends_via_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
) -> None:
    calls: list[str] = []

    def fake_outline12_backend(**kwargs):
        calls.append(backend_name)
        return _outline12_reference(**kwargs)

    monkeypatch.setitem(raster_kernel_module.OUTLINE12_DISPATCH, backend_name, fake_outline12_backend)

    rng = np.random.default_rng(23)
    zpix = np.minimum(rng.normal(loc=-2.2, scale=1.4, size=(18, 20)).astype(np.float32), 0.0)
    atom_buf = rng.integers(0, 8, size=(18, 20), dtype=np.int32)
    bio_buf = rng.integers(1, 4, size=(18, 20), dtype=np.int32)
    su_lookup = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    res_lookup = np.array([0, 10, 20, 30, 40, 50, 60, 70], dtype=np.int32)

    for kernel in (1, 2):
        actual = run_outline12_kernel(
            backend=backend_name,
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

    assert calls == [backend_name, backend_name]


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

    if not backend_available("mlx"):
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

    if not backend_available("mlx"):
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
    monkeypatch.setattr(render_module, "_host_platform_backend_candidates", lambda: ("mlx", "numpy"))
    monkeypatch.setattr(render_module, "backend_available", lambda name: name == "mlx")
    assert render_module._resolve_render_backend(None) == "mlx"


def test_resolve_render_backend_auto_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ILLUSTRATE_RENDER_BACKEND", raising=False)
    monkeypatch.setattr(render_module, "_host_platform_backend_candidates", lambda: ("cupy", "numpy"))
    monkeypatch.setattr(render_module, "backend_available", lambda name: name == "numpy")
    assert render_module._resolve_render_backend(None) == "numpy"


def test_resolve_render_backend_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILLUSTRATE_RENDER_BACKEND", "numpy")
    monkeypatch.setattr(render_module, "backend_available", lambda _name: False)
    assert render_module._resolve_render_backend(None) == "numpy"


def test_render_program_gpu_fast_path_skips_numpy_materialization_when_outlines_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    params.outlines.enabled = False
    atoms = load_pdb(pdb_path, params.rules)
    program = render_module._program_from_params(params)

    calls = {"precompute": 0, "fast_path": 0, "materialize": 0}

    def fake_initialize_backend_buffers(backend_name: str, width: int, height: int) -> render_pipeline_module.BackendBuffers:
        assert backend_name == "mlx"
        return render_pipeline_module.BackendBuffers(
            backend_name=backend_name,
            zpix=np.full((width, height), -10000.0, dtype=np.float32),
            atom_buf=np.zeros((width, height), dtype=np.int32),
            bio_buf=np.ones((width, height), dtype=np.int32),
        )

    def fake_precompute_outline(scene, _atoms, _buffers):
        calls["precompute"] += 1
        assert scene.outlines.enabled is False
        return np.ones((scene.layout.width, scene.layout.height), dtype=np.float32)

    def fake_render_precomputed_outline(scene, _atoms, _buffers, _pconetot, precomputed_outline):
        calls["fast_path"] += 1
        assert precomputed_outline is not None
        return render_pipeline_module.RenderResult(
            rgb=np.zeros((scene.layout.height, scene.layout.width, 3), dtype=np.uint8),
            opacity=np.zeros((scene.layout.height, scene.layout.width), dtype=np.uint8),
            width=scene.layout.width,
            height=scene.layout.height,
        )

    def fake_materialize_numpy(*_args, **_kwargs):
        calls["materialize"] += 1
        raise AssertionError("NumPy materialization fallback should not run on the GPU fast path")

    monkeypatch.setattr(render_pipeline_module, "_initialize_backend_buffers", fake_initialize_backend_buffers)
    monkeypatch.setattr(render_pipeline_module, "_rasterize_atoms", lambda *args, **kwargs: None)
    monkeypatch.setattr(render_pipeline_module, "_precompute_outline", fake_precompute_outline)
    monkeypatch.setattr(
        render_pipeline_module,
        "_shadow_mask",
        lambda scene, _buffers: np.ones((scene.layout.width, scene.layout.height), dtype=np.float32),
    )
    monkeypatch.setattr(render_pipeline_module, "_render_precomputed_outline", fake_render_precomputed_outline)
    monkeypatch.setattr(render_pipeline_module, "_materialize_numpy", fake_materialize_numpy)

    result = render_pipeline_module.render_program(
        program,
        atoms,
        backend_name="mlx",
        sphere_lookup=render_module._precompute_sphere,
    )
    expected_width, expected_height = render_module.estimate_render_size(atoms, params)

    assert calls == {"precompute": 1, "fast_path": 1, "materialize": 0}
    assert (result.width, result.height) == (expected_width, expected_height)
    assert result.rgb.shape == (expected_height, expected_width, 3)
    assert result.opacity.shape == (expected_height, expected_width)


def test_host_platform_backend_candidates_for_apple_silicon() -> None:
    assert render_module._host_platform_backend_candidates(platform_name="darwin", machine_name="arm64") == (
        "mlx",
        "numpy",
    )


def test_host_platform_backend_candidates_for_linux() -> None:
    assert render_module._host_platform_backend_candidates(platform_name="linux", machine_name="x86_64") == (
        "cupy",
        "numpy",
    )


def test_host_platform_backend_candidates_for_windows() -> None:
    assert render_module._host_platform_backend_candidates(platform_name="win32", machine_name="AMD64") == (
        "cupy",
        "numpy",
    )


def test_render_cupy_backend_contract(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    atoms = load_pdb(pdb_path, params.rules)

    if not backend_available("cupy"):
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

    if not backend_available("mlx"):
        with pytest.raises(RuntimeError, match="MLX backend requested but MLX is unavailable"):
            render_from_atoms(atoms, params, backend="mlx")
        return

    cpu = render_from_atoms(atoms, params, backend="numpy")
    mlx = render_from_atoms(atoms, params, backend="mlx")

    assert cpu.rgb.shape == mlx.rgb.shape
    assert cpu.opacity.shape == mlx.opacity.shape
    assert int(np.max(np.abs(cpu.rgb.astype(np.int16) - mlx.rgb.astype(np.int16)))) <= 1
    assert int(np.max(np.abs(cpu.opacity.astype(np.int16) - mlx.opacity.astype(np.int16)))) <= 1


@pytest.mark.parametrize("backend_name", ["cupy", "mlx"])
def test_render_gpu_backend_without_outlines_avoids_numpy_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
) -> None:
    if not backend_available(backend_name):
        pytest.skip(f"{backend_name} backend unavailable")

    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    params.outlines.enabled = False
    atoms = load_pdb(pdb_path, params.rules)
    cpu = render_from_atoms(atoms, params, backend="numpy")

    render_pipeline_module = importlib.import_module("illustrate.render_pipeline")

    def fail_materialize(*_args, **_kwargs):
        raise AssertionError("_materialize_numpy should not run for GPU no-outline renders")

    monkeypatch.setattr(render_pipeline_module, "_materialize_numpy", fail_materialize)

    gpu = render_from_atoms(atoms, params, backend=backend_name)

    assert cpu.rgb.shape == gpu.rgb.shape
    assert cpu.opacity.shape == gpu.opacity.shape
    assert int(np.max(np.abs(cpu.rgb.astype(np.int16) - gpu.rgb.astype(np.int16)))) <= 1
    assert int(np.max(np.abs(cpu.opacity.astype(np.int16) - gpu.opacity.astype(np.int16)))) <= 1


@pytest.mark.parametrize("backend_name", ["cupy", "mlx"])
@pytest.mark.parametrize("kernel", [1, 2])
def test_render_gpu_backend_outline12_path_skips_numpy_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    kernel: int,
) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    params.outlines.enabled = True
    params.outlines.kernel = kernel
    atoms = load_pdb(pdb_path, params.rules)
    program = render_module._program_from_params(params)
    calls = {"outline12": 0, "fast_path": 0, "materialize": 0}

    def fake_outline12_backend(**kwargs):
        calls["outline12"] += 1
        return _outline12_reference(**kwargs)

    def fake_initialize_backend_buffers(backend_name_arg: str, width: int, height: int) -> render_pipeline_module.BackendBuffers:
        assert backend_name_arg == backend_name
        return render_pipeline_module.BackendBuffers(
            backend_name=backend_name_arg,
            zpix=np.minimum(np.zeros((width, height), dtype=np.float32), 0.0),
            atom_buf=np.zeros((width, height), dtype=np.int32),
            bio_buf=np.ones((width, height), dtype=np.int32),
        )

    def fake_precompute_outline(scene, _atoms, buffers):
        assert scene.outlines.enabled is True
        assert scene.outlines.kernel in (1, 2)
        return run_outline12_kernel(
            backend=backend_name,
            zpix=np.asarray(buffers.zpix, dtype=np.float32),
            atom_buf=np.asarray(buffers.atom_buf, dtype=np.int32),
            bio_buf=np.asarray(buffers.bio_buf, dtype=np.int32),
            su_lookup=atoms.su,
            res_lookup=atoms.res,
            residue_diff=float(scene.outlines.residue_diff),
            residue_low=float(scene.outlines.residue_low),
            residue_high=float(scene.outlines.residue_high),
            subunit_low=float(scene.outlines.subunit_low),
            subunit_high=float(scene.outlines.subunit_high),
            contour_low=float(scene.outlines.contour_low),
            contour_high=float(scene.outlines.contour_high),
            kernel=int(scene.outlines.kernel),
        )

    def fake_render_precomputed_outline(scene, _atoms, _buffers, _pconetot, precomputed_outline):
        calls["fast_path"] += 1
        assert precomputed_outline is not None
        return render_pipeline_module.RenderResult(
            rgb=np.zeros((scene.layout.height, scene.layout.width, 3), dtype=np.uint8),
            opacity=np.zeros((scene.layout.height, scene.layout.width), dtype=np.uint8),
            width=scene.layout.width,
            height=scene.layout.height,
        )

    def fail_materialize(*_args, **_kwargs):
        calls["materialize"] += 1
        raise AssertionError("NumPy materialization fallback should not run for GPU outline12 renders")

    monkeypatch.setitem(raster_kernel_module.OUTLINE12_DISPATCH, backend_name, fake_outline12_backend)
    monkeypatch.setattr(render_pipeline_module, "_initialize_backend_buffers", fake_initialize_backend_buffers)
    monkeypatch.setattr(render_pipeline_module, "_rasterize_atoms", lambda *args, **kwargs: None)
    monkeypatch.setattr(render_pipeline_module, "_precompute_outline", fake_precompute_outline)
    monkeypatch.setattr(
        render_pipeline_module,
        "_shadow_mask",
        lambda scene, _buffers: np.ones((scene.layout.width, scene.layout.height), dtype=np.float32),
    )
    monkeypatch.setattr(render_pipeline_module, "_render_precomputed_outline", fake_render_precomputed_outline)
    monkeypatch.setattr(render_pipeline_module, "_materialize_numpy", fail_materialize)

    result = render_pipeline_module.render_program(
        program,
        atoms,
        backend_name=backend_name,
        sphere_lookup=render_module._precompute_sphere,
    )
    expected_width, expected_height = render_module.estimate_render_size(atoms, params)

    assert calls == {"outline12": 1, "fast_path": 1, "materialize": 0}
    assert (result.width, result.height) == (expected_width, expected_height)
    assert result.rgb.shape == (expected_height, expected_width, 3)
    assert result.opacity.shape == (expected_height, expected_width)
