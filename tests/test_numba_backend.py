"""The numba backend must reproduce the NumPy renderer byte for byte."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from textwrap import dedent
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
import pytest

from illustrate import prepare_render_session, render_params_from_preset
from illustrate.formats.pdb import load_pdb
from illustrate.io.preset_store import builtin_preset_items
from illustrate.rendering.api import host_backend_candidates
from illustrate.rendering.kernels import (
    backend_available,
    run_composite_kernel,
    run_outline34_kernel,
    run_raster_kernel,
    run_shadow_kernel,
)

pytestmark = pytest.mark.skipif(not backend_available("numba"), reason="Numba backend unavailable")

_REPO = Path(__file__).resolve().parents[1]
_PDB = _REPO / "examples" / "2hhb.pdb"


def _raster(backend: str, **inputs: object) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
    zpix = np.full((40, 46), -10000.0, dtype=np.float32)[:, 1:45]
    atom_buf = np.zeros((40, 46), dtype=np.int32)[:, 1:45]
    bio_buf = np.ones((40, 46), dtype=np.int32)[:, 1:45]
    run_raster_kernel(backend=backend, zpix=zpix, atom_buf=atom_buf, bio_buf=bio_buf, **inputs)  # pyright: ignore[reportArgumentType]
    return zpix, atom_buf, bio_buf


@pytest.mark.parametrize("fractional", [False, True])
def test_raster_matches_numpy_with_ties_clipping_and_strided_buffers(fractional: bool) -> None:
    rng = np.random.default_rng(3)
    stamp = np.arange(-3, 4, dtype=np.float32)
    sx, sy = (grid.ravel() for grid in np.meshgrid(stamp, stamp, indexing="ij"))
    if fractional:
        # A stamp row with gaps takes the per-sample path. The others update
        # whole column spans, which must clip at a fractional fiy.
        sx = np.concatenate((sx, np.full(4, 4.0, dtype=np.float32)))
        sy = np.concatenate((sy, np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)))
    # Integer depths and centers make many samples land on equal depths.
    sz = np.round(np.sqrt(np.maximum(0.0, 12.0 - sx * sx - sy * sy))).astype(np.float32)
    count = 60
    centers = (
        rng.uniform(-24.0, 24.0, size=(2, count)) if fractional else rng.integers(-26, 26, size=(2, count))
    ).astype(np.float32)
    inputs: dict[str, object] = {
        "sx": sx,
        "sy": sy,
        "sz": sz,
        "c_cx": centers[0],
        "c_cy": centers[1],
        "c_cz": rng.integers(-6, -2, size=count).astype(np.float32),
        "c_ia": np.arange(1, count + 1, dtype=np.int32),
        "half_ix": 20.0,
        "half_iy": 22.0,
        "fix": 39.5 if fractional else 40.0,
        "fiy": 43.5 if fractional else 44.0,
        "nv": sx.size,
        "ibio": 3,
    }

    expected = _raster("numpy", **inputs)
    actual = _raster("numba", **inputs)

    for want, got in zip(expected, actual):
        np.testing.assert_array_equal(got, want)
    assert np.count_nonzero(expected[1]) > 0


@pytest.mark.parametrize("dense", [False, True])
def test_shadow_matches_numpy_on_cropped_depth(dense: bool) -> None:
    rng = np.random.default_rng(5)
    depth = rng.normal(loc=-30.0, scale=12.0, size=(150, 170)).astype(np.float32)
    atoms = (rng.uniform(size=depth.shape) < (0.9 if dense else 0.2)).astype(np.int32)
    crop = np.s_[4:140, 7:166]
    settings: dict[str, float] = {
        "shadow_strength": 0.0023,
        "shadow_angle": 2.0,
        "shadow_min_z": 1.0,
        "shadow_max_dark": 0.2,
    }

    expected = run_shadow_kernel(backend="numpy", zpix=depth[crop], atom_buf=atoms[crop], **settings)
    actual = run_shadow_kernel(backend="numba", zpix=depth[crop], atom_buf=atoms[crop], **settings)

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert np.asarray(expected).min() < 1.0


@pytest.mark.parametrize("kernel", (3, 4))
def test_outline34_matches_numpy(kernel: int) -> None:
    rng = np.random.default_rng(13)
    depth = np.minimum(rng.normal(loc=-3.0, scale=4.0, size=(48, 52)).astype(np.float32), 0.0)
    atoms = rng.integers(0, 8, size=depth.shape, dtype=np.int32)
    assemblies = rng.integers(1, 4, size=depth.shape, dtype=np.int32)
    crop = np.s_[2:46, 3:50]
    inputs: dict[str, object] = {
        "su_lookup": np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32),
        "res_lookup": np.array([0, 10, 20, 30, 40, 50, 60, 70], dtype=np.int32),
        "residue_diff": 15.0,
        "residue_low": 3.0,
        "residue_high": 8.0,
        "subunit_low": 3.0,
        "subunit_high": 10.0,
        "z_diff_min": 0.5,
        "z_diff_max": 5.0,
        "contour_low": 1.0,
        "contour_high": 10.0,
        "kernel": kernel,
    }

    expected = run_outline34_kernel(
        backend="numpy", zpix=depth[crop], atom_buf=atoms[crop], bio_buf=assemblies[crop], **inputs  # pyright: ignore[reportArgumentType]
    )
    actual = run_outline34_kernel(
        backend="numba", zpix=depth[crop], atom_buf=atoms[crop], bio_buf=assemblies[crop], **inputs  # pyright: ignore[reportArgumentType]
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert 0.0 < float(np.asarray(expected).max()) <= 1.0


def test_composite_matches_numpy() -> None:
    rng = np.random.default_rng(21)
    depth = rng.normal(loc=-20.0, scale=8.0, size=(30, 34)).astype(np.float32)
    depth[:4] = -10000.0  # Background rows take the far-fog branch.
    inputs: dict[str, object] = {
        "zpix": depth,
        "atom_buf": rng.integers(0, 7, size=depth.shape, dtype=np.int32),
        "pconetot": rng.uniform(0.6, 1.0, size=depth.shape).astype(np.float32),
        "l_opacity": rng.uniform(0.0, 0.9, size=depth.shape).astype(np.float32),
        "type_lookup": np.array([0, 3, 5, 9, 15, 23, 37], dtype=np.int32),
        "colortype": rng.uniform(0.0, 1.0, size=(1001, 3)).astype(np.float32),
        "fog_color": np.array([0.8, 0.7, 0.9], dtype=np.float32),
        "fog_front": 0.9,
        "fog_back": 0.3,
        "zbuf_bg": -10000.0,
    }

    expected = run_composite_kernel(backend="numpy", **inputs)  # pyright: ignore[reportArgumentType]
    actual = run_composite_kernel(backend="numba", **inputs)  # pyright: ignore[reportArgumentType]

    for want, got in zip(expected, actual):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


@pytest.mark.parametrize("preset", [name for name, _ in builtin_preset_items()])
def test_render_matches_numpy_for_builtin_presets(preset: str) -> None:
    params = render_params_from_preset(preset, str(_PDB))
    params = replace(params, transform=replace(params.transform, scale=3.0))
    atoms = load_pdb(_PDB, params.rules)

    expected = prepare_render_session(atoms, backend="numpy").render(params)
    actual = prepare_render_session(atoms, backend="numba").render(params)

    np.testing.assert_array_equal(actual.rgb, expected.rgb)
    np.testing.assert_array_equal(actual.opacity, expected.opacity)


def test_render_completes_for_concurrent_callers() -> None:
    # Concurrent launches abort the process under Numba's default threading
    # layer, so run them in a child to keep the test runner alive.
    script = dedent("""\
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        import numpy as np
        from illustrate import render_params_from_preset
        from illustrate.rendering.api import load_atoms, render_from_atoms

        params = render_params_from_preset("Default", "examples/2hhb.pdb")
        atoms = load_atoms(params)
        render_from_atoms(atoms, params, backend="numba")
        barrier = Barrier(4)

        def render_frame():
            barrier.wait(timeout=30)
            return render_from_atoms(atoms, params, backend="numba")

        with ThreadPoolExecutor(max_workers=4) as pool:
            frames = [task.result(timeout=60) for task in [pool.submit(render_frame) for _ in range(4)]]
        for frame in frames[1:]:
            np.testing.assert_array_equal(frame.rgb, frames[0].rgb)
            np.testing.assert_array_equal(frame.opacity, frames[0].opacity)
        """)
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("platform_name", "machine_name"),
    [("linux", "x86_64"), ("win32", "AMD64"), ("darwin", "x86_64")],
)
def test_hosts_without_mlx_prefer_compiled_cpu_kernels(platform_name: str, machine_name: str) -> None:
    assert host_backend_candidates(platform_name=platform_name, machine_name=machine_name) == ("numba", "numpy")


def test_availability_probe_does_not_import_numba() -> None:
    # Status probes run on every desktop launch, including hosts that render with MLX.
    script = dedent("""\
        import sys
        from illustrate.rendering.kernels import backend_available
        assert backend_available("numba")
        assert "numba" not in sys.modules
        """)
    result = subprocess.run([sys.executable, "-c", script], cwd=_REPO, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_auto_selection_skips_numba_when_it_fails_to_import(monkeypatch: pytest.MonkeyPatch) -> None:
    import illustrate.rendering.api as api

    monkeypatch.delenv("ILLUSTRATE_RENDER_BACKEND", raising=False)
    monkeypatch.setattr(api, "host_backend_candidates", lambda: ("numba", "numpy"))
    def importable(name: str) -> bool:
        return name != "numba"

    monkeypatch.setattr(api, "backend_importable", importable)

    assert api.resolve_render_backend(None) == "numpy"


@pytest.mark.skipif(
    os.environ.get("ILLUSTRATE_RUN_RENDER_PERF") != "1",
    reason="set ILLUSTRATE_RUN_RENDER_PERF=1 to run the render latency probe",
)
def test_numba_detailed_render_latency() -> None:
    default = render_params_from_preset("Default", str(_PDB))
    params = replace(default, transform=replace(default.transform, scale=24.0))
    session = prepare_render_session(load_pdb(_PDB, params.rules), backend="numba")
    result = session.render(params)
    assert (result.width, result.height) == (1784, 1440)

    # M1 Max budget for a warm frame with the default shadows and outlines.
    budget_ms = float(os.environ.get("ILLUSTRATE_NUMBA_RENDER_BUDGET_MS", "150"))
    samples: list[float] = []
    for _ in range(7):
        start = perf_counter()
        session.render(params)
        samples.append((perf_counter() - start) * 1000)
    elapsed_ms = median(samples)
    assert elapsed_ms <= budget_ms, f"Numba median {elapsed_ms:.1f} ms exceeds {budget_ms:.1f} ms"
