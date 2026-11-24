from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import importlib
from operator import index as integer_index
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from textwrap import dedent
from time import perf_counter
from types import ModuleType
from typing import TYPE_CHECKING, SupportsIndex, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from illustrate import prepare_render_session, render_params_from_preset
from illustrate.formats.pdb import load_pdb
import illustrate.rendering.kernels as raster_kernel_module
import illustrate.rendering.pipeline as render_pipeline_module
from illustrate.rendering.kernels import (
    ArrayBuffer,
    backend_available,
    run_composite_kernel,
    run_raster_kernel,
    run_outline12_kernel,
    run_outline34_kernel,
    run_shadow_kernel,
)
from illustrate.rendering.api import load_atoms, render, render_from_atoms
from illustrate.core.types import AtomTable, RenderParams, RenderResult, SelectionRule


render_module = importlib.import_module("illustrate.rendering.api")


def _optional_backend_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def _only_backend(backend: str) -> Callable[[str], bool]:
    def available(candidate: str) -> bool:
        return candidate == backend

    return available


def _skip_rasterization(*_args: object, **_kwargs: object) -> None:
    pass


def _unchanged_buffers(**kwargs: object) -> tuple[object, object, object]:
    return kwargs["zpix"], kwargs["atom_buf"], kwargs["bio_buf"]


def _test_result(scene: render_pipeline_module.PreparedScene) -> RenderResult:
    width, height = scene.layout.width, scene.layout.height
    return RenderResult(
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        opacity=np.zeros((height, width), dtype=np.uint8),
        width=width,
        height=height,
    )


def _patch_backend_buffers(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], render_pipeline_module.BackendBuffers],
) -> None:
    def initialize(_backend: str, _width: int, _height: int) -> render_pipeline_module.BackendBuffers:
        return factory()

    monkeypatch.setattr(render_pipeline_module, "_initialize_backend_buffers", initialize)


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


if TYPE_CHECKING:
    _Slice = slice[int | None, int | None, int | None]
    _IndexArray = NDArray[np.bool_ | np.int32 | np.int64]
    _IndexPart = int | _Slice | _IndexArray | None
    _Index = _IndexPart | tuple[_IndexPart, ...]


def _array_index(value: object) -> _Index:
    if isinstance(value, tuple):
        return tuple(_index_part(item) for item in cast(tuple[object, ...], value))
    return _index_part(value)


def _index_part(value: object) -> _IndexPart:
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, slice):
        return cast("_Slice", value)
    if hasattr(type(value), "__array__"):
        index = np.asarray(value)
        if index.dtype in (np.dtype(np.bool_), np.dtype(np.int32), np.dtype(np.int64)):
            return cast("_IndexArray", index)
    if isinstance(value, SupportsIndex):
        return integer_index(value)
    raise TypeError(f"Unsupported test array index {type(value).__name__}")


def _array_dtype(value: object) -> np.dtype[np.generic]:
    if isinstance(value, np.dtype):
        return cast(np.dtype[np.generic], value)
    if isinstance(value, type) and issubclass(value, np.generic):
        return np.dtype(cast(type[np.generic], value))
    raise TypeError(f"Unsupported test array dtype {value}")


class _FakeGPUArray:
    def __init__(self, data: object, *, dtype: np.dtype[np.generic] | type[np.generic] | None = None) -> None:
        self._data = np.asarray(data, dtype=dtype)

    def __array__(
        self,
        dtype: np.dtype[np.generic] | None = None,
        copy: bool | None = None,
    ) -> NDArray[np.generic]:
        del copy
        return np.asarray(self._data, dtype=dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype[np.generic]:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: object) -> _FakeGPUArray:
        return _FakeGPUArray(self._data[_array_index(item)], dtype=self._data.dtype)

    def __setitem__(self, item: object, value: object) -> None:
        self._data[_array_index(item)] = np.asarray(value)

    @property
    def at(self) -> "_FakeArrayAt":
        return _FakeArrayAt(self)

    def _binary(self, other: object, op: Callable[[NDArray[np.generic], NDArray[np.generic]], NDArray[np.generic]]) -> _FakeGPUArray:
        return _FakeGPUArray(op(self._data, np.asarray(other)), dtype=self._data.dtype)

    def __add__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.add)

    def __sub__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.subtract)

    def __mul__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.multiply)

    def __rmul__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.multiply)

    def __truediv__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.divide)

    def __and__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.bitwise_and)

    def __or__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.bitwise_or)

    def __lt__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.less)

    def __le__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.less_equal)

    def __gt__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.greater)

    def __ge__(self, other: object) -> _FakeGPUArray:
        return self._binary(other, np.greater_equal)

    def __rsub__(self, other: object) -> _FakeGPUArray:
        return _FakeGPUArray(np.subtract(np.asarray(other), self._data))

    def astype(self, dtype: object, /) -> _FakeGPUArray:
        return _FakeGPUArray(self._data, dtype=_array_dtype(dtype))

    def reshape(self, shape: tuple[int, ...], /) -> _FakeGPUArray:
        return _FakeGPUArray(np.reshape(self._data, shape))

    def copy(self) -> _FakeGPUArray:
        return _FakeGPUArray(np.array(self._data, copy=True), dtype=self._data.dtype)


class _FakeArrayAt:
    def __init__(self, target: _FakeGPUArray) -> None:
        self._target = target

    def __getitem__(self, item: object) -> "_FakeArrayAtIndexer":
        return _FakeArrayAtIndexer(self._target, _array_index(item))


class _FakeArrayAtIndexer:
    def __init__(self, target: _FakeGPUArray, item: _Index) -> None:
        self._target = target
        self._item = item

    def subtract(self, value: object) -> _FakeGPUArray:
        updated = np.array(self._target, copy=True)
        updated[self._item] = updated[self._item] - np.asarray(value)
        return _FakeGPUArray(updated, dtype=updated.dtype)

    def maximum(self, value: object) -> _FakeGPUArray:
        updated = np.array(self._target, copy=True)
        updated[self._item] = np.maximum(updated[self._item], np.asarray(value))
        return _FakeGPUArray(updated, dtype=updated.dtype)


class _FakeBackendModule:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.float32 = np.float32
        self.int32 = np.int32
        self.uint8 = np.uint8
        self.calls = {"array": 0, "asarray": 0, "asnumpy": 0, "eval": 0, "clip": 0, "swapaxes": 0}

    def array(self, values: object, /, *, dtype: object) -> _FakeGPUArray:
        self.calls["array"] += 1
        return _FakeGPUArray(values, dtype=_array_dtype(dtype))

    def asarray(self, values: object, /, *, dtype: object) -> _FakeGPUArray:
        self.calls["asarray"] += 1
        return _FakeGPUArray(values, dtype=_array_dtype(dtype))

    def zeros(self, shape: tuple[int, ...], /, *, dtype: object) -> _FakeGPUArray:
        return _FakeGPUArray(np.zeros(shape, dtype=_array_dtype(dtype)))

    def ones(self, shape: tuple[int, ...], /, *, dtype: object) -> _FakeGPUArray:
        return _FakeGPUArray(np.ones(shape, dtype=_array_dtype(dtype)))

    def full(self, shape: tuple[int, ...], value: float, /, *, dtype: object) -> _FakeGPUArray:
        return _FakeGPUArray(np.full(shape, value, dtype=_array_dtype(dtype)))

    def minimum(self, values: object, limit: float, /) -> _FakeGPUArray:
        return _FakeGPUArray(np.minimum(np.asarray(values), limit))

    def asnumpy(self, values: object, dtype: np.dtype[np.generic] | type[np.generic] | None = None) -> NDArray[np.generic]:
        self.calls["asnumpy"] += 1
        return np.asarray(values, dtype=dtype)

    def eval(self, *_args: object) -> None:
        self.calls["eval"] += 1

    def clip(self, values: object, min_value: object, max_value: object) -> _FakeGPUArray:
        self.calls["clip"] += 1
        return _FakeGPUArray(np.asarray(values).clip(np.asarray(min_value), np.asarray(max_value)))

    def swapaxes(self, values: object, axis1: int, axis2: int) -> _FakeGPUArray:
        self.calls["swapaxes"] += 1
        return _FakeGPUArray(np.swapaxes(np.asarray(values), axis1, axis2))


def test_raster_kernel_dispatch_rejects_unknown_backend() -> None:
    samples = np.zeros(1, dtype=np.float32)
    indices = np.ones(1, dtype=np.int32)
    with pytest.raises(ValueError, match="Unsupported backend"):
        run_raster_kernel(
            backend="unknown",
            sx=samples,
            sy=samples,
            sz=samples,
            c_cx=samples,
            c_cy=samples,
            c_cz=samples,
            c_ia=indices,
            half_ix=0.5,
            half_iy=0.5,
            fix=1.0,
            fiy=1.0,
            nv=1,
            ibio=1,
            zpix=np.zeros((1, 1), dtype=np.float32),
            atom_buf=np.zeros((1, 1), dtype=np.int32),
            bio_buf=np.ones((1, 1), dtype=np.int32),
        )


@pytest.mark.parametrize("column_stride", [1, 2])
def test_numpy_raster_clips_samples_and_uses_last_equal_depth_atom(column_stride: int) -> None:
    shape = (4, 4 * column_stride)
    zpix = np.full(shape, -9.0, dtype=np.float32)[:, ::column_stride]
    atom_buf = np.zeros(shape, dtype=np.int32)[:, ::column_stride]
    bio_buf = np.zeros(shape, dtype=np.int32)[:, ::column_stride]
    # A sample at the existing depth can replace an earlier assembly's atom.
    zpix[2, 2], atom_buf[2, 2], bio_buf[2, 2] = -2.0, 99, 1

    run_raster_kernel(
        backend="numpy",
        sx=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        sy=np.zeros(3, dtype=np.float32),
        sz=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        c_cx=np.array([1.0, 1.0, 3.0, 5.0], dtype=np.float32),
        c_cy=np.full(4, 3.0, dtype=np.float32),
        c_cz=np.full(4, -3.0, dtype=np.float32),
        c_ia=np.array([11, 22, 33, 44], dtype=np.int32),
        half_ix=0.0,
        half_iy=0.0,
        fix=4.0,
        fiy=4.0,
        nv=3,
        ibio=2,
        zpix=zpix,
        atom_buf=atom_buf,
        bio_buf=bio_buf,
    )

    expected_atoms = np.zeros_like(atom_buf)
    expected_atoms[:, 2] = [22, 33, 33, 44]
    expected_depth = np.full_like(zpix, -9.0)
    expected_depth[:, 2] = [-2.0, -3.0, -2.0, -3.0]
    np.testing.assert_array_equal(atom_buf, expected_atoms)
    np.testing.assert_array_equal(zpix, expected_depth)
    np.testing.assert_array_equal(bio_buf, np.where(expected_atoms != 0, 2, 0))


def test_backend_available_handles_missing_parent_package(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_find_spec(name: str):
        if name == "mlx":
            return None
        return object()

    raster_kernel_module.clear_optional_module_probe_cache()
    monkeypatch.setattr("importlib.util.find_spec", fake_find_spec)
    assert backend_available("numpy") is True
    assert backend_available("mlx") is False
    raster_kernel_module.clear_optional_module_probe_cache()


@pytest.mark.parametrize("dense", [False, True])
def test_numpy_shadow_samples_across_strips_in_cropped_depth(dense: bool) -> None:
    zpix = np.full((256, 272), -30.0, dtype=np.float32)[:, 1:271]
    atom_buf = np.full((256, 272), int(dense), dtype=np.int32)[:, 1:271]
    zpix[244, 100] = -1.0
    zpix[239, 105] = -21.0  # Exactly on the cone threshold at distance 5.
    atom_buf[239, 100] = 1
    atom_buf[239, 95] = 0

    result = run_shadow_kernel(
        backend="numpy",
        zpix=zpix,
        atom_buf=atom_buf,
        shadow_strength=0.4,
        shadow_angle=2.0,
        shadow_min_z=1.0,
        shadow_max_dark=0.3,
    )

    shaded = np.asarray(result)
    assert shaded[239, 100] == np.float32(0.6)
    assert shaded[239, 95] == np.float32(1.0)


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
    mx = _optional_backend_module("mlx.core")

    mx.eval(mlx_out)
    mlx_np = np.array(mlx_out, dtype=np.float32)
    assert numpy_out.shape == mlx_np.shape
    assert np.max(np.abs(np.asarray(numpy_out, dtype=np.float32) - mlx_np)) <= 1e-5


def test_shadow_kernel_mlx_handles_cropped_depth_buffers() -> None:
    if not backend_available("mlx"):
        pytest.skip("MLX backend unavailable")

    mx = _optional_backend_module("mlx.core")
    base_depth = np.full((24, 27), -10000.0, dtype=np.float32)
    crop = np.s_[2:20, 3:24]
    depth = base_depth[crop]
    depth[0, 0] = -25.0
    depth[0, 5] = depth[5, 0] = depth[5, 5] = -1.0
    depth[17, 20] = -12.0
    depth[17, 15] = 0.0
    base_atoms = (base_depth != -10000.0).astype(np.int32)

    expected = np.ones(depth.shape, dtype=np.float32)
    expected[0, 0] = 0.3  # Three occluders reach the darkness clamp.
    expected[17, 20] = 0.6  # One occluder reduces visibility by 0.4.
    devices = (mx.cpu, mx.gpu) if mx.metal.is_available() else (mx.cpu,)

    for device in devices:
        with mx.stream(device):
            actual = run_shadow_kernel(
                backend="mlx",
                zpix=mx.array(base_depth)[crop],
                atom_buf=mx.array(base_atoms)[crop],
                shadow_strength=0.4,
                shadow_angle=2.0,
                shadow_min_z=1.0,
                shadow_max_dark=0.3,
            )
            mx.eval(actual)
            np.testing.assert_allclose(np.array(actual), np.asarray(expected, dtype=np.float32), rtol=0, atol=1e-6)


def test_mlx_render_completes_for_concurrent_callers() -> None:
    if not backend_available("mlx") or not _optional_backend_module("mlx.core").metal.is_available():
        pytest.skip("Metal device unavailable")

    # Keep the test runner alive if concurrent Metal encoding aborts again.
    script = dedent("""\
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        import numpy as np
        from illustrate import render_params_from_preset
        from illustrate.rendering.api import load_atoms, render_from_atoms

        params = render_params_from_preset("Default", "examples/2hhb.pdb")
        atoms = load_atoms(params)
        barrier = Barrier(2)

        def render_frame():
            barrier.wait(timeout=5)
            return render_from_atoms(atoms, params, backend="mlx")

        with ThreadPoolExecutor(max_workers=2) as pool:
            renders = [pool.submit(render_frame) for _ in range(2)]
            first, second = [task.result(timeout=15) for task in renders]
        assert (first.width, first.height) == (922, 750)
        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(first.opacity, second.opacity)
        """)
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    os.environ.get("ILLUSTRATE_RUN_RENDER_PERF") != "1",
    reason="set ILLUSTRATE_RUN_RENDER_PERF=1 to run the render latency probe",
)
def test_mlx_detailed_render_latency() -> None:
    if not backend_available("mlx"):
        pytest.skip("MLX backend unavailable")
    mx = _optional_backend_module("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal device unavailable")

    pdb = Path(__file__).resolve().parents[1] / "examples" / "2hhb.pdb"
    default = render_params_from_preset("Default", str(pdb))
    params = replace(default, transform=replace(default.transform, scale=24.0))
    atoms = load_atoms(params)
    # M1 Max budget for a warm 1784 × 1440 frame, including RGB readback.
    budget_ms = float(os.environ.get("ILLUSTRATE_MLX_RENDER_BUDGET_MS", "110"))
    samples: list[float] = []
    with mx.stream(mx.gpu):
        result = render_from_atoms(atoms, params, backend="mlx")
        for _ in range(2):
            render_from_atoms(atoms, params, backend="mlx")
        assert (result.width, result.height) == (1784, 1440)
        for _ in range(7):
            start = perf_counter()
            render_from_atoms(atoms, params, backend="mlx")
            samples.append((perf_counter() - start) * 1000)
    elapsed_ms = median(samples)
    assert elapsed_ms <= budget_ms, f"MLX median {elapsed_ms:.1f} ms exceeds {budget_ms:.1f} ms"


@pytest.mark.skipif(
    os.environ.get("ILLUSTRATE_RUN_RENDER_PERF") != "1",
    reason="set ILLUSTRATE_RUN_RENDER_PERF=1 to run the render latency probe",
)
def test_numpy_detailed_render_latency() -> None:
    pdb = Path(__file__).resolve().parents[1] / "examples" / "2hhb.pdb"
    default = render_params_from_preset("Default", str(pdb))
    params = replace(default, transform=replace(default.transform, scale=24.0))
    session = prepare_render_session(load_atoms(params), backend="numpy")
    result = session.render(params)
    assert (result.width, result.height) == (1784, 1440)

    # M1 Max budget for a warm frame with the default shadows and outlines.
    budget_ms = float(os.environ.get("ILLUSTRATE_NUMPY_RENDER_BUDGET_MS", "1500"))
    samples: list[float] = []
    for _ in range(5):
        start = perf_counter()
        session.render(params)
        samples.append((perf_counter() - start) * 1000)
    elapsed_ms = median(samples)
    assert elapsed_ms <= budget_ms, f"NumPy median {elapsed_ms:.1f} ms exceeds {budget_ms:.1f} ms"


def test_numpy_depth_outline_keeps_zero_range_contours_inside_the_border() -> None:
    shape = (7, 9)
    empty = np.zeros(shape, dtype=np.int32)
    lookup = np.zeros(1, dtype=np.int32)
    opacity = run_outline34_kernel(
        backend="numpy",
        zpix=np.zeros(shape, dtype=np.float32),
        atom_buf=empty,
        bio_buf=empty,
        su_lookup=lookup,
        res_lookup=lookup,
        residue_diff=1.0,
        residue_low=0.0,
        residue_high=1.0,
        subunit_low=0.0,
        subunit_high=1.0,
        z_diff_min=1.0,
        z_diff_max=1.0,
        contour_low=-1.0,
        contour_high=1.0,
        kernel=4,
    )
    expected = np.zeros(shape, dtype=np.float32)
    expected[2:-2, 2:-2] = 0.75
    np.testing.assert_array_equal(np.asarray(opacity), expected)


def _outline12_reference(
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
    h, w = zpix.shape
    su_map = su_lookup[atom_buf]
    res_map = res_lookup[atom_buf]
    res_map_f = res_map.astype(np.float32)

    r_count = np.zeros((h, w), dtype=np.float32)
    g_count = np.zeros((h, w), dtype=np.float32)
    rg_pad = 2
    su_padded = np.pad(su_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="edge")
    bio_padded = np.pad(bio_buf, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="edge")
    res_padded = np.pad(res_map, ((rg_pad, rg_pad), (rg_pad, rg_pad)), mode="edge")
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
        g_opacity = ((g_count - residue_low) / (residue_high - residue_low)).clip(0.0, 1.0)
    else:
        g_opacity = np.zeros((h, w), dtype=np.float32)
    if subunit_high != subunit_low:
        r_opacity = ((r_count - subunit_low) / (subunit_high - subunit_low)).clip(0.0, 1.0)
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
                l_v = ((shifted_l - contour_low) / (contour_high - contour_low)).clip(0.0, 1.0)
            else:
                l_v = np.zeros_like(shifted_l)
            rl += (l_v > 0).astype(np.float32)
            l_opacity_ave += l_v
            if di == 0 and dj == 0:
                l_center = l_v

    l_opacity = np.where(rl >= 6.0, l_opacity_ave / 6.0, l_center)
    l_opacity = l_opacity.clip(0.0, 1.0)
    l_opacity[:2, :] = 0.0
    l_opacity[-2:, :] = 0.0
    l_opacity[:, :2] = 0.0
    l_opacity[:, -2:] = 0.0
    return np.maximum(l_opacity, g_opacity)


def test_outline_kernel12_matches_the_reference_on_each_backend() -> None:
    rng = np.random.default_rng(17)
    zpix = np.minimum(rng.normal(loc=-2.5, scale=1.7, size=(24, 26)).astype(np.float32), 0.0)
    atom_buf = rng.integers(0, 8, size=(24, 26), dtype=np.int32)
    bio_buf = rng.integers(1, 4, size=(24, 26), dtype=np.int32)
    su_lookup = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    res_lookup = np.array([0, 10, 20, 30, 40, 50, 60, 70], dtype=np.int32)
    settings: dict[str, float] = {
        "residue_diff": 5.0,
        "residue_low": 3.0,
        "residue_high": 8.0,
        "subunit_low": 3.0,
        "subunit_high": 10.0,
        "contour_low": 3.0,
        "contour_high": 10.0,
    }
    inputs: list[tuple[str, ArrayBuffer, ArrayBuffer, ArrayBuffer]] = [("numpy", zpix, atom_buf, bio_buf)]
    if backend_available("mlx"):
        mx = raster_kernel_module.require_mlx()
        inputs.append(("mlx", mx.array(zpix), mx.array(atom_buf), mx.array(bio_buf)))

    for kernel in (1, 2):
        expected = _outline12_reference(
            zpix=zpix,
            atom_buf=atom_buf,
            bio_buf=bio_buf,
            su_lookup=su_lookup,
            res_lookup=res_lookup,
            kernel=kernel,
            **settings,
        )
        for backend, depth, atoms, assemblies in inputs:
            actual = run_outline12_kernel(
                backend=backend,
                zpix=depth,
                atom_buf=atoms,
                bio_buf=assemblies,
                su_lookup=su_lookup,
                res_lookup=res_lookup,
                kernel=kernel,
                **settings,
            )
            np.testing.assert_allclose(
                np.asarray(actual, dtype=np.float32), expected, rtol=0, atol=1e-5, err_msg=f"{backend} kernel {kernel}"
            )


@pytest.mark.parametrize("kernel", (3, 4))
def test_outline_kernel34_contract_numpy_vs_mlx(kernel: int) -> None:
    rng = np.random.default_rng(13)
    depth = rng.normal(loc=-3.0, scale=1.8, size=(28, 30)).astype(np.float32)
    atoms = rng.integers(0, 8, size=(28, 30), dtype=np.int32)
    assemblies = rng.integers(1, 4, size=(28, 30), dtype=np.int32)
    crop = np.s_[2:26, 3:27]
    su_lookup = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int32)
    res_lookup = np.array([0, 10, 20, 30, 40, 50, 60, 70], dtype=np.int32)

    def outline(backend: str, zpix: ArrayBuffer, atom_buf: ArrayBuffer, bio_buf: ArrayBuffer) -> ArrayBuffer:
        return run_outline34_kernel(
            backend=backend,
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
            kernel=kernel,
        )

    expected = outline("numpy", depth[crop], atoms[crop], assemblies[crop])

    if not backend_available("mlx"):
        with pytest.raises(RuntimeError, match="MLX backend requested but MLX is unavailable"):
            outline("mlx", depth[crop], atoms[crop], assemblies[crop])
        return

    mx = raster_kernel_module.require_mlx()
    devices = (mx.cpu, mx.gpu) if mx.metal.is_available() else (mx.cpu,)
    for device in devices:
        with mx.stream(device):
            actual = outline(
                "mlx",
                mx.array(depth, dtype=mx.float32)[crop],
                mx.array(atoms, dtype=mx.int32)[crop],
                mx.array(assemblies, dtype=mx.int32)[crop],
            )
            mx.eval(actual)
            np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32), rtol=0, atol=1e-6)


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
    mx = _optional_backend_module("mlx.core")

    mx.eval(rgb_mlx, alpha_mlx)
    rgb_mlx_np = np.array(rgb_mlx, dtype=np.float32)
    alpha_mlx_np = np.array(alpha_mlx, dtype=np.float32)
    assert rgb_np.shape == rgb_mlx_np.shape
    assert alpha_np.shape == alpha_mlx_np.shape
    assert np.max(np.abs(np.asarray(rgb_np, dtype=np.float32) - rgb_mlx_np)) <= 1e-5
    assert np.max(np.abs(np.asarray(alpha_np, dtype=np.float32) - alpha_mlx_np)) <= 1e-5


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
    monkeypatch.setattr(render_module, "host_backend_candidates", lambda: ("mlx", "numpy"))
    monkeypatch.setattr(render_module, "backend_available", _only_backend("mlx"))
    monkeypatch.setattr(render_module, "backend_importable", _only_backend("mlx"))
    assert render_module.resolve_render_backend(None) == "mlx"


def test_resolve_render_backend_auto_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ILLUSTRATE_RENDER_BACKEND", raising=False)
    monkeypatch.setattr(render_module, "host_backend_candidates", lambda: ("mlx", "numpy"))
    monkeypatch.setattr(render_module, "backend_available", _only_backend("numpy"))
    monkeypatch.setattr(render_module, "backend_importable", _only_backend("numpy"))
    assert render_module.resolve_render_backend(None) == "numpy"


def test_resolve_render_backend_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILLUSTRATE_RENDER_BACKEND", "numpy")
    monkeypatch.setattr(render_module, "backend_available", _only_backend("numpy"))
    assert render_module.resolve_render_backend(None) == "numpy"


def test_resolve_render_backend_env_override_requires_available_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ILLUSTRATE_RENDER_BACKEND", "mlx")
    monkeypatch.setattr(render_module, "backend_available", _only_backend("numpy"))
    with pytest.raises(RuntimeError, match="MLX backend requested but MLX is unavailable"):
        render_module.resolve_render_backend(None)


def test_public_resolve_render_backend_normalizes_explicit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module, "backend_available", _only_backend("numpy"))

    assert render_module.resolve_render_backend(" NumPy ") == "numpy"


def test_prepare_render_session_renders_from_loaded_atoms_after_source_path_changes(tmp_path: Path) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    atoms = load_pdb(pdb_path, params.rules)
    expected = render_from_atoms(atoms, params, backend="numpy")
    params.pdb_path = str(tmp_path / "missing-after-load.pdb")

    session = render_module.prepare_render_session(atoms, backend="numpy")
    result = session.render(params)

    assert session.backend == "numpy"
    assert np.array_equal(result.rgb, expected.rgb)
    assert np.array_equal(result.opacity, expected.opacity)


def test_host_platform_backend_candidates_for_apple_silicon() -> None:
    assert render_module.host_backend_candidates(platform_name="darwin", machine_name="arm64") == (
        "mlx",
        "numba",
        "numpy",
    )


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


def test_rasterize_atoms_tracks_shadow_bounds_for_visible_atoms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    atoms = load_pdb(pdb_path, params.rules)
    program = render_module.program_from_params(params)
    scene = render_pipeline_module.prepare_scene(program, atoms)
    buffers = render_pipeline_module.BackendBuffers(
        backend_name="numpy",
        zpix=np.full((scene.layout.width, scene.layout.height), -10000.0, dtype=np.float32),
        atom_buf=np.zeros((scene.layout.width, scene.layout.height), dtype=np.int32),
        bio_buf=np.ones((scene.layout.width, scene.layout.height), dtype=np.int32),
    )

    monkeypatch.setattr(
        render_pipeline_module,
        "run_raster_kernel",
        _unchanged_buffers,
    )

    _patch_backend_buffers(monkeypatch, lambda: buffers)
    render_pipeline_module.render_program(
        program, atoms, backend_name="numpy", sphere_lookup=render_module.precompute_sphere,
    )

    assert buffers.shadow_bounds is not None
    x0, x1, y0, y1 = buffers.shadow_bounds
    assert 0 <= x0 < x1 <= scene.layout.width
    assert 0 <= y0 < y1 <= scene.layout.height
    assert (x0, x1) != (0, scene.layout.width)
    assert (y0, y1) != (0, scene.layout.height)


@pytest.mark.parametrize("backend_name", ["numpy", "mlx"])
def test_shadow_mask_crops_to_shadow_bounds_and_reembeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
) -> None:
    pdb_path = tmp_path / "mini.pdb"
    _write_minimal_pdb(pdb_path)
    params = _minimal_params(pdb_path)
    params.world.shadows = True
    atoms = load_pdb(pdb_path, params.rules)
    program = render_module.program_from_params(params)
    scene = render_pipeline_module.prepare_scene(program, atoms)
    width, height = scene.layout.width, scene.layout.height
    shadow_bounds = (5, 14, 7, 19)

    zpix_np = np.full((width, height), -10000.0, dtype=np.float32)
    atom_np = np.zeros((width, height), dtype=np.int32)
    atom_np[shadow_bounds[0] : shadow_bounds[1], shadow_bounds[2] : shadow_bounds[3]] = 1
    bio_np = np.ones((width, height), dtype=np.int32)

    fake_backend = _FakeBackendModule(mode=backend_name)
    if backend_name == "numpy":
        buffers = render_pipeline_module.BackendBuffers(
            backend_name=backend_name,
            zpix=np.array(zpix_np, copy=True),
            atom_buf=np.array(atom_np, copy=True),
            bio_buf=np.array(bio_np, copy=True),
            shadow_bounds=shadow_bounds,
        )
    else:
        buffers = render_pipeline_module.BackendBuffers(
            backend_name=backend_name,
            zpix=_FakeGPUArray(np.array(zpix_np, copy=True)),
            atom_buf=_FakeGPUArray(np.array(atom_np, copy=True)),
            bio_buf=_FakeGPUArray(np.array(bio_np, copy=True)),
            mlx_mod=fake_backend if backend_name == "mlx" else None,
            shadow_bounds=shadow_bounds,
        )

    crop_shape = (shadow_bounds[1] - shadow_bounds[0], shadow_bounds[3] - shadow_bounds[2])
    calls = {"count": 0}

    def fake_shadow_kernel(**kwargs: object) -> NDArray[np.float32] | _FakeGPUArray:
        calls["count"] += 1
        assert np.asarray(kwargs["zpix"], dtype=np.float32).shape == crop_shape
        assert np.asarray(kwargs["atom_buf"], dtype=np.int32).shape == crop_shape
        shadow_crop = np.full(crop_shape, 0.25, dtype=np.float32)
        if backend_name == "numpy":
            return shadow_crop
        return _FakeGPUArray(shadow_crop, dtype=np.float32)

    monkeypatch.setattr(render_pipeline_module, "run_shadow_kernel", fake_shadow_kernel)

    def check_shadow(
        _scene: render_pipeline_module.PreparedScene, _atoms: AtomTable,
        _buffers: render_pipeline_module.BackendBuffers, shadow: object, _outline: object,
    ) -> RenderResult:
        shadow_np = np.asarray(shadow, dtype=np.float32)
        expected = np.ones((width, height), dtype=np.float32)
        expected[shadow_bounds[0] : shadow_bounds[1], shadow_bounds[2] : shadow_bounds[3]] = 0.25
        np.testing.assert_array_equal(shadow_np, expected)
        return _test_result(scene)

    _patch_backend_buffers(monkeypatch, lambda: buffers)
    monkeypatch.setattr(render_pipeline_module, "_rasterize_atoms", _skip_rasterization)
    monkeypatch.setattr(render_pipeline_module, "_render_precomputed_outline", check_shadow)
    render_pipeline_module.render_program(
        program, atoms, backend_name=backend_name, sphere_lookup=render_module.precompute_sphere,
    )
    assert calls == {"count": 1}


