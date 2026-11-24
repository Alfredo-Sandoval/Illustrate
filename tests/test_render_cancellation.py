from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from threading import get_ident

import numpy as np
import pytest

from illustrate import AtomTable, RenderParams
from illustrate.rendering.api import render_from_atoms
from illustrate.rendering.kernels import run_shadow_kernel
from illustrate.rendering.pipeline import RenderCancelled


_REPO_ROOT = Path(__file__).resolve().parents[1]


class _CancelledToken:
    def is_cancelled(self) -> bool:
        return True


def _empty_atoms() -> AtomTable:
    return AtomTable(
        coord=np.zeros((1, 3), dtype=np.float32),
        type_idx=np.zeros(1, dtype=np.int32),
        res=np.zeros(1, dtype=np.int32),
        su=np.zeros(1, dtype=np.int32),
        biomat=np.zeros((1, 3, 4), dtype=np.float32),
        n=0,
        nbiomat=0,
    )


def test_render_from_atoms_accepts_optional_cancellation_token() -> None:
    params = RenderParams(pdb_path="dummy.pdb", rules=[])
    with pytest.raises(RenderCancelled, match="render cancelled"):
        render_from_atoms(
            _empty_atoms(),
            params,
            backend="numpy",
            cancellation_token=_CancelledToken(),
        )


def test_numpy_shadow_polls_cancellation_on_the_render_thread() -> None:
    depth = np.full((512, 640), -20.0, dtype=np.float32)
    atoms = np.ones(depth.shape, dtype=np.int32)
    caller = get_ident()
    checks = 0

    def cancel_between_batches() -> None:
        nonlocal checks
        assert get_ident() == caller
        checks += 1
        if checks == 2:
            raise RenderCancelled("render cancelled")

    with pytest.raises(RenderCancelled, match="render cancelled"):
        run_shadow_kernel(
            cancel_between_batches,
            backend="numpy",
            zpix=depth,
            atom_buf=atoms,
            shadow_strength=0.4,
            shadow_angle=2.0,
            shadow_min_z=1.0,
            shadow_max_dark=0.3,
        )
    assert checks == 2


@pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="PySide6 not installed; skipping GUI worker cancellation regression",
)
def test_render_worker_suppresses_cancelled_stale_result() -> None:
    script = """
        import threading
        import time
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication
        from illustrate.gui.workers import RenderRequest, RenderWorker
        import illustrate.gui.workers.render as worker_module

        app = QApplication([])
        started = threading.Event()
        calls = []

        orig = worker_module.render_from_atoms
        def fake(_atoms, params, *, backend=None, cancellation_token=None):
            calls.append((params, cancellation_token is not None))
            if params == "A":
                started.set()
                deadline = time.monotonic() + 1.0
                while (
                    cancellation_token is not None
                    and not cancellation_token.is_cancelled()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
            return params

        worker_module.render_from_atoms = fake
        try:
            worker = RenderWorker()
            done = []
            failed = []
            worker.result_ready.connect(done.append)
            worker.failed.connect(failed.append)

            worker.submit(RenderRequest(params="A", atoms=object()))

            def submit_newer_when_running():
                if started.is_set():
                    worker.submit(RenderRequest(params="B", atoms=object()))
                    return
                QTimer.singleShot(5, submit_newer_when_running)

            QTimer.singleShot(5, submit_newer_when_running)
            loop = QEventLoop()
            QTimer.singleShot(700, loop.quit)
            loop.exec()

            if worker.isRunning():
                worker.wait()
            print("FAILED", failed)
            print("DONE", done)
            print("CALLS", calls)
        finally:
            worker_module.render_from_atoms = orig
    """
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "FAILED []" in output
    assert "DONE ['B']" in output
    assert "CALLS [('A', True), ('B', True)]" in output
