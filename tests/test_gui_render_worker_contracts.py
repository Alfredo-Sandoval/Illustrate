from __future__ import annotations

import importlib.util
import os
import time
from functools import partial
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if importlib.util.find_spec("PySide6") is None:
    raise pytest.skip.Exception(
        "PySide6 not installed; skipping GUI render worker contracts",
        allow_module_level=True,
    )

from PySide6.QtCore import QCoreApplication, QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow

from illustrate import AtomTable, RenderParams, RenderResult, WorldParams
from illustrate.gui.controllers.render import RenderController
from illustrate.gui.controllers.settings import SettingsController
from illustrate.gui.controllers.structure import StructureController
from illustrate.gui.image_frame import GuiImageFrame, image_frame_from_arrays
from illustrate.gui.main_window import MainWindow
from illustrate.gui.workers import RenderJobResult, RenderRequest, RenderWorker
from illustrate.gui.workers.render import (
    LoadJobResult, LoadRequest, LoadWorker, RenderJobStale, SizeEstimateJobResult,
    SizeEstimateRequest, SizeEstimateWorker,
)
from illustrate.rendering.pipeline import RenderCancellationToken
import illustrate.gui.workers.render as worker_module


@pytest.fixture(scope="module", autouse=True)
def _qt_app() -> QApplication:
    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


class _ManualRenderWorker(RenderWorker):
    def submit(self, request: RenderRequest) -> None:
        # Completion is delivered explicitly to control its order in the tests.
        pass


class _RenderWindow(MainWindow):
    def __init__(self) -> None:
        QMainWindow.__init__(self)
        self.worker = _ManualRenderWorker(self)
        self.atoms = _atoms()
        self.pdb_path = "mini.pdb"
        self.render_request_id = 0
        self.last_result = None
        self.params_dirty = True
        # create_actions binds these controllers' methods; the tests never trigger them.
        self.structure_controller = StructureController(self)
        self._settings_controller = SettingsController(self)
        self.main_actions = self.create_actions()
        self.base_images: list[tuple[tuple[int, ...], object]] = []
        self.busy_states: list[bool] = []
        self.render_style_updates = 0
        self.dim_update_count = 0

    def set_render_busy(self, busy: bool) -> None:
        self.busy_states.append(bool(busy))

    def render_elapsed_suffix(self) -> str:
        return ""

    def update_render_button_style(self) -> None:
        self.render_style_updates += 1

    def display_image(self, rgb: NDArray[np.uint8], opacity: object = None) -> None:
        self.base_images.append((tuple(rgb.shape), opacity))

    def schedule_render_dimensions_update(self) -> None:
        self.dim_update_count += 1

    def build_params(self) -> RenderParams:
        return _params("mini.pdb")

    def load_atoms_if_needed(
        self, *, status_message: str | None = None, render_after_load: bool = False,
    ) -> bool:
        return True


def _controller_window(
    *, latest_request_id: int, active_request_id: int | None = None,
) -> tuple[_RenderWindow, RenderController]:
    window = _RenderWindow()
    controller = RenderController(window)
    if active_request_id is not None:
        window.render_request_id = active_request_id - 1
        controller.render()
    window.render_request_id = latest_request_id
    return window, controller


def _result() -> RenderResult:
    return RenderResult(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        opacity=np.zeros((2, 2), dtype=np.uint8),
        width=2,
        height=2,
    )


def _params(label: str) -> RenderParams:
    return RenderParams(pdb_path=label, rules=[], world=WorldParams(width=16, height=16))


def _atoms() -> AtomTable:
    return AtomTable(
        coord=np.zeros((1, 3), dtype=np.float32), type_idx=np.zeros(1, dtype=np.int32),
        res=np.zeros(1, dtype=np.int32), su=np.zeros(1, dtype=np.int32),
        biomat=np.zeros((1, 3, 4), dtype=np.float32), n=0, nbiomat=0,
    )


def test_stale_full_result_drops_export_state_and_clears_active_busy() -> None:
    window, controller = _controller_window(latest_request_id=6, active_request_id=5)

    controller.on_render_done(
        RenderJobResult(request_id=5, interactive=False, result=_result())
    )

    assert window.base_images == []
    assert window.last_result is None
    assert window.main_actions.copy_to_clipboard.isEnabled() is False
    assert window.main_actions.render.isEnabled() is True
    assert window.busy_states[-1] is False
    assert window.statusBar().currentMessage() == "Render superseded."


def test_stale_drag_result_preserves_the_current_image() -> None:
    window, controller = _controller_window(latest_request_id=6)

    controller.on_render_done(
        RenderJobResult(request_id=5, interactive=True, result=_result())
    )

    assert window.base_images == []
    assert window.last_result is None
    assert window.main_actions.copy_to_clipboard.isEnabled() is False


def test_current_drag_result_updates_the_exportable_image() -> None:
    window, controller = _controller_window(latest_request_id=7)
    result = _result()

    controller.on_render_done(
        RenderJobResult(request_id=7, interactive=True, result=result)
    )

    assert len(window.base_images) == 1
    assert window.last_result is result
    assert window.main_actions.copy_to_clipboard.isEnabled() is True
    assert window.params_dirty is False


def test_current_full_result_updates_exportable_state() -> None:
    window, controller = _controller_window(latest_request_id=8, active_request_id=8)
    result = _result()

    controller.on_render_done(
        RenderJobResult(request_id=8, interactive=False, result=result)
    )

    assert window.last_result is result
    assert len(window.base_images) == 1
    assert window.main_actions.copy_to_clipboard.isEnabled() is True
    assert window.params_dirty is False
    assert window.dim_update_count == 1


def test_stale_callback_for_active_full_clears_busy_without_export_update() -> None:
    window, controller = _controller_window(latest_request_id=10, active_request_id=9)

    controller.on_render_stale(
        RenderJobStale(request_id=9, interactive=False, reason="finished-stale")
    )

    assert window.last_result is None
    assert window.base_images == []
    assert window.main_actions.render.isEnabled() is True
    assert window.busy_states[-1] is False


def test_render_worker_stale_signal_covers_coalesced_and_finished_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_render(_atoms: AtomTable, params: RenderParams, *, backend: str | None = None) -> RenderResult:
        calls.append(params.pdb_path)
        if params.pdb_path == "A":
            time.sleep(0.12)
        return _result()

    monkeypatch.setattr(worker_module, "render_from_atoms", fake_render)

    worker = RenderWorker()
    done: list[RenderJobResult] = []
    stale: list[RenderJobStale] = []
    worker.result_ready.connect(done.append)
    worker.stale.connect(stale.append)

    worker.submit(RenderRequest(params=_params("A"), atoms=_atoms(), request_id=1))
    QTimer.singleShot(
        20,
        lambda: worker.submit(
            RenderRequest(params=_params("B"), atoms=_atoms(), request_id=2)
        ),
    )
    QTimer.singleShot(
        40,
        lambda: worker.submit(
            RenderRequest(params=_params("C"), atoms=_atoms(), request_id=3)
        ),
    )

    loop = QEventLoop()
    worker.result_ready.connect(loop.quit)
    QTimer.singleShot(700, loop.quit)
    loop.exec()
    if worker.isRunning():
        worker.wait()

    assert calls == ["A", "C"]
    assert [payload.request_id for payload in done] == [3]
    assert sorted(payload.request_id for payload in stale) == [1, 2]
    reasons = {payload.request_id: payload.reason for payload in stale}
    assert reasons[1] == "finished-stale"
    assert reasons[2] == "coalesced"


def test_render_worker_uses_the_startup_backend_for_pending_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    backends: list[str | None] = []
    preparing = Event()
    preparing.set()

    def fake_render(_atoms: AtomTable, _settings: RenderParams, *, backend: str | None = None) -> RenderResult:
        backends.append(backend)
        return _result()

    def startup_backend(_token: RenderCancellationToken) -> str | None:
        return "numpy" if preparing.is_set() else None

    monkeypatch.setattr(worker_module, "render_from_atoms", fake_render)
    worker = RenderWorker(startup_backend=startup_backend)
    worker.submit(RenderRequest(params=_params("initial"), atoms=_atoms()))
    assert worker.wait(2000)
    preparing.clear()
    worker.submit(RenderRequest(params=_params("prepared"), atoms=_atoms()))
    assert worker.wait(2000)
    assert backends == ["numpy", None]


def test_render_worker_packs_gui_frame_off_main_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = QCoreApplication.instance()
    assert app is not None
    main_thread = app.thread()
    result = _result()
    frame = image_frame_from_arrays(result.rgb, result.opacity)
    assert frame is not None
    pack_seen_on_main: list[bool] = []

    def fake_render(
        _atoms: object,
        _params: object,
        *,
        backend: str | None = None,
    ) -> RenderResult:
        return result

    def fake_pack(rgb: NDArray[np.uint8], opacity: NDArray[np.uint8] | None) -> GuiImageFrame:
        pack_seen_on_main.append(QThread.currentThread() is main_thread)
        assert rgb is result.rgb
        assert opacity is result.opacity
        return frame

    monkeypatch.setattr(worker_module, "render_from_atoms", fake_render)
    monkeypatch.setattr(worker_module, "image_frame_from_arrays", fake_pack)

    worker = RenderWorker()
    done: list[RenderJobResult] = []
    worker.result_ready.connect(done.append)

    worker.submit(RenderRequest(params=_params("A"), atoms=_atoms(), request_id=1))

    loop = QEventLoop()
    worker.result_ready.connect(loop.quit)
    QTimer.singleShot(700, loop.quit)
    loop.exec()
    if worker.isRunning():
        worker.wait()

    assert pack_seen_on_main == [False]
    assert len(done) == 1
    assert done[0].frame is frame


@pytest.mark.parametrize("kind", ["render", "load", "size"])
def test_submission_during_thread_exit_is_eventually_processed(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ILLUSTRATE_RENDER_BACKEND", "numpy")
    atoms = _atoms()
    params = _params("mini.pdb")
    worker: RenderWorker | LoadWorker | SizeEstimateWorker
    submit: Callable[[], None]
    if kind == "render":
        render_worker = RenderWorker()
        worker = render_worker
        submit = partial(render_worker.submit, RenderRequest(params=params, atoms=atoms, request_id=1))
    elif kind == "load":
        pdb = tmp_path / "mini.pdb"
        pdb.write_text("END\n", encoding="utf-8")
        load_worker = LoadWorker()
        worker = load_worker
        submit = partial(load_worker.submit, LoadRequest(pdb_path=str(pdb), rules=[], rules_signature="empty", request_id=1))
    else:
        size_worker = SizeEstimateWorker()
        worker = size_worker
        submit = partial(size_worker.submit, SizeEstimateRequest(params=params, atoms=atoms, request_id=1))
    empty_seen, release = Event(), Event()
    original_run = worker.run

    def pause_before_thread_exit() -> None:
        original_run()
        if not empty_seen.is_set():
            empty_seen.set()
            if not release.wait(3):
                raise RuntimeError("Test did not release the exiting worker")

    monkeypatch.setattr(worker, "run", pause_before_thread_exit)
    done: list[int] = []
    failed: list[object] = []
    loop = QEventLoop()
    def record_result(payload: RenderJobResult | LoadJobResult | SizeEstimateJobResult) -> None:
        done.append(payload.request_id)
        loop.quit()

    worker.result_ready.connect(record_result)
    worker.failed.connect(failed.append)
    try:
        worker.start()
        assert empty_seen.wait(3), "Worker never reached its empty-queue exit"
        submit()
        release.set()
        assert worker.wait(3000), "Initial worker run did not finish"
        QTimer.singleShot(3000, loop.quit)
        loop.exec()
        assert done == [1], "A request submitted during thread exit was stranded"
        assert failed == []
    finally:
        release.set()
        worker.cancel_pending()
        assert worker.wait(3000), "Worker did not shut down"


def test_label_refreshes_cannot_drop_a_queued_fit_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    running, release = Event(), Event()

    def estimate(_atoms: AtomTable, params: RenderParams) -> tuple[int, int]:
        if params.pdb_path == "running":
            running.set()
            release.wait(5)
        return (100, 80)

    monkeypatch.setattr(worker_module, "estimate_render_size", estimate)
    worker = SizeEstimateWorker()
    done: list[SizeEstimateJobResult] = []
    worker.result_ready.connect(done.append)

    worker.submit(SizeEstimateRequest(params=_params("running"), atoms=_atoms(), request_id=1))
    assert running.wait(5)
    # Queued behind the running estimate: a fit, then two label refreshes.
    worker.submit(SizeEstimateRequest(params=_params("fit"), atoms=_atoms(), request_id=2, purpose="fit"))
    worker.submit(SizeEstimateRequest(params=_params("label"), atoms=_atoms(), request_id=3))
    worker.submit(SizeEstimateRequest(params=_params("label"), atoms=_atoms(), request_id=4))
    release.set()
    assert worker.wait(5000)
    QCoreApplication.processEvents()

    assert [(payload.request_id, payload.purpose) for payload in done] == [(1, "label"), (2, "fit"), (4, "label")]
