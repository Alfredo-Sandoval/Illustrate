from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

if importlib.util.find_spec("PySide6") is None:
    raise pytest.skip.Exception(
        "PySide6 not installed; skipping GUI regressions",
        allow_module_level=True,
    )


_REPO_ROOT = Path(__file__).resolve().parents[1]


def close_test_window(window: QWidget) -> None:
    from PySide6.QtTest import QTest

    for _ in range(300):
        if window.close():
            return
        QTest.qWait(10)
    raise AssertionError("The desktop window did not finish closing")


def run_gui_script(script: str) -> str:
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
    if proc.returncode != 0:
        raise AssertionError(
            f"GUI regression subprocess failed with code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout + proc.stderr


def test_viewport_shows_final_render_at_device_pixel_resolution() -> None:
    output = run_gui_script(
        """
        import os

        os.environ["QT_SCALE_FACTOR"] = "2"

        from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
        from PySide6.QtWidgets import QApplication
        from illustrate.gui.viewport import RenderViewport

        app = QApplication([])
        viewport = RenderViewport()
        viewport.resize(400, 300)
        print("RATIO", viewport.devicePixelRatioF())

        # Alternating one-pixel columns survive only if the 800x400 render is
        # shown 1:1 in device pixels; a logical-size copy blurs them to grey.
        source = QImage(800, 400, QImage.Format.Format_RGB32)
        source.fill(QColor(255, 255, 255))
        painter = QPainter(source)
        painter.setPen(QColor(0, 0, 0))
        for x in range(0, 800, 2):
            painter.drawLine(x, 0, x, 399)
        painter.end()

        viewport.update_image(QPixmap.fromImage(source))
        frame = viewport.grab().toImage()
        print("FRAME", frame.width(), frame.height())
        # The render is centered vertically: device rows 100-499.
        print("VALUES", sorted({frame.pixelColor(x, 300).red() for x in range(100, 110)}))
        """
    )
    assert "RATIO 2.0" in output
    assert "FRAME 800 600" in output
    assert "VALUES [0, 255]" in output


def test_autocomplete_fetch_executes_off_main_thread() -> None:
    output = run_gui_script(
        """
        from PySide6.QtCore import QEventLoop, QThread, QTimer
        from PySide6.QtWidgets import QApplication, QLineEdit
        from illustrate.gui.pdb_completer import PdbCompleter, _FetchWorker

        app = QApplication([])
        main_thread = app.thread()
        seen = []

        orig = _FetchWorker._suggest_ids
        def fake(self, query):
            seen.append(QThread.currentThread() is main_thread)
            return []

        _FetchWorker._suggest_ids = fake
        try:
            line = QLineEdit()
            c = PdbCompleter(line)
            line.setText("2hh")
            c._last_query = "2hh"
            c._on_timer()
            loop = QEventLoop()
            QTimer.singleShot(250, loop.quit)
            loop.exec()
            c.stop()
            assert c._thread.wait(3000)
        finally:
            _FetchWorker._suggest_ids = orig

        print("SEEN", seen)
        """
    )
    assert "SEEN [False]" in output


def test_render_size_estimate_executes_off_main_thread() -> None:
    output = run_gui_script(
        """
        from PySide6.QtCore import QEventLoop, QThread, QTimer
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.workers.render as worker_module
        from illustrate.gui.workers import SizeEstimateRequest, SizeEstimateWorker

        app = QApplication([])
        main_thread = app.thread()
        seen = []

        orig = worker_module.estimate_render_size
        def fake(_atoms, _params):
            seen.append(QThread.currentThread() is main_thread)
            return (123, 456)

        worker_module.estimate_render_size = fake
        try:
            worker = SizeEstimateWorker()
            done = []
            worker.result_ready.connect(lambda payload: done.append((payload.width, payload.height)))
            worker.submit(SizeEstimateRequest(params=object(), atoms=object(), request_id=1))
            loop = QEventLoop()
            worker.result_ready.connect(loop.quit)
            worker.failed.connect(loop.quit)
            QTimer.singleShot(500, loop.quit)
            loop.exec()
            worker.wait()
        finally:
            worker_module.estimate_render_size = orig

        print("SEEN", seen)
        print("DONE", done)
        """
    )
    assert "SEEN [False]" in output
    assert "DONE [(123, 456)]" in output


def test_update_check_executes_off_main_thread() -> None:
    output = run_gui_script(
        """
        from PySide6.QtCore import QEventLoop, QThread, QTimer
        from PySide6.QtWidgets import QApplication
        from illustrate.gui.updater import UpdateCheckResult
        import illustrate.gui.workers.update as update_worker_module
        from illustrate.gui.workers import UpdateCheckWorker

        app = QApplication([])
        main_thread = app.thread()
        seen = []

        orig = update_worker_module.check_for_updates
        def fake():
            seen.append(QThread.currentThread() is main_thread)
            return UpdateCheckResult(status="up_to_date", current_version="0.0.0", message="ok")

        update_worker_module.check_for_updates = fake
        try:
            worker = UpdateCheckWorker()
            done = []
            worker.finished.connect(lambda payload: done.append(payload.status))
            worker.start()
            loop = QEventLoop()
            worker.finished.connect(loop.quit)
            QTimer.singleShot(500, loop.quit)
            loop.exec()
            worker.wait()
        finally:
            update_worker_module.check_for_updates = orig

        print("SEEN", seen)
        print("DONE", done)
        """
    )
    assert "SEEN [False]" in output
    assert "DONE ['up_to_date']" in output


def test_update_check_worker_reports_unexpected_failures() -> None:
    output = run_gui_script(
        """
        from PySide6.QtCore import QEventLoop, QThread, QTimer
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.workers.update as update_worker_module
        from illustrate.gui.workers import UpdateCheckFailure, UpdateCheckWorker

        app = QApplication([])
        main_thread = app.thread()
        seen = []

        orig = update_worker_module.check_for_updates
        def fake():
            seen.append(QThread.currentThread() is main_thread)
            raise RuntimeError("network boom")

        update_worker_module.check_for_updates = fake
        try:
            worker = UpdateCheckWorker()
            failed = []
            worker.failed.connect(lambda payload: failed.append(payload))
            worker.start()
            loop = QEventLoop()
            worker.finished.connect(loop.quit)
            worker.failed.connect(loop.quit)
            QTimer.singleShot(500, loop.quit)
            loop.exec()
            worker.wait()
        finally:
            update_worker_module.check_for_updates = orig

        print("SEEN", seen)
        print("FAILED_TYPE", type(failed[0]).__name__ if failed else "none")
        print("FAILED_MESSAGE", failed[0].message if failed else "none")
        print("IS_FAILURE", isinstance(failed[0], UpdateCheckFailure) if failed else False)
        """
    )
    assert "SEEN [False]" in output
    assert "FAILED_TYPE UpdateCheckFailure" in output
    assert "FAILED_MESSAGE network boom" in output
    assert "IS_FAILURE True" in output


def test_image_export_executes_off_main_thread() -> None:
    output = run_gui_script(
        """
        import numpy as np
        from PySide6.QtCore import QEventLoop, QThread, QTimer
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.workers.export as export_worker_module
        from illustrate.gui.workers import ExportRequest, ExportWorker

        app = QApplication([])
        main_thread = app.thread()
        seen = []

        orig = export_worker_module.export_png
        def fake(_path, _rgb, _opacity):
            seen.append(QThread.currentThread() is main_thread)

        export_worker_module.export_png = fake
        try:
            worker = ExportWorker()
            done = []
            worker.finished.connect(lambda payload: done.append((payload.kind, payload.path)))
            worker.submit(
                ExportRequest(
                    path="/tmp/illustrate-test-export.png",
                    kind="png",
                    rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                    opacity=None,
                )
            )
            loop = QEventLoop()
            worker.finished.connect(loop.quit)
            worker.failed.connect(loop.quit)
            QTimer.singleShot(500, loop.quit)
            loop.exec()
            worker.wait()
        finally:
            export_worker_module.export_png = orig

        print("SEEN", seen)
        print("DONE", done)
        """
    )
    assert "SEEN [False]" in output
    assert "DONE [('png', '/tmp/illustrate-test-export.png')]" in output


def test_settings_io_executes_off_main_thread() -> None:
    output = run_gui_script(
        """
        from pathlib import Path
        from PySide6.QtCore import QEventLoop, QThread, QTimer
        from PySide6.QtWidgets import QApplication
        from illustrate import RenderParams
        import illustrate.gui.workers.settings_io as settings_worker_module
        from illustrate.gui.workers import SettingsIORequest, SettingsIOWorker

        app = QApplication([])
        main_thread = app.thread()
        seen = []

        orig = settings_worker_module.save_settings
        def fake(params, path):
            seen.append(QThread.currentThread() is main_thread)
            return path

        settings_worker_module.save_settings = fake
        try:
            worker = SettingsIOWorker()
            done = []
            worker.finished.connect(lambda payload: done.append((payload.kind, str(payload.path))))
            worker.submit(
                SettingsIORequest(
                    path=Path("/tmp/illustrate-test-settings.json"),
                    kind="save",
                    params=RenderParams(pdb_path="dummy.pdb", rules=[]),
                )
            )
            loop = QEventLoop()
            worker.finished.connect(loop.quit)
            worker.failed.connect(loop.quit)
            QTimer.singleShot(500, loop.quit)
            loop.exec()
            worker.wait()
        finally:
            settings_worker_module.save_settings = orig

        print("SEEN", seen)
        print("DONE", done)
        """
    )
    assert "SEEN [False]" in output
    assert "DONE [('save', '/tmp/illustrate-test-settings.json')]" in output


def test_saved_structure_path_is_resolved_in_settings_worker() -> None:
    run_gui_script(
        """
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from PySide6.QtCore import QEventLoop, QThread, QTimer
        from PySide6.QtWidgets import QApplication
        from illustrate import RenderParams
        from illustrate.gui.workers import SettingsIORequest, SettingsIOWorker
        import illustrate.gui.workers.settings_io as settings_io
        from illustrate.io.params_json import save_settings

        app = QApplication([])
        gui_thread = app.thread()
        original = settings_io._resolve_saved_structure
        on_gui_thread = []
        def resolve(params, source):
            on_gui_thread.append(QThread.currentThread() is gui_thread)
            return original(params, source)
        settings_io._resolve_saved_structure = resolve

        with TemporaryDirectory() as directory:
            source = Path(directory) / "look.json"
            pdb = source.with_name("model.pdb")
            pdb.touch()
            save_settings(RenderParams(pdb_path=pdb.name, rules=[]), source)
            worker = SettingsIOWorker()
            results = []
            loop = QEventLoop()
            worker.finished.connect(results.append)
            worker.finished.connect(loop.quit)
            worker.failed.connect(loop.quit)
            assert worker.submit(SettingsIORequest(path=source, kind="load"))
            QTimer.singleShot(3000, loop.quit)
            loop.exec()
            assert worker.wait(3000)
            assert on_gui_thread == [False], on_gui_thread
            assert [result.resolved_pdb for result in results] == [pdb.resolve()], results
        """
    )


def test_desktop_startup_and_preset_save_keep_event_loop_running() -> None:
    run_gui_script(
        """
        from threading import Event

        from PySide6.QtCore import QEventLoop, QThread, QTimer, Qt
        from PySide6.QtWidgets import QApplication
        from illustrate import RenderParams
        from illustrate.gui import main_window
        from illustrate.gui.widgets import backend_status
        from illustrate.gui.workers import presets
        from illustrate.io.preset_store import CustomPresetLoadResult

        app = QApplication([])
        app.setQuitOnLastWindowClosed(False)
        gui_thread = app.thread()
        release_startup = Event()
        release_save = Event()
        threads = {}
        def gated(operation, release):
            threads[operation] = QThread.currentThread() is gui_thread
            if not release.wait(3):
                raise OSError(f"{operation} did not get its event-loop callback")
        def load():
            gated("load", release_startup)
            return CustomPresetLoadResult([("Existing", RenderParams("", []))], [])
        def probe():
            gated("probe", release_startup)
            available = backend_status.BackendAvailability("numpy", "NumPy", True)
            return backend_status.BackendStatus("numpy", "auto", None, (available,))
        saved = []
        def write(items):
            gated("save", release_save)
            saved.extend(items)
        presets.load_custom_presets = load
        presets.write_custom_presets = write
        backend_status.collect_backend_status = probe

        win = main_window.MainWindow()
        loop = QEventLoop()
        try:
            win.show()
            assert not win.main_actions.save_preset.isEnabled()
            assert win.backend_status_label.text() == "Backend: checking…"
            def check_startup():
                if win.main_actions.save_preset.isEnabled() and win.backend_status_label.text() == "Backend: NumPy":
                    loop.quit()
            win._preset_worker.finished.connect(check_startup)
            win._backend_status_worker.finished.connect(check_startup)
            QTimer.singleShot(0, release_startup.set)
            deadline = QTimer()
            deadline.setSingleShot(True)
            deadline.timeout.connect(loop.quit)
            deadline.start(3000)
            loop.exec()
            assert win.main_actions.save_preset.isEnabled()
            assert win.backend_status_label.text() == "Backend: NumPy"
            assert win.preset_list.findItems("Custom: Existing", Qt.MatchFlag.MatchExactly)

            win.pdb_path = "model.pdb"
            main_window.QInputDialog.getText = lambda *_args: ("New", True)
            win.main_actions.save_preset.trigger()
            assert not win.main_actions.save_preset.isEnabled()
            assert [name for name, _ in win._custom_presets] == ["Existing"]
            win._preset_worker.finished.connect(loop.quit)
            QTimer.singleShot(0, release_save.set)
            deadline.start(3000)
            loop.exec()
            assert threads == {"load": False, "probe": False, "save": False}, threads
            assert [name for name, _ in saved] == ["Existing", "New"], saved
            assert saved[-1][1].pdb_path == ""
            assert win.preset_list.currentItem().text() == "Custom: New"
        finally:
            release_startup.set()
            release_save.set()
            win.close()
            QTimer.singleShot(500, loop.quit)
            loop.exec()
        assert not win.isVisible()
        """
    )


@pytest.mark.parametrize(
    ("attr", "label"),
    [
        ("worker", "render"),
        ("fetch_worker", "structure download"),
        ("load_worker", "structure load"),
    ],
)
def test_mainwindow_close_defers_without_blocking_until_worker_stops(attr: str, label: str) -> None:
    output = run_gui_script(
        f"""
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication
        from illustrate.gui.main_window import MainWindow

        app = QApplication([])
        app.setQuitOnLastWindowClosed(False)
        win = MainWindow()
        win.show()

        class BusyWorker:
            running = True
            waits = []
            def cancel_pending(self):
                pass
            def isRunning(self):
                return self.running
            def wait(self, *args):
                self.waits.append(args)
                return False

        busy = BusyWorker()
        win.{attr} = busy
        print("CLOSED_NOW", win.close())
        print("ENABLED", win.isEnabled())
        print("STATUS", win.statusBar().currentMessage())
        print("WAITS", busy.waits)

        busy.running = False
        loop = QEventLoop()
        QTimer.singleShot(500, loop.quit)
        loop.exec()
        print("VISIBLE_AFTER", win.isVisible())
        """
    )
    assert "CLOSED_NOW False" in output
    assert "ENABLED False" in output
    assert f"STATUS Closing after the {label} finishes..." in output
    assert "WAITS []" in output
    assert "VISIBLE_AFTER False" in output


def test_autocomplete_skips_superseded_queries_and_shows_ids_before_titles() -> None:
    output = run_gui_script(
        """
        from illustrate.gui.pdb_suggestions import PdbSuggestion
        from illustrate.gui.workers.pdb_autocomplete import PdbAutocompleteWorker

        looked_up = []

        class Worker(PdbAutocompleteWorker):
            def _suggest_ids(self, query):
                looked_up.append(query)
                return ["2HHB"]

            def _fetch_titles(self, pdb_ids):
                return [PdbSuggestion(pid, "Hemoglobin") for pid in pdb_ids]

        worker = Worker()
        results = []
        worker.finished.connect(lambda query, found: results.append((query, [s.title for s in found])))
        worker.set_latest_query("2hhb")
        worker.fetch("2h")
        worker.fetch("2hhb")
        print("LOOKED_UP", looked_up)
        print("RESULTS", results)
        """
    )
    assert "LOOKED_UP ['2hhb']" in output
    assert "RESULTS [('2hhb', ['']), ('2hhb', ['Hemoglobin'])]" in output


def test_mainwindow_close_defers_while_pdb_search_is_in_flight() -> None:
    output = run_gui_script(
        """
        import threading
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication
        from illustrate.gui.pdb_completer import _FetchWorker
        from illustrate.gui.main_window import MainWindow

        started = threading.Event()
        release = threading.Event()

        def slow_lookup(self, query):
            started.set()
            release.wait(10)
            return []

        _FetchWorker._suggest_ids = slow_lookup
        app = QApplication([])
        app.setQuitOnLastWindowClosed(False)
        win = MainWindow()
        win.show()
        win._pdb_completer._last_query = "hemoglobin"
        win._pdb_completer._on_timer()
        print("STARTED", started.wait(5))

        waits = []
        win._pdb_completer._thread.wait = lambda *args: waits.append(args) or False
        print("CLOSED_NOW", win.close())
        print("STATUS", win.statusBar().currentMessage())
        print("WAITS", waits)

        release.set()
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec()
        print("VISIBLE_AFTER", win.isVisible())
        print("SEARCH_RUNNING", win._pdb_completer.is_running())
        """
    )
    assert "STARTED True" in output
    assert "CLOSED_NOW False" in output
    assert "STATUS Closing after the PDB search finishes..." in output
    assert "WAITS []" in output
    assert "VISIBLE_AFTER False" in output
    assert "SEARCH_RUNNING False" in output


def test_opening_a_structure_fits_it_before_the_first_render() -> None:
    output = run_gui_script(
        """
        from pathlib import Path
        from PySide6.QtCore import QEventLoop, QTimer
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        win = app_module.MainWindow()
        win.resize(1200, 800)
        win.show()
        fits = []
        fit_view = win.render_size.fit_view
        win.render_size.fit_view = lambda **kwargs: (fits.append(kwargs), fit_view(**kwargs))
        start_scale = win.transform_panel.value["scale"]
        win.structure_controller.activate_loaded_structure(
            path=str(Path("examples/2hhb.pdb").resolve()), render_after_load=True, fit_view=True
        )
        loop = QEventLoop()
        timer = QTimer()
        timer.timeout.connect(lambda: win.last_result is not None and loop.quit())
        timer.start(20)
        QTimer.singleShot(15000, loop.quit)
        loop.exec()
        timer.stop()
        result = win.last_result
        view = win.viewport.size() * win.viewport.devicePixelRatioF()
        print("FITS", fits)
        print("SCALE_CHANGED", win.transform_panel.value["scale"] != start_scale)
        print("FITS_VIEW", result.width <= view.width() and result.height <= view.height())
        close_test_window(win)
        """
    )
    assert "FITS [{'render': True}]" in output
    assert "SCALE_CHANGED True" in output
    assert "FITS_VIEW True" in output


def test_preset_keeps_the_view_of_a_loaded_structure() -> None:
    output = run_gui_script(
        """
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        win = app_module.MainWindow()
        win.render_frame = lambda *args, **kwargs: None
        keys = ("scale", "xrot", "yrot", "zrot", "xtran", "ytran", "ztran")
        view = dict(zip(keys, (7.5, 10.0, 20.0, 30.0, 1.0, 2.0, 3.0)))

        win.transform_panel.set_value(view)
        win._apply_preset(1)
        print("NO_STRUCTURE_KEEPS_VIEW", win.transform_panel.value == view)

        win.pdb_path = "structure.pdb"
        win.transform_panel.set_value(view)
        background = win.world_panel.value["background"]
        win._apply_preset(0)
        print("LOADED_KEEPS_VIEW", win.transform_panel.value == view)
        print("STYLE_CHANGED", win.world_panel.value["background"] != background)
        win.pdb_path = None
        close_test_window(win)
        """
    )
    assert "NO_STRUCTURE_KEEPS_VIEW False" in output
    assert "LOADED_KEEPS_VIEW True" in output
    assert "STYLE_CHANGED True" in output


def test_repeated_requests_do_not_restart_a_running_atom_load() -> None:
    output = run_gui_script(
        """
        from types import SimpleNamespace
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate.gui.workers import LoadJobResult

        app = QApplication([])
        win = app_module.MainWindow()
        submitted = []
        rendered = []
        win.load_worker.submit = submitted.append
        win.render_frame = lambda *args, **kwargs: rendered.append(args)
        win.pdb_path = "structure.pdb"
        # Atoms loaded for other rules, as after a preset or rule change.
        win.atoms = object()
        win.atoms_signature = "other rules"

        win.load_atoms_if_needed(status_message="Updating structure for render size...")
        win.load_atoms_if_needed(status_message="Updating structure for render size...")
        win.load_atoms_if_needed(render_after_load=True)
        print("SUBMITTED", len(submitted), win.atoms_loading)
        win.structure_controller.on_atoms_loaded(LoadJobResult(
            request_id=submitted[0].request_id,
            rules_signature=submitted[0].rules_signature,
            atoms=SimpleNamespace(n=0),
        ))
        print("RENDERED_AFTER_LOAD", len(rendered), win.atoms_loading)
        win.pdb_path = None
        close_test_window(win)
        """
    )
    assert "SUBMITTED 1 True" in output
    assert "RENDERED_AFTER_LOAD 1 False" in output


def test_activate_loaded_structure_queues_async_atom_load() -> None:
    output = run_gui_script(
        """
        from types import SimpleNamespace
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate.gui.render_params import rules_signature
        from illustrate.gui.workers import LoadJobResult

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            rendered = []

            class DummyLoadWorker:
                def __init__(self):
                    self.requests = []
                def cancel_pending(self):
                    self.requests.clear()
                def submit(self, request):
                    self.requests.append(request)
                def isRunning(self):
                    return False
                def wait(self):
                    return None

            win.load_worker = DummyLoadWorker()
            win.render_frame = lambda *args, **kwargs: rendered.append((args, kwargs))
            win.update_rule_match_counts = lambda: None
            win.schedule_render_dimensions_update = lambda: None
            win.sync_render_size_controls = lambda: None
            win.set_loaded_model_label = lambda _path: None
            win.clear_last_result = lambda: None
            win.setWindowTitle = lambda *args, **kwargs: None

            win.structure_controller.activate_loaded_structure(
                path="/tmp/example.pdb",
                title_name="example.pdb",
                status_message="Loaded example.pdb. Click Render.",
                render_after_load=True,
            )

            print("SUBMIT_COUNT", len(win.load_worker.requests))
            print("RENDER_BEFORE_LOAD", len(rendered))

            payload = LoadJobResult(
                request_id=win.load_worker.requests[0].request_id,
                rules_signature=rules_signature(win.current_rules()),
                atoms=SimpleNamespace(n=0),
            )
            win.structure_controller.on_atoms_loaded(payload)
            print("RENDER_AFTER_LOAD", len(rendered))
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "SUBMIT_COUNT 1" in output
    assert "RENDER_BEFORE_LOAD 0" in output
    assert "RENDER_AFTER_LOAD 1" in output


def test_mainwindow_save_settings_writes_json_file() -> None:
    output = run_gui_script(
        """
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from PySide6.QtCore import QEventLoop, QTimer
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate import OutlineParams, RenderParams, SelectionRule, Transform, WorldParams

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        orig_get_save = app_module.QFileDialog.getSaveFileName
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            with TemporaryDirectory() as tmp_dir:
                out_path = Path(tmp_dir) / "settings.json"
                app_module.QFileDialog.getSaveFileName = staticmethod(
                    lambda *_args, **_kwargs: (str(out_path), "JSON settings (*.json)")
                )
                win = app_module.MainWindow()
                win.pdb_path = "dummy.pdb"
                win.build_params = lambda: RenderParams(
                    pdb_path="dummy.pdb",
                    rules=[SelectionRule("ATOM  ", "----------", 0, 9999, (1.0, 0.7, 0.5), 1.5)],
                    transform=Transform(),
                    world=WorldParams(),
                    outlines=OutlineParams(),
                )
                win._settings_controller.save_settings()
                loop = QEventLoop()
                win._settings_controller.settings_worker.finished.connect(loop.quit)
                win._settings_controller.settings_worker.failed.connect(loop.quit)
                QTimer.singleShot(500, loop.quit)
                loop.exec()
                print("SETTINGS_EXISTS", out_path.exists())
                print("HAS_RULES_KEY", '"rules"' in out_path.read_text(encoding="utf-8"))
                close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
            app_module.QFileDialog.getSaveFileName = orig_get_save
        """
    )
    assert "SETTINGS_EXISTS True" in output
    assert "HAS_RULES_KEY True" in output


def test_viewport_drag_coalesces_requests_while_a_frame_is_in_flight() -> None:
    output = run_gui_script(
        """
        from types import SimpleNamespace
        import numpy as np
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        from illustrate import RenderResult
        import illustrate.gui.main_window as app_module
        from illustrate.gui.workers import RenderJobResult

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            requests = []
            win.pdb_path = "dummy.pdb"
            win.atoms = SimpleNamespace(n=5000)
            win.load_atoms_if_needed = lambda **kwargs: True
            win.worker = SimpleNamespace(
                submit=requests.append,
                cancel_pending=lambda: None,
                isRunning=lambda: False,
            )
            win.auto_render_on_drag.setChecked(True)

            for _ in range(3):
                win._on_viewport_rotation(1.0, 2.0)
            print("IN_FLIGHT", len(requests))

            frame = RenderResult(
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                opacity=np.zeros((2, 2), dtype=np.uint8),
                width=2,
                height=2,
            )
            win._render_controller.on_render_done(RenderJobResult(
                request_id=requests[0].request_id, interactive=True, result=frame,
            ))
            print("REQUEST_IDS", [request.request_id for request in requests])
            print("INTERACTIVE", [request.interactive for request in requests])
            print("CURRENT_PARAMS", requests[-1].params == win.build_params())
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "IN_FLIGHT 1" in output
    assert "REQUEST_IDS [1, 2]" in output
    assert "INTERACTIVE [True, True]" in output
    assert "CURRENT_PARAMS True" in output


def test_transform_field_edits_follow_the_live_render_setting() -> None:
    output = run_gui_script(
        """
        from types import SimpleNamespace
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            requests = []
            win.pdb_path = "dummy.pdb"
            win.atoms = SimpleNamespace(n=5000)
            win.load_atoms_if_needed = lambda **kwargs: True
            win.worker = SimpleNamespace(
                submit=requests.append,
                cancel_pending=lambda: None,
                isRunning=lambda: False,
            )

            win.auto_render_on_drag.setChecked(True)
            win.params_dirty = False
            win.transform_panel.xrot.setValue(30.0)
            print("LIVE", len(requests), requests[-1].interactive, win.params_dirty)
            print("LIVE_PARAMS", requests[-1].params == win.build_params())

            win._render_controller._finish_drag_frame(requests[-1].request_id)
            win.auto_render_on_drag.setChecked(False)
            win.transform_panel.scale.setValue(20.0)
            print("MANUAL", len(requests), win.params_dirty)
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "LIVE 1 True False" in output
    assert "LIVE_PARAMS True" in output
    assert "MANUAL 1 True" in output


def test_orientation_gizmo_x_axis_points_where_the_render_puts_world_x(tmp_path: Path) -> None:
    # Two atoms along world +X, colored red then blue: after rotation, the
    # red-to-blue direction in the render is where world +X lands on screen.
    def atom(serial: int, chain: str, x: float) -> str:
        return (
            f"{'ATOM':<6}{serial:5d} {'CA':<4}{' ':1}{'GLY':>3} {chain:1}{serial:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        )

    pdb_path = tmp_path / "axis.pdb"
    pdb_path.write_text(f"{atom(1, 'A', 0.0)}\n{atom(2, 'B', 8.0)}\nEND\n", encoding="utf-8")
    output = run_gui_script(
        f"""
        import numpy as np
        from PySide6.QtWidgets import QApplication
        from illustrate import render
        from illustrate.core.types import SelectionRule
        from illustrate.gui.render_params import render_params_from_panels
        from illustrate.gui.viewport import RenderViewport, _gizmo_offset

        def centroid(rgb, channel):
            others = [c for c in range(3) if c != channel]
            mask = rgb[..., channel] - rgb[..., others].max(axis=-1) > 60
            rows, cols = np.nonzero(mask)
            return np.array([cols.mean(), rows.mean()])

        app = QApplication([])
        rules = [
            SelectionRule("ATOM  ", "---------A", 0, 9999, (1.0, 0.0, 0.0), 1.5),
            SelectionRule("ATOM  ", "---------B", 0, 9999, (0.0, 0.0, 1.0), 1.5),
        ]
        view = RenderViewport()
        for xrot, yrot, zrot in [(20.0, 30.0, 60.0), (-40.0, 10.0, 150.0)]:
            transform = dict(scale=12.0, xrot=xrot, yrot=yrot, zrot=zrot, xtran=0.0, ytran=0.0, ztran=0.0)
            params = render_params_from_panels(
                {str(pdb_path)!r}, rules, transform, {{}}, {{}}, render_width=-30, render_height=-30
            )
            rgb = np.asarray(render(params).rgb, dtype=np.int16)
            on_screen = centroid(rgb, 2) - centroid(rgb, 0)

            view.set_orientation(xrot=xrot, yrot=yrot, zrot=zrot)
            x_axis = next(vector for name, _color, vector in view._orientation_basis() if name == "X")
            offset = _gizmo_offset(x_axis)
            gizmo = np.array([offset.x(), offset.y()])
            cosine = float(on_screen @ gizmo / (np.linalg.norm(on_screen) * np.linalg.norm(gizmo)))
            print("COSINE", round(cosine, 3))
        """
    )
    cosines = [float(line.split()[1]) for line in output.splitlines() if line.startswith("COSINE")]
    assert len(cosines) == 2
    assert all(cosine > 0.99 for cosine in cosines), output


def test_viewport_drag_turns_the_front_of_the_structure_with_the_mouse() -> None:
    # The point facing the viewer should follow the drag, like grabbing a ball.
    output = run_gui_script(
        """
        import numpy as np
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate.gui.render_params import render_params_from_panels
        from illustrate.rendering.api import rotation_matrix

        def rm(transform):
            params = render_params_from_panels("x.pdb", [], transform, {}, {}, render_width=-30, render_height=-30)
            return np.asarray(rotation_matrix(params.transform.rotations), dtype=np.float64)[:3, :3]

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.auto_render_on_drag.setChecked(False)
            for label, dx, dy in (("RIGHT", 10.0, 0.0), ("DOWN", 0.0, 10.0)):
                win.transform_panel.set_value(xrot=0.0, yrot=0.0, zrot=90.0)
                start = dict(win.transform_panel.value)
                win.viewport.rotation_requested.emit(dy * 0.5, dx * 0.5)
                front = rm(start) @ np.array([0.0, 0.0, 1.0])
                view = front @ rm(win.transform_panel.value)
                # Rotated x runs down the image and rotated y across it.
                print(label, round(float(view[1]), 3), round(float(view[0]), 3))
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    moves = {
        line.split()[0]: (float(line.split()[1]), float(line.split()[2]))
        for line in output.splitlines()
        if line.startswith(("RIGHT", "DOWN"))
    }
    right, down = moves["RIGHT"], moves["DOWN"]
    assert right[0] > 0.05 and abs(right[1]) < 1e-3, output
    assert down[1] > 0.05 and abs(down[0]) < 1e-3, output


def test_worker_packed_image_frame_preserves_non_square_dimensions() -> None:
    output = run_gui_script(
        """
        import numpy as np
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate.gui.image_frame import image_frame_from_arrays, pixmap_from_frame

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            rgb = np.zeros((228, 480, 3), dtype=np.uint8)
            opacity = np.full((228, 480), 255, dtype=np.uint8)
            frame = image_frame_from_arrays(rgb, opacity)
            pix = pixmap_from_frame(frame)
            print("FRAME", frame.width, frame.height, frame.has_alpha)
            print("PIX", pix.width(), pix.height())
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "FRAME 480 228 True" in output
    assert "PIX 480 228" in output


def test_viewport_keeps_checkerboard_for_alpha_images() -> None:
    output = run_gui_script(
        """
        from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            vp = win.viewport
            vp.resize(320, 240)
            vp.show()

            transparent = QImage(60, 60, QImage.Format.Format_RGBA8888)
            transparent.fill(QColor(0, 0, 0, 0))
            qpix = QPixmap.fromImage(transparent)
            vp.update_image(qpix)
            frame = QImage(vp.size(), QImage.Format.Format_ARGB32)
            frame.fill(QColor(0, 0, 0))
            painter = QPainter(frame)
            vp._paint_content(painter)
            painter.end()

            corner = frame.pixelColor(5, 5)
            adjacent = frame.pixelColor(23, 5)
            diagonal = frame.pixelColor(23, 23)
            print("CHECKERBOARD", corner == diagonal and corner != adjacent)
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "CHECKERBOARD True" in output


def test_viewport_rotation_keeps_the_rest_of_the_transform() -> None:
    output = run_gui_script(
        """
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win._render_controller.render = lambda *, interactive: None
            win.transform_panel.set_value(
                {"scale": 20.0, "xrot": 10.0, "yrot": 20.0, "zrot": 30.0,
                 "xtran": 1.0, "ytran": 2.0, "ztran": 3.0}
            )
            win._on_viewport_rotation(1.0, 2.0)
            value = win.transform_panel.value
            print("TRANSFORM", [round(value[k], 2) for k in
                                ("scale", "xrot", "yrot", "zrot", "xtran", "ytran", "ztran")])
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "TRANSFORM [20.0, 11.0, 22.0, 30.0, 1.0, 2.0, 3.0]" in output


def test_wheel_magnifies_the_view_without_changing_the_render() -> None:
    output = run_gui_script(
        """
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QPixmap, QWheelEvent
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            renders = []
            win._render_controller.render = lambda *a, **k: renders.append(k)
            win.resize(1200, 800)
            win.viewport.resize(600, 400)
            image = QPixmap(300, 200)
            image.fill(Qt.GlobalColor.darkCyan)
            win.viewport.update_image(image)
            scale_before = win.transform_panel.value["scale"]

            def wheel(delta):
                center = QPointF(300.0, 200.0)
                event = QWheelEvent(
                    center, center, QPoint(), QPoint(0, delta),
                    Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                    Qt.ScrollPhase.NoScrollPhase, False,
                )
                QApplication.sendEvent(win.viewport, event)

            for _ in range(3):
                wheel(120)
            print("ZOOMED", round(win.viewport.view_zoom, 3))
            print("SCALE_KEPT", win.transform_panel.value["scale"] == scale_before)
            print("RENDERS", len(renders))
            for _ in range(10):
                wheel(-120)
            print("FLOOR", win.viewport.view_zoom)
            wheel(240)
            win._fit_view()
            print("RESET", win.viewport.view_zoom)
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "ZOOMED 1.331" in output
    assert "SCALE_KEPT True" in output
    assert "RENDERS 0" in output
    assert "FLOOR 1.0" in output
    assert "RESET 1.0" in output


def test_scale_is_capped_where_the_largest_atom_fits_the_renderer() -> None:
    output = run_gui_script(
        """
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate.core.types import SelectionRule

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win._render_controller.render = lambda *a, **k: None
            win.transform_panel.set_value(scale=74.0)
            rule = SelectionRule("ATOM", "----------", 0, 9999, (1.0, 1.0, 1.0), 2.0)
            win.rule_panel.set_value([rule])
            print("CAP", win.transform_panel.scale.maximum())
            print("CLAMPED", win.transform_panel.value["scale"])
            win.transform_panel.set_value({**win.transform_panel.value, "scale": 80.0})
            print("TYPED", win.transform_panel.value["scale"])
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    # 2 Å atoms reach the renderer's 100 px radius limit at scale 50.
    assert "CAP 50.0" in output
    assert "CLAMPED 50.0" in output
    assert "TYPED 50.0" in output


def test_failed_live_render_says_why_instead_of_up_to_date() -> None:
    output = run_gui_script(
        """
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate.gui.workers.render import RenderJobFailure

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.pdb_path = "x.pdb"
            win.params_dirty = False
            win.update_render_button_style()
            print("BEFORE", win.render_bar.heading.text())
            message = "Scale is too high for these atom radii. Use a scale of at most 50.00."
            win._render_controller.on_render_failed(
                RenderJobFailure(request_id=win.render_request_id, interactive=True, message=message)
            )
            print("HEADING", win.render_bar.heading.text())
            print("TOOLTIP", win.render_bar.heading.toolTip() == message)
            print("STATUS", win.statusBar().currentMessage())
            print("DIRTY", win.params_dirty)
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "BEFORE Render is up to date" in output
    assert "HEADING Render failed: Scale is too high for these atom radii" in output
    assert "TOOLTIP True" in output
    assert "STATUS Render failed: Scale is too high" in output
    assert "DIRTY True" in output


def test_output_width_and_height_show_only_for_a_custom_size() -> None:
    output = run_gui_script(
        """
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            form = win.render_width_spin.parentWidget().layout()
            rows = lambda: [form.isRowVisible(w) for w in (win.render_width_spin, win.render_height_spin)]
            win.render_size_mode_combo.setCurrentText("Auto")
            print("AUTO", rows())
            win.render_size_mode_combo.setCurrentText("Custom")
            print("CUSTOM", rows())
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "AUTO [False, False]" in output
    assert "CUSTOM [True, True]" in output


def test_build_params_respects_render_size_controls() -> None:
    output = run_gui_script(
        """
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.pdb_path = "dummy.pdb"
            win.render_size_mode_combo.setCurrentText("Custom")
            win.render_width_spin.setValue(2000)
            win.render_height_spin.setValue(1500)
            custom = win.build_params()
            win.render_size_mode_combo.setCurrentText("Auto")
            auto = win.build_params()
            print("CUSTOM", custom.world.width, custom.world.height)
            print("AUTO", auto.world.width, auto.world.height)
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "CUSTOM 2000 1500" in output
    assert "AUTO -30 -30" in output


def test_apply_loaded_params_sets_custom_render_size_mode() -> None:
    output = run_gui_script(
        """
        from pathlib import Path
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        from illustrate import OutlineParams, RenderParams, SelectionRule, Transform, WorldParams
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            params = RenderParams(
                pdb_path="",
                rules=[
                    SelectionRule(
                        record_name="ATOM  ",
                        descriptor="----------",
                        res_low=0,
                        res_high=9999,
                        color=(1.0, 1.0, 1.0),
                        radius=1.5,
                    )
                ],
                transform=Transform(scale=12.0, translate=(0.0, 0.0, 0.0), rotations=[("z", 90.0), ("y", 0.0), ("x", 0.0)], autocenter="auto"),
                world=WorldParams(width=1234, height=876),
                outlines=OutlineParams(),
            )
            win._settings_controller.apply_loaded_params(params, source=Path("."))
            print("MODE", win.render_size_mode_combo.currentText())
            print("WH", win.render_width_spin.value(), win.render_height_spin.value())
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "MODE Custom" in output
    assert "WH 1234 876" in output


def test_structure_switch_discards_late_render_success_and_failure() -> None:
    output = run_gui_script(
        """
        import numpy as np
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        from illustrate import RenderResult
        from illustrate.gui.main_window import MainWindow
        from illustrate.gui.workers import RenderJobFailure, RenderJobResult

        app = QApplication([])
        win = MainWindow()
        try:
            win.render_request_id = 7
            win._render_controller._active_full_request_id = 7
            win.set_render_busy(True)
            win.structure_controller.activate_loaded_structure(path="missing-second-structure.pdb")
            old = RenderResult(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8), 2, 2)
            win._render_controller.on_render_done(RenderJobResult(7, False, old))
            win._render_controller.on_render_failed(RenderJobFailure(7, False, "old failure"))
            print("CURRENT", win.pdb_path)
            print("OLD_EXPORT_REJECTED", win.last_result is None and not win.main_actions.copy_to_clipboard.isEnabled())
            print("BUSY", win._render_started_at is not None)
        finally:
            close_test_window(win)
        """
    )
    assert "CURRENT missing-second-structure.pdb" in output
    assert "OLD_EXPORT_REJECTED True" in output
    assert "BUSY False" in output


def test_fit_view_resets_translation_and_adjusts_scale() -> None:
    output = run_gui_script(
        """
        from types import SimpleNamespace
        import numpy as np
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate.gui.render_params import rules_signature

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.resize(1400, 900)
            win.pdb_path = "dummy.pdb"
            win.auto_render_on_drag.setChecked(False)
            win.transform_panel.set_value(scale=12.0, xtran=40.0, ytran=-20.0, ztran=5.0)
            win.atoms = SimpleNamespace(
                n=1,
                coord=np.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]], dtype=np.float32),
                type_idx=np.asarray([0, 1], dtype=np.int32),
            )
            win.atoms_signature = rules_signature(win.current_rules())
            params = win.build_params()
            params.world.width = 0
            params.world.height = 0
            params.transform.translate = (0.0, 0.0, 0.0)
            cache_key = win.render_size._cache_key(win.atoms, params)
            win.render_size._store_estimate(cache_key, (300, 200))
            win._fit_view()
            t = win.transform_panel.value
            print("TRAN", t["xtran"], t["ytran"], t["ztran"])
            print("SCALE_GT", t["scale"] > 12.0)
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "TRAN 0.0 0.0 0.0" in output
    assert "SCALE_GT True" in output


def test_rule_panel_match_counts_are_populated_from_atom_types() -> None:
    output = run_gui_script(
        """
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.atoms = object()
            counts = [0] * len(win.current_rules())
            counts[0] = 2
            counts[2] = 3
            win.atom_match_counts = counts
            win.update_rule_match_counts()
            row0 = win.rule_panel.table.item(0, 8).text()
            row1 = win.rule_panel.table.item(1, 8).text()
            row2 = win.rule_panel.table.item(2, 8).text()
            print("COUNTS", row0, row1, row2)
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "COUNTS 2 0 3" in output


def test_rule_panel_pattern_columns_edit_the_descriptor() -> None:
    output = run_gui_script(
        """
        from PySide6.QtCore import Qt
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate.gui.panels import rules

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            table = win.rule_panel.table
            before = win.rule_panel.value[0].descriptor
            table.item(0, rules.ATOM_COLUMN).setText("fe")
            table.item(0, rules.RESIDUE_COLUMN).setText("hem")
            table.item(0, rules.CHAIN_COLUMN).setText("any")
            table.item(0, rules.RADIUS_COLUMN).setData(Qt.ItemDataRole.EditRole, 2.5)
            rule = win.rule_panel.value[0]
            print("BEFORE", before)
            print("DESC_AFTER", rule.descriptor)
            print("RADIUS", rule.radius)
            print("SHOWN", table.item(0, rules.ATOM_COLUMN).text())
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "BEFORE -----HOH--" in output
    assert "DESC_AFTER FE---HEM--" in output
    assert "RADIUS 2.5" in output
    assert "SHOWN FE--" in output
