from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

if importlib.util.find_spec("PySide6") is None:
    pytest.skip("PySide6 not installed; skipping GUI regressions", allow_module_level=True)


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_gui_script(script: str) -> str:
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


def test_autocomplete_fetch_executes_off_main_thread() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtCore import QEventLoop, QThread, QTimer
        from PySide6.QtWidgets import QApplication, QLineEdit
        from illustrate_gui.autocomplete import PdbCompleter, _FetchWorker

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
            c.cleanup()
        finally:
            _FetchWorker._suggest_ids = orig

        print("SEEN", seen)
        """
    )
    assert "SEEN [False]" in output


def test_mainwindow_close_event_waits_for_running_render_worker() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        from illustrate_gui.app import MainWindow

        app = QApplication([])
        win = MainWindow()
        waited = {"called": False}

        class DummyWorker:
            def isRunning(self):
                return True
            def wait(self):
                waited["called"] = True

        win.worker = DummyWorker()
        win.closeEvent(QCloseEvent())
        print("WAITED", waited["called"])
        """
    )
    assert "WAITED True" in output


def test_collapsible_section_toggle_shows_state_arrows() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication, QLabel
        from illustrate_gui.app import CollapsibleSection

        app = QApplication([])
        content = QLabel("content")
        section = CollapsibleSection("Transform", content, expanded=False)
        toggle = section._toggle

        print("INITIAL_ARROW_RIGHT", toggle.arrowType() == Qt.ArrowType.RightArrow)
        print("INITIAL_CONTENT_HIDDEN", content.isHidden())

        toggle.setChecked(True)
        app.processEvents()
        print("EXPANDED_ARROW_DOWN", toggle.arrowType() == Qt.ArrowType.DownArrow)
        print("EXPANDED_CONTENT_HIDDEN", content.isHidden())

        toggle.setChecked(False)
        app.processEvents()
        print("COLLAPSED_ARROW_RIGHT", toggle.arrowType() == Qt.ArrowType.RightArrow)
        print("COLLAPSED_CONTENT_HIDDEN", content.isHidden())
        """
    )
    assert "INITIAL_ARROW_RIGHT True" in output
    assert "INITIAL_CONTENT_HIDDEN True" in output
    assert "EXPANDED_ARROW_DOWN True" in output
    assert "EXPANDED_CONTENT_HIDDEN False" in output
    assert "COLLAPSED_ARROW_RIGHT True" in output
    assert "COLLAPSED_CONTENT_HIDDEN True" in output


def test_mainwindow_selection_rules_dock_starts_minimized() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.show()
            app.processEvents()
            print("DOCK_VISIBLE", win.rules_dock.isVisible())
            print("TOGGLE_CHECKED", win.toggle_rules_action.isChecked())
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "DOCK_VISIBLE False" in output
    assert "TOGGLE_CHECKED False" in output


def test_mainwindow_save_settings_writes_json_file() -> None:
    output = _run_gui_script(
        """
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

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
                win._build_params = lambda: app_module.RenderParams(
                    pdb_path="dummy.pdb",
                    rules=[app_module.SelectionRule("ATOM  ", "----------", 0, 9999, (1.0, 0.7, 0.5), 1.5)],
                    transform=app_module.Transform(),
                    world=app_module.WorldParams(),
                    outlines=app_module.OutlineParams(),
                )
                win._save_settings()
                print("SETTINGS_EXISTS", out_path.exists())
                print("HAS_RULES_KEY", '"rules"' in out_path.read_text(encoding="utf-8"))
                win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
            app_module.QFileDialog.getSaveFileName = orig_get_save
        """
    )
    assert "SETTINGS_EXISTS True" in output
    assert "HAS_RULES_KEY True" in output


def test_render_worker_coalesces_rapid_submissions() -> None:
    output = _run_gui_script(
        """
        import time
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication
        from illustrate_gui.worker import RenderRequest, RenderWorker
        import illustrate_gui.worker as worker_module

        app = QApplication([])

        orig = worker_module.render_from_atoms
        def fake(_atoms, params):
            time.sleep(0.12)
            return params
        worker_module.render_from_atoms = fake
        try:
            worker = RenderWorker()
            done = []
            failed = []
            worker.finished.connect(done.append)
            worker.failed.connect(failed.append)

            worker.submit(RenderRequest(params="A", atoms=object()))
            QTimer.singleShot(20, lambda: worker.submit(RenderRequest(params="B", atoms=object())))
            QTimer.singleShot(40, lambda: worker.submit(RenderRequest(params="C", atoms=object())))

            loop = QEventLoop()
            QTimer.singleShot(700, loop.quit)
            loop.exec()

            if worker.isRunning():
                worker.wait()
            print("FAILED", failed)
            print("DONE", done)
        finally:
            worker_module.render_from_atoms = orig
        """
    )
    assert "FAILED []" in output
    assert "DONE ['A', 'C']" in output


def test_render_worker_emits_request_id_payload_for_preview_jobs() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication
        from illustrate_gui.worker import RenderRequest, RenderWorker
        import illustrate_gui.worker as worker_module

        app = QApplication([])

        orig = worker_module.render_from_atoms
        def fake(_atoms, params):
            return params
        worker_module.render_from_atoms = fake
        try:
            worker = RenderWorker()
            done = []
            worker.finished.connect(done.append)
            worker.submit(RenderRequest(params="preview", atoms=object(), request_id=17))

            loop = QEventLoop()
            QTimer.singleShot(350, loop.quit)
            loop.exec()

            if worker.isRunning():
                worker.wait()
            print("DONE", done)
        finally:
            worker_module.render_from_atoms = orig
        """
    )
    assert "DONE [(17, 'preview')]" in output


def test_viewport_uses_raster_fallback_in_offscreen_mode() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtWidgets import QApplication
        from illustrate_gui.viewport import RenderViewport, is_opengl_viewport

        app = QApplication([])
        viewport = RenderViewport()
        print("OPENGL", is_opengl_viewport(viewport))
        """
    )
    assert "OPENGL False" in output


def test_preview_quality_modes_change_preview_dimensions() -> None:
    output = _run_gui_script(
        """
        from types import SimpleNamespace
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.resize(1400, 900)
            default_mode = win.preview_quality_combo.currentText()
            win._atoms = SimpleNamespace(n=45000)

            win.preview_quality_combo.setCurrentText("Fast")
            fast = win._preview_dimensions()
            win.preview_quality_combo.setCurrentText("Balanced")
            balanced = win._preview_dimensions()
            win.preview_quality_combo.setCurrentText("High")
            high = win._preview_dimensions()

            print("DEFAULT", default_mode)
            print("ORDER_W", fast[0] < balanced[0] < high[0])
            print("ORDER_H", fast[1] < balanced[1] < high[1])
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset

        """
    )
    assert "DEFAULT Fast" in output
    assert "ORDER_W True" in output
    assert "ORDER_H True" in output


def test_fast_preview_disables_expensive_passes_for_mid_sized_models() -> None:
    output = _run_gui_script(
        """
        from types import SimpleNamespace
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.pdb_path = "dummy.pdb"
            win._atoms = SimpleNamespace(n=5000)

            win.preview_quality_combo.setCurrentText("Fast")
            fast = win._build_preview_params()
            print("FAST", fast.world.shadows, fast.outlines.enabled)

            win.preview_quality_combo.setCurrentText("Balanced")
            balanced = win._build_preview_params()
            print("BALANCED", balanced.world.shadows, balanced.outlines.enabled)
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "FAST False False" in output
    assert "BALANCED True True" in output


def test_interactive_auto_rerender_preserves_framing_and_outlines() -> None:
    output = _run_gui_script(
        """
        from types import SimpleNamespace
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.pdb_path = "dummy.pdb"
            win._atoms = SimpleNamespace(n=8000)
            win.render_size_mode_combo.setCurrentText("Custom")
            win.render_width_spin.setValue(2000)
            win.render_height_spin.setValue(2000)
            full = win._build_params()
            interactive = win._build_interactive_rerender_params()
            print("FULL", full.world.width, full.world.height)
            print("INTER", interactive.world.width, interactive.world.height)
            print("INTER_MATCH_FULL", interactive.world.width == full.world.width and interactive.world.height == full.world.height)
            print("INTER_FLAGS", interactive.world.shadows, interactive.outlines.enabled)
            print("INTER_SCALE_SAME", abs(interactive.transform.scale - full.transform.scale) < 1e-6)
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "FULL 2000 2000" in output
    assert "INTER_MATCH_FULL True" in output
    assert "INTER_FLAGS False True" in output
    assert "INTER_SCALE_SAME True" in output


def test_interactive_rerender_disables_shadows_for_large_models() -> None:
    output = _run_gui_script(
        """
        from types import SimpleNamespace
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.pdb_path = "dummy.pdb"
            win._atoms = SimpleNamespace(n=50000)
            interactive = win._build_interactive_rerender_params()
            print("INTER_FLAGS_LARGE", interactive.world.shadows, interactive.outlines.enabled)
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "INTER_FLAGS_LARGE False True" in output


def test_render_interactive_still_uses_interactive_mode_before_settle_render() -> None:
    output = _run_gui_script(
        """
        from types import SimpleNamespace
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            calls = []
            def fake_render(_checked=False, *, interactive=False):
                calls.append(bool(interactive))
            win._render = fake_render

            win._atoms = SimpleNamespace(n=5000)
            win._render_interactive()
            print("MID_INTERACTIVE", calls[-1])

            win._atoms = SimpleNamespace(n=50000)
            win._render_interactive()
            print("LARGE_INTERACTIVE", calls[-1])
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "MID_INTERACTIVE True" in output
    assert "LARGE_INTERACTIVE True" in output


def test_preview_scale_in_auto_mode_uses_estimated_output_dimensions() -> None:
    output = _run_gui_script(
        """
        from types import SimpleNamespace
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        orig_estimate = app_module.estimate_render_size
        app_module.MainWindow._apply_preset = lambda self, _index: None
        app_module.estimate_render_size = lambda _atoms, _params: (1000, 500)
        try:
            win = app_module.MainWindow()
            win.pdb_path = "dummy.pdb"
            win._atoms = SimpleNamespace(n=8000)
            win.render_size_mode_combo.setCurrentText("Auto")
            full = win._build_params()
            win.preview_quality_combo.setCurrentText("Fast")
            preview = win._build_preview_params()
            expected_ratio = min(preview.world.width / 1000.0, preview.world.height / 500.0)
            scale_ratio = preview.transform.scale / full.transform.scale
            print("AUTO_PREVIEW", preview.world.width, preview.world.height)
            print("AUTO_SCALE_RATIO", f"{scale_ratio:.3f}")
            print("AUTO_EXPECTED_RATIO", f"{expected_ratio:.3f}")
            win.closeEvent(QCloseEvent())
        finally:
            app_module.estimate_render_size = orig_estimate
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    match_scale = re.search(r"AUTO_SCALE_RATIO ([0-9.]+)", output)
    match_expected = re.search(r"AUTO_EXPECTED_RATIO ([0-9.]+)", output)
    assert match_scale is not None and match_expected is not None
    scale_ratio = float(match_scale.group(1))
    expected_ratio = float(match_expected.group(1))
    assert abs(scale_ratio - expected_ratio) <= 0.02


def test_pixmap_conversion_preserves_non_square_render_dimensions() -> None:
    output = _run_gui_script(
        """
        import numpy as np
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            rgb = np.zeros((228, 480, 3), dtype=np.uint8)
            opacity = np.full((228, 480), 255, dtype=np.uint8)
            pix = win._pixmap_from_image(rgb, opacity)
            print("PIX", pix.width(), pix.height())
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "PIX 480 228" in output


def test_preview_mode_keeps_checkerboard_for_alpha_images() -> None:
    output = _run_gui_script(
        """
        import numpy as np
        from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

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
            vp.set_preview_scene(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), None, None)

            # Preview branch should keep checkerboard for alpha images instead
            # of filling the viewport with preview background.
            frame = QImage(vp.size(), QImage.Format.Format_ARGB32)
            frame.fill(QColor(0, 0, 0))
            painter = QPainter(frame)
            vp._paint_content(painter)
            painter.end()

            c = frame.pixelColor(5, 5)
            print("PIXEL", c.red(), c.green(), c.blue())
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    # checkerboard top-left tile color
    assert "PIXEL 48 48 52" in output


def test_push_preview_scene_marks_scene_presence_for_renderer_preview() -> None:
    output = _run_gui_script(
        """
        import numpy as np
        from types import SimpleNamespace
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        from illustrate import SelectionRule
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            coords = np.zeros((5, 3), dtype=np.float32)
            coords[1:, 0] = np.array([-1.0, 0.5, 1.0, 0.0], dtype=np.float32)
            coords[1:, 1] = np.array([0.0, 1.0, -0.5, -1.0], dtype=np.float32)
            type_idx = np.array([0, 1, 1, 1, 1], dtype=np.int32)
            win._atoms = SimpleNamespace(
                n=4,
                coord=coords,
                type_idx=type_idx,
            )
            win.rule_panel.set_value([
                SelectionRule(
                    record_name="ATOM  ",
                    descriptor="----------",
                    res_low=0,
                    res_high=9999,
                    color=(0.2, 0.7, 0.9),
                    radius=1.5,
                )
            ])
            win._push_preview_scene()
            vp = win.viewport
            print("HAS_SCENE", vp._has_preview_scene)
            print("SCENE_SHAPE", None if vp._preview_coords is None else vp._preview_coords.shape)
            print("COLOR_SHAPE", None if vp._preview_colors is None else vp._preview_colors.shape)
            print("RAD_SHAPE", None if vp._preview_radii is None else vp._preview_radii.shape)
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "HAS_SCENE True" in output
    assert "SCENE_SHAPE (1, 3)" in output
    assert "COLOR_SHAPE (1, 3)" in output
    assert "RAD_SHAPE (1,)" in output


def test_panel_changed_can_skip_preview_worker_for_viewport_drag_updates() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            calls = {"preview": 0}

            def fake_preview():
                calls["preview"] += 1

            win._request_preview_render = fake_preview
            win._suppress_preview_render_once = True
            win._panel_changed({})
            print("FIRST_CALLS", calls["preview"])
            win._panel_changed({})
            print("SECOND_CALLS", calls["preview"])
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "FIRST_CALLS 0" in output
    assert "SECOND_CALLS 1" in output


def test_rotation_submits_interactive_render_and_skips_preview_worker() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            calls = {"preview": 0, "render": 0}

            def fake_preview():
                calls["preview"] += 1

            def fake_render(*, interactive=False):
                calls["render"] += 1
                print("RENDER_INTERACTIVE", interactive)

            win._request_preview_render = fake_preview
            win._render = fake_render
            win._suppress_preview_render_once = False
            win._on_viewport_rotation(1.0, 2.0)
            print("PREVIEW_CALLS", calls["preview"])
            print("RENDER_CALLS", calls["render"])
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "PREVIEW_CALLS 0" in output
    assert "RENDER_CALLS 1" in output
    assert "RENDER_INTERACTIVE True" in output


def test_build_params_respects_render_size_controls() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win.pdb_path = "dummy.pdb"
            win.render_size_mode_combo.setCurrentText("Custom")
            win.render_width_spin.setValue(2000)
            win.render_height_spin.setValue(1500)
            custom = win._build_params()
            win.render_size_mode_combo.setCurrentText("Auto")
            auto = win._build_params()
            print("CUSTOM", custom.world.width, custom.world.height)
            print("AUTO", auto.world.width, auto.world.height)
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "CUSTOM 2000 1500" in output
    assert "AUTO -30 -30" in output


def test_apply_loaded_params_sets_custom_render_size_mode() -> None:
    output = _run_gui_script(
        """
        from pathlib import Path
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        from illustrate import OutlineParams, RenderParams, SelectionRule, Transform, WorldParams
        import illustrate_gui.app as app_module

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
            win._apply_loaded_params(params, source=Path("."))
            print("MODE", win.render_size_mode_combo.currentText())
            print("WH", win.render_width_spin.value(), win.render_height_spin.value())
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "MODE Custom" in output
    assert "WH 1234 876" in output


def test_fit_view_resets_translation_and_adjusts_scale() -> None:
    output = _run_gui_script(
        """
        from types import SimpleNamespace
        import numpy as np
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        orig_estimate = app_module.estimate_render_size
        app_module.MainWindow._apply_preset = lambda self, _index: None
        app_module.estimate_render_size = lambda _atoms, _params: (300, 200)
        try:
            win = app_module.MainWindow()
            win.resize(1400, 900)
            win.pdb_path = "dummy.pdb"
            win.transform_panel.set_value(scale=12.0, xtran=40.0, ytran=-20.0, ztran=5.0)
            win._atoms = SimpleNamespace(
                n=1,
                coord=np.asarray([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]], dtype=np.float32),
                type_idx=np.asarray([0, 1], dtype=np.int32),
            )
            win._atoms_signature = app_module._rules_signature(win._current_rules())
            win._fit_view()
            t = win.transform_panel.value
            print("TRAN", t["xtran"], t["ytran"], t["ztran"])
            print("SCALE_GT", t["scale"] > 12.0)
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
            app_module.estimate_render_size = orig_estimate
        """
    )
    assert "TRAN 0.0 0.0 0.0" in output
    assert "SCALE_GT True" in output


def test_rule_panel_match_counts_are_populated_from_atom_types() -> None:
    output = _run_gui_script(
        """
        from types import SimpleNamespace
        import numpy as np
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            win._atoms = SimpleNamespace(type_idx=np.asarray([0, 1, 1, 3, 3, 3], dtype=np.int32))
            win._update_rule_match_counts()
            row0 = win.rule_panel.table.item(0, 6).text()
            row1 = win.rule_panel.table.item(1, 6).text()
            row2 = win.rule_panel.table.item(2, 6).text()
            print("COUNTS", row0, row1, row2)
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "COUNTS 2 0 3" in output


def test_rule_panel_descriptor_column_is_editable_and_normalized() -> None:
    output = _run_gui_script(
        """
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate_gui.app as app_module

        app = QApplication([])
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            headers = [
                win.rule_panel.table.horizontalHeaderItem(i).text()
                for i in range(win.rule_panel.table.columnCount())
            ]
            descriptor_edit = win.rule_panel.table.cellWidget(0, 1)
            descriptor_edit.setText("fe---hem--")
            after = win.rule_panel.value[0].descriptor
            print("COLS", win.rule_panel.table.columnCount())
            print("HAS_DESC", "Descriptor" in headers)
            print("DESC_AFTER", after)
            win.closeEvent(QCloseEvent())
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "COLS 7" in output
    assert "HAS_DESC True" in output
    assert "DESC_AFTER FE---HEM--" in output
