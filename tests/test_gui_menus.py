from __future__ import annotations

from tests.test_gui_regressions import run_gui_script


def test_dark_appearance_recolors_the_viewport_and_edits_flag_the_render_button() -> None:
    output = run_gui_script(
        """
        from illustrate.gui import kit
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        from illustrate.gui.themes import apply_appearance

        app = QApplication([])
        apply_appearance("light", app)
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            bar_actions = win.menuBar().actions()
            view_actions = next(a for a in bar_actions if a.text() == "&View").menu().actions()
            appearance = next(a for a in view_actions if a.text() == "&Appearance").menu().actions()
            next(a for a in appearance if a.data() == "dark").trigger()
            print("VIEWPORT_DARK", win.viewport._empty_bg.name() == kit.colors("dark")["surface-sunken"])
            win.mark_params_dirty()
            print("DIRTY_WITHOUT_STRUCTURE", bool(win._render_btn.property("attention")))
            win.pdb_path = "model.pdb"
            win.mark_params_dirty()
            print("RENDER_DIRTY", bool(win._render_btn.property("attention")))
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "VIEWPORT_DARK True" in output
    assert "DIRTY_WITHOUT_STRUCTURE False" in output
    assert "RENDER_DIRTY True" in output


def test_close_waits_for_a_running_size_estimate_without_blocking() -> None:
    # A size estimate on a large structure must not freeze the window on close.
    output = run_gui_script(
        """
        import threading
        import time
        from PySide6.QtGui import QCloseEvent
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module
        import illustrate.gui.workers.render as render_workers
        from illustrate import RenderParams
        from illustrate.gui.workers import SizeEstimateRequest

        app = QApplication([])
        release = threading.Event()
        started = threading.Event()

        def slow_estimate(_atoms, _params):
            started.set()
            release.wait(10.0)
            return (100, 100)

        render_workers.estimate_render_size = slow_estimate
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, _index: None
        try:
            win = app_module.MainWindow()
            worker = win.render_size.estimate_worker
            worker.submit(SizeEstimateRequest(params=RenderParams(pdb_path="", rules=[]), atoms=None))
            started.wait(5.0)
            event = QCloseEvent()
            t0 = time.monotonic()
            win.closeEvent(event)
            print("FAST", time.monotonic() - t0 < 2.0)
            print("DEFERRED", not event.isAccepted(), win.statusBar().currentMessage())
            release.set()
            worker.wait(5000)
            event = QCloseEvent()
            win.closeEvent(event)
            print("CLOSED", event.isAccepted())
        finally:
            release.set()
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "FAST True" in output
    assert "DEFERRED True Closing after the output size estimate finishes..." in output
    assert "CLOSED True" in output


def test_preset_list_applies_rows_and_keeps_saved_selection() -> None:
    output = run_gui_script(
        """
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        import illustrate.gui.main_window as app_module

        app = QApplication([])
        applied = []
        orig_apply_preset = app_module.MainWindow._apply_preset
        app_module.MainWindow._apply_preset = lambda self, index: applied.append(index)
        try:
            win = app_module.MainWindow()
            applied.clear()
            win.preset_list.setCurrentRow(1)
            print("APPLIED", applied)
            win._custom_presets.append(("Mine", win._builtin_presets[0][1]))
            win._refresh_preset_list(preferred_label="Custom: Mine")
            print("SELECTED", win.preset_list.currentItem().text(), applied)
            close_test_window(win)
        finally:
            app_module.MainWindow._apply_preset = orig_apply_preset
        """
    )
    assert "APPLIED [1]" in output
    assert "SELECTED Custom: Mine [1]" in output


def test_rotation_controls_trim_display_but_keep_precision() -> None:
    output = run_gui_script(
        """
        from PySide6.QtWidgets import QApplication
        from illustrate.gui.panels.transform import TransformPanel

        app = QApplication([])
        panel = TransformPanel()
        print("DEFAULT", repr(panel.zrot.text()), repr(panel.xrot.text()))
        panel.set_value(xrot=12.345678, yrot=-0.0)
        print("PRECISE", panel.value["xrot"], repr(panel.xrot.text()), repr(panel.yrot.text()))
        panel.yrot.lineEdit().setText("45.5°")
        panel.yrot.interpretText()
        print("TYPED", panel.value["yrot"])
        """
    )
    assert "DEFAULT '90°' '0°'" in output
    assert "PRECISE 12.345678 '12.345678°' '0°'" in output
    assert "TYPED 45.5" in output
