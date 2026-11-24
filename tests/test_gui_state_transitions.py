from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_gui_regressions import run_gui_script


def test_panel_batches_preserve_all_values_and_emit_one_complete_change() -> None:
    run_gui_script("""
        from PySide6.QtWidgets import QApplication
        from illustrate.gui.panels.world import WorldPanel
        from illustrate.gui.panels.outlines import OutlinesPanel
        app = QApplication([])
        for panel_type, updates in (
            (WorldPanel, dict(fog_front=0.45, fog_back=0.65, shadows=False, shadow_strength=0.125, shadow_angle=4.0)),
            (OutlinesPanel, dict(enabled=False, contour_low=2.0, contour_high=12.0, kernel=2, z_diff_min=2.0, z_diff_max=15.0)),
        ):
            changes = []
            panel = panel_type(on_changed=lambda values: changes.append(dict(values)))
            requested = dict(panel.value, **updates)
            changes.clear()
            panel.set_value(requested)
            assert panel.value == requested, (panel_type.__name__, panel.value, requested)
            assert changes == [requested], changes
    """)


def test_gui_presets_preserve_orientation_world_and_outline_parameters() -> None:
    run_gui_script("""
        import numpy as np
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        from illustrate.gui.main_window import MainWindow
        from illustrate.rendering.api import rotation_matrix
        app = QApplication([])
        win = MainWindow()
        try:
            win.pdb_path = 'examples/2hhb.pdb'
            for name, preset in win._builtin_presets:
                win.apply_params_to_panels(preset)
                rebuilt = win.build_params()
                np.testing.assert_allclose(rotation_matrix(rebuilt.transform.rotations), rotation_matrix(preset.transform.rotations), atol=1e-6, rtol=0, err_msg=name+'.rotations')
                assert rebuilt.outlines == preset.outlines, (name, rebuilt.outlines, preset.outlines)
                for field in ('fog_front', 'fog_back', 'shadows', 'shadow_strength', 'shadow_angle', 'shadow_min_z', 'shadow_max_dark'):
                    np.testing.assert_allclose(getattr(rebuilt.world, field), getattr(preset.world, field), atol=1e-6, rtol=0, err_msg=name+'.'+field)
        finally:
            close_test_window(win)
    """)


@pytest.mark.parametrize("activation", ["open", "settings"])
@pytest.mark.parametrize("outcome", ["success", "failure"])
def test_late_fetch_cannot_replace_a_newer_structure(
    activation: str, outcome: str, tmp_path: Path,
) -> None:
    run_gui_script(f"""
        from pathlib import Path
        from threading import Event
        from PySide6.QtCore import QEventLoop, QTimer
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        from illustrate import RenderParams, WorldParams
        from illustrate.gui.main_window import MainWindow
        import illustrate.gui.workers.render as worker_module
        from tests.test_render_validation import _write_minimal_pdb
        app = QApplication([])
        win = MainWindow()
        entered, release = Event(), Event()
        fixture = Path({str(tmp_path / 'later.pdb')!r})
        _write_minimal_pdb(fixture)
        def delayed_fetch(pdb_id):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Fetch gate timed out')
            if {outcome!r} == 'failure':
                raise ConnectionError('old fetch failed')
            return Path('examples/2hhb.pdb').resolve()
        worker_module.fetch_pdb = delayed_fetch
        try:
            win.render_size_mode_combo.setCurrentText('Custom')
            win.render_width_spin.setValue(64)
            win.render_height_spin.setValue(64)
            win.pdb_id_input.setText('2hhb')
            win.structure_controller.fetch_pdb()
            assert entered.wait(3), 'Fetch did not start'
            if {activation!r} == 'settings':
                win._settings_controller.apply_loaded_params(
                    RenderParams(pdb_path=str(fixture), rules=win.current_rules(), world=WorldParams(width=64, height=64)),
                    source=fixture.with_suffix('.json'),
                    resolved_pdb=fixture,
                )
            else:
                win.structure_controller.activate_loaded_structure(path=str(fixture))
            loop = QEventLoop()
            win.fetch_worker.fetched.connect(loop.quit)
            win.fetch_worker.failed.connect(loop.quit)
            QTimer.singleShot(3000, loop.quit)
            release.set()
            loop.exec()
            assert win.fetch_worker.wait(3000), 'Fetch did not finish'
            assert win.pdb_path == str(fixture), win.pdb_path
            assert win.windowTitle() == 'later.pdb'
            assert 'Fetch failed' not in win.statusBar().currentMessage()
            assert win.main_actions.fetch_pdb.isEnabled()
        finally:
            release.set()
            close_test_window(win)
    """)


def test_control_edit_invalidates_full_render_without_marking_controls_clean() -> None:
    run_gui_script("""
        from threading import Event
        from PySide6.QtCore import QEventLoop, QTimer
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        from illustrate.formats.pdb import read_and_classify_atoms
        from illustrate.gui.main_window import MainWindow
        from illustrate.gui.render_params import rules_signature
        import illustrate.gui.workers.render as worker_module
        app = QApplication([])
        win = MainWindow()
        computed, release = Event(), Event()
        original = worker_module.render_from_atoms
        rotations = []
        def gated_render(atoms, params, *, backend=None, cancellation_token=None):
            result = original(atoms, params, backend='numpy', cancellation_token=cancellation_token)
            rotations.append(list(params.transform.rotations))
            computed.set()
            if not release.wait(5):
                raise RuntimeError('Render gate timed out')
            return result
        worker_module.render_from_atoms = gated_render
        try:
            win.pdb_path = 'examples/2hhb.pdb'
            rules = win.current_rules()
            win.atoms = read_and_classify_atoms(win.pdb_path, rules)
            win.atoms_signature = rules_signature(rules)
            win.render_size_mode_combo.setCurrentText('Custom')
            win.render_width_spin.setValue(64)
            win.render_height_spin.setValue(64)
            win.render_frame()
            assert computed.wait(3), 'Render did not compute'
            win.transform_panel.set_value(dict(win.transform_panel.value), xrot=15.0)
            loop = QEventLoop()
            win.worker.result_ready.connect(loop.quit)
            win.worker.stale.connect(loop.quit)
            QTimer.singleShot(3000, loop.quit)
            release.set()
            loop.exec()
            assert win.worker.wait(3000), 'Render did not finish'
            assert rotations[0][-1] == ('x', 0.0)
            assert win.transform_panel.value['xrot'] == 15.0
            assert win.params_dirty is True
            assert win.last_result is None
            assert not win.main_actions.copy_to_clipboard.isEnabled()
            assert win.main_actions.render.isEnabled()
            assert win._render_started_at is None
            win.render_frame()
            loop = QEventLoop()
            win.worker.result_ready.connect(loop.quit)
            QTimer.singleShot(3000, loop.quit)
            loop.exec()
            assert win.worker.wait(3000), 'Fresh render did not finish'
            assert win.last_result.width == 64
            assert win.params_dirty is False
            assert win.main_actions.copy_to_clipboard.isEnabled()
            assert rotations[-1][-1] == ('x', 15.0)
        finally:
            release.set()
            close_test_window(win)
    """)


def test_current_fetch_results_activate_the_model_and_report_errors(tmp_path: Path) -> None:
    run_gui_script(f"""
        from pathlib import Path
        from PySide6.QtCore import QEventLoop, QTimer
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        from illustrate.gui.main_window import MainWindow
        import illustrate.gui.workers.render as worker_module
        from tests.test_render_validation import _write_minimal_pdb
        app = QApplication([])
        win = MainWindow()
        fixture = Path({str(tmp_path / 'fetched.pdb')!r})
        _write_minimal_pdb(fixture)
        def fetch(pdb_id):
            if pdb_id == 'bad0':
                raise ConnectionError('current fetch failed')
            return fixture
        worker_module.fetch_pdb = fetch
        messages = []
        win.statusBar().messageChanged.connect(messages.append)
        try:
            win.render_size_mode_combo.setCurrentText('Custom')
            win.render_width_spin.setValue(64)
            win.render_height_spin.setValue(64)
            loop = QEventLoop()
            win.fetch_worker.fetched.connect(loop.quit)
            win.fetch_worker.failed.connect(loop.quit)
            for pdb_id in ('2hhb', 'bad0'):
                win.pdb_id_input.setText(pdb_id)
                win.structure_controller.fetch_pdb()
                QTimer.singleShot(3000, loop.quit)
                loop.exec()
                assert win.fetch_worker.wait(3000), 'Fetch did not finish'
                assert win.pdb_path == str(fixture)
                assert win.windowTitle() == '2HHB.pdb'
                assert win.main_actions.fetch_pdb.isEnabled()
            assert 'Fetch failed: current fetch failed' in messages, messages
        finally:
            close_test_window(win)
    """)


@pytest.mark.parametrize("field", ["scale", "translate", "width", "shadow_angle", "kernel"])
def test_unsupported_settings_ranges_fail_before_any_controls_change(field: str) -> None:
    run_gui_script(f"""
        from pathlib import Path
        from tests.test_gui_regressions import close_test_window
        from PySide6.QtWidgets import QApplication
        from illustrate import params_from_json, params_to_json
        from illustrate.gui.main_window import MainWindow
        from illustrate.gui.workers import SettingsIOResult
        import illustrate.gui.controllers.settings as settings_module
        app = QApplication([])
        win = MainWindow()
        errors = []
        settings_module.QMessageBox.critical = lambda *args: errors.append(args[2])
        try:
            win.pdb_path = 'examples/2hhb.pdb'
            before = params_to_json(win.build_params())
            params = params_from_json(before)
            params.transform.rotations = [('x', 35.0)]
            field = {field!r}
            if field == 'scale':
                params.transform.scale = 0.5
            elif field == 'translate':
                params.transform.translate = (10001.0, 0.0, 0.0)
            elif field == 'width':
                params.world.width, params.world.height = 32, 64
            elif field == 'shadow_angle':
                params.world.shadow_angle = 20.0
            else:
                params.outlines.kernel = 9
            win._settings_controller._on_settings_io_done(SettingsIOResult(path=Path('settings.json'), kind='load', params=params))
            assert len(errors) == 1, errors
            assert 'GUI settings' in errors[0]
            assert params_to_json(win.build_params()) == before
            assert win.pdb_path == 'examples/2hhb.pdb'
        finally:
            close_test_window(win)
    """)
