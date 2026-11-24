from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QComboBox, QLabel, QSpinBox

from illustrate import AtomTable, RenderParams, params_to_json

from illustrate.gui.render_params import AUTO_FRAME_PAD
from illustrate.gui.workers import (
    SizeEstimateJobFailure,
    SizeEstimateJobResult,
    SizeEstimateRequest,
    SizeEstimateWorker,
)

if TYPE_CHECKING:
    from illustrate.gui.main_window import MainWindow


@dataclass(frozen=True)
class _FitTarget:
    """The view a fit was requested for, applied once its size estimate arrives."""

    view_w: int
    view_h: int
    current_scale: float
    render: bool


class RenderSizeToolbar(QObject):
    """Connect output-size controls to debounced estimates and Fit View."""

    def __init__(
        self,
        window: MainWindow,
        *,
        mode_combo: QComboBox,
        width_spin: QSpinBox,
        height_spin: QSpinBox,
        dims_label: QLabel,
        build_params: Callable[[], RenderParams],
        debounce_ms: int = 60,
    ) -> None:
        super().__init__(window)
        self._window = window
        self._mode_combo = mode_combo
        self._width_spin = width_spin
        self._height_spin = height_spin
        self._dims_label = dims_label
        self.build_params = build_params

        self._dimensions_timer = QTimer(self)
        self._dimensions_timer.setSingleShot(True)
        self._dimensions_timer.timeout.connect(self._update_dimensions_label)
        self._debounce_ms = debounce_ms
        self._estimate_worker = SizeEstimateWorker(self)
        self._estimate_worker.result_ready.connect(self._on_estimate_done)
        self._estimate_worker.failed.connect(self._on_estimate_failed)
        self._estimate_request_id = 0
        self._latest_label_request_id = 0
        # The newest fit still waiting for its estimate; older fits are dropped.
        self._pending_fit: tuple[int, _FitTarget] | None = None
        self._estimate_cache: OrderedDict[str, tuple[int, int]] = OrderedDict()

    def selected_dimensions(self) -> tuple[int, int]:
        if self._mode_combo.currentText() != "Custom":
            # Negative values request Fortran-style auto-frame padding.
            return (-AUTO_FRAME_PAD, -AUTO_FRAME_PAD)
        return (int(self._width_spin.value()), int(self._height_spin.value()))

    def sync_controls_enabled(self) -> None:
        custom = self._mode_combo.currentText() == "Custom"
        self._width_spin.setEnabled(custom)
        self._height_spin.setEnabled(custom)

    def on_size_changed(self, _value: object = None) -> None:
        self.sync_controls_enabled()
        if self._window.panel_callbacks_suspended:
            return
        self._window.mark_params_dirty()
        self.schedule_dimensions_update()
        self._window.statusBar().showMessage("Render size updated.")

    def schedule_dimensions_update(self) -> None:
        if self._dimensions_timer.isActive():
            self._dimensions_timer.stop()
        self._dimensions_timer.start(self._debounce_ms)

    @property
    def estimate_worker(self) -> SizeEstimateWorker:
        return self._estimate_worker

    def stop(self) -> None:
        """Stop scheduling estimates and cancel queued ones.

        An estimate already running finishes on its own. The window can
        defer closing until the worker exits.
        """
        self._dimensions_timer.stop()
        self._estimate_worker.cancel_pending()

    def cached_dimensions(self, atoms: AtomTable, params: RenderParams) -> tuple[int, int]:
        key = self._cache_key(atoms, params)
        cached = self._estimate_cache.get(key)
        if cached is None:
            return (0, 0)
        self._estimate_cache.move_to_end(key)
        return cached

    def _update_dimensions_label(self) -> None:
        width, height = self.selected_dimensions()
        if width > 0 and height > 0:
            self._dims_label.setText(f"Output: {width}×{height}")
            return

        window = self._window
        if window.pdb_path is None:
            self._dims_label.setText("Output: auto")
            return
        if window.atoms is None:
            self._dims_label.setText("Output: auto")
            window.load_atoms_if_needed(
                status_message="Loading structure for render size...",
            )
            return
        atoms_ready = window.load_atoms_if_needed(
            status_message="Updating structure for render size...",
        )
        if atoms_ready is False:
            self._dims_label.setText("Output: auto")
            return
        params = self.build_params()
        est_w, est_h = self.cached_dimensions(window.atoms, params)
        if est_w > 0 and est_h > 0:
            self._dims_label.setText(f"Output: {est_w}×{est_h}")
            return
        self._dims_label.setText("Output: estimating...")
        self._submit_estimate("label", window.atoms, params)

    def fit_view(self, *, render: bool = False) -> None:
        """Request a fit to the viewport in device pixels.

        A missing size estimate is computed in the worker. Pass ``render=True``
        when the structure still needs its first image, even if live rendering
        is disabled.
        """
        window = self._window
        if window.pdb_path is None:
            window.statusBar().showMessage("Load a PDB first.")
            return
        atoms_ready = window.load_atoms_if_needed(
            status_message="Loading structure before fit...",
            render_after_load=render,
        )
        if atoms_ready is False or window.atoms is None:
            window.statusBar().showMessage("Structure is still loading...")
            return
        if window.atoms.n <= 0:
            window.statusBar().showMessage("No atoms available to fit.")
            if render:
                window.render_frame()
            return

        # Fit in device pixels: the viewport shows one render pixel per device
        # pixel, so a logical-size fit would look half-size on HiDPI screens.
        pixel_ratio = float(window.viewport.devicePixelRatioF())
        view_w = int(max(320, window.viewport.width()) * pixel_ratio)
        view_h = int(max(240, window.viewport.height()) * pixel_ratio)

        params = self.build_params()
        params.world.width = 0
        params.world.height = 0
        params.transform.translate = (0.0, 0.0, 0.0)

        est_w, est_h = self.cached_dimensions(window.atoms, params)
        if est_w <= 0 or est_h <= 0:
            request_id = self._submit_estimate("fit", window.atoms, params)
            self._pending_fit = (
                request_id,
                _FitTarget(
                    view_w=view_w,
                    view_h=view_h,
                    current_scale=float(window.transform_panel.value["scale"]),
                    render=render,
                ),
            )
            window.statusBar().showMessage("Fitting view...")
            return
        self._apply_fit_view(
            est_w,
            est_h,
            view_w=view_w,
            view_h=view_h,
            current_scale=float(window.transform_panel.value["scale"]),
            render=render,
        )

    def _submit_estimate(self, purpose: str, atoms: AtomTable, params: RenderParams) -> int:
        self._estimate_request_id += 1
        request_id = self._estimate_request_id
        if purpose == "label":
            self._latest_label_request_id = request_id
        self._estimate_worker.submit(
            SizeEstimateRequest(
                params=params,
                atoms=atoms,
                request_id=request_id,
                purpose=purpose,
                cache_key=self._cache_key(atoms, params),
            )
        )
        return request_id

    def _take_pending_fit(self, request_id: int) -> _FitTarget | None:
        if self._pending_fit is None or self._pending_fit[0] != request_id:
            return None
        target = self._pending_fit[1]
        self._pending_fit = None
        return target

    def _on_estimate_done(self, payload: object) -> None:
        if not isinstance(payload, SizeEstimateJobResult):
            return
        if payload.cache_key:
            self._store_estimate(payload.cache_key, (payload.width, payload.height))
        if payload.purpose == "label":
            if payload.request_id != self._latest_label_request_id:
                return
            if payload.width > 0 and payload.height > 0:
                self._dims_label.setText(f"Output: {payload.width}×{payload.height}")
            else:
                self._dims_label.setText("Output: auto")
            return
        if payload.purpose == "fit":
            target = self._take_pending_fit(payload.request_id)
            if target is None:
                return
            self._apply_fit_view(
                payload.width,
                payload.height,
                view_w=target.view_w,
                view_h=target.view_h,
                current_scale=target.current_scale,
                render=target.render,
            )

    def _on_estimate_failed(self, payload: object) -> None:
        if not isinstance(payload, SizeEstimateJobFailure):
            return
        if payload.purpose == "label":
            if payload.request_id == self._latest_label_request_id:
                self._dims_label.setText("Output: auto")
            return
        target = self._take_pending_fit(payload.request_id)
        if target is None:
            return
        self._window.statusBar().showMessage(f"Fit to view failed: {payload.message}")
        if target.render:
            self._window.render_frame()

    def _apply_fit_view(
        self,
        est_w: int,
        est_h: int,
        *,
        view_w: int,
        view_h: int,
        current_scale: float,
        render: bool = False,
    ) -> None:
        window = self._window
        if est_w <= 0 or est_h <= 0:
            window.statusBar().showMessage("Fit to view failed: invalid bounds.")
            if render:
                window.render_frame()
            return

        # The estimate is the bare structure. An auto-sized render adds a fixed
        # frame on every side that does not grow with the scale.
        frame = 2.0 * AUTO_FRAME_PAD if self._mode_combo.currentText() != "Custom" else 0.0
        fit_factor = min(
            (0.92 * float(view_w) - frame) / float(est_w),
            (0.92 * float(view_h) - frame) / float(est_h),
        )
        fit_factor = max(0.05, min(20.0, fit_factor))
        next_scale = max(1.0, current_scale * fit_factor)
        # Same path as a viewport drag: re-render now when render-on-drag is on.
        # The Scale control caps the value where the atom radii allow.
        window.apply_viewport_transform(
            scale=next_scale, xtran=0.0, ytran=0.0, ztran=0.0
        )
        fitted = float(window.transform_panel.value["scale"])
        window.statusBar().showMessage(f"Fitted view (scale {fitted:.2f})")
        if render and not window.auto_render_on_drag.isChecked():
            window.render_frame()

    def _cache_key(self, atoms: AtomTable, params: RenderParams) -> str:
        try:
            params_key = params_to_json(params)
        except (AttributeError, TypeError, ValueError):
            params_key = repr(params)
        return ":".join(
            (
                str(id(atoms)),
                str(atoms.n),
                self._window.atoms_signature,
                params_key,
            )
        )

    def _store_estimate(self, key: str, dims: tuple[int, int]) -> None:
        self._estimate_cache[key] = dims
        self._estimate_cache.move_to_end(key)
        while len(self._estimate_cache) > 32:
            self._estimate_cache.popitem(last=False)
