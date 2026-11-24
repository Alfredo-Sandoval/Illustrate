"""Desktop workers for rendering, structure loading, and size estimates."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Generic, TypeVar

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from illustrate import (
    AtomTable,
    RenderParams,
    RenderResult,
    SelectionRule,
    estimate_render_size,
    render_from_atoms,
)
from illustrate.io.rcsb import fetch_pdb
from illustrate.formats.pdb import PdbRecords, classify_atoms, read_pdb_records
from illustrate.gui.image_frame import GuiImageFrame, image_frame_from_arrays
from illustrate.rendering.pipeline import RenderCancellationToken, RenderCancelled


@dataclass
class RenderRequest:
    params: RenderParams
    atoms: AtomTable
    request_id: int = 0
    interactive: bool = False


@dataclass(frozen=True)
class _QueuedRenderRequest:
    request: RenderRequest
    epoch: int


@dataclass(frozen=True)
class RenderJobResult:
    request_id: int
    interactive: bool
    result: RenderResult
    frame: GuiImageFrame | None = None


@dataclass(frozen=True)
class RenderJobFailure:
    request_id: int
    interactive: bool
    message: str


@dataclass(frozen=True)
class RenderJobStale:
    request_id: int
    interactive: bool
    reason: str


@dataclass(frozen=True)
class LoadJobResult:
    request_id: int
    rules_signature: str
    atoms: AtomTable
    match_counts: list[int] | None = None


@dataclass(frozen=True)
class LoadJobFailure:
    request_id: int
    rules_signature: str
    message: str


@dataclass
class LoadRequest:
    pdb_path: str
    rules: list[SelectionRule]
    rules_signature: str
    request_id: int = 0


@dataclass
class SizeEstimateRequest:
    params: RenderParams
    atoms: AtomTable
    request_id: int = 0
    purpose: str = "label"
    # Identifies the atoms and settings, so the caller can cache the result.
    cache_key: str = ""


@dataclass(frozen=True)
class SizeEstimateJobResult:
    request_id: int
    purpose: str
    width: int
    height: int
    cache_key: str = ""


@dataclass(frozen=True)
class SizeEstimateJobFailure:
    request_id: int
    purpose: str
    message: str


@dataclass(frozen=True)
class FetchRequest:
    request_id: int
    pdb_id: str


@dataclass(frozen=True)
class FetchJobResult:
    request_id: int
    path: str
    pdb_id: str


@dataclass(frozen=True)
class FetchJobFailure:
    request_id: int
    message: str


class _RenderCancellationState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch = 0

    def advance(self) -> int:
        with self._lock:
            self._epoch += 1
            return self._epoch

    def is_current(self, epoch: int) -> bool:
        with self._lock:
            return epoch == self._epoch


class _RenderCancellationToken:
    def __init__(self, state: _RenderCancellationState, epoch: int) -> None:
        self._state = state
        self._epoch = int(epoch)

    def is_cancelled(self) -> bool:
        return not self._state.is_current(self._epoch)


def _render_from_atoms_accepts_cancellation_token() -> bool:
    try:
        parameters = signature(render_from_atoms).parameters
    except (TypeError, ValueError):
        return True
    return "cancellation_token" in parameters or any(
        parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _render_from_atoms(
    atoms: AtomTable,
    params: RenderParams,
    *,
    backend: str | None,
    cancellation_token: _RenderCancellationToken,
) -> tuple[RenderResult, bool]:
    if _render_from_atoms_accepts_cancellation_token():
        return (
            render_from_atoms(
                atoms,
                params,
                backend=backend,
                cancellation_token=cancellation_token,
            ),
            True,
        )
    return render_from_atoms(atoms, params, backend=backend), False


_Request = TypeVar("_Request")


class _CoalescingWorker(QThread, Generic[_Request]):
    """Keep the newest queued request and restart if one arrives during shutdown."""

    result_ready = Signal(object)
    failed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._request: _Request | None = None
        self._request_lock = threading.Lock()
        self.finished.connect(self._start_pending)

    @Slot()
    def _start_pending(self) -> None:
        with self._request_lock:
            if self._request is not None and not self.isRunning():
                self.start()

    def _submit_request(self, request: _Request) -> None:
        with self._request_lock:
            self._request = request
        self._start_pending()

    def _pop_request(self) -> _Request | None:
        with self._request_lock:
            request = self._request
            self._request = None
        return request

    def cancel_pending(self) -> None:
        with self._request_lock:
            self._request = None


class RenderWorker(_CoalescingWorker[_QueuedRenderRequest]):
    """Render in the background and cancel work superseded by a newer request."""

    stale = Signal(object)

    def __init__(
        self, parent: QObject | None = None,
        *, startup_backend: Callable[[RenderCancellationToken], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._latest_request_id = 0
        self._cancellation_state = _RenderCancellationState()
        self._startup_backend = startup_backend

    def submit(self, request: RenderRequest) -> None:
        replaced_request: _QueuedRenderRequest | None = None
        stale_request: RenderRequest | None = None
        with self._request_lock:
            if request.request_id != 0 and int(request.request_id) < self._latest_request_id:
                stale_request = request
            else:
                if request.request_id != 0:
                    self._latest_request_id = max(
                        self._latest_request_id,
                        int(request.request_id),
                    )
                epoch = self._cancellation_state.advance()
                replaced_request = self._request
                self._request = _QueuedRenderRequest(request=request, epoch=epoch)
        if replaced_request is not None and replaced_request.request.request_id != 0:
            self._emit_stale(replaced_request.request, "coalesced")
        if stale_request is not None:
            self._emit_stale(stale_request, "stale-submit")
        self._start_pending()

    def cancel_pending(self) -> None:
        with self._request_lock:
            self._cancellation_state.advance()
            queued = self._request
            self._request = None
        if queued is not None:
            self._emit_stale(queued.request, "cancelled")

    def _request_is_stale(self, request: RenderRequest) -> bool:
        if request.request_id == 0:
            return False
        with self._request_lock:
            return int(request.request_id) < self._latest_request_id

    def _emit_stale(self, request: RenderRequest, reason: str) -> None:
        if request.request_id == 0:
            return
        self.stale.emit(
            RenderJobStale(
                request_id=int(request.request_id),
                interactive=bool(request.interactive),
                reason=reason,
            )
        )

    def run(self) -> None:
        while True:
            queued_request = self._pop_request()
            if queued_request is None:
                return
            request = queued_request.request
            if self._request_is_stale(request):
                self._emit_stale(request, "before-start")
                continue
            cancellation_token = _RenderCancellationToken(
                self._cancellation_state,
                queued_request.epoch,
            )
            try:
                result, supports_cancellation = _render_from_atoms(
                    request.atoms,
                    request.params,
                    backend=self._startup_backend(cancellation_token) if self._startup_backend is not None else None,
                    cancellation_token=cancellation_token,
                )
            except RenderCancelled:
                self._emit_stale(request, "cancelled")
                continue
            except (RuntimeError, TypeError, ValueError) as exc:
                if self._request_is_stale(request):
                    self._emit_stale(request, "failed-stale")
                    continue
                if (
                    _render_from_atoms_accepts_cancellation_token()
                    and cancellation_token.is_cancelled()
                ):
                    self._emit_stale(request, "cancelled")
                    continue
                if request.request_id != 0:
                    self.failed.emit(
                        RenderJobFailure(
                            request_id=request.request_id,
                            interactive=bool(request.interactive),
                            message=str(exc),
                        )
                    )
                else:
                    self.failed.emit(str(exc))
                continue
            if self._request_is_stale(request):
                self._emit_stale(request, "finished-stale")
                continue
            if supports_cancellation and cancellation_token.is_cancelled():
                self._emit_stale(request, "cancelled")
                continue
            frame = None
            if hasattr(result, "rgb"):
                frame = image_frame_from_arrays(
                    result.rgb,
                    result.opacity if hasattr(result, "opacity") else None,
                )
            if self._request_is_stale(request):
                self._emit_stale(request, "packed-stale")
                continue
            if supports_cancellation and cancellation_token.is_cancelled():
                self._emit_stale(request, "cancelled")
                continue
            if request.request_id != 0:
                self.result_ready.emit(
                    RenderJobResult(
                        request_id=request.request_id,
                        interactive=bool(request.interactive),
                        result=result,
                        frame=frame,
                    )
                )
            else:
                self.result_ready.emit(result)


class LoadWorker(_CoalescingWorker[LoadRequest]):
    """Reuse parsed records for rule changes and return per-rule atom counts."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # The last file read, so a rule change re-classifies its atoms instead
        # of parsing the file again. Only the worker thread touches these.
        self._records_key: tuple[str, int, int] | None = None
        self._records: PdbRecords | None = None

    def submit(self, request: LoadRequest) -> None:
        self._submit_request(request)

    def _read_records(self, pdb_path: str) -> PdbRecords:
        stat = os.stat(pdb_path)
        # A file edited on disk has a new modification time or size.
        key = (os.path.abspath(pdb_path), stat.st_mtime_ns, stat.st_size)
        if self._records is None or key != self._records_key:
            self._records = read_pdb_records(pdb_path)
            self._records_key = key
        return self._records

    def run(self) -> None:
        while True:
            request = self._pop_request()
            if request is None:
                return
            try:
                atoms = classify_atoms(self._read_records(request.pdb_path), request.rules)
            except (OSError, TypeError, ValueError) as exc:
                self.failed.emit(
                    LoadJobFailure(
                        request_id=request.request_id,
                        rules_signature=request.rules_signature,
                        message=str(exc),
                    )
                )
                continue
            match_counts = _rule_match_counts(atoms, len(request.rules))
            self.result_ready.emit(
                LoadJobResult(
                    request_id=request.request_id,
                    rules_signature=request.rules_signature,
                    atoms=atoms,
                    match_counts=match_counts,
                )
            )


class SizeEstimateWorker(_CoalescingWorker[SizeEstimateRequest]):
    """Estimate render sizes for the output label and for fit-to-view.

    Each purpose keeps its own newest request, so a label refresh cannot
    replace a fit that is still queued.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queued: dict[str, SizeEstimateRequest] = {}

    @Slot()
    def _start_pending(self) -> None:
        with self._request_lock:
            if self._queued and not self.isRunning():
                self.start()

    def submit(self, request: SizeEstimateRequest) -> None:
        with self._request_lock:
            self._queued[request.purpose] = request
        self._start_pending()

    def _pop_request(self) -> SizeEstimateRequest | None:
        with self._request_lock:
            if not self._queued:
                return None
            return self._queued.pop(next(iter(self._queued)))

    def cancel_pending(self) -> None:
        with self._request_lock:
            self._queued.clear()

    def run(self) -> None:
        while True:
            request = self._pop_request()
            if request is None:
                return
            try:
                width, height = estimate_render_size(request.atoms, request.params)
            except (
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                self.failed.emit(
                    SizeEstimateJobFailure(
                        request_id=request.request_id,
                        purpose=request.purpose,
                        message=str(exc),
                    )
                )
                continue
            self.result_ready.emit(
                SizeEstimateJobResult(
                    request_id=request.request_id,
                    purpose=request.purpose,
                    width=int(width),
                    height=int(height),
                    cache_key=request.cache_key,
                )
            )


def _rule_match_counts(atoms: AtomTable, rule_count: int) -> list[int]:
    if rule_count <= 0:
        return []
    counts = [0] * rule_count
    atom_types = np.asarray(atoms.type_idx, dtype=np.int32)
    if atom_types.size <= 1:
        return counts
    hist = np.bincount(atom_types[1:], minlength=rule_count + 1)
    for index in range(rule_count):
        counts[index] = int(hist[index + 1]) if index + 1 < len(hist) else 0
    return counts


class FetchWorker(QThread):
    """Fetch one structure at a time and return its local cache path."""

    fetched = Signal(object)
    failed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._request: FetchRequest | None = None
        self._request_lock = threading.Lock()

    def submit(self, request: FetchRequest) -> None:
        normalized = request.pdb_id.strip()
        if normalized == "":
            raise ValueError("pdb_id must not be empty")
        with self._request_lock:
            if self.isRunning():
                raise RuntimeError("fetch already in progress")
            self._request = FetchRequest(request_id=request.request_id, pdb_id=normalized)
        self.start()

    def _pop_request(self) -> FetchRequest | None:
        with self._request_lock:
            request = self._request
            self._request = None
        return request

    def run(self) -> None:
        request = self._pop_request()
        if request is None:
            return
        try:
            path = fetch_pdb(request.pdb_id)
        except (OSError, RuntimeError, ValueError) as exc:
            self.failed.emit(FetchJobFailure(request_id=request.request_id, message=str(exc)))
            return
        self.fetched.emit(FetchJobResult(request_id=request.request_id, path=str(path), pdb_id=request.pdb_id.upper()))
