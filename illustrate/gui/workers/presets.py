from __future__ import annotations

from dataclasses import dataclass, replace

from PySide6.QtCore import QObject, QThread, Signal

from illustrate import RenderParams, params_from_json, params_to_json
from illustrate.io.preset_store import load_custom_presets, write_custom_presets


@dataclass(frozen=True, slots=True)
class PresetSaveRequest:
    name: str
    params: RenderParams
    presets: tuple[tuple[str, RenderParams], ...]
    replace_index: int


def _save_preset(request: PresetSaveRequest) -> list[tuple[str, RenderParams]]:
    """Write the proposed list before publishing it back to the preset picker."""
    params = replace(params_from_json(params_to_json(request.params)), pdb_path="")
    presets = list(request.presets)
    if request.replace_index >= 0:
        presets[request.replace_index] = (request.name, params)
    else:
        presets.append((request.name, params))
    write_custom_presets(presets)
    return presets


class PresetWorker(QThread):
    """Load the custom preset store at startup and serialize later saves."""

    loaded = Signal(object)
    saved = Signal(str, list)
    failed = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._request: PresetSaveRequest | None = None

    def submit_save(self, request: PresetSaveRequest) -> bool:
        if self.isRunning():
            return False
        self._request = request
        self.start()
        return True

    def run(self) -> None:
        request = self._request
        self._request = None
        if request is None:
            try:
                result = load_custom_presets()
            except (OSError, TypeError, ValueError) as exc:
                self.failed.emit("load", str(exc))
            else:
                self.loaded.emit(result)
            return
        try:
            presets = _save_preset(request)
        except (OSError, TypeError, ValueError) as exc:
            self.failed.emit("save", str(exc))
        else:
            self.saved.emit(request.name, presets)
