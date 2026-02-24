from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class SelectionRule:
    record_name: str
    descriptor: str
    res_low: int
    res_high: int
    color: tuple[float, float, float]
    radius: float


@dataclass(slots=True)
class WorldParams:
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fog_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fog_front: float = 1.0
    fog_back: float = 1.0
    shadows: bool = False
    shadow_strength: float = 0.0023
    shadow_angle: float = 2.0
    shadow_min_z: float = 1.0
    shadow_max_dark: float = 0.2
    width: int = 0
    height: int = 0


@dataclass(slots=True)
class OutlineParams:
    enabled: bool = False
    contour_low: float = 1.0
    contour_high: float = 10.0
    kernel: int = 4
    z_diff_min: float = 1.0
    z_diff_max: float = 50.0
    subunit_low: float = 3.0
    subunit_high: float = 10.0
    residue_low: float = 3.0
    residue_high: float = 8.0
    residue_diff: float = 6000.0


@dataclass(slots=True)
class TransformState:
    scale: float = 1.0
    xtran: float = 0.0
    ytran: float = 0.0
    ztran: float = 0.0
    autocenter: int = 0
    rm: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32),
    )


@dataclass(slots=True)
class CommandProgram:
    pdb_file: Path | None = None
    selection_rules: list[SelectionRule] = field(default_factory=list)
    transform: TransformState = field(default_factory=TransformState)
    world: WorldParams = field(default_factory=WorldParams)
    outlines: OutlineParams = field(default_factory=OutlineParams)
    output_file: Path | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AtomTable:
    coord: np.ndarray
    type_idx: np.ndarray
    res: np.ndarray
    su: np.ndarray
    biomat: np.ndarray
    n: int
    nbiomat: int


@dataclass(slots=True)
class RenderResult:
    rgb: np.ndarray
    opacity: np.ndarray
    width: int
    height: int


@dataclass(slots=True)
class Transform:
    scale: float = 12.0
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotations: list[tuple[str, float]] = field(default_factory=list)
    autocenter: str = "auto"


@dataclass(slots=True)
class RenderParams:
    pdb_path: str
    rules: list[SelectionRule]
    transform: Transform = field(default_factory=Transform)
    world: WorldParams = field(default_factory=WorldParams)
    outlines: OutlineParams = field(default_factory=OutlineParams)


def _resolve_path(value: str | Path) -> str:
    return str(Path(value))
