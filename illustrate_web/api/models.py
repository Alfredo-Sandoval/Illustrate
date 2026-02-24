"""Pydantic models used by web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TransformPayload(BaseModel):
    scale: float = 12.0
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotations: list[tuple[str, float]] = Field(default_factory=list)
    autocenter: str = "auto"


class WorldPayload(BaseModel):
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


class OutlinePayload(BaseModel):
    enabled: bool = True
    contour_low: float = 3.0
    contour_high: float = 10.0
    kernel: int = 4
    z_diff_min: float = 0.0
    z_diff_max: float = 5.0
    subunit_low: float = 3.0
    subunit_high: float = 10.0
    residue_low: float = 3.0
    residue_high: float = 8.0
    residue_diff: float = 6000.0


class RenderRequest(BaseModel):
    pdb_id: str
    rules: list[dict[str, Any]] = Field(default_factory=list)
    transform: TransformPayload = Field(default_factory=TransformPayload)
    world: WorldPayload = Field(default_factory=WorldPayload)
    outlines: OutlinePayload = Field(default_factory=OutlinePayload)
    output_format: str = Field(default="png")


class UploadResponse(BaseModel):
    pdb_id: str
