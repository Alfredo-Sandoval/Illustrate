"""Preset route."""

from __future__ import annotations

from fastapi import APIRouter

from illustrate.presets import preset_payloads

router = APIRouter(prefix="/api")


@router.get("/presets")
async def list_presets() -> list[dict[str, object]]:
    return preset_payloads("default")
