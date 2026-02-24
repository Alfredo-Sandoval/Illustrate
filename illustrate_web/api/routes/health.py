"""Health route."""

from __future__ import annotations

try:
    from fastapi import APIRouter
except Exception:  # pragma: no cover
    APIRouter = None  # type: ignore

if APIRouter is None:  # pragma: no cover
    raise RuntimeError("fastapi is required to import API routes")

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
