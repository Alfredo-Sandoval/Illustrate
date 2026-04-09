"""Render route implementation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import hashlib
from io import BytesIO
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image as PILImage

from illustrate import OutlineParams, RenderParams, SelectionRule, Transform, WorldParams, render_from_atoms
from illustrate.pdb import load_pdb as _load_pdb

from illustrate_web.api.deps import RateLimitExceeded, enforce_rate_limit, get_upload_path
from illustrate_web.api.models import RenderRequest

router = APIRouter(prefix="/api")
_ATOM_CACHE_MAX = 16
_ATOM_CACHE: OrderedDict[tuple[str, tuple[int, int, int, int], str], object] = OrderedDict()
_ATOM_CACHE_LOCK = threading.Lock()


def _to_rules(raw_rules: list[dict]) -> list[SelectionRule]:
    rules: list[SelectionRule] = []
    for rule in raw_rules:
        color = rule.get("color", (1.0, 1.0, 1.0))
        if len(color) < 3:
            raise ValueError("Each rule must provide a three-element color tuple.")
        rules.append(
            SelectionRule(
                record_name=str(rule["record_name"]),
                descriptor=str(rule["descriptor"]),
                res_low=int(rule["res_low"]),
                res_high=int(rule["res_high"]),
                color=(float(color[0]), float(color[1]), float(color[2])),
                radius=float(rule["radius"]),
            )
        )
    return rules


def _rules_signature(rules: list[SelectionRule]) -> str:
    payload: list[tuple[str, str, int, int, tuple[float, float, float], float]] = []
    for rule in rules:
        payload.append(
            (
                str(rule.record_name),
                str(rule.descriptor),
                int(rule.res_low),
                int(rule.res_high),
                (float(rule.color[0]), float(rule.color[1]), float(rule.color[2])),
                float(rule.radius),
            )
    )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def _upload_signature(pdb_path: str | Path) -> tuple[int, int, int, int]:
    stat_result = Path(pdb_path).stat()
    inode = int(getattr(stat_result, "st_ino", 0) or 0)
    return (
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
        inode,
    )


def _load_atoms_cached(
    *,
    pdb_id: str,
    pdb_path: str,
    pdb_signature: tuple[int, int, int, int],
    rules: list[SelectionRule],
):
    key = (str(pdb_id), pdb_signature, _rules_signature(rules))
    with _ATOM_CACHE_LOCK:
        cached = _ATOM_CACHE.get(key)
        if cached is not None:
            _ATOM_CACHE.move_to_end(key)
            return cached

    atoms = _load_pdb(pdb_path, rules)
    current_signature = _upload_signature(pdb_path)
    if current_signature != pdb_signature:
        raise RuntimeError("uploaded PDB changed while rendering")

    with _ATOM_CACHE_LOCK:
        _ATOM_CACHE[key] = atoms
        _ATOM_CACHE.move_to_end(key)
        while len(_ATOM_CACHE) > _ATOM_CACHE_MAX:
            _ATOM_CACHE.popitem(last=False)
    return atoms


def _to_image_bytes(rgb_image, opacity_image, output_format: str) -> bytes:
    buffer = BytesIO()
    pil = PILImage.fromarray(rgb_image, mode="RGB")
    if output_format == "PNG" and opacity_image is not None:
        alpha = PILImage.fromarray(opacity_image, mode="L")
        pil.putalpha(alpha)
    pil.save(buffer, format=output_format)
    buffer.seek(0)
    return buffer.read()


def _normalize_format(value: str) -> str:
    normalized = (value or "png").lower().strip()
    if normalized in {"png", "image/png"}:
        return "PNG"
    if normalized in {"ppm", "pnm", "image/x-portable-pixmap"}:
        return "PPM"
    raise ValueError(f"Unsupported output format: {value}")


@router.post("/render")
async def render_endpoint(payload: RenderRequest, request: Request) -> StreamingResponse:
    try:
        enforce_rate_limit("render", client_host=request.client.host if request.client is not None else None)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc

    try:
        pdb_path = get_upload_path(payload.pdb_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pdb_signature = _upload_signature(pdb_path)

    try:
        rules = _to_rules(payload.rules)
        output_format = _normalize_format(payload.output_format)
    except (TypeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    transform = payload.transform
    world = payload.world
    outlines = payload.outlines
    tx, ty, tz = transform.translate

    params = RenderParams(
        pdb_path=str(pdb_path),
        rules=rules,
        transform=Transform(
            scale=float(transform.scale),
            translate=(
                float(tx),
                float(ty),
                float(tz),
            ),
            rotations=[(str(axis), float(angle)) for axis, angle in transform.rotations],
            autocenter=str(transform.autocenter),
        ),
        world=WorldParams(
            background=tuple(world.background),
            fog_color=tuple(world.fog_color),
            fog_front=float(world.fog_front),
            fog_back=float(world.fog_back),
            shadows=bool(world.shadows),
            shadow_strength=float(world.shadow_strength),
            shadow_angle=float(world.shadow_angle),
            shadow_min_z=float(world.shadow_min_z),
            shadow_max_dark=float(world.shadow_max_dark),
            width=int(world.width),
            height=int(world.height),
        ),
        outlines=OutlineParams(
            enabled=bool(outlines.enabled),
            contour_low=float(outlines.contour_low),
            contour_high=float(outlines.contour_high),
            kernel=int(outlines.kernel),
            z_diff_min=float(outlines.z_diff_min),
            z_diff_max=float(outlines.z_diff_max),
            subunit_low=float(outlines.subunit_low),
            subunit_high=float(outlines.subunit_high),
            residue_low=float(outlines.residue_low),
            residue_high=float(outlines.residue_high),
            residue_diff=float(outlines.residue_diff),
        ),
    )

    try:
        atoms = await asyncio.to_thread(
            _load_atoms_cached,
            pdb_id=payload.pdb_id,
            pdb_path=params.pdb_path,
            pdb_signature=pdb_signature,
            rules=params.rules,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid render input: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"render failed: {exc}") from exc

    try:
        result = await asyncio.to_thread(render_from_atoms, atoms, params)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid render parameters: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"render failed: {exc}") from exc

    opacity = result.opacity if hasattr(result, "opacity") else None
    image = _to_image_bytes(result.rgb, opacity, output_format)
    media_type = "image/png" if output_format == "PNG" else "image/x-portable-pixmap"
    return StreamingResponse(BytesIO(image), media_type=media_type)
