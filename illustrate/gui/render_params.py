"""Helpers for translating panel state into RenderParams."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping

from illustrate import (
    OutlineParams,
    RenderParams,
    SelectionRule,
    Transform,
    WorldParams,
)
from illustrate.rendering.api import rotation_matrix


AUTO_FRAME_PAD = 30


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    hex_value = value.lstrip("#")
    if len(hex_value) != 6:
        return (1.0, 1.0, 1.0)
    return (
        int(hex_value[0:2], 16) / 255.0,
        int(hex_value[2:4], 16) / 255.0,
        int(hex_value[4:6], 16) / 255.0,
    )


def rules_signature(rules: list[SelectionRule]) -> str:
    """Hash the ordered rule fields that can change atom classification.

    A radius contributes only whether it is zero, which hides matched atoms.
    """
    payload: list[tuple[str, str, int, int, bool]] = []
    for rule in rules:
        payload.append(
            (
                rule.record_name,
                rule.descriptor,
                rule.res_low,
                rule.res_high,
                float(rule.radius) == 0.0,
            )
        )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def transform_panel_values(transform: Transform) -> dict[str, float]:
    """Convert rotations to GUI angles in degrees, applied X, then Y, then Z.

    The panels require automatic centering. Other modes raise ``ValueError``.
    """
    if transform.autocenter.strip().lower() not in {"auto", "aut"}:
        raise ValueError("GUI settings require autocenter='auto'. Use the Python API for other centering modes.")
    matrix = rotation_matrix(transform.rotations)
    cos_y = math.hypot(float(matrix[0, 0]), float(matrix[0, 1]))
    y_angle = math.atan2(-float(matrix[0, 2]), cos_y)
    if cos_y > 1e-6:
        x_angle = math.atan2(float(matrix[1, 2]), float(matrix[2, 2]))
        z_angle = math.atan2(float(matrix[0, 1]), float(matrix[0, 0]))
    else:
        # At gimbal lock, X and Z share a degree of freedom. Choose X=0.
        x_angle = 0.0
        z_angle = math.atan2(-float(matrix[1, 0]), float(matrix[1, 1]))
    return {
        "scale": transform.scale,
        "xtran": transform.translate[0],
        "ytran": transform.translate[1],
        "ztran": transform.translate[2],
        "xrot": math.degrees(x_angle),
        "yrot": math.degrees(y_angle),
        "zrot": math.degrees(z_angle),
    }


def render_params_from_panels(
    pdb_path: str,
    rules: list[SelectionRule],
    transform_state: Mapping[str, float],
    world_state: Mapping[str, float | bool | str],
    outline_state: Mapping[str, float | int | bool],
    *,
    render_width: int,
    render_height: int,
) -> RenderParams:
    return RenderParams(
        pdb_path=pdb_path,
        rules=rules,
        transform=Transform(
            scale=float(transform_state.get("scale", 12.0)),
            translate=(
                float(transform_state.get("xtran", 0.0)),
                float(transform_state.get("ytran", 0.0)),
                float(transform_state.get("ztran", 0.0)),
            ),
            rotations=[
                ("z", float(transform_state.get("zrot", 90.0))),
                ("y", float(transform_state.get("yrot", 0.0))),
                ("x", float(transform_state.get("xrot", 0.0))),
            ],
            autocenter="auto",
        ),
        world=WorldParams(
            background=_hex_to_rgb(str(world_state.get("background", "#ffffff"))),
            fog_color=_hex_to_rgb(str(world_state.get("fog", "#ffffff"))),
            fog_front=float(world_state.get("fog_front", 1.0)),
            fog_back=float(world_state.get("fog_back", 1.0)),
            shadows=bool(world_state.get("shadows", False)),
            shadow_strength=float(world_state.get("shadow_strength", 0.0023)),
            shadow_angle=float(world_state.get("shadow_angle", 2.0)),
            shadow_min_z=float(world_state.get("shadow_min_z", 1.0)),
            shadow_max_dark=float(world_state.get("shadow_max_dark", 0.2)),
            width=int(render_width),
            height=int(render_height),
        ),
        outlines=OutlineParams(
            enabled=bool(outline_state.get("enabled", True)),
            contour_low=float(outline_state.get("contour_low", 3.0)),
            contour_high=float(outline_state.get("contour_high", 10.0)),
            kernel=int(outline_state.get("kernel", 4)),
            z_diff_min=float(outline_state.get("z_diff_min", 0.0)),
            z_diff_max=float(outline_state.get("z_diff_max", 5.0)),
            subunit_low=float(outline_state.get("subunit_low", 3.0)),
            subunit_high=float(outline_state.get("subunit_high", 10.0)),
            residue_low=float(outline_state.get("residue_low", 3.0)),
            residue_high=float(outline_state.get("residue_high", 8.0)),
            residue_diff=float(outline_state.get("residue_diff", 6000.0)),
        ),
    )
