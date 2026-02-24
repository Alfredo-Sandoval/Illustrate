"""Built-in rendering presets."""

from __future__ import annotations

from illustrate import OutlineParams, RenderParams, Transform, WorldParams
from illustrate.types import SelectionRule


def default_preset(pdb_path: str, rules: list[SelectionRule]) -> RenderParams:
    """Return a safe baseline preset.

    This intentionally matches the command-file example defaults from the project
    docs and can be used by GUI/web clients as a starter.
    """

    return RenderParams(
        pdb_path=pdb_path,
        rules=rules,
        transform=Transform(
            scale=12.0,
            translate=(0.0, 0.0, 0.0),
            rotations=[("z", 90.0)],
            autocenter="auto",
        ),
        world=WorldParams(
            background=(1.0, 1.0, 1.0),
            fog_color=(1.0, 1.0, 1.0),
            fog_front=1.0,
            fog_back=1.0,
            shadows=True,
            shadow_strength=0.0023,
            shadow_angle=2.0,
            shadow_min_z=1.0,
            shadow_max_dark=0.2,
            width=-30,
            height=-30,
        ),
        outlines=OutlineParams(enabled=True),
    )
