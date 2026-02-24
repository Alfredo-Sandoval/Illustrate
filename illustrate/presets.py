"""Built-in rendering presets for desktop and web frontends."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from illustrate.types import (
    OutlineParams,
    RenderParams,
    SelectionRule,
    Transform,
    WorldParams,
)

PRESET_NAMES = [
    "Default",
    "Black Background",
    "Wireframe",
    "Dark Chain Colors",
    "Cool Blues",
    "High Contrast",
    "Pen & Ink",
    "Earth Tones",
]


def _copy_rule(rule: SelectionRule) -> SelectionRule:
    return SelectionRule(
        record_name=str(rule.record_name),
        descriptor=str(rule.descriptor),
        res_low=int(rule.res_low),
        res_high=int(rule.res_high),
        color=(float(rule.color[0]), float(rule.color[1]), float(rule.color[2])),
        radius=float(rule.radius),
    )


def _copy_rules(rules: list[SelectionRule]) -> list[SelectionRule]:
    return [_copy_rule(rule) for rule in rules]


def _copy_transform(transform: Transform) -> Transform:
    return Transform(
        scale=float(transform.scale),
        translate=(
            float(transform.translate[0]),
            float(transform.translate[1]),
            float(transform.translate[2]),
        ),
        rotations=[(str(axis), float(angle)) for axis, angle in transform.rotations],
        autocenter=str(transform.autocenter),
    )


def _copy_world(world: WorldParams) -> WorldParams:
    return WorldParams(
        background=(float(world.background[0]), float(world.background[1]), float(world.background[2])),
        fog_color=(float(world.fog_color[0]), float(world.fog_color[1]), float(world.fog_color[2])),
        fog_front=float(world.fog_front),
        fog_back=float(world.fog_back),
        shadows=bool(world.shadows),
        shadow_strength=float(world.shadow_strength),
        shadow_angle=float(world.shadow_angle),
        shadow_min_z=float(world.shadow_min_z),
        shadow_max_dark=float(world.shadow_max_dark),
        width=int(world.width),
        height=int(world.height),
    )


def _copy_outlines(outlines: OutlineParams) -> OutlineParams:
    return OutlineParams(
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
    )


def _chain_color_rules() -> list[SelectionRule]:
    """Chain-aware atom palette tuned for dark backgrounds."""

    chain_palette: list[tuple[float, float, float]] = [
        (0.95, 0.45, 0.55),  # coral
        (0.45, 0.72, 1.00),  # sky
        (0.55, 0.92, 0.62),  # mint
        (1.00, 0.82, 0.40),  # amber
        (0.86, 0.58, 1.00),  # violet
        (0.45, 0.95, 0.95),  # cyan
        (1.00, 0.60, 0.40),  # orange
        (0.76, 0.90, 0.48),  # lime
    ]
    chain_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    rules: list[SelectionRule] = [
        SelectionRule(
            record_name="HETATM",
            descriptor="-----HOH--",
            res_low=0,
            res_high=9999,
            color=(0.5, 0.5, 0.5),
            radius=0.0,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="-H--------",
            res_low=0,
            res_high=9999,
            color=(0.5, 0.5, 0.5),
            radius=0.0,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="H---------",
            res_low=0,
            res_high=9999,
            color=(0.5, 0.5, 0.5),
            radius=0.0,
        ),
    ]

    for idx, chain in enumerate(chain_chars):
        rules.append(
            SelectionRule(
                record_name="ATOM  ",
                descriptor=f"---------{chain}",
                res_low=0,
                res_high=9999,
                color=chain_palette[idx % len(chain_palette)],
                radius=1.5,
            )
        )

    rules.extend(
        [
            SelectionRule(
                record_name="ATOM  ",
                descriptor="----------",
                res_low=0,
                res_high=9999,
                color=(0.82, 0.76, 0.70),
                radius=1.45,
            ),
            SelectionRule(
                record_name="HETATM",
                descriptor="----------",
                res_low=0,
                res_high=9999,
                color=(1.00, 0.75, 0.25),
                radius=1.55,
            ),
        ]
    )
    return rules


def _recolor_default(palette: dict[int, tuple[float, float, float]]) -> list[SelectionRule]:
    """Return default_rules() with specific indices recolored."""
    rules = default_rules()
    for idx, color in palette.items():
        r = rules[idx]
        rules[idx] = SelectionRule(
            record_name=r.record_name,
            descriptor=r.descriptor,
            res_low=r.res_low,
            res_high=r.res_high,
            color=color,
            radius=r.radius,
        )
    return rules


def _cool_blues_rules() -> list[SelectionRule]:
    """Steel blue / teal chain-differentiated palette."""
    return _recolor_default({
        3:  (0.55, 0.75, 0.90),   # helix C (A) — light steel blue
        4:  (0.40, 0.60, 0.80),   # helix S (A)
        5:  (0.45, 0.65, 0.85),   # helix other (A)
        6:  (0.35, 0.80, 0.75),   # sheet C (C) — teal
        7:  (0.25, 0.70, 0.65),   # sheet S (C)
        8:  (0.30, 0.75, 0.70),   # sheet other (C)
        9:  (0.70, 0.82, 0.95),   # general C — pale blue
        10: (0.60, 0.72, 0.85),   # general S
        11: (0.65, 0.77, 0.90),   # general other
        12: (0.90, 0.70, 0.20),   # heme Fe — gold
        13: (0.30, 0.50, 0.70),   # heme C — dark blue
        14: (0.20, 0.40, 0.65),   # heme other — deep blue
    })


def _high_contrast_rules() -> list[SelectionRule]:
    """Vivid saturated red/blue/amber palette for presentations."""
    return _recolor_default({
        3:  (0.95, 0.35, 0.35),   # helix C (A) — red
        4:  (0.85, 0.25, 0.25),
        5:  (0.90, 0.30, 0.30),
        6:  (0.30, 0.65, 0.95),   # sheet C (C) — blue
        7:  (0.20, 0.55, 0.85),
        8:  (0.25, 0.60, 0.90),
        9:  (0.95, 0.75, 0.30),   # general — amber
        10: (0.85, 0.65, 0.20),
        11: (0.90, 0.70, 0.25),
        12: (0.20, 0.90, 0.20),   # heme Fe — green
        13: (0.70, 0.20, 0.70),   # heme C — purple
        14: (0.60, 0.15, 0.60),
    })


def _monochrome_rules() -> list[SelectionRule]:
    """Grayscale with subtle value variation for pen-and-ink style."""
    return _recolor_default({
        3:  (0.75, 0.75, 0.75),
        4:  (0.65, 0.65, 0.65),
        5:  (0.70, 0.70, 0.70),
        6:  (0.60, 0.60, 0.60),
        7:  (0.50, 0.50, 0.50),
        8:  (0.55, 0.55, 0.55),
        9:  (0.80, 0.80, 0.80),
        10: (0.70, 0.70, 0.70),
        11: (0.75, 0.75, 0.75),
        12: (0.90, 0.90, 0.90),   # bright Fe
        13: (0.45, 0.45, 0.45),   # dark heme
        14: (0.35, 0.35, 0.35),
    })


def _earth_tones_rules() -> list[SelectionRule]:
    """Warm sandstone / sage green textbook palette."""
    return _recolor_default({
        3:  (0.85, 0.65, 0.45),   # helix — sandstone
        4:  (0.75, 0.55, 0.35),
        5:  (0.80, 0.60, 0.40),
        6:  (0.65, 0.75, 0.50),   # sheet — sage green
        7:  (0.55, 0.65, 0.40),
        8:  (0.60, 0.70, 0.45),
        9:  (0.90, 0.75, 0.55),   # general — warm tan
        10: (0.80, 0.65, 0.45),
        11: (0.85, 0.70, 0.50),
        12: (0.95, 0.75, 0.15),   # heme Fe — gold
        13: (0.75, 0.40, 0.30),   # heme — terracotta
        14: (0.65, 0.30, 0.25),
    })


def default_rules() -> list[SelectionRule]:
    """Return the original Illustrate Fortran-style default rule stack.

    Order matters: rules are matched top-to-bottom and first match wins.
    """

    return [
        SelectionRule(
            record_name="HETATM",
            descriptor="-----HOH--",
            res_low=0,
            res_high=9999,
            color=(0.5, 0.5, 0.5),
            radius=0.0,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="-H--------",
            res_low=0,
            res_high=9999,
            color=(0.5, 0.5, 0.5),
            radius=0.0,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="H---------",
            res_low=0,
            res_high=9999,
            color=(0.5, 0.5, 0.5),
            radius=0.0,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="-C-------A",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.6, 0.6),
            radius=1.6,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="-S-------A",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.5, 0.5),
            radius=1.8,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="---------A",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.5, 0.5),
            radius=1.5,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="-C-------C",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.6, 0.6),
            radius=1.6,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="-S-------C",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.5, 0.5),
            radius=1.8,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="---------C",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.5, 0.5),
            radius=1.5,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="-C--------",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.8, 0.6),
            radius=1.6,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="-S--------",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.7, 0.5),
            radius=1.8,
        ),
        SelectionRule(
            record_name="ATOM  ",
            descriptor="----------",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.7, 0.5),
            radius=1.5,
        ),
        SelectionRule(
            record_name="HETATM",
            descriptor="FE---HEM--",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.8, 0.0),
            radius=1.8,
        ),
        SelectionRule(
            record_name="HETATM",
            descriptor="-C---HEM--",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.3, 0.3),
            radius=1.6,
        ),
        SelectionRule(
            record_name="HETATM",
            descriptor="-----HEM--",
            res_low=0,
            res_high=9999,
            color=(1.0, 0.1, 0.1),
            radius=1.5,
        ),
    ]


def _world(background: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> WorldParams:
    return WorldParams(
        background=background,
        fog_color=background,
        fog_front=1.0,
        fog_back=1.0,
        shadows=True,
        shadow_strength=0.0023,
        shadow_angle=2.0,
        shadow_min_z=1.0,
        shadow_max_dark=0.2,
        width=-30,
        height=-30,
    )


def preset_library(rules: list[SelectionRule] | None = None) -> list[RenderParams]:
    """Return a small collection of reusable preset parameter sets."""

    rule_list = default_rules() if rules is None else _copy_rules(list(rules))
    chain_rule_list = _chain_color_rules()

    return [
        RenderParams(
            pdb_path="",
            rules=_copy_rules(rule_list),
            transform=Transform(
                scale=12.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_world(),
            outlines=OutlineParams(enabled=True),
        ),
        RenderParams(
            pdb_path="",
            rules=_copy_rules(rule_list),
            transform=Transform(
                scale=14.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_world((0.0, 0.0, 0.0)),
            outlines=OutlineParams(enabled=True),
        ),
        RenderParams(
            pdb_path="",
            rules=_copy_rules(rule_list),
            transform=Transform(
                scale=10.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_world(),
            outlines=OutlineParams(
                enabled=True,
                contour_low=2.0,
                contour_high=16.0,
                kernel=4,
                z_diff_min=0.0,
                z_diff_max=12.0,
                subunit_low=2.5,
                subunit_high=8.0,
                residue_low=2.5,
                residue_high=7.0,
                residue_diff=4000.0,
            ),
        ),
        RenderParams(
            pdb_path="",
            rules=_copy_rules(chain_rule_list),
            transform=Transform(
                scale=12.5,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_world((0.05, 0.06, 0.08)),
            outlines=OutlineParams(
                enabled=True,
                contour_low=2.2,
                contour_high=11.5,
                kernel=4,
                z_diff_min=0.0,
                z_diff_max=8.0,
                subunit_low=2.5,
                subunit_high=8.0,
                residue_low=2.5,
                residue_high=7.0,
                residue_diff=4000.0,
            ),
        ),
        # Cool Blues — steel blue / teal on pale blue background
        RenderParams(
            pdb_path="",
            rules=_copy_rules(_cool_blues_rules()),
            transform=Transform(
                scale=12.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 15.0), ("y", -10.0)],
                autocenter="auto",
            ),
            world=WorldParams(
                background=(0.95, 0.97, 1.0),
                fog_color=(0.95, 0.97, 1.0),
                fog_front=1.0, fog_back=1.0,
                shadows=True, shadow_strength=0.002,
                shadow_angle=2.0, shadow_min_z=1.0, shadow_max_dark=0.15,
                width=-30, height=-30,
            ),
            outlines=OutlineParams(enabled=True),
        ),
        # High Contrast — vivid red/blue/amber for presentations
        RenderParams(
            pdb_path="",
            rules=_copy_rules(_high_contrast_rules()),
            transform=Transform(
                scale=12.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 5.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_world(),
            outlines=OutlineParams(enabled=True),
        ),
        # Pen & Ink — grayscale with heavy outlines
        RenderParams(
            pdb_path="",
            rules=_copy_rules(_monochrome_rules()),
            transform=Transform(
                scale=12.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=WorldParams(
                background=(1.0, 1.0, 1.0),
                fog_color=(1.0, 1.0, 1.0),
                fog_front=1.0, fog_back=1.0,
                shadows=True, shadow_strength=0.003,
                shadow_angle=2.5, shadow_min_z=1.0, shadow_max_dark=0.3,
                width=-30, height=-30,
            ),
            outlines=OutlineParams(
                enabled=True,
                contour_low=2.0, contour_high=16.0,
                kernel=4,
                z_diff_min=0.0, z_diff_max=12.0,
                subunit_low=2.5, subunit_high=8.0,
                residue_low=2.5, residue_high=7.0,
                residue_diff=4000.0,
            ),
        ),
        # Earth Tones — sandstone / sage on warm cream
        RenderParams(
            pdb_path="",
            rules=_copy_rules(_earth_tones_rules()),
            transform=Transform(
                scale=13.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 45.0), ("x", 20.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=WorldParams(
                background=(0.98, 0.96, 0.93),
                fog_color=(0.98, 0.96, 0.93),
                fog_front=1.0, fog_back=1.0,
                shadows=True, shadow_strength=0.0025,
                shadow_angle=2.0, shadow_min_z=1.0, shadow_max_dark=0.2,
                width=-30, height=-30,
            ),
            outlines=OutlineParams(enabled=True),
        ),
    ]


def preset_payloads(pdb_path: str, rules: list[SelectionRule] | None = None) -> list[dict[str, object]]:
    """Return API-friendly preset payloads, injecting the provided pdb path."""

    payloads: list[dict[str, object]] = []
    for name, params in zip(PRESET_NAMES, preset_library(rules)):
        payload = params_to_payload(params, pdb_path)
        payload["name"] = name
        payloads.append(payload)
    return payloads


def params_to_payload(params: RenderParams, pdb_path: str) -> dict[str, object]:
    """Convert RenderParams into a JSON-safe payload."""

    return {
        "pdb_path": pdb_path,
        "rules": [_selection_rule_payload(r) for r in params.rules],
        "transform": asdict(params.transform),
        "world": asdict(params.world),
        "outlines": asdict(params.outlines),
    }


def _selection_rule_payload(rule: SelectionRule) -> dict[str, Any]:
    return asdict(rule)


def make_render_params(payload: dict[str, Any], pdb_path: str) -> RenderParams:
    """Create RenderParams from payload while keeping missing fields defaulted."""

    rules_payload = payload.get("rules", [])
    rules = [
        SelectionRule(
            rule["record_name"],
            rule["descriptor"],
            int(rule["res_low"]),
            int(rule["res_high"]),
            tuple(rule["color"]),
            float(rule["radius"]),
        )
        for rule in rules_payload
    ]
    transform_payload = payload.get("transform", {})
    world_payload = payload.get("world", {})
    outlines_payload = payload.get("outlines", {})

    return RenderParams(
        pdb_path=pdb_path,
        rules=rules,
        transform=Transform(
            scale=float(transform_payload.get("scale", 12.0)),
            translate=tuple(transform_payload.get("translate", (0.0, 0.0, 0.0))),
            rotations=[(str(axis), float(angle)) for axis, angle in transform_payload.get("rotations", [])],
            autocenter=str(transform_payload.get("autocenter", "auto")),
        ),
        world=WorldParams(
            background=tuple(world_payload.get("background", (1.0, 1.0, 1.0))),
            fog_color=tuple(world_payload.get("fog_color", (1.0, 1.0, 1.0))),
            fog_front=float(world_payload.get("fog_front", 1.0)),
            fog_back=float(world_payload.get("fog_back", 1.0)),
            shadows=bool(world_payload.get("shadows", False)),
            shadow_strength=float(world_payload.get("shadow_strength", 0.0023)),
            shadow_angle=float(world_payload.get("shadow_angle", 2.0)),
            shadow_min_z=float(world_payload.get("shadow_min_z", 1.0)),
            shadow_max_dark=float(world_payload.get("shadow_max_dark", 0.2)),
            width=int(world_payload.get("width", -30)),
            height=int(world_payload.get("height", -30)),
        ),
        outlines=OutlineParams(
            enabled=bool(outlines_payload.get("enabled", True)),
            contour_low=float(outlines_payload.get("contour_low", 3.0)),
            contour_high=float(outlines_payload.get("contour_high", 10.0)),
            kernel=int(outlines_payload.get("kernel", 4)),
            z_diff_min=float(outlines_payload.get("z_diff_min", 0.0)),
            z_diff_max=float(outlines_payload.get("z_diff_max", 5.0)),
            subunit_low=float(outlines_payload.get("subunit_low", 3.0)),
            subunit_high=float(outlines_payload.get("subunit_high", 10.0)),
            residue_low=float(outlines_payload.get("residue_low", 3.0)),
            residue_high=float(outlines_payload.get("residue_high", 8.0)),
            residue_diff=float(outlines_payload.get("residue_diff", 6000.0)),
        ),
    )


def render_params_from_preset(
    name: str,
    pdb_path: str,
    rules: list[SelectionRule] | None = None,
) -> RenderParams:
    """Load a preset by name and bind it to a concrete PDB path."""

    presets = preset_library(rules)
    by_name = {label: preset for label, preset in zip(PRESET_NAMES, presets)}
    base = by_name.get(name, presets[0])
    params = RenderParams(
        pdb_path=pdb_path,
        rules=_copy_rules(base.rules),
        transform=_copy_transform(base.transform),
        world=_copy_world(base.world),
        outlines=_copy_outlines(base.outlines),
    )
    return params
