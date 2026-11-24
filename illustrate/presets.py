"""Built-in rendering presets for the Python renderer and desktop GUI."""

from __future__ import annotations

import copy

from illustrate.core.types import (
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

__all__ = [
    "PRESET_NAMES",
    "default_rules",
    "preset_library",
    "render_params_from_preset",
]


def _chain_color_rules() -> list[SelectionRule]:
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


def _recolor_default(
    palette: dict[int, tuple[float, float, float]],
) -> list[SelectionRule]:
    """Recolor selected rules by zero-based position in the default list."""
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
    return _recolor_default(
        {
            3: (0.55, 0.75, 0.90),  # chain A carbon pattern
            4: (0.40, 0.60, 0.80),  # chain A sulfur pattern
            5: (0.45, 0.65, 0.85),  # remaining ATOM records in chain A
            6: (0.35, 0.80, 0.75),  # chain C carbon pattern
            7: (0.25, 0.70, 0.65),  # chain C sulfur pattern
            8: (0.30, 0.75, 0.70),  # remaining ATOM records in chain C
            9: (0.70, 0.82, 0.95),  # general C, pale blue
            10: (0.60, 0.72, 0.85),  # general S
            11: (0.65, 0.77, 0.90),  # general other
            12: (0.90, 0.70, 0.20),  # heme Fe, gold
            13: (0.30, 0.50, 0.70),  # heme C, dark blue
            14: (0.20, 0.40, 0.65),  # heme other, deep blue
        }
    )


def _high_contrast_rules() -> list[SelectionRule]:
    return _recolor_default(
        {
            3: (0.95, 0.35, 0.35),  # chain A
            4: (0.85, 0.25, 0.25),
            5: (0.90, 0.30, 0.30),
            6: (0.30, 0.65, 0.95),  # chain C
            7: (0.20, 0.55, 0.85),
            8: (0.25, 0.60, 0.90),
            9: (0.95, 0.75, 0.30),  # general, amber
            10: (0.85, 0.65, 0.20),
            11: (0.90, 0.70, 0.25),
            12: (0.20, 0.90, 0.20),  # heme Fe, green
            13: (0.70, 0.20, 0.70),  # heme C, purple
            14: (0.60, 0.15, 0.60),
        }
    )


def _monochrome_rules() -> list[SelectionRule]:
    return _recolor_default(
        {
            3: (0.75, 0.75, 0.75),
            4: (0.65, 0.65, 0.65),
            5: (0.70, 0.70, 0.70),
            6: (0.60, 0.60, 0.60),
            7: (0.50, 0.50, 0.50),
            8: (0.55, 0.55, 0.55),
            9: (0.80, 0.80, 0.80),
            10: (0.70, 0.70, 0.70),
            11: (0.75, 0.75, 0.75),
            12: (0.90, 0.90, 0.90),  # bright Fe
            13: (0.45, 0.45, 0.45),  # dark heme
            14: (0.35, 0.35, 0.35),
        }
    )


def _earth_tones_rules() -> list[SelectionRule]:
    return _recolor_default(
        {
            3: (0.85, 0.65, 0.45),  # chain A
            4: (0.75, 0.55, 0.35),
            5: (0.80, 0.60, 0.40),
            6: (0.65, 0.75, 0.50),  # chain C
            7: (0.55, 0.65, 0.40),
            8: (0.60, 0.70, 0.45),
            9: (0.90, 0.75, 0.55),  # general, warm tan
            10: (0.80, 0.65, 0.45),
            11: (0.85, 0.70, 0.50),
            12: (0.95, 0.75, 0.15),  # heme Fe, gold
            13: (0.75, 0.40, 0.30),  # heme, terracotta
            14: (0.65, 0.30, 0.25),
        }
    )


def default_rules() -> list[SelectionRule]:
    """Return the classic Illustrate atom styles in first-match order.

    Keep water and hydrogen exclusions ahead of broader drawing rules.
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


def _auto_sized_world(background: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> WorldParams:
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
    """Return independent settings in ``PRESET_NAMES`` order.

    The first three presets use copies of the supplied rules or defaults.
    The remaining presets use their own palettes.
    """
    rule_list = default_rules() if rules is None else copy.deepcopy(list(rules))
    chain_rule_list = _chain_color_rules()

    return [
        RenderParams(
            pdb_path="",
            rules=copy.deepcopy(rule_list),
            transform=Transform(
                scale=12.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_auto_sized_world(),
            outlines=OutlineParams(enabled=True),
        ),
        RenderParams(
            pdb_path="",
            rules=copy.deepcopy(rule_list),
            transform=Transform(
                scale=14.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_auto_sized_world((0.0, 0.0, 0.0)),
            outlines=OutlineParams(enabled=True),
        ),
        RenderParams(
            pdb_path="",
            rules=copy.deepcopy(rule_list),
            transform=Transform(
                scale=10.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_auto_sized_world(),
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
            rules=copy.deepcopy(chain_rule_list),
            transform=Transform(
                scale=12.5,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_auto_sized_world((0.05, 0.06, 0.08)),
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
        # Cool Blues: steel blue / teal on pale blue background
        RenderParams(
            pdb_path="",
            rules=copy.deepcopy(_cool_blues_rules()),
            transform=Transform(
                scale=12.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 15.0), ("y", -10.0)],
                autocenter="auto",
            ),
            world=WorldParams(
                background=(0.95, 0.97, 1.0),
                fog_color=(0.95, 0.97, 1.0),
                fog_front=1.0,
                fog_back=1.0,
                shadows=True,
                shadow_strength=0.002,
                shadow_angle=2.0,
                shadow_min_z=1.0,
                shadow_max_dark=0.15,
                width=-30,
                height=-30,
            ),
            outlines=OutlineParams(enabled=True),
        ),
        # High Contrast: vivid red, blue and amber
        RenderParams(
            pdb_path="",
            rules=copy.deepcopy(_high_contrast_rules()),
            transform=Transform(
                scale=12.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 5.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=_auto_sized_world(),
            outlines=OutlineParams(enabled=True),
        ),
        # Pen & Ink: grayscale with heavy outlines
        RenderParams(
            pdb_path="",
            rules=copy.deepcopy(_monochrome_rules()),
            transform=Transform(
                scale=12.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=WorldParams(
                background=(1.0, 1.0, 1.0),
                fog_color=(1.0, 1.0, 1.0),
                fog_front=1.0,
                fog_back=1.0,
                shadows=True,
                shadow_strength=0.003,
                shadow_angle=2.5,
                shadow_min_z=1.0,
                shadow_max_dark=0.3,
                width=-30,
                height=-30,
            ),
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
        # Earth Tones: sandstone / sage on warm cream
        RenderParams(
            pdb_path="",
            rules=copy.deepcopy(_earth_tones_rules()),
            transform=Transform(
                scale=13.0,
                translate=(0.0, 0.0, 0.0),
                rotations=[("z", 45.0), ("x", 20.0), ("y", 0.0)],
                autocenter="auto",
            ),
            world=WorldParams(
                background=(0.98, 0.96, 0.93),
                fog_color=(0.98, 0.96, 0.93),
                fog_front=1.0,
                fog_back=1.0,
                shadows=True,
                shadow_strength=0.0025,
                shadow_angle=2.0,
                shadow_min_z=1.0,
                shadow_max_dark=0.2,
                width=-30,
                height=-30,
            ),
            outlines=OutlineParams(enabled=True),
        ),
    ]


def render_params_from_preset(
    name: str,
    pdb_path: str,
    rules: list[SelectionRule] | None = None,
) -> RenderParams:
    """Bind a named preset to a PDB path, requiring an exact ``PRESET_NAMES`` match."""

    presets = preset_library(rules)
    by_name = {label: preset for label, preset in zip(PRESET_NAMES, presets)}
    try:
        base = by_name[name]
    except KeyError as exc:
        available = ", ".join(PRESET_NAMES)
        raise ValueError(
            f"unknown preset {name!r}; expected one of: {available}"
        ) from exc
    params = RenderParams(
        pdb_path=pdb_path,
        rules=copy.deepcopy(base.rules),
        transform=copy.deepcopy(base.transform),
        world=copy.deepcopy(base.world),
        outlines=copy.deepcopy(base.outlines),
    )
    return params
