"""Render hemoglobin (2HHB) with the gallery's own rendering settings."""

from pathlib import Path
from typing import Callable, TypedDict
from PIL import Image

from illustrate import RenderParams, RenderResult, load_pdb, render_from_atoms
from illustrate.core.types import OutlineParams, SelectionRule, Transform, WorldParams
from illustrate.presets import default_rules

REPO_ROOT = Path(__file__).resolve().parents[1]
PDB = REPO_ROOT / "examples" / "2hhb.pdb"
OUT = REPO_ROOT / "renders"
OUT.mkdir(exist_ok=True)


def save(result: RenderResult, name: str) -> None:
    Image.fromarray(result.rgb).save(OUT / f"{name}.png")
    print(f"  Saved {name}.png ({result.width}x{result.height})")


# Color palettes

def _classic_goodsell() -> list[SelectionRule]:
    return default_rules()


def _cool_blues() -> list[SelectionRule]:
    rules = default_rules()
    palette = {
        # Chain A carbon pattern
        3:  (0.55, 0.75, 0.90),  # light steel blue
        # Chain A sulfur pattern
        4:  (0.40, 0.60, 0.80),
        # Remaining ATOM records in chain A
        5:  (0.45, 0.65, 0.85),
        # Chain C rules
        6:  (0.35, 0.80, 0.75),  # teal
        7:  (0.25, 0.70, 0.65),
        8:  (0.30, 0.75, 0.70),
        # Catch-all ATOM rules for other chains
        9:  (0.70, 0.82, 0.95),  # pale blue
        10: (0.60, 0.72, 0.85),
        11: (0.65, 0.77, 0.90),
        # Heme
        12: (0.90, 0.70, 0.20),  # gold iron
        13: (0.30, 0.50, 0.70),  # dark blue carbons
        14: (0.20, 0.40, 0.65),  # deep blue other
    }
    for idx, color in palette.items():
        rules[idx] = SelectionRule(
            record_name=rules[idx].record_name,
            descriptor=rules[idx].descriptor,
            res_low=rules[idx].res_low,
            res_high=rules[idx].res_high,
            color=color,
            radius=rules[idx].radius,
        )
    return rules


def _earth_tones() -> list[SelectionRule]:
    rules = default_rules()
    palette = {
        3:  (0.85, 0.65, 0.45),  # sandstone
        4:  (0.75, 0.55, 0.35),
        5:  (0.80, 0.60, 0.40),
        6:  (0.65, 0.75, 0.50),  # sage green
        7:  (0.55, 0.65, 0.40),
        8:  (0.60, 0.70, 0.45),
        9:  (0.90, 0.75, 0.55),  # warm tan
        10: (0.80, 0.65, 0.45),
        11: (0.85, 0.70, 0.50),
        12: (0.95, 0.75, 0.15),  # gold Fe
        13: (0.75, 0.40, 0.30),  # terracotta heme
        14: (0.65, 0.30, 0.25),
    }
    for idx, color in palette.items():
        rules[idx] = SelectionRule(
            record_name=rules[idx].record_name,
            descriptor=rules[idx].descriptor,
            res_low=rules[idx].res_low,
            res_high=rules[idx].res_high,
            color=color,
            radius=rules[idx].radius,
        )
    return rules


def _high_contrast() -> list[SelectionRule]:
    rules = default_rules()
    palette = {
        3:  (0.95, 0.35, 0.35),  # red
        4:  (0.85, 0.25, 0.25),
        5:  (0.90, 0.30, 0.30),
        6:  (0.30, 0.65, 0.95),  # blue
        7:  (0.20, 0.55, 0.85),
        8:  (0.25, 0.60, 0.90),
        9:  (0.95, 0.75, 0.30),  # amber
        10: (0.85, 0.65, 0.20),
        11: (0.90, 0.70, 0.25),
        12: (0.20, 0.90, 0.20),  # green Fe
        13: (0.70, 0.20, 0.70),  # purple heme
        14: (0.60, 0.15, 0.60),
    }
    for idx, color in palette.items():
        rules[idx] = SelectionRule(
            record_name=rules[idx].record_name,
            descriptor=rules[idx].descriptor,
            res_low=rules[idx].res_low,
            res_high=rules[idx].res_high,
            color=color,
            radius=rules[idx].radius,
        )
    return rules


def _monochrome() -> list[SelectionRule]:
    rules = default_rules()
    palette = {
        3:  (0.75, 0.75, 0.75),
        4:  (0.65, 0.65, 0.65),
        5:  (0.70, 0.70, 0.70),
        6:  (0.60, 0.60, 0.60),
        7:  (0.50, 0.50, 0.50),
        8:  (0.55, 0.55, 0.55),
        9:  (0.80, 0.80, 0.80),
        10: (0.70, 0.70, 0.70),
        11: (0.75, 0.75, 0.75),
        12: (0.90, 0.90, 0.90),  # bright Fe
        13: (0.45, 0.45, 0.45),  # dark heme
        14: (0.35, 0.35, 0.35),
    }
    for idx, color in palette.items():
        rules[idx] = SelectionRule(
            record_name=rules[idx].record_name,
            descriptor=rules[idx].descriptor,
            res_low=rules[idx].res_low,
            res_high=rules[idx].res_high,
            color=color,
            radius=rules[idx].radius,
        )
    return rules


class RenderConfig(TypedDict):
    rules: Callable[[], list[SelectionRule]]
    transform: Transform
    world: WorldParams
    outlines: OutlineParams


# Render configs

CONFIGS: dict[str, RenderConfig] = {
    # 1. Classic Goodsell
    "01_classic_goodsell": {
        "rules": _classic_goodsell,
        "transform": Transform(
            scale=12.0,
            rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(1.0, 1.0, 1.0),
            fog_color=(1.0, 1.0, 1.0),
            fog_front=1.0, fog_back=1.0,
            shadows=True, shadow_strength=0.0023,
            shadow_angle=2.0, shadow_min_z=1.0, shadow_max_dark=0.2,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(enabled=True),
    },

    # 2. Black background
    "02_dark_mode": {
        "rules": _classic_goodsell,
        "transform": Transform(
            scale=14.0,
            rotations=[("z", 90.0), ("x", 10.0), ("y", 0.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(0.0, 0.0, 0.0),
            fog_color=(0.0, 0.0, 0.0),
            fog_front=1.0, fog_back=1.0,
            shadows=True, shadow_strength=0.003,
            shadow_angle=2.5, shadow_min_z=1.0, shadow_max_dark=0.4,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(enabled=True),
    },

    # 3. Cool blue palette
    "03_cool_blues": {
        "rules": _cool_blues,
        "transform": Transform(
            scale=12.0,
            rotations=[("z", 90.0), ("x", 15.0), ("y", -10.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(0.95, 0.97, 1.0),  # very faint blue
            fog_color=(0.95, 0.97, 1.0),
            fog_front=1.0, fog_back=1.0,
            shadows=True, shadow_strength=0.002,
            shadow_angle=2.0, shadow_min_z=1.0, shadow_max_dark=0.15,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(enabled=True),
    },

    # 4. Earth tones
    "04_earth_tones": {
        "rules": _earth_tones,
        "transform": Transform(
            scale=13.0,
            rotations=[("z", 45.0), ("x", 20.0), ("y", 0.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(0.98, 0.96, 0.93),  # warm cream
            fog_color=(0.98, 0.96, 0.93),
            fog_front=1.0, fog_back=1.0,
            shadows=True, shadow_strength=0.0025,
            shadow_angle=2.0, shadow_min_z=1.0, shadow_max_dark=0.2,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(enabled=True),
    },

    # 5. High contrast
    "05_high_contrast": {
        "rules": _high_contrast,
        "transform": Transform(
            scale=12.0,
            rotations=[("z", 90.0), ("x", 5.0), ("y", 0.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(1.0, 1.0, 1.0),
            fog_color=(1.0, 1.0, 1.0),
            fog_front=1.0, fog_back=1.0,
            shadows=True, shadow_strength=0.0023,
            shadow_angle=2.0, shadow_min_z=1.0, shadow_max_dark=0.2,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(enabled=True),
    },

    # 6. Pen and ink: grayscale with heavy outlines
    "06_pen_and_ink": {
        "rules": _monochrome,
        "transform": Transform(
            scale=12.0,
            rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(1.0, 1.0, 1.0),
            fog_color=(1.0, 1.0, 1.0),
            fog_front=1.0, fog_back=1.0,
            shadows=True, shadow_strength=0.003,
            shadow_angle=2.5, shadow_min_z=1.0, shadow_max_dark=0.3,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(
            enabled=True,
            contour_low=2.0, contour_high=16.0,
            kernel=4,
            z_diff_min=0.0, z_diff_max=12.0,
            subunit_low=2.5, subunit_high=8.0,
            residue_low=2.5, residue_high=7.0,
            residue_diff=4000.0,
        ),
    },

    # 7. Close-up at a higher scale
    "07_closeup_dramatic": {
        "rules": _classic_goodsell,
        "transform": Transform(
            scale=20.0,
            rotations=[("z", 60.0), ("x", 30.0), ("y", -15.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(0.12, 0.12, 0.14),  # dark slate
            fog_color=(0.12, 0.12, 0.14),
            fog_front=1.0, fog_back=0.7,
            shadows=True, shadow_strength=0.004,
            shadow_angle=3.0, shadow_min_z=1.0, shadow_max_dark=0.5,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(enabled=True),
    },

    # 8. No outlines
    "08_spacefill_no_outlines": {
        "rules": _classic_goodsell,
        "transform": Transform(
            scale=12.0,
            rotations=[("z", 90.0), ("x", 0.0), ("y", 0.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(1.0, 1.0, 1.0),
            fog_color=(1.0, 1.0, 1.0),
            fog_front=1.0, fog_back=1.0,
            shadows=True, shadow_strength=0.0023,
            shadow_angle=2.0, shadow_min_z=1.0, shadow_max_dark=0.2,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(enabled=False),
    },

    # 9. Heavy fog
    "09_foggy_depth": {
        "rules": _cool_blues,
        "transform": Transform(
            scale=12.0,
            rotations=[("z", 90.0), ("x", 10.0), ("y", 0.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(0.92, 0.94, 0.96),
            fog_color=(0.92, 0.94, 0.96),
            fog_front=0.6, fog_back=0.3,
            shadows=True, shadow_strength=0.002,
            shadow_angle=2.0, shadow_min_z=1.0, shadow_max_dark=0.15,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(enabled=True),
    },

    # 10. Top-down view
    "10_top_view": {
        "rules": _high_contrast,
        "transform": Transform(
            scale=11.0,
            rotations=[("x", 90.0), ("z", 0.0), ("y", 0.0)],
            autocenter="auto",
        ),
        "world": WorldParams(
            background=(1.0, 1.0, 1.0),
            fog_color=(1.0, 1.0, 1.0),
            fog_front=1.0, fog_back=1.0,
            shadows=True, shadow_strength=0.002,
            shadow_angle=2.0, shadow_min_z=1.0, shadow_max_dark=0.2,
            width=-30, height=-30,
        ),
        "outlines": OutlineParams(enabled=True),
    },
}


def main() -> None:
    print(f"Rendering {len(CONFIGS)} presets of 2HHB...")
    for name, cfg in CONFIGS.items():
        print(f"\n[{name}]")
        rules = cfg["rules"]()
        atoms = load_pdb(str(PDB), rules)
        params = RenderParams(
            pdb_path=str(PDB),
            rules=rules,
            transform=cfg["transform"],
            world=cfg["world"],
            outlines=cfg["outlines"],
        )
        result = render_from_atoms(atoms, params)
        save(result, name)

    print(f"\nDone! All renders saved to {OUT}/")


if __name__ == "__main__":
    main()
