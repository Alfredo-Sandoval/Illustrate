"""Read and write versioned JSON render settings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
import math
from typing import cast

from illustrate.core.types import (
    OutlineParams,
    RenderParams,
    SelectionRule,
    Transform,
    WorldParams,
)
from illustrate.rendering.api import autocenter_mode_code

PARAMS_JSON_VERSION = 1


def params_to_json(params: RenderParams) -> str:
    """Serialize render settings to JSON with the current schema version."""
    payload: object = asdict(params) if hasattr(params, "__dataclass_fields__") else params
    if not isinstance(payload, dict):
        raise TypeError("RenderParams serialization must produce an object")
    serialized = {str(key): value for key, value in cast(dict[object, object], payload).items()}
    serialized["schema_version"] = PARAMS_JSON_VERSION
    return json.dumps(serialized, sort_keys=True)


def _coerce_params_schema_version(payload: object) -> int:
    if payload is None:
        return PARAMS_JSON_VERSION
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise TypeError("schema_version must be an integer")
    if payload != PARAMS_JSON_VERSION:
        raise ValueError(f"unsupported RenderParams schema_version: {payload}")
    return payload


def _coerce_int_value(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer-compatible value")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"{field_name} must be an integer-compatible value")
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be an integer-compatible value"
            ) from exc
    raise TypeError(f"{field_name} must be an integer-compatible value")


def _coerce_float_value(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a float-compatible value")
    if isinstance(value, (str, bytes, bytearray, int, float)):
        try:
            coerced = float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a float-compatible value") from exc
        if math.isfinite(coerced):
            return coerced
        raise ValueError(f"{field_name} must be finite")
    raise TypeError(f"{field_name} must be a float-compatible value")


def _coerce_str_value(value: object, field_name: str) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f"{field_name} must be a string")


def _coerce_bool_value(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a boolean")


def _coerce_color(
    payload: object,
    default: tuple[float, float, float],
    field_name: str,
) -> tuple[float, float, float]:
    if payload is None:
        values: Sequence[object] = default
    elif isinstance(payload, (list, tuple)):
        values = cast(Sequence[object], payload)
    else:
        raise TypeError(f"{field_name} must be a 3-component array")
    if len(values) < 3:
        raise ValueError(f"{field_name} must contain at least 3 components")
    return (
        _coerce_float_value(values[0], f"{field_name}[0]"),
        _coerce_float_value(values[1], f"{field_name}[1]"),
        _coerce_float_value(values[2], f"{field_name}[2]"),
    )


def _coerce_translate(payload: object, field_name: str) -> tuple[float, float, float]:
    if payload is None:
        values: Sequence[object] = (0.0, 0.0, 0.0)
    elif isinstance(payload, (list, tuple)):
        values = cast(Sequence[object], payload)
    else:
        raise TypeError(f"{field_name} must be a 3-component array")
    if len(values) < 3:
        raise ValueError(f"{field_name} must contain at least 3 components")
    return (
        _coerce_float_value(values[0], f"{field_name}[0]"),
        _coerce_float_value(values[1], f"{field_name}[1]"),
        _coerce_float_value(values[2], f"{field_name}[2]"),
    )


def _coerce_object(payload: object, field_name: str) -> dict[str, object]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return {str(key): value for key, value in cast(dict[object, object], payload).items()}
    raise TypeError(f"{field_name} must be an object")


def _coerce_rotations(payload: object, field_name: str) -> list[tuple[str, float]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise TypeError(f"{field_name} must be a list")

    rotations: list[tuple[str, float]] = []
    for idx, item in enumerate(cast(list[object], payload)):
        item_name = f"{field_name}[{idx}]"
        if not isinstance(item, (list, tuple)):
            raise TypeError(f"{item_name} must be a 2-item array")
        components = cast(Sequence[object], item)
        if len(components) < 2:
            raise ValueError(f"{item_name} must contain an axis and angle")
        axis = _coerce_str_value(components[0], f"{item_name}[0]")
        if axis.strip().lower() not in {"x", "y", "z"}:
            raise ValueError(f"{item_name}[0] must be one of: x, y, z")
        rotations.append((axis, _coerce_float_value(components[1], f"{item_name}[1]")))
    return rotations


def _coerce_autocenter(value: object) -> str:
    autocenter = _coerce_str_value(value, "transform.autocenter")
    autocenter_mode_code(autocenter)
    return autocenter


def params_from_json(json_str: str) -> RenderParams:
    """Parse and validate saved render settings.

    An omitted or null schema version uses the current version. An explicit
    version must match it. Missing transform, world, and outline settings
    use the dataclass defaults.
    """
    payload_raw: object = json.loads(json_str)
    if not isinstance(payload_raw, dict):
        raise TypeError("top-level JSON payload must be an object")
    payload = {str(key): value for key, value in cast(dict[object, object], payload_raw).items()}
    _coerce_params_schema_version(payload.get("schema_version"))

    rules_payload = payload.get("rules", [])
    if not isinstance(rules_payload, list):
        raise TypeError("rules must be a list")

    transform_payload = _coerce_object(payload.get("transform", {}), "transform")
    world_payload = _coerce_object(payload.get("world", {}), "world")
    outlines_payload = _coerce_object(payload.get("outlines", {}), "outlines")
    default_transform = Transform()
    default_world = WorldParams()
    default_outlines = OutlineParams()

    normalized_rules: list[SelectionRule] = []
    for idx, rule in enumerate(cast(list[object], rules_payload)):
        rule_payload = _coerce_object(rule, f"rules[{idx}]")
        normalized_rules.append(
            SelectionRule(
                record_name=_coerce_str_value(
                    rule_payload.get("record_name"), f"rules[{idx}].record_name"
                ),
                descriptor=_coerce_str_value(
                    rule_payload.get("descriptor"), f"rules[{idx}].descriptor"
                ),
                res_low=_coerce_int_value(
                    rule_payload.get("res_low"), f"rules[{idx}].res_low"
                ),
                res_high=_coerce_int_value(
                    rule_payload.get("res_high"), f"rules[{idx}].res_high"
                ),
                color=_coerce_color(
                    rule_payload.get("color"), (1.0, 1.0, 1.0), f"rules[{idx}].color"
                ),
                radius=_coerce_float_value(
                    rule_payload.get("radius"), f"rules[{idx}].radius"
                ),
            )
        )

    return RenderParams(
        pdb_path=_coerce_str_value(payload.get("pdb_path"), "pdb_path"),
        rules=normalized_rules,
        transform=Transform(
            scale=_coerce_float_value(
                transform_payload.get("scale", default_transform.scale),
                "transform.scale",
            ),
            translate=_coerce_translate(
                transform_payload.get("translate", default_transform.translate),
                "transform.translate",
            ),
            rotations=_coerce_rotations(
                transform_payload.get("rotations", default_transform.rotations),
                "transform.rotations",
            ),
            autocenter=_coerce_autocenter(
                transform_payload.get("autocenter", default_transform.autocenter)
            ),
        ),
        world=WorldParams(
            background=_coerce_color(
                world_payload.get("background", default_world.background),
                default_world.background,
                "world.background",
            ),
            fog_color=_coerce_color(
                world_payload.get("fog_color", default_world.fog_color),
                default_world.fog_color,
                "world.fog_color",
            ),
            fog_front=_coerce_float_value(
                world_payload.get("fog_front", default_world.fog_front),
                "world.fog_front",
            ),
            fog_back=_coerce_float_value(
                world_payload.get("fog_back", default_world.fog_back), "world.fog_back"
            ),
            shadows=_coerce_bool_value(
                world_payload.get("shadows", default_world.shadows), "world.shadows"
            ),
            shadow_strength=_coerce_float_value(
                world_payload.get("shadow_strength", default_world.shadow_strength),
                "world.shadow_strength",
            ),
            shadow_angle=_coerce_float_value(
                world_payload.get("shadow_angle", default_world.shadow_angle),
                "world.shadow_angle",
            ),
            shadow_min_z=_coerce_float_value(
                world_payload.get("shadow_min_z", default_world.shadow_min_z),
                "world.shadow_min_z",
            ),
            shadow_max_dark=_coerce_float_value(
                world_payload.get("shadow_max_dark", default_world.shadow_max_dark),
                "world.shadow_max_dark",
            ),
            width=_coerce_int_value(
                world_payload.get("width", default_world.width), "world.width"
            ),
            height=_coerce_int_value(
                world_payload.get("height", default_world.height), "world.height"
            ),
        ),
        outlines=OutlineParams(
            enabled=_coerce_bool_value(
                outlines_payload.get("enabled", default_outlines.enabled),
                "outlines.enabled",
            ),
            contour_low=_coerce_float_value(
                outlines_payload.get("contour_low", default_outlines.contour_low),
                "outlines.contour_low",
            ),
            contour_high=_coerce_float_value(
                outlines_payload.get("contour_high", default_outlines.contour_high),
                "outlines.contour_high",
            ),
            kernel=_coerce_int_value(
                outlines_payload.get("kernel", default_outlines.kernel),
                "outlines.kernel",
            ),
            z_diff_min=_coerce_float_value(
                outlines_payload.get("z_diff_min", default_outlines.z_diff_min),
                "outlines.z_diff_min",
            ),
            z_diff_max=_coerce_float_value(
                outlines_payload.get("z_diff_max", default_outlines.z_diff_max),
                "outlines.z_diff_max",
            ),
            subunit_low=_coerce_float_value(
                outlines_payload.get("subunit_low", default_outlines.subunit_low),
                "outlines.subunit_low",
            ),
            subunit_high=_coerce_float_value(
                outlines_payload.get("subunit_high", default_outlines.subunit_high),
                "outlines.subunit_high",
            ),
            residue_low=_coerce_float_value(
                outlines_payload.get("residue_low", default_outlines.residue_low),
                "outlines.residue_low",
            ),
            residue_high=_coerce_float_value(
                outlines_payload.get("residue_high", default_outlines.residue_high),
                "outlines.residue_high",
            ),
            residue_diff=_coerce_float_value(
                outlines_payload.get("residue_diff", default_outlines.residue_diff),
                "outlines.residue_diff",
            ),
        ),
    )


def save_settings(params: RenderParams, path: Path) -> Path:
    output_path = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
    output_path.write_text(params_to_json(params), encoding="utf-8")
    return output_path


def load_settings(path: Path) -> RenderParams:
    return params_from_json(path.read_text(encoding="utf-8"))
