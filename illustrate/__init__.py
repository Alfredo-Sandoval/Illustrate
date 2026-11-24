"""Render PDB structures as Goodsell-style molecular illustrations.

Start with ``render_params_from_preset`` and ``render``, then write the image
with ``export_png`` or ``export_svg``. To render one structure repeatedly,
classify it once with ``load_pdb`` and use ``prepare_render_session``.
"""

from illustrate.io.rcsb import fetch_pdb
from illustrate.formats.command_stream import (
    ParseError,
    parse_command_file,
    parse_command_stream,
)
from illustrate.formats.pdb import load_pdb, read_and_classify_atoms
from illustrate.rendering.api import (
    estimate_render_size,
    prepare_render_session,
    render,
    render_from_atoms,
    render_from_command_file,
    RenderSession,
    resolve_render_backend,
)
from illustrate.io.params_json import (
    PARAMS_JSON_VERSION,
    params_from_json,
    params_to_json,
)
from illustrate.io.image_export import (
    export_png,
    export_svg,
    to_rgba,
    write_p3_pnm,
    write_png,
    write_svg,
)
from illustrate.core.types import (
    AtomTable,
    OutlineParams,
    RenderParams,
    RenderResult,
    SelectionRule,
    Transform,
    WorldParams,
)
from illustrate.presets import (
    PRESET_NAMES,
    default_rules,
    preset_library,
    render_params_from_preset,
)

__all__ = [
    "AtomTable",
    "fetch_pdb",
    "OutlineParams",
    "RenderParams",
    "RenderResult",
    "RenderSession",
    "SelectionRule",
    "Transform",
    "WorldParams",
    "load_pdb",
    "parse_command_file",
    "parse_command_stream",
    "prepare_render_session",
    "ParseError",
    "read_and_classify_atoms",
    "render",
    "render_from_atoms",
    "render_from_command_file",
    "resolve_render_backend",
    "estimate_render_size",
    "PARAMS_JSON_VERSION",
    "params_to_json",
    "params_from_json",
    "write_p3_pnm",
    "write_png",
    "write_svg",
    "export_png",
    "export_svg",
    "to_rgba",
    "PRESET_NAMES",
    "default_rules",
    "preset_library",
    "render_params_from_preset",
]
