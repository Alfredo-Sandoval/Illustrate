# API Reference

This page documents public Python types/functions and CLI surface that are implemented in this repository.

## Core types

### `SelectionRule`

```python
SelectionRule(
    record_name: str,
    descriptor: str,
    res_low: int,
    res_high: int,
    color: tuple[float, float, float],
    radius: float,
)
```

Matching behavior (from `illustrate.pdb.read_and_classify_atoms`):

- `record_name` is compared against the first 6 PDB characters (space-padded/truncated to 6).
- `descriptor` is compared character-by-character against PDB columns 13-22.
- `-` in descriptor is treated as wildcard.
- `res_low <= residue <= res_high` must hold.
- `radius == 0.0` suppresses matching atoms.
- Rules are evaluated top-to-bottom; first match wins.

### `Transform`

| Field | Type | Default |
|------|------|---------|
| `scale` | `float` | `12.0` |
| `translate` | `tuple[float, float, float]` | `(0.0, 0.0, 0.0)` |
| `rotations` | `list[tuple[str, float]]` | `[]` |
| `autocenter` | `str` | `"auto"` |

Supported `autocenter` values:

- `"auto"` / `"aut"`
- `"center"` / `"cen"`
- `"none"` / `"off"` / `"0"`

### `WorldParams`

| Field | Type | Default |
|------|------|---------|
| `background` | `tuple[float, float, float]` | `(1.0, 1.0, 1.0)` |
| `fog_color` | `tuple[float, float, float]` | `(1.0, 1.0, 1.0)` |
| `fog_front` | `float` | `1.0` |
| `fog_back` | `float` | `1.0` |
| `shadows` | `bool` | `False` |
| `shadow_strength` | `float` | `0.0023` |
| `shadow_angle` | `float` | `2.0` |
| `shadow_min_z` | `float` | `1.0` |
| `shadow_max_dark` | `float` | `0.2` |
| `width` | `int` | `0` |
| `height` | `int` | `0` |

Render-size semantics:

- `width > 0` and `height > 0`: explicit output size.
- `width <= 0` or `height <= 0`: autosize.
- Negative values add margin via renderer formula (`-2 * width` / `-2 * height`).

### `OutlineParams`

| Field | Type | Default |
|------|------|---------|
| `enabled` | `bool` | `False` |
| `contour_low` | `float` | `1.0` |
| `contour_high` | `float` | `10.0` |
| `kernel` | `int` | `4` |
| `z_diff_min` | `float` | `1.0` |
| `z_diff_max` | `float` | `50.0` |
| `subunit_low` | `float` | `3.0` |
| `subunit_high` | `float` | `10.0` |
| `residue_low` | `float` | `3.0` |
| `residue_high` | `float` | `8.0` |
| `residue_diff` | `float` | `6000.0` |

### `RenderParams`

```python
RenderParams(
    pdb_path: str,
    rules: list[SelectionRule],
    transform: Transform = Transform(),
    world: WorldParams = WorldParams(),
    outlines: OutlineParams = OutlineParams(),
)
```

### `RenderResult`

| Field | Type | Meaning |
|------|------|---------|
| `rgb` | `np.ndarray` | `H x W x 3`, `uint8` |
| `opacity` | `np.ndarray` | `H x W`, `uint8` |
| `width` | `int` | output width |
| `height` | `int` | output height |

## Rendering functions

### `render(params)`

```python
from illustrate import render
result = render(params)
```

Loads atoms from `params.pdb_path` and renders.

### `render_from_atoms(atoms, params)`

```python
from illustrate import load_pdb, render_from_atoms
atoms = load_pdb("data/2hhb.pdb", params.rules)
result = render_from_atoms(atoms, params)
```

Renders from preloaded/classified atoms.

### `estimate_render_size(atoms, params)`

Returns `(width, height)` without full sphere rasterization.

### `render_from_command_file(text, strict_input=False)`

Parses classic command-stream text and renders.

## Serialization

### `params_to_json(params)`

Serializes `RenderParams` to JSON.

### `params_from_json(json_str)`

Parses JSON into `RenderParams`.

## Presets

### `PRESET_NAMES`

```python
[
  "Default",
  "Black Background",
  "Wireframe",
  "Dark Chain Colors",
  "Cool Blues",
  "High Contrast",
  "Pen & Ink",
  "Earth Tones",
]
```

### `default_rules()`

Returns the built-in 15-rule default stack (`list[SelectionRule]`).

### `preset_library(rules=None)`

Returns `list[RenderParams]` in `PRESET_NAMES` order.

### `render_params_from_preset(name, pdb_path, rules=None)`

Returns one `RenderParams` bound to `pdb_path`.

## PDB and fetch utilities

### `read_and_classify_atoms(path, rules, strict_input=False)`

Parses a PDB file, applies `SelectionRule` matching, and returns `AtomTable`.

### `load_pdb(path, rules, strict_input=False)`

Alias of `read_and_classify_atoms`.

### `fetch_pdb(pdb_id)`

Downloads from RCSB and caches at:

- `~/.cache/illustrate/<PDB_ID>.pdb` (uppercase ID in filename)

`pdb_id` must be exactly 4 alphanumeric characters.

## CLI

### `illustrate-py`

```bash
illustrate-py [--strict-input] < command_file.inp
```

Behavior:

- Reads stdin command stream.
- Normalizes NetPBM/no-suffix output names to `.png`.
- Writes main PNG output and `opacity.png` in the same output directory.

Exit codes:

- `0`: success
- `1`: render error
- `2`: parse/input/IO error

### `illustrate-gui`

Launches the desktop GUI.

```bash
illustrate-gui
```
