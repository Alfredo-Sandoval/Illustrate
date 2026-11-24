# Python and CLI reference

## Render a PDB

### `render(params)`

Reads the PDB and classifies its atoms using `params.rules`.

```python
from illustrate import export_png, render, render_params_from_preset

params = render_params_from_preset("Cool Blues", "examples/2hhb.pdb")
result = render(params, backend="numpy")
export_png("hemoglobin.png", result.rgb, result.opacity)
```

### `RenderResult`

`rgb` is a uint8 array of shape `(height, width, 3)`. `opacity` is a matching uint8 plane with 0 for transparent and 255 for opaque. Image rows follow the renderer’s X axis and columns follow Y.

### `fetch_pdb(pdb_id)`

Accepts four ASCII letters or digits and returns a local PDB path. Reuses a complete file in `~/.cache/illustrate/` and uses uppercase IDs in cache filenames.

## Reuse loaded atoms

### `load_pdb(path, rules, strict_input=False)`

Loads and classifies the PDB for repeated use.

```python
from illustrate import load_pdb, prepare_render_session, render_params_from_preset

params = render_params_from_preset("Default", "examples/2hhb.pdb")
atoms = load_pdb(params.pdb_path, params.rules)
session = prepare_render_session(atoms, backend="numpy")
result = session.render(params)
```

### `read_and_classify_atoms(path, rules, strict_input=False)`

Same operation as `load_pdb`. Strict mode checks coordinate fields on retained atoms.

### `render_from_atoms(atoms, params)`

Renders a classified atom table. Each atom stores the index of the rule it matched. Reclassify if you change rule order, matching criteria, or which atoms are hidden by rules.

### `prepare_render_session(atoms, *, backend=None)`

Returns a `RenderSession` that keeps the classified atom table by reference and selects the backend at construction.

### `estimate_render_size(atoms, params)`

Returns `(width, height)` in pixels before rasterization.

## Backend

### `resolve_render_backend(backend=None)`

An explicit backend takes precedence over `ILLUSTRATE_RENDER_BACKEND`. Automatic selection tries MLX first on Apple Silicon, then Numba and NumPy in that order on all hosts. An unsupported name raises `ValueError`. An explicit request for an unavailable backend raises `RuntimeError`.

### NumPy shadow rendering

The NumPy renderer can assign dense shadow strips to up to four threads, with one parallel shadow pass per process. Each worker reads the depth plane and owns its scratch space and output rows. The render caller joins each batch and checks cancellation between batches. The desktop app runs that caller in its background render worker.

Threads share the depth plane directly. A separate process pool would require transferring or sharing the raster buffers across processes, and parallel teams for simultaneous renders would multiply CPU use. Sparse shadow passes gather neighboring depths only for occupied pixels.

### Desktop CPU startup

When the desktop selects Numba, the backend status worker builds its kernel cache in a child process using a small in-memory scene, then loads the cached kernels. Process isolation keeps LLVM compilation off the desktop interpreter. With automatic selection, the render worker uses NumPy at the requested resolution while Numba prepares. If preparation fails, the desktop keeps using NumPy and shows the error in the backend tooltip. An explicit Numba override waits in the render worker for preparation and reports a render error if preparation fails. Closing the app cancels preparation.

## Render settings

See the dataclass definitions for field types and defaults. [Core types](https://github.com/Alfredo-Sandoval/Illustrate/blob/main/illustrate/core/types.py)

### `RenderParams`

Holds the PDB path, ordered rules, transform, world, and outline settings.

### `SelectionRule`

Sets a color and radius for atoms matched by record type, descriptor, and residue range. The first match wins. The renderer checks scaled radii in the whole list, including unused rules. Keep `radius × scale` at or below 100 pixels. [Selection rules](guide/selection-rules.md)

### `Transform`

Scale is in pixels per ångström. Translation is in view coordinates in ångströms. Rotations are `(axis, degrees)` pairs, and the last listed rotation acts first. `autocenter="auto"` centers X and Y and places the front near the origin. `"center"` centers the depth bounds as well. `"none"` skips automatic translation.

### `WorldParams`

Two positive dimensions set the image size in pixels. With automatic centering and at least one retained atom, a nonpositive dimension triggers autosizing on both axes. Zero fits the structure bounds, `-30` adds 30 pixels per edge, and a positive dimension in this mode subtracts that amount per edge. Each dimension is capped at 3000 and rounded down to an even number.

### `OutlineParams`

Kernels 1 and 2 use a discrete Laplacian of depth for contours. Kernels 3 and 4 use neighboring depth differences.

## Settings files

### `PARAMS_JSON_VERSION`

The current settings schema version is `1`.

### `params_to_json(params)`

Writes a JSON string including `schema_version`.

### `params_from_json(json_str)`

Uses the current version when the version is omitted or null and rejects other explicit versions. Missing transform, world, and outline fields use dataclass defaults.

## Image export

### `export_png(path, rgb, opacity)`

Pass `None` for RGB output. A supplied opacity plane must match the image height and width.

### `export_svg(path, rgb, opacity)`

Embeds the rendered image as a PNG, with the same opacity handling as `export_png`.

### `to_rgba(rgb, opacity)`

Returns the combined array, or `None` if no opacity was supplied.

### `write_png(path, rgb)`

Writes an RGB array directly. `write_svg` accepts RGB or RGBA. `write_p3_pnm` writes the original Fortran program’s plain-text P3 layout.

## Presets

### `PRESET_NAMES`

Lists the built-in names in `preset_library` order.

### `default_rules()`

Returns the classic Illustrate rules with water and hydrogen exclusions first.

### `preset_library(rules=None)`

Returns independent settings. The first three presets use copies of the supplied rules or defaults. The remaining presets use their own palettes.

### `render_params_from_preset(name, pdb_path, rules=None)`

Binds a copy of a preset to a structure path. Python uses exact preset names. [Presets](guide/presets.md)

## Classic command streams

### `parse_command_stream(text, strict_input=False)`

The lenient parser records warnings for unknown cards and some malformed fields. Strict mode raises `ParseError` for those cases and requires `read` and `calculate` cards. Missing card payloads and nonfinite numeric values can raise in either mode.

### `parse_command_file(path, strict_input=False)`

Reads UTF-8 text and leaves paths as written in the cards.

### `render_from_command_file(text, strict_input=False)`

Renders from command-stream text. Tries relative PDB paths in the working directory first and then in the optional `base_dir`. The `strict_input` flag applies to both inputs.

## CLI

```bash
illustrate-py --help
illustrate-py --list-presets
illustrate-py examples/2hhb.pdb --preset "Cool Blues" --width 640 -o hemoglobin.png
```

The CLI accepts local file paths and four-character PDB IDs. By default, it writes one PNG per structure using the input stem. Use `--outdir` for a batch, `--format svg` to change the format, or `--output` to name a single output. PNG and SVG are opaque unless you pass `--transparent`.

- **Settings.** Choose either `--preset` or `--settings`. The CLI matches preset names case-insensitively. A settings file supplies the input path when you omit structure arguments. A relative path is resolved from the working directory. `--save-settings` writes the settings beside each image.
- **Size and rotation.** Use `--scale` in pixels per ångström or fit to `--width` and/or `--height` in the range 100–3000. Fitting uses an even size and may give a smaller image when the maximum scale is reached. `--rotate` appends rotations to the current list.
- **Classic inputs.** Use `--command-file examples/2hhb.inp -o hemoglobin.png` from the checkout, or `--command-file -` to read cards from standard input. With a command file on disk, relative PDB paths are tried in the working directory and then beside the file.
- **Exit status.** `0` means the requested work succeeded. `1` means at least one input in a multi-structure batch failed. `2` means the request was invalid or the single input failed.

Launch the installed desktop app with `illustrate-gui`.
