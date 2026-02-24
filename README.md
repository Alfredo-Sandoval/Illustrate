<p align="center">
  <img src="renders/01_classic_goodsell.png" alt="Illustrate — hemoglobin 2HHB" width="480">
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)

# Illustrate

Molecular structure renderer using sphere-based projection, optional shadows, fog depth cueing, and edge outlines. This repository provides a Python implementation of David Goodsell's renderer with CLI, desktop GUI, web API, and Python interfaces.

> DS Goodsell & AJ Olson (1992) "Molecular Illustration in Black and White" *J Mol Graphics* 10, 235-240.

---

## Quickstart

```bash
pip install -e .
```

## Easiest Way (Non-Dev)

Use one command to set up (if needed) and launch the desktop app:

```bash
make start
```

On macOS, you can also double-click:

- `Illustrate.command`

When release installers are published, non-developers can skip terminal setup and use:

- macOS `.dmg` packages from the GitHub Releases page
- Windows `.zip` bundles from the GitHub Releases page

What this does:

- checks/creates the managed environment via micromamba/mamba
- installs dependencies from `environment/<os>/requirements.txt`
- launches `illustrate-gui`

Render hemoglobin from the included example:

```bash
cd data
illustrate-py < 2hhb.inp
```

Or use the Python API directly:

```python
from illustrate import (
    RenderParams,
    SelectionRule,
    Transform,
    WorldParams,
    render,
    write_png,
)

params = RenderParams(
    pdb_path="data/2hhb.pdb",
    rules=[SelectionRule("ATOM  ", "----------", 0, 9999, (1.0, 0.7, 0.5), 1.5)],
    transform=Transform(scale=12.0, rotations=[("z", 90.0)]),
    world=WorldParams(),
)
result = render(params)
write_png("hemoglobin.png", result.rgb)
```

---

## Installation

Requires Python 3.9+. Core dependencies are just **numpy** and **pillow**.

```bash
# Core rendering + CLI
pip install -e .

# Desktop GUI (adds PySide6)
pip install -e ".[gui]"

# Web API (adds FastAPI)
pip install -e ".[web]"

# Jupyter support (adds ipywidgets)
pip install -e ".[notebook]"
```

Cross-platform setup script (uses micromamba/mamba + uv):

```bash
make env
```

Launch GUI via managed environment:

```bash
make start
```

Build desktop installer artifacts:

```bash
# macOS: .app + .dmg + zipped app bundle
make package-macos

# Windows (run on Windows host): zipped .exe bundle
make package-windows
```

---

## Frontends

### CLI — `illustrate-py`

Compatible with the original Fortran command-stream format. Reads commands from stdin:

```bash
cd data
illustrate-py < 2hhb.inp
illustrate-py --strict-input < 2hhb.inp   # strict validation
```

Outputs a PNG image and `opacity.png` mask. Netpbm extensions (`.pnm`/`.ppm`/`.pgm`/`.pbm`) in the command file are automatically normalized to `.png`.

### Desktop GUI — `illustrate-gui`

```bash
illustrate-gui
```

PySide6 application with:

- Interactive viewport — drag to rotate, scroll to zoom
- Live parameter editing — rules, transforms, world, outlines
- PDB loading — local files or fetch by ID from RCSB
- Preset switching — Default, Black Background, Wireframe
- Check Updates action — reads release `latest.json` manifest

GPU backend selection is automatic by default. The renderer picks the first
available backend in this order: `mlx` -> `cupy` -> `numpy`.

You can still force a specific backend explicitly with an environment variable.
Dispatch is strict: unsupported backend names raise errors.

```bash
# Automatic backend selection (default): mlx -> cupy -> numpy
illustrate-gui

# GPU backend (requires cupy installed and a supported CUDA runtime)
ILLUSTRATE_RENDER_BACKEND=cupy illustrate-gui

# Apple GPU backend (requires mlx installed on Apple Silicon)
ILLUSTRATE_RENDER_BACKEND=mlx illustrate-gui
```
- Threaded rendering — UI stays responsive during renders
- PNG export

### Web API

```bash
uvicorn illustrate_web.api.main:app
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/render` | POST | Render with custom parameters, returns PNG stream |
| `/api/upload-pdb` | POST | Upload a PDB file |
| `/api/fetch-pdb` | POST | Fetch a PDB from RCSB by 4-character ID |
| `/api/presets` | GET | List preset configurations |
| `/api/health` | GET | Health check |

Includes a Next.js/React frontend in `illustrate_web/frontend/` with panels for rules, transforms, world parameters, and outlines.

### Jupyter Notebook

See [`illustrate.ipynb`](illustrate.ipynb) for an interactive walkthrough. Build parameters, render, and display inline:

```python
from PIL import Image
from illustrate import render
from IPython.display import display

result = render(params)
display(Image.fromarray(result.rgb))               # main render
display(Image.fromarray(result.opacity, mode="L")) # opacity mask
```

---

## Architecture

```
illustrate/               Shared rendering engine
  types.py                  RenderParams, SelectionRule, Transform, WorldParams, OutlineParams
  render.py                 render(), render_from_atoms(), render_from_command_file()
  pdb.py                    PDB parsing and atom classification
  fetch.py                  RCSB download with local cache (~/.cache/illustrate/)
  presets.py                Default, Black Background, Wireframe, and style variants
  parser.py                 Command file parsing
  math3d.py                 Rotation matrices, 3D transforms
  io.py                     Image I/O

illustrate_gui/           Desktop frontend (PySide6)
  app.py                    Main window, dark theme, panel layout
  viewport.py               Interactive render viewport (mouse rotation/zoom)
  worker.py                 Threaded RenderWorker
  panels/                   Rule, Transform, World, Outlines editors

illustrate_web/           Web frontend (FastAPI + Next.js)
  api/                      REST endpoints for render, upload, fetch, presets
  frontend/                 React UI with parameter panels
```

**Data flow:** PDB file → `load_pdb()` → `AtomTable` → `render()` → `RenderResult` (`rgb` + `opacity` as `uint8` arrays)

---

## Python API

### Core functions

```python
from illustrate import (
    render,                    # RenderParams → RenderResult
    render_from_atoms,         # AtomTable + RenderParams → RenderResult
    render_from_command_file,  # command file text → RenderResult
    load_pdb,                  # path + rules → AtomTable
    fetch_pdb,                 # "2hhb" → Path (downloads from RCSB, caches locally)
    parse_command_file,        # path → parsed program
    parse_command_stream,      # text → parsed program
    params_to_json,            # RenderParams → JSON string
    params_from_json,          # JSON string → RenderParams
)
```

### Selection rules

Rules match atoms sequentially — first match wins. Use `-` as wildcard in the atom descriptor.

```python
SelectionRule(
    record_name="ATOM  ",       # "ATOM  " or "HETATM" (6 chars, matched to PDB cols 1-6)
    descriptor="----------",    # 10-char pattern matched to PDB cols 13-22
    res_low=0,
    res_high=9999,
    color=(1.0, 0.7, 0.5),     # RGB, 0.0–1.0
    radius=1.5,                 # angstroms; 0.0 hides the atom
)
```

### Presets

```python
from illustrate.presets import preset_library, render_params_from_preset

presets = preset_library()  # list of 8 RenderParams
params = render_params_from_preset("Default", "data/2hhb.pdb", rules=my_rules)
```

| Preset | Background | Scale | Notes |
|--------|-----------|-------|-------|
| Default | White | 12.0 | Classic illustration style |
| Black Background | Black | 14.0 | Dark theme variant |
| Wireframe | White | 10.0 | Sharper outlines, more surface detail |
| Dark Chain Colors | Near-black | 12.5 | Chain-aware palette on dark background |
| Cool Blues | Pale blue | 12.0 | Steel-blue and teal palette |
| High Contrast | White | 12.0 | Saturated red/blue/amber palette |
| Pen & Ink | White | 12.0 | Grayscale with heavier outlines |
| Earth Tones | Warm cream | 13.0 | Sandstone and sage palette |

### Serialization

```python
json_str = params_to_json(params)   # save/transmit parameters
params = params_from_json(json_str) # restore
```

---

## Command file format

For compatibility with the original Fortran program. Commands are issued in order:

1. **`read`** — PDB filename, then selection/rendering cards, terminated by `END`
2. **`center`** / **`translate`** / **`xrot`** / **`yrot`** / **`zrot`** / **`scale`** — any order, rotations accumulate
3. **`world`** — background, fog, and shadow parameters
4. **`illustrate`** — outline thresholds and kernels
5. **`calculate`** — output filename

<details>
<summary><strong>Full command reference</strong></summary>

### read

```
read
filename.pdb
ATOM  -C-------A 0,9999, 1.0,0.6,0.6, 1.6    ← selection cards
...
END
```

Selection card format: `record(A6) descriptor(A10) res_low,res_high, r,g,b, radius`

- Record: `ATOM  ` or `HETATM`, matched against PDB columns 1–6
- Descriptor: 10 characters matched against PDB columns 13–22; `-` is wildcard
- Residue range: integer low, high
- Color: RGB floats 0.0–1.0
- Radius: angstroms (0.0 = omit atom)

Cards are tested in order — first match wins.

### center

```
center
auto     ← "aut" = autocenter (no clipping), "cen" = center on coordinate extents
```

### translate

```
trans
0.,0.,0.    ← x, y, z in angstroms
```

### xrot, yrot, zrot

```
zrot
90.         ← degrees; multiple rotation cards are concatenated (applied last-to-first)
```

### scale

```
scale
12.0        ← pixels per angstrom
```

### world

```
wor
1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0     ← bg_r,g,b, fog_r,g,b, fog_front, fog_back
1,0.0023,2.0,1.0,0.2                  ← shadows_on, strength, cone_angle, z_threshold, max_shadow
-30,-30                                ← image width, height (negative = autosize with margin)
```

### illustrate

```
illustrate
3.0,10.0,4,0.0,5.0     ← contour: low,high thresholds, kernel(1-4), z_diff_min,max
3.0,10.0                ← subunit: low,high thresholds
3.0,8.0,6000.           ← residue: low,high thresholds, residue_number_diff
```

### calculate

```
calculate
output.pnm     ← output filename (.pnm/.ppm/.pgm/.pbm auto-normalized to .png)
```

</details>

<details>
<summary><strong>Example: hemoglobin (2HHB)</strong></summary>

```
read
2hhb.pdb
HETATM-----HOH-- 0,9999, 0.5,0.5,0.5, 0.0        hide water
ATOM  -H-------- 0,9999, 0.5,0.5,0.5, 0.0        hide hydrogens
ATOM  H--------- 0,9999, 0.5,0.5,0.5, 0.0        hide hydrogens (long names)
ATOM  -C-------A 0,9999, 1.0,0.6,0.6, 1.6        chain A carbons — pink
ATOM  -S-------A 0,9999, 1.0,0.5,0.5, 1.8
ATOM  ---------A 0,9999, 1.0,0.5,0.5, 1.5        chain A other atoms
ATOM  -C-------C 0,9999, 1.0,0.6,0.6, 1.6        chain C carbons — pink
ATOM  -S-------C 0,9999, 1.0,0.5,0.5, 1.8
ATOM  ---------C 0,9999, 1.0,0.5,0.5, 1.5
ATOM  -C-------- 0,9999, 1.0,0.8,0.6, 1.6        remaining carbons — light orange
ATOM  -S-------- 0,9999, 1.0,0.7,0.5, 1.8
ATOM  ---------- 0,9999, 1.0,0.7,0.5, 1.5
HETATMFE---HEM-- 0,9999, 1.0,0.8,0.0, 1.8        heme iron — yellow
HETATM-C---HEM-- 0,9999, 1.0,0.3,0.3, 1.6        heme carbons — dark red
HETATM-----HEM-- 0,9999, 1.0,0.1,0.1, 1.5        heme other — near black
END
center
auto
trans
0.,0.,0.
scale
12.0
zrot
90.
wor
1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
1,0.0023,2.0,1.0,0.2
-30,-30
illustrate
3.0,10.0,4,0.0,5.0
3.0,10.0
3.0,8.0,6000.
calculate
2hhb.pnm
```

</details>

### Coordinate system

Origin at upper left. +x down, +y left to right, +z towards viewer. Molecules are clipped at z=0. Rotations are applied first (last-to-first order), then centering, then translation.

### Outlines

Three outline types, all derived from z-buffer derivatives:

- **Contour** — surface shape and detail; z-diff range controls scope (small = every atom, large = silhouette only)
- **Subunit** — boundaries between chains
- **Residue** — boundaries between residues (controlled by residue number difference threshold)

Threshold pairs (low, high) set the gray-to-black range. Kernel (1–4) controls smoothness.

---

## Development

```bash
make lint        # ruff
make typecheck   # ty
make test        # pytest
make qa          # lint + typecheck + test
make loc         # lines of code by module
```

Test markers: `@pytest.mark.port` (porting workflow), `@pytest.mark.parity` (Fortran vs Python comparison).

---

## Project history

Originally written in Fortran by David S. Goodsell at The Scripps Research Institute (released 2019). This repository provides a Python implementation with command-file compatibility, plus GUI and web frontends built on a shared rendering core.

## License

[Apache 2.0](LICENSE). Original Fortran implementation copyright 2019 David S. Goodsell. Python implementation and frontends copyright 2026 Alfredo Sandoval.
