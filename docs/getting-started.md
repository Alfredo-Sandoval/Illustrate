# Getting Started

## Quick install

```bash
git clone https://github.com/Alfredo-Sandoval/Illustrate.git
cd Illustrate
pip install -e .
```

Requires Python 3.9+.

Core runtime dependencies:

- `numpy`
- `pillow`

Optional extras:

```bash
pip install -e ".[gui]"       # desktop GUI (PySide6)
pip install -e ".[web]"       # FastAPI models/routes package
pip install -e ".[notebook]"  # ipython + ipywidgets
```

For full development tooling (lint/typecheck/tests/docs), run:

```bash
make env
```

## Easiest desktop launch

```bash
make start
```

On macOS, double-click `Illustrate.command` to run the same launcher.

If release installers are available, you can use packaged desktop builds instead:

- macOS `.dmg`
- Windows `.zip` app bundle

## Render from the CLI

The CLI reads an Illustrate command stream from stdin.

The bundled demo command file (`data/2hhb.inp`) references `2hhb.pdb` relative to the `data/` directory, so run it from there:

```bash
cd data
illustrate-py < 2hhb.inp
```

This writes:

- `2hhb.png`
- `opacity.png`

## Render from Python

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
    rules=[
        SelectionRule("HETATM", "-----HOH--", 0, 9999, (0.5, 0.5, 0.5), 0.0),
        SelectionRule("ATOM  ", "---------A", 0, 9999, (1.0, 0.5, 0.5), 1.5),
        SelectionRule("ATOM  ", "---------B", 0, 9999, (0.5, 0.7, 1.0), 1.5),
        SelectionRule("ATOM  ", "----------", 0, 9999, (1.0, 0.8, 0.6), 1.5),
    ],
    transform=Transform(scale=12.0, rotations=[("z", 90.0)]),
    world=WorldParams(shadows=True),
)
result = render(params)
write_png("output.png", result.rgb)
```

Or load a named preset:

```python
from illustrate import render, write_png
from illustrate.presets import render_params_from_preset

params = render_params_from_preset("Default", "data/2hhb.pdb")
result = render(params)
write_png("output.png", result.rgb)
```

## Launch the desktop GUI

```bash
illustrate-gui
```

Use **Open PDB** for local files or **Fetch** for a 4-character RCSB ID.

## Known limitations

- `radius * scale` must stay `<= 100` or rendering fails with `atoms radius * scale > 100`.
- The viewport is image-based (not editable molecular geometry). Interactions update transform/render state; they do not directly manipulate a 3D mesh.

Running into issues? See [Troubleshooting](troubleshooting.md).
