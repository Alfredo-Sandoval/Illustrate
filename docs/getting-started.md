# Getting started

## Install on macOS

Install in an environment with Python 3.10 or newer and Git on your path.

```bash
python -m pip install "illustrate-py[gui] @ git+https://github.com/Alfredo-Sandoval/Illustrate.git"
illustrate-gui
```

Once the package is published on PyPI, use `python -m pip install "illustrate-py[gui]"`. Omit `[gui]` for the CLI and Python API.

### Install from a checkout

```bash
git clone https://github.com/Alfredo-Sandoval/Illustrate.git
cd Illustrate
make start
```

`make start` sets up the `illustrate` environment and launches the desktop app. It uses Mamba or Micromamba and installs Micromamba if neither is available. Run `make env` to set up the environment without launching. In an existing Python environment, run `python -m pip install -e ".[gui]"` from the checkout.

## Render the bundled example

From the checkout, render `examples/2hhb.pdb` without downloading a structure.

```bash
illustrate-py examples/2hhb.pdb --preset "Pen & Ink" --width 640 -o hemoglobin.png
```

If you installed without a checkout, use `2hhb` in place of `examples/2hhb.pdb` to fetch the structure from RCSB.

In Python, the preset supplies the scale and margins. [Presets](guide/presets.md)

```python
from illustrate import export_png, render, render_params_from_preset

params = render_params_from_preset("Pen & Ink", "examples/2hhb.pdb")
result = render(params)
export_png("hemoglobin.png", result.rgb, result.opacity)
```

To render the same structure repeatedly, [reuse a classified atom table](reference.md#reuse-loaded-atoms).

## Open a structure in the desktop app

Launch `illustrate-gui`. Use **Open PDB** for a local file or **Fetch** for a four-character PDB ID. See the [desktop guide](guide/gui.md) for controls and export.
