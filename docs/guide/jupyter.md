# Jupyter

Illustrate returns NumPy arrays, so renders can be displayed directly in notebooks.

## Install notebook extras

```bash
pip install -e ".[notebook]"
```

This installs:

- `ipython`
- `ipywidgets`

## Inline rendering example

```python
from PIL import Image
from IPython.display import display

from illustrate import RenderParams, SelectionRule, Transform, WorldParams, OutlineParams, render

params = RenderParams(
    pdb_path="data/2hhb.pdb",
    rules=[
        SelectionRule("HETATM", "-----HOH--", 0, 9999, (0.5, 0.5, 0.5), 0.0),
        SelectionRule("ATOM  ", "---------A", 0, 9999, (1.0, 0.5, 0.5), 1.5),
        SelectionRule("ATOM  ", "---------B", 0, 9999, (0.5, 0.7, 1.0), 1.5),
        SelectionRule("ATOM  ", "----------", 0, 9999, (1.0, 0.8, 0.6), 1.5),
    ],
    transform=Transform(scale=12.0, rotations=[("z", 90.0)]),
    world=WorldParams(shadows=True, width=-30, height=-30),
    outlines=OutlineParams(enabled=True),
)

result = render(params)
display(Image.fromarray(result.rgb))
```

## Using presets

```python
from PIL import Image
from IPython.display import display

from illustrate import render
from illustrate.presets import render_params_from_preset

params = render_params_from_preset("Cool Blues", "data/2hhb.pdb")
result = render(params)
display(Image.fromarray(result.rgb))
```

## Raw arrays

`RenderResult` includes:

- `result.rgb` (`H x W x 3`, `uint8`)
- `result.opacity` (`H x W`, `uint8`)

Opacity display example:

```python
from PIL import Image
from IPython.display import display

display(Image.fromarray(result.opacity, mode="L"))
```

## Included notebook

The repository includes `illustrate.ipynb` at repo root.
