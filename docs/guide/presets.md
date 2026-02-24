# Presets

Presets are complete parameter bundles (`rules`, `transform`, `world`, `outlines`). Applying a preset replaces current values with preset values.

## Available preset names

```python
from illustrate.presets import PRESET_NAMES

# [
#   "Default",
#   "Black Background",
#   "Wireframe",
#   "Dark Chain Colors",
#   "Cool Blues",
#   "High Contrast",
#   "Pen & Ink",
#   "Earth Tones",
# ]
```

## Preset summary (from `illustrate.presets`)

| Preset | Scale | Background | Outlines enabled |
|------|------:|------------|------------------|
| Default | 12.0 | `(1.0, 1.0, 1.0)` | `True` |
| Black Background | 14.0 | `(0.0, 0.0, 0.0)` | `True` |
| Wireframe | 10.0 | `(1.0, 1.0, 1.0)` | `True` |
| Dark Chain Colors | 12.5 | `(0.05, 0.06, 0.08)` | `True` |
| Cool Blues | 12.0 | `(0.95, 0.97, 1.0)` | `True` |
| High Contrast | 12.0 | `(1.0, 1.0, 1.0)` | `True` |
| Pen & Ink | 12.0 | `(1.0, 1.0, 1.0)` | `True` |
| Earth Tones | 13.0 | `(0.98, 0.96, 0.93)` | `True` |

## In the desktop GUI

Use the preset dropdown in the top toolbar. Applying a preset updates rules, transform, world, and outline settings.

## In Python

```python
from illustrate import render, write_png
from illustrate.presets import render_params_from_preset

params = render_params_from_preset("Default", "data/2hhb.pdb")
result = render(params)
write_png("output.png", result.rgb)
```

## Customizing a preset

```python
from dataclasses import replace
from illustrate.presets import render_params_from_preset

params = render_params_from_preset("Cool Blues", "data/2hhb.pdb")
params = replace(
    params,
    world=replace(params.world, background=(0.05, 0.05, 0.1)),
)
```

## Saving custom configurations

GUI workflow:

- **Save Settings** exports full params JSON.
- **Load Settings** restores from JSON.

Python workflow:

```python
from illustrate.render import params_to_json, params_from_json

json_str = params_to_json(params)
params2 = params_from_json(json_str)
```
