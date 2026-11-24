# Presets

Choose a look in the desktop sidebar or with `illustrate-py --preset NAME`. With a structure loaded in the app, applying a preset keeps your current rotation and translation and preserves scale within the new radius limit.

The built-in looks are Default, Black Background, Wireframe, Dark Chain Colors, Cool Blues, High Contrast, Pen & Ink, and Earth Tones. Run `illustrate-py --list-presets` to print the names. Wireframe uses sharper sphere contours. Dark Chain Colors assigns its palette by chain ID.

## Make your own

Use **Save as preset…** in the desktop app to reuse your current look. Use **Save Settings** when you also want to record the current structure path and share the settings with the CLI.

In Python, preset names are case-sensitive. The returned settings are independent of the built-in preset.

```python
from illustrate import render, render_params_from_preset

params = render_params_from_preset("Cool Blues", "examples/2hhb.pdb")
params.world.background = (0.05, 0.05, 0.10)
result = render(params)
```

To edit colors and hide specific atoms, see [selection rules](selection-rules.md).
