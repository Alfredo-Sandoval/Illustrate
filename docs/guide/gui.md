# Desktop GUI

## Overview

`illustrate-gui` is a PySide6 desktop interface over the same renderer used by CLI and Python APIs.

## Launch

```bash
illustrate-gui
```

## Main controls

Top toolbar includes:

- `Open PDB`
- `Fetch`
- preset selector + `Save Preset`
- `Load Settings` / `Save Settings`
- `Check Updates`
- theme selector (`Dark`, `Light`)
- preview quality (`Fast`, `Balanced`, `High`)
- render size controls (`Auto`/`Custom`, width/height)
- `Render`
- `Fit View`
- `Export PNG`
- `Export SVG`
- `Copy to Clipboard`
- `Selection Rules` toggle action (shows/hides bottom dock)

## Layout

![Desktop GUI screenshot](../images/gui_desktop_ui.png)

_Actual screenshot of the running `illustrate-gui` desktop app._

- Left sidebar: `Transform`, `World / Lighting`, `Outlines` sections.
- Sidebar sections are collapsible and show state arrows (right when collapsed, down when expanded).
- Center/right: image viewport.
- Bottom dock: `Selection Rules` table (starts minimized/hidden by default).

## Viewport behavior

- Drag emits rotation updates.
- Mouse wheel emits zoom updates.
- The viewport displays rendered images (not an editable molecular mesh).

By default, **Render on drag/zoom** is enabled, so drag/zoom interactions schedule interactive rerenders.

## Sidebar panels

### Transform

Fields:

- scale
- x/y/z rotation
- x/y/z translation

Includes `Reset`.

### World / Lighting

Fields:

- background color
- fog color
- fog front/back
- shadows enabled
- shadow strength
- shadow angle
- shadow start Z
- max shadow

### Outlines

Fields:

- outlines enabled/disabled
- contour low/high
- kernel (`1`..`4`)
- z diff min/max
- subunit low/high
- residue low/high
- residue diff

## Selection Rules dock

Columns:

- `Record`
- `Descriptor`
- `Res Low`
- `Res High`
- `Color`
- `Radius`
- `Matches`

Buttons:

- `Add`
- `Remove`
- `Up`
- `Down`

Descriptor is editable directly in the `Descriptor` column.

## Settings and export

- **Load/Save Settings** reads/writes JSON parameter bundles.
- **Export PNG/SVG** is available after a render.
- **Copy to Clipboard** copies the latest rendered image.
