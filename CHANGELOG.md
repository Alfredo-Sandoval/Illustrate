# Changelog

## 0.1.0 · 2026-11-24

Initial release of the Python port of David Goodsell’s Illustrate. Includes the
`illustrate-py` CLI, the macOS desktop app, and the Python API.

- Render local PDB files or fetch structures by PDB ID. The CLI also reads classic Illustrate command files.
- Start from eight presets and customize atom selection, lighting, and outlines. Share JSON render settings between the desktop app and CLI.
- Render on the CPU with Numba or NumPy, or with MLX on Apple Silicon. Export PNG or an SVG with an embedded PNG. The Python API also writes P3 PNM.
