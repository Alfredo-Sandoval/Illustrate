from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def _iter_chunks(values: list[int], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def write_p3_pnm(path: str | Path, rgb: np.ndarray) -> None:
    """Write P3 PNM with Fortran-like scanline ordering and integer width formatting."""
    output_path = Path(path)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 array")

    ixsize, iysize, _ = rgb.shape

    with output_path.open("w", encoding="ascii", newline="\n") as fh:
        fh.write("P3\n")
        fh.write(f"{iysize:5d}{ixsize:5d}\n")
        fh.write(f"{255:5d}\n")

        for ix in range(ixsize):
            scanline: list[int] = []
            for iout in range(iysize):
                r, g, b = rgb[ix, iout]
                scanline.append(int(r))
                scanline.append(int(g))
                scanline.append(int(b))
            for chunk in _iter_chunks(scanline, 20):
                fh.write("".join(f"{value:4d}" for value in chunk))
                fh.write("\n")


def write_optional_png(path: str | Path, rgb: np.ndarray) -> None:
    output_path = Path(path)
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    image.save(output_path)


def write_svg(path: str | Path, rgb: np.ndarray) -> None:
    """Write SVG with the rendered image embedded as a base64 PNG.

    Accepts either RGB (HxWx3) or RGBA (HxWx4).
    """
    output_path = Path(path)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError("rgb must be HxWx3 or HxWx4 array")

    mode = "RGBA" if rgb.shape[2] == 4 else "RGB"
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode=mode)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    width, height = image.size

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <image width="{width}" height="{height}" href="data:image/png;base64,{encoded}" />\n'
        "</svg>\n"
    )
    output_path.write_text(svg, encoding="utf-8")


def write_ppm(path: str | Path, rgb: np.ndarray) -> None:
    """Compatibility alias for legacy API."""
    write_p3_pnm(path, rgb)


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    """Compatibility alias for legacy API."""
    write_optional_png(path, rgb)
