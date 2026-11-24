"""Write rendered images as PNG, SVG or PNM files."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image



def _iter_chunks(values: list[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def write_p3_pnm(path: str | Path, rgb: NDArray[np.generic]) -> None:
    """Write a P3 image in the legacy Fortran text layout.

    Pass ``(height, width, 3)`` pixels with channels in the 0–255 range. Values
    are written as integers in row order.
    """
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


def write_png(path: str | Path, rgb: NDArray[np.generic]) -> None:
    """Write ``(height, width, 3)`` RGB pixels, casting channels to uint8."""
    output_path = Path(path)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 array")

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    image.save(output_path)


def write_svg(path: str | Path, rgb: NDArray[np.generic]) -> None:
    """Write SVG with the rendered image embedded as a base64 PNG.

    Accepts RGB or RGBA in row, column, channel order and casts to uint8.
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


def _validate_rgb(rgb: NDArray[np.uint8]) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 array")


def _validate_opacity(opacity: NDArray[np.uint8], rgb: NDArray[np.uint8]) -> None:
    if opacity.ndim != 2:
        raise ValueError("opacity must be HxW array")
    if opacity.shape != rgb.shape[:2]:
        raise ValueError("opacity shape must match rgb height and width")


def to_rgba(rgb: NDArray[np.uint8], opacity: NDArray[np.uint8] | None) -> NDArray[np.uint8] | None:
    """Append a matching opacity plane as alpha, or return None if absent."""
    if opacity is None:
        return None
    _validate_rgb(rgb)
    _validate_opacity(opacity, rgb)
    return np.dstack((rgb, opacity))


def export_png(path: str | Path, rgb: NDArray[np.uint8], opacity: NDArray[np.uint8] | None) -> None:
    """Write RGB pixels to a PNG path, adding alpha from a matching opacity plane."""
    rgba = to_rgba(rgb, opacity)
    if rgba is None:
        write_png(path, rgb)
    else:
        Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")


def export_svg(path: str | Path, rgb: NDArray[np.uint8], opacity: NDArray[np.uint8] | None) -> None:
    """Embed the RGB image as a PNG in an SVG, preserving any supplied opacity."""
    rgba = to_rgba(rgb, opacity)
    write_svg(path, rgba if rgba is not None else rgb)
