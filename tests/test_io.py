from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from illustrate import export_png
from illustrate.io.image_export import write_p3_pnm, write_png, write_svg


def test_write_p3_pnm_requires_rgb_channel_last(tmp_path: Path) -> None:
    bad_rgb = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="rgb must be HxWx3 array"):
        write_p3_pnm(tmp_path / "bad.p3", bad_rgb)


def test_write_png_requires_rgb_channel_last(tmp_path: Path) -> None:
    bad_rgb = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="rgb must be HxWx3 array"):
        write_png(tmp_path / "bad.png", bad_rgb)


def test_write_p3_pnm_emits_fortran_ordered_header_and_pixels(tmp_path: Path) -> None:
    rgb = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=np.uint8,
    )
    out = tmp_path / "out.p3"
    write_p3_pnm(out, rgb)

    lines = out.read_text(encoding="ascii").splitlines()
    assert lines[0] == "P3"
    assert lines[1] == "    3    2"
    assert lines[2] == "  255"
    assert "   1   2   3   4   5   6" in lines[3]


def test_write_svg_requires_rgb_channel_last(tmp_path: Path) -> None:
    bad_rgb = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="rgb must be HxWx3 or HxWx4 array"):
        write_svg(tmp_path / "bad.svg", bad_rgb)


def test_write_svg_embeds_png_data_uri(tmp_path: Path) -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 255], [0, 0, 0], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )
    out = tmp_path / "out.svg"
    write_svg(out, rgb)

    text = out.read_text(encoding="utf-8")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<svg " in text
    assert 'width="3"' in text
    assert 'height="2"' in text
    assert "data:image/png;base64," in text


def test_root_export_png_preserves_opacity_channel(tmp_path: Path) -> None:
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    opacity = np.array([[0, 255]], dtype=np.uint8)
    out = tmp_path / "rgba.png"

    export_png(out, rgb, opacity)

    image = Image.open(out)
    assert image.mode == "RGBA"
    pixels = np.asarray(image)
    assert pixels.shape == (1, 2, 4)
    assert pixels[:, :, 3].tolist() == [[0, 255]]


def test_root_export_png_rejects_mismatched_opacity_shape(tmp_path: Path) -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    opacity = np.zeros((1, 2), dtype=np.uint8)

    with pytest.raises(
        ValueError, match="opacity shape must match rgb height and width"
    ):
        export_png(tmp_path / "bad.png", rgb, opacity)
