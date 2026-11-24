"""Qt image display helpers."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QImage, QPixmap

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GuiImageFrame:
    """QImage plus the Python-owned buffer that keeps its pixels alive."""

    qimage: QImage
    buffer: NDArray[np.uint8]
    width: int
    height: int
    has_alpha: bool


def _contiguous_u8(array: NDArray[np.generic]) -> NDArray[np.uint8]:
    return np.ascontiguousarray(array, dtype=np.uint8)


def _pack_rgba(rgb: NDArray[np.uint8], opacity: NDArray[np.uint8]) -> NDArray[np.uint8]:
    height, width = rgb.shape[:2]
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    # One channel at a time: a single three-channel slice assignment is about
    # four times slower, and this runs for every frame of a drag.
    for channel in range(3):
        rgba[:, :, channel] = rgb[:, :, channel]
    rgba[:, :, 3] = opacity
    return rgba


def image_frame_from_arrays(
    rgb: NDArray[np.generic], opacity: NDArray[np.generic] | None = None
) -> GuiImageFrame | None:
    if rgb.size == 0:
        return None
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None

    # Renderer outputs row-major image buffers ([height, width, channels]),
    # matching QImage expectations.
    rgb_u8 = _contiguous_u8(rgb)
    height, width = rgb_u8.shape[:2]

    if opacity is not None and opacity.ndim == 2 and opacity.shape == (height, width):
        rgba_u8 = _pack_rgba(rgb_u8, _contiguous_u8(opacity))
        qimage = QImage(
            rgba_u8.data,
            width,
            height,
            int(rgba_u8.strides[0]),
            QImage.Format.Format_RGBA8888,
        )
        return GuiImageFrame(
            qimage=qimage,
            buffer=rgba_u8,
            width=width,
            height=height,
            has_alpha=True,
        )

    # Normalize transposed alpha masks before QImage export.
    if opacity is not None and opacity.ndim == 2 and opacity.shape == (width, height):
        opacity_u8 = _contiguous_u8(np.transpose(opacity, (1, 0)))
        rgba_u8 = _pack_rgba(rgb_u8, opacity_u8)
        qimage = QImage(
            rgba_u8.data,
            width,
            height,
            int(rgba_u8.strides[0]),
            QImage.Format.Format_RGBA8888,
        )
        return GuiImageFrame(
            qimage=qimage,
            buffer=rgba_u8,
            width=width,
            height=height,
            has_alpha=True,
        )

    qimage = QImage(
        rgb_u8.data,
        width,
        height,
        int(rgb_u8.strides[0]),
        QImage.Format.Format_RGB888,
    )
    return GuiImageFrame(
        qimage=qimage,
        buffer=rgb_u8,
        width=width,
        height=height,
        has_alpha=False,
    )


def pixmap_from_frame(frame: GuiImageFrame) -> QPixmap | None:
    if frame.qimage.isNull():
        return None
    return QPixmap.fromImage(frame.qimage)


def pixmap_from_image(
    rgb: NDArray[np.generic], opacity: NDArray[np.generic] | None = None
) -> QPixmap | None:
    frame = image_frame_from_arrays(rgb, opacity)
    if frame is None:
        return None
    return pixmap_from_frame(frame)
