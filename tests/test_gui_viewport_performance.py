from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if importlib.util.find_spec("PySide6") is None:
    raise pytest.skip.Exception(
        "PySide6 not installed; skipping GUI viewport performance tests",
        allow_module_level=True,
    )

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from illustrate.gui.viewport import RenderViewport


@pytest.fixture
def app() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


@pytest.fixture
def viewport(app: QApplication) -> Iterator[RenderViewport]:
    viewport = RenderViewport()
    viewport.resize(400, 300)
    viewport.show()
    app.processEvents()
    try:
        yield viewport
    finally:
        viewport.close()


def _solid_pixmap(width: int, height: int, color: QColor) -> QPixmap:
    pixmap = QPixmap(width, height)
    pixmap.fill(color)
    return pixmap


def test_viewport_refits_the_image_when_resized(
    app: QApplication,
    viewport: RenderViewport,
) -> None:
    red = QColor(220, 40, 40)
    viewport.update_image(_solid_pixmap(800, 400, red))
    frame = viewport.grab().toImage()
    assert frame.pixelColor(200, 49) != red
    assert frame.pixelColor(200, 50) == red

    viewport.resize(500, 300)
    app.processEvents()

    frame = viewport.grab().toImage()
    assert frame.pixelColor(250, 24) != red
    assert frame.pixelColor(250, 25) == red
    assert frame.pixelColor(499, 25) == red


def test_render_smaller_than_view_is_shown_without_upscaling(
    viewport: RenderViewport,
) -> None:
    # Upscaling blurs the render; a small render keeps one pixel per device pixel.
    red = QColor(220, 40, 40)
    viewport.update_image(_solid_pixmap(160, 80, red))
    frame = viewport.grab().toImage()
    assert frame.pixelColor(119, 110) != red
    assert frame.pixelColor(120, 109) != red
    assert frame.pixelColor(120, 110) == red
    assert frame.pixelColor(279, 189) == red
