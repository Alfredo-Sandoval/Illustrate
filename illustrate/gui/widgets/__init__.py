"""Reusable embedded widgets for the Illustrate desktop app."""

from illustrate.gui.widgets.menus import (
    OutputSizeControls,
    build_main_menus,
    build_output_size_controls,
)
from illustrate.gui.widgets.render_bar import RenderBar
from illustrate.gui.widgets.render_size_toolbar import RenderSizeToolbar
from illustrate.gui.widgets.sidebar import NavigationControls, fill_navigation

__all__ = [
    "NavigationControls",
    "OutputSizeControls",
    "RenderBar",
    "RenderSizeToolbar",
    "build_main_menus",
    "build_output_size_controls",
    "fill_navigation",
]
