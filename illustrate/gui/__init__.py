"""PySide6 desktop app for Illustrate."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from illustrate.gui.main import main as main


def __getattr__(name: str) -> object:
    if name == "main":
        from illustrate.gui.main import main

        return main
    raise AttributeError(name)

__all__ = ["main"]
