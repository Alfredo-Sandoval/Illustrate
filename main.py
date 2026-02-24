"""Repository entrypoint for launching the desktop GUI."""

from __future__ import annotations


def main() -> int:
    """Launch the PySide6 desktop GUI.

    Returns:
        Exit code from the GUI event loop.
    """

    from illustrate_gui.main import main as launch_gui

    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
