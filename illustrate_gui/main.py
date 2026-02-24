"""Desktop app entry point."""

from __future__ import annotations


def main() -> int:
    try:
        from illustrate_gui.app import run
    except ImportError as exc:
        raise SystemExit(
            "PySide6 is required for the desktop app. Install pyside6 and retry."
        ) from exc

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
