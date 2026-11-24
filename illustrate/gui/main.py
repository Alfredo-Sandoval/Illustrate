"""Desktop app entry point."""

from __future__ import annotations

import sys

PREPARE_NUMBA_ARGUMENT = "--prepare-numba-cache"


def main() -> int:
    if sys.argv[1:] == [PREPARE_NUMBA_ARGUMENT]:
        from illustrate.rendering.api import prepare_numba_backend

        prepare_numba_backend()
        return 0

    try:
        from illustrate.gui.main_window import run
    except ImportError as exc:
        missing = exc.name or "a GUI dependency"
        raise SystemExit(
            f"The desktop app cannot start: {missing} is not installed. "
            "Run `make env`, or install the `gui` extra."
        ) from exc

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
