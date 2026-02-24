"""PySide6 desktop shell package."""


def __getattr__(name: str):
    if name == "main":
        from illustrate_gui.main import main

        return main
    raise AttributeError(name)

__all__ = ["main"]
