"""FastAPI backend skeleton for web rendering."""

from __future__ import annotations


def create_app():
    try:
        from fastapi import FastAPI
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "fastapi is required for the API backend. Install fastapi and retry."
        ) from exc

    app = FastAPI(title="Illustrate API")

    from illustrate_web.api.routes import (
        fetch,
        health,
        presets,
        render as render_route,
        suggest,
        upload,
    )

    app.include_router(render_route.router)
    app.include_router(upload.router)
    app.include_router(fetch.router)
    app.include_router(suggest.router)
    app.include_router(presets.router)
    app.include_router(health.router)
    return app


app = create_app()
