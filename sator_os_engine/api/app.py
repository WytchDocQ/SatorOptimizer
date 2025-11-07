from __future__ import annotations

from fastapi import FastAPI

from ..settings import Settings
from ..security.ip_filters import IPFilterMiddleware
from .routes.optimize import router as optimize_router
from .routes.reconstruct import router as reconstruct_router
from .routes.jobs import router as jobs_router
from .errors import register_error_handlers


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="SATOR OS Engine", version="0.1.0")

    # Middlewares
    app.add_middleware(IPFilterMiddleware, settings=settings)

    # Routes
    app.include_router(optimize_router, prefix="/v1", tags=["optimize"])
    app.include_router(reconstruct_router, prefix="/v1", tags=["reconstruct"])
    app.include_router(jobs_router, prefix="/v1", tags=["jobs"])
    register_error_handlers(app)

    @app.get("/livez")
    def livez() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict:
        return {"status": "ready"}

    return app


