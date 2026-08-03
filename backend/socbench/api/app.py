"""FastAPI application factory."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from socbench.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


def _cors_origins() -> list[str]:
    origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    return origins or ["*"]


def create_app() -> FastAPI:
    cors_origins = _cors_origins()
    app = FastAPI(
        title="Socbench API",
        description="Scientific dataset intelligence. The unexamined dataset is not worth training on.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    from socbench.api.routes import router

    app.include_router(router, prefix="/api")

    return app


app = create_app()
