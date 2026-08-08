"""FastAPI application factory."""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy import text

from socbench.db import engine, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    try:
        yield
    finally:
        await engine.dispose()


def _cors_origins() -> list[str]:
    raw_origins = os.getenv("CORS_ORIGINS", "")
    production = os.getenv("APP_ENV", "development").lower() == "production"
    if not raw_origins and production:
        return []
    origins = [
        origin.strip()
        for origin in (raw_origins or "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ]
    if production and "*" in origins:
        raise RuntimeError("CORS_ORIGINS must list explicit origins in production")
    return origins


def _trusted_hosts() -> list[str]:
    raw_hosts = os.getenv("TRUSTED_HOSTS", "")
    if not raw_hosts and os.getenv("APP_ENV", "development").lower() == "production":
        raise RuntimeError("TRUSTED_HOSTS is required when APP_ENV=production")
    return [
        host.strip()
        for host in (raw_hosts or "localhost,127.0.0.1,testserver").split(",")
        if host.strip()
    ]


def create_app() -> FastAPI:
    cors_origins = _cors_origins()
    production = os.getenv("APP_ENV", "development").lower() == "production"
    app = FastAPI(
        title="Socbench API",
        description="Scientific dataset intelligence. The unexamined dataset is not worth training on.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_trusted_hosts())
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Accept", "Authorization", "Content-Type", "X-Request-ID"],
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail="database unavailable") from exc
        return {"status": "ready"}

    from socbench.api.routes import router

    app.include_router(router, prefix="/api")

    return app


app = create_app()
