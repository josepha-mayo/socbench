"""Async SQLAlchemy engine and session factory.

Defaults to a local SQLite file so Socbench runs with zero external
infrastructure. Set DATABASE_URL to a postgresql+asyncpg:// URL to use
Postgres in production.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DEFAULT_SQLITE = "sqlite+aiosqlite:///./socbench.db"

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE)

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    connect_args=connect_args,
)

async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    from socbench.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
