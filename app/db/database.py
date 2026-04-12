from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _sqlite_url() -> str:
    # 允许用户使用相对路径
    db_path = settings.SQLITE_DB_PATH
    if db_path.startswith("{{"):
        # 保持占位符语义：本地启动前用户必须填充
        db_path = os.path.join("data", "sqlite", "medical.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return f"sqlite+aiosqlite:///{db_path}"


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is not None:
        return _engine

    if settings.DB_TYPE == "sqlite" or settings.DB_TYPE.startswith("{{"):
        url = _sqlite_url()
    else:
        # MySQL 生产配置（需用户手动填充）
        url = (
            f"mysql+aiomysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
            f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}?charset=utf8mb4"
        )

    _engine = create_async_engine(url, pool_pre_ping=True)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is not None:
        return _sessionmaker
    _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async_session = get_sessionmaker()
    async with async_session() as session:
        yield session
