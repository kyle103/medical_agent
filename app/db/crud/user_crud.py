from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from app.db.database import get_sessionmaker
from app.db.models import UserInfo


class UserCRUD:
    """用户数据访问层（异步）。

    说明：
    - 本项目使用 async SQLAlchemy + SQLite（aiosqlite）。
    - 这里提供 user_router 需要的最小能力：创建用户、按 user_id/phone 查询。
    """

    async def create_user(
        self,
        *,
        user_id: str,
        user_nickname: str,
        phone: str | None = None,
        password_hash: str | None = None,
    ) -> UserInfo:
        async_session = get_sessionmaker()
        async with async_session() as session:
            user = UserInfo(
                user_id=user_id,
                user_nickname=user_nickname,
                phone=phone,
                password_hash=password_hash,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def get_user(self, *, user_id: str) -> Optional[UserInfo]:
        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(
                select(UserInfo).where(UserInfo.user_id == user_id)
            )
            return result.scalars().first()

    async def get_user_by_phone(self, *, phone: str) -> Optional[UserInfo]:
        async_session = get_sessionmaker()
        async with async_session() as session:
            result = await session.execute(select(UserInfo).where(UserInfo.phone == phone))
            return result.scalars().first()
