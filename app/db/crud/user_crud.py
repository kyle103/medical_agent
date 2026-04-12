from __future__ import annotations

from sqlalchemy import select

from app.db.database import get_sessionmaker
from app.db.models import UserInfo


class UserCRUD:
    async def create_user(self, *, user_id: str, user_nickname: str) -> None:
        async_session = get_sessionmaker()
        async with async_session() as session:
            session.add(UserInfo(user_id=user_id, user_nickname=user_nickname))
            await session.commit()

    async def get_user(self, *, user_id: str) -> UserInfo | None:
        async_session = get_sessionmaker()
        async with async_session() as session:
            res = await session.execute(
                select(UserInfo).where(UserInfo.user_id == user_id, UserInfo.is_deleted == 0)
            )
            return res.scalar_one_or_none()
