from __future__ import annotations

from sqlalchemy import delete, select

from app.common.exceptions import UserAuthException
from app.db.database import get_sessionmaker
from app.db.models import UserChatRecord


class MemoryService:
    async def get_user_memory(self, user_id: str, session_id: str, limit: int = 10):
        if not user_id:
            raise UserAuthException("用户ID不能为空")

        async_session = get_sessionmaker()
        async with async_session() as session:
            res = await session.execute(
                select(UserChatRecord)
                .where(
                    UserChatRecord.user_id == user_id,
                    UserChatRecord.session_id == session_id,
                    UserChatRecord.is_deleted == 0,
                )
                .order_by(UserChatRecord.chat_id.desc())
                .limit(limit)
            )
            rows = list(res.scalars().all())
            return [
                {"role": r.role, "content": r.content, "create_time": str(r.create_time)}
                for r in reversed(rows)
            ]

    async def update_user_memory(self, user_id: str, session_id: str, role: str, content: str):
        if not user_id:
            raise UserAuthException("用户ID不能为空")

        async_session = get_sessionmaker()
        async with async_session() as session:
            session.add(
                UserChatRecord(user_id=user_id, session_id=session_id, role=role, content=content)
            )
            await session.commit()

    async def clear_user_memory(self, user_id: str, session_id: str | None = None):
        if not user_id:
            raise UserAuthException("用户ID不能为空")

        async_session = get_sessionmaker()
        async with async_session() as session:
            q = delete(UserChatRecord).where(UserChatRecord.user_id == user_id)
            if session_id:
                q = q.where(UserChatRecord.session_id == session_id)
            await session.execute(q)
            await session.commit()

    async def get_memory_summary(self, user_id: str, session_id: str):
        # MVP：不生成摘要（避免未配置LLM时阻塞）；可在生产环境接入 LLM 总结
        return ""
