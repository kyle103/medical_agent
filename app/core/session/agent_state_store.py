from __future__ import annotations

import json
from typing import Any, Dict

from sqlalchemy import select

from app.db.database import get_sessionmaker
from app.db.models import AgentSessionState


class AgentStateStore:
    """会话级Agent运行状态持久化（最小实现）。"""

    async def get_state(self, *, user_id: str, session_id: str) -> Dict[str, Any]:
        if not user_id or not session_id:
            return {}

        async_session = get_sessionmaker()
        async with async_session() as session:
            stmt = (
                select(AgentSessionState)
                .where(
                    AgentSessionState.user_id == user_id,
                    AgentSessionState.session_id == session_id,
                    AgentSessionState.is_deleted == 0,
                )
                .order_by(AgentSessionState.id.desc())
                .limit(1)
            )
            res = await session.execute(stmt)
            row = res.scalars().first()
            if not row or not row.state_json:
                return {}
            try:
                data = json.loads(row.state_json)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

    async def upsert_state(self, *, user_id: str, session_id: str, state: Dict[str, Any]) -> None:
        if not user_id or not session_id:
            return

        payload = json.dumps(state or {}, ensure_ascii=False)
        async_session = get_sessionmaker()
        async with async_session() as session:
            stmt = (
                select(AgentSessionState)
                .where(
                    AgentSessionState.user_id == user_id,
                    AgentSessionState.session_id == session_id,
                    AgentSessionState.is_deleted == 0,
                )
                .order_by(AgentSessionState.id.desc())
                .limit(1)
            )
            res = await session.execute(stmt)
            row = res.scalars().first()
            if row:
                row.state_json = payload
            else:
                session.add(
                    AgentSessionState(
                        user_id=user_id,
                        session_id=session_id,
                        state_json=payload,
                    )
                )
            await session.commit()

    async def clear_pending_confirmation(self, *, user_id: str, session_id: str) -> None:
        data = await self.get_state(user_id=user_id, session_id=session_id)
        if not data:
            return
        data.pop("pending_confirmation", None)
        await self.upsert_state(user_id=user_id, session_id=session_id, state=data)

