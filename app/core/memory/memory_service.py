from __future__ import annotations

from sqlalchemy import delete, func, select

from app.common.exceptions import UserAuthException
from app.config.settings import settings
from app.core.llm.llm_service import LLMService
from app.core.prompts import Prompts
from app.db.database import get_sessionmaker
from app.db.models import UserChatRecord, UserChatSummary

# 会话摘要生成参数
SUMMARY_MIN_TOTAL = 8       # 少于 8 条消息（约 4 轮）不生成摘要：最近窗口已覆盖
SUMMARY_MIN_NEW = 6         # 距上次摘要新增 ≥6 条才重新生成，避免每轮都调 LLM
SUMMARY_FETCH_LIMIT = 40    # 每次增量总结最多读取的新增消息数
SUMMARY_TIMEOUT_S = 10.0


def _summary_llm_enabled() -> bool:
    def _ok(v: str) -> bool:
        v = (v or "").strip()
        return bool(v) and not (v.startswith("{{") and v.endswith("}}"))
    return _ok(settings.LLM_API_BASE) and _ok(settings.LLM_API_KEY) and _ok(settings.LLM_MODEL_NAME)


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
                {
                    "chat_id": r.chat_id,
                    "role": r.role,
                    "content": r.content,
                    "create_time": str(r.create_time),
                }
                for r in reversed(rows)
            ]

    async def get_latest_user_chat_id(self, user_id: str, session_id: str) -> int | None:
        """查询该会话最新一条用户消息的 chat_id（长期记忆游标用）。"""
        async_session = get_sessionmaker()
        async with async_session() as session:
            res = await session.execute(
                select(UserChatRecord.chat_id)
                .where(
                    UserChatRecord.user_id == user_id,
                    UserChatRecord.session_id == session_id,
                    UserChatRecord.role == "user",
                    UserChatRecord.is_deleted == 0,
                )
                .order_by(UserChatRecord.chat_id.desc())
                .limit(1)
            )
            return res.scalar()

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

    async def get_memory_summary(self, user_id: str, session_id: str) -> str:
        """获取会话摘要（懒生成 + 增量维护）。

        - 已总结到最新消息 → 直接返回缓存，零 LLM 开销。
        - 距上次摘要新增 ≥ SUMMARY_MIN_NEW 条 → 用 LLM 增量合并更新并持久化。
        - 短对话（< SUMMARY_MIN_TOTAL）→ 不生成（最近 4 轮窗口已覆盖）。
        - LLM 不可用 / 生成失败 → 返回已有摘要或空串，绝不阻塞主流程。
        """
        if not user_id or not session_id:
            return ""
        if not _summary_llm_enabled():
            rec = await self._get_summary_record(user_id, session_id)
            return rec.summary if rec else ""

        latest_chat_id = await self._get_latest_chat_id(user_id, session_id)
        if latest_chat_id is None:
            rec = await self._get_summary_record(user_id, session_id)
            return rec.summary if rec else ""

        existing = await self._get_summary_record(user_id, session_id)
        covered = existing.covered_chat_id if existing else 0

        # 缓存命中：距上次摘要不足阈值 → 直接用缓存，零 LLM 调用
        if existing and (latest_chat_id - covered) < SUMMARY_MIN_NEW:
            return existing.summary

        # 首建门槛：消息太少，最近窗口已覆盖，不值得生成
        if not existing and latest_chat_id < SUMMARY_MIN_TOTAL:
            return ""

        new_msgs = await self._get_messages_after(
            user_id, session_id, covered, limit=SUMMARY_FETCH_LIMIT
        )
        if not new_msgs:
            return existing.summary if existing else ""

        summary = await self._summarize(existing.summary if existing else "", new_msgs)
        if not summary:
            return existing.summary if existing else ""

        last_covered = new_msgs[-1][0]  # 实际总结到的 chat_id（而非 latest，避免漏总结被截断部分）
        await self._upsert_summary(user_id, session_id, summary, last_covered)
        return summary

    async def _get_latest_chat_id(self, user_id: str, session_id: str) -> int | None:
        async_session = get_sessionmaker()
        async with async_session() as session:
            res = await session.execute(
                select(func.max(UserChatRecord.chat_id)).where(
                    UserChatRecord.user_id == user_id,
                    UserChatRecord.session_id == session_id,
                    UserChatRecord.is_deleted == 0,
                )
            )
            return res.scalar()

    async def _get_summary_record(self, user_id: str, session_id: str):
        async_session = get_sessionmaker()
        async with async_session() as session:
            res = await session.execute(
                select(UserChatSummary)
                .where(
                    UserChatSummary.user_id == user_id,
                    UserChatSummary.session_id == session_id,
                )
                .order_by(UserChatSummary.id.desc())
                .limit(1)
            )
            return res.scalar_one_or_none()

    async def _get_messages_after(
        self, user_id: str, session_id: str, after_chat_id: int, limit: int
    ) -> list[tuple]:
        async_session = get_sessionmaker()
        async with async_session() as session:
            res = await session.execute(
                select(UserChatRecord.chat_id, UserChatRecord.role, UserChatRecord.content)
                .where(
                    UserChatRecord.user_id == user_id,
                    UserChatRecord.session_id == session_id,
                    UserChatRecord.is_deleted == 0,
                    UserChatRecord.chat_id > after_chat_id,
                )
                .order_by(UserChatRecord.chat_id.asc())
                .limit(limit)
            )
            return [(r[0], r[1], (r[2] or "")) for r in res.all()]

    async def _summarize(self, existing_summary: str, new_msgs: list[tuple]) -> str:
        lines = []
        for _, role, content in new_msgs:
            who = "用户" if role == "user" else "助手"
            lines.append(f"{who}: {content[:500]}")
        new_text = "\n".join(lines)
        if existing_summary:
            user_prompt = (
                f"已有会话摘要：\n{existing_summary}\n\n新增对话：\n{new_text}\n\n"
                "请将新增信息合并进已有摘要，输出更新后的完整摘要。"
            )
        else:
            user_prompt = f"对话内容：\n{new_text}\n\n请输出这段医疗咨询会话的摘要。"
        try:
            llm = LLMService()
            raw = await llm.chat_completion(
                prompt=user_prompt,
                system_prompt=Prompts.get_prompt("SESSION_SUMMARY"),
                stream=False,
                timeout_s=SUMMARY_TIMEOUT_S,
                max_tokens=500,
            )
            return (raw or "").strip()
        except Exception:
            # 生成失败保留旧摘要，不阻塞主流程
            return existing_summary

    async def _upsert_summary(
        self, user_id: str, session_id: str, summary: str, covered_chat_id: int
    ) -> None:
        async_session = get_sessionmaker()
        async with async_session() as session:
            rec = await session.execute(
                select(UserChatSummary)
                .where(
                    UserChatSummary.user_id == user_id,
                    UserChatSummary.session_id == session_id,
                )
                .order_by(UserChatSummary.id.desc())
                .limit(1)
            )
            rec = rec.scalar_one_or_none()
            if rec:
                rec.summary = summary
                rec.covered_chat_id = covered_chat_id
            else:
                session.add(
                    UserChatSummary(
                        user_id=user_id,
                        session_id=session_id,
                        summary=summary,
                        covered_chat_id=covered_chat_id,
                    )
                )
            await session.commit()
