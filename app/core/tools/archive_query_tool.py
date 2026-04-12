from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import Select, select

from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord


class ArchiveQueryTool:
    """档案查询工具（MVP，仅查询用户自己录入/同步的用药记录）。

    合规说明：
    - 仅返回用户数据，不做诊断推断
    - 如果没有记录，返回引导用户通过档案接口录入
    """

    async def query_recent_drugs(self, *, user_id: str, days: int = 7, limit: int = 20) -> dict:
        if days < 1:
            days = 1
        if days > 365:
            days = 365

        since = date.today() - timedelta(days=days)

        async_session = get_sessionmaker()
        async with async_session() as session:
            stmt: Select[tuple[UserDrugRecord]] = (
                select(UserDrugRecord)
                .where(
                    UserDrugRecord.user_id == user_id,
                    UserDrugRecord.is_deleted == 0,
                )
                .order_by(UserDrugRecord.drug_record_id.desc())
                .limit(limit)
            )
            rows = list((await session.execute(stmt)).scalars().all())

        # 目前表里没有严格的用药时间字段（start_date/end_date 可能为空），
        # MVP 先按创建时间/插入顺序返回最近 N 条。
        if not rows:
            return {
                "final_desc": "我没有在你的用药档案中查询到记录。你可以先通过档案接口录入用药记录（药名、时间等），之后我才能帮你按时间回溯查询。",
                "items": [],
            }

        names = [r.drug_name for r in rows if r.drug_name]
        uniq = []
        seen = set()
        for n in names:
            if n not in seen:
                uniq.append(n)
                seen.add(n)

        return {
            "final_desc": "根据你的用药档案，最近记录的药品包括：" + "、".join(uniq[:10]) + "。如需更精确，请告诉我大致日期范围或补充录入用药时间。",
            "items": [
                {
                    "drug_record_id": r.drug_record_id,
                    "drug_name": r.drug_name,
                    "dosage": r.dosage,
                    "frequency": r.frequency,
                    "start_date": str(r.start_date) if r.start_date else None,
                    "end_date": str(r.end_date) if r.end_date else None,
                    "create_time": str(r.create_time),
                }
                for r in rows
            ],
        }
