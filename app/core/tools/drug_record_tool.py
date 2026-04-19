from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy import and_, select, update

from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord


class DrugRecordTool:
    """用药记录工具：确定性 CRUD 能力。"""

    @staticmethod
    def _parse_date_text(time_text: str | None) -> Optional[date]:
        t = (time_text or "").strip()
        if not t:
            return None
        today = date.today()
        if "今天" in t:
            return today
        if "昨天" in t:
            return today - timedelta(days=1)
        if "前天" in t:
            return today - timedelta(days=2)
        return None

    async def add_record(
        self,
        *,
        user_id: str,
        drug_name: str,
        dosage: str = "",
        frequency: str = "",
        time_text: str = "",
    ) -> dict:
        if not user_id or not drug_name:
            return {"ok": False, "message": "缺少必要参数"}

        async_session = get_sessionmaker()
        async with async_session() as session:
            stmt = select(UserDrugRecord).where(
                and_(
                    UserDrugRecord.user_id == user_id,
                    UserDrugRecord.drug_name == drug_name,
                    UserDrugRecord.is_deleted == 0,
                )
            )
            existing = (await session.execute(stmt)).scalars().first()
            if existing:
                return {"ok": True, "created": False, "message": "用药记录已存在，无需重复添加。"}

            parsed_date = self._parse_date_text(time_text)
            record = UserDrugRecord(
                user_id=user_id,
                drug_name=drug_name,
                dosage=dosage or "未指定",
                frequency=frequency or "未指定",
                start_date=parsed_date,
                end_date=None,
                remark=f"用户描述时间: {time_text or '未提供'}",
            )
            session.add(record)
            await session.commit()
            return {"ok": True, "created": True, "message": "已添加用药记录"}

    async def list_recent(self, *, user_id: str, limit: int = 10) -> list[dict]:
        async_session = get_sessionmaker()
        async with async_session() as session:
            stmt = (
                select(UserDrugRecord)
                .where(UserDrugRecord.user_id == user_id, UserDrugRecord.is_deleted == 0)
                .order_by(UserDrugRecord.drug_record_id.desc())
                .limit(limit)
            )
            rows = list((await session.execute(stmt)).scalars().all())
            return [
                {
                    "drug_record_id": r.drug_record_id,
                    "drug_name": r.drug_name,
                    "dosage": r.dosage,
                    "frequency": r.frequency,
                    "start_date": str(r.start_date) if r.start_date else None,
                    "remark": r.remark,
                }
                for r in rows
            ]

    async def soft_delete_latest_by_name(self, *, user_id: str, drug_name: str) -> dict:
        if not user_id or not drug_name:
            return {"ok": False, "message": "缺少参数"}

        async_session = get_sessionmaker()
        async with async_session() as session:
            stmt = (
                select(UserDrugRecord)
                .where(
                    UserDrugRecord.user_id == user_id,
                    UserDrugRecord.drug_name == drug_name,
                    UserDrugRecord.is_deleted == 0,
                )
                .order_by(UserDrugRecord.drug_record_id.desc())
                .limit(1)
            )
            row = (await session.execute(stmt)).scalars().first()
            if not row:
                return {"ok": False, "message": "未找到可删除的记录"}

            await session.execute(
                update(UserDrugRecord)
                .where(UserDrugRecord.drug_record_id == row.drug_record_id)
                .values(is_deleted=1)
            )
            await session.commit()
            return {"ok": True, "message": f"已删除最近一条“{drug_name}”记录"}

