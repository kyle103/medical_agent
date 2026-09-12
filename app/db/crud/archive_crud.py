from __future__ import annotations

from datetime import date

from sqlalchemy import insert, select

from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord, UserLabReportItem, UserLabReportRecord


class ArchiveCRUD:
    async def list_drug_entries(self, *, user_id: str, limit: int = 60) -> list[dict]:
        """列出用户档案中的用药记录（按药名聚合，带时间与次数元数据）。

        用于跨轮冲突检测。与「只给药名」不同，这里必须带来源信息——
        提醒时要能告诉用户"这个药是哪天记录的、记过几次"，而不是和本轮
        提到的药混为一谈。

        硬排除：`end_date` 已过的记录视为明确结束，不参与。
        """
        if not user_id:
            return []
        today = date.today()
        async_session = get_sessionmaker()
        async with async_session() as session:
            stmt = (
                select(
                    UserDrugRecord.drug_name,
                    UserDrugRecord.start_date,
                    UserDrugRecord.create_time,
                    UserDrugRecord.end_date,
                )
                .where(UserDrugRecord.user_id == user_id, UserDrugRecord.is_deleted == 0)
                .order_by(UserDrugRecord.drug_record_id.desc())
                .limit(limit)
            )
            res = await session.execute(stmt)
            rows = res.all()

        agg: dict[str, dict] = {}
        for name, start_date, create_time, end_date in rows:
            name = str(name or "").strip()
            if not name:
                continue
            if end_date is not None and end_date < today:
                continue  # 明确已结束
            d = start_date or (create_time.date() if create_time is not None else None)
            entry = agg.get(name)
            if entry is None:
                agg[name] = {"latest_date": d, "record_count": 1}
            else:
                entry["record_count"] += 1
                if d and (entry["latest_date"] is None or d > entry["latest_date"]):
                    entry["latest_date"] = d

        out: list[dict] = []
        for name, entry in agg.items():
            d = entry["latest_date"]
            out.append({
                "name": name,
                "record_date": d.isoformat() if d else None,
                "days_ago": (today - d).days if d else None,
                "record_count": entry["record_count"],
            })
        out.sort(key=lambda e: (e["days_ago"] if e["days_ago"] is not None else 10**6))
        return out
    async def sync_drugs(self, *, user_id: str, drug_names: list[str]) -> None:
        async_session = get_sessionmaker()
        async with async_session() as session:
            for dn in drug_names:
                session.add(UserDrugRecord(user_id=user_id, drug_name=dn))
            await session.commit()

    async def sync_lab_items(self, *, user_id: str, items: list[dict]) -> None:
        # 简化：每次同步创建一条化验单记录，并写入 items
        async_session = get_sessionmaker()
        async with async_session() as session:
            report = UserLabReportRecord(
                user_id=user_id,
                report_name="化验单（由解读同步）",
                test_time=date.today(),
                report_content=None,
            )
            session.add(report)
            await session.flush()

            for it in items:
                session.add(
                    UserLabReportItem(
                        report_id=report.report_id,
                        user_id=user_id,
                        item_name=it.get("item_name") or "",
                        item_en_name=None,
                        test_value=str(it.get("test_value") or ""),
                        unit=None,
                        reference_range=it.get("reference_range"),
                        abnormal_flag=it.get("abnormal_flag"),
                    )
                )
            await session.commit()
