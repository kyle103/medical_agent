from __future__ import annotations

from datetime import date

from sqlalchemy import insert

from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord, UserLabReportItem, UserLabReportRecord


class ArchiveCRUD:
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
