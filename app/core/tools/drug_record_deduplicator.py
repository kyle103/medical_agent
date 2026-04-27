from __future__ import annotations

import hashlib
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import and_, select

from app.common.logger import get_logger
from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord

logger = get_logger(__name__)


class DrugRecordDeduplicator:
    FUZZY_MATCH_THRESHOLD_DAYS = 3

    @staticmethod
    def compute_idempotent_key(
        user_id: str,
        drug_name: str,
        dosage: str = "",
        frequency: str = "",
        start_date: Optional[date] = None,
    ) -> str:
        payload = f"{user_id}|{drug_name}|{dosage}|{frequency}|{start_date or ''}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    async def check_duplicate(
        user_id: str,
        drug_name: str,
        dosage: str = "",
        frequency: str = "",
        start_date: Optional[date] = None,
    ) -> dict:
        async_session = get_sessionmaker()
        async with async_session() as session:
            conditions = [
                UserDrugRecord.user_id == user_id,
                UserDrugRecord.drug_name == drug_name,
                UserDrugRecord.is_deleted == 0,
            ]

            if start_date:
                threshold = timedelta(days=DrugRecordDeduplicator.FUZZY_MATCH_THRESHOLD_DAYS)
                conditions.append(
                    UserDrugRecord.start_date.between(
                        start_date - threshold,
                        start_date + threshold,
                    )
                )

            if dosage and dosage != "未指定":
                conditions.append(UserDrugRecord.dosage == dosage)

            stmt = select(UserDrugRecord).where(and_(*conditions))
            existing = (await session.execute(stmt)).scalars().first()

            if existing:
                return {
                    "is_duplicate": True,
                    "reason": (
                        f"已存在相似记录：{existing.drug_name} "
                        f"剂量={existing.dosage} 频次={existing.frequency} "
                        f"开始日期={existing.start_date}"
                    ),
                    "existing_record_id": existing.drug_record_id,
                }

            return {"is_duplicate": False}
