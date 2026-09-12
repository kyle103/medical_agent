from __future__ import annotations

import re
from datetime import date, datetime, timedelta, time as dtime
from typing import Optional

from sqlalchemy import and_, select, update

from app.common.logger import get_logger
from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord
from app.core.tools.drug_record_deduplicator import DrugRecordDeduplicator

logger = get_logger(__name__)


class DrugRecordTool:

    _CN_HOUR = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}
    # 中文时段：(默认小时, 是否需对 <12 的具体小时 +12)
    _PERIOD = {
        "凌晨": (1, False), "早上": (7, False), "早晨": (7, False), "上午": (9, False),
        "中午": (12, False), "午间": (12, False),
        "下午": (15, True), "晚上": (20, True), "晚间": (20, True), "半夜": (23, True),
    }

    @staticmethod
    def _parse_intake_time(time_text: str | None) -> tuple[Optional[date], Optional[datetime]]:
        """把"昨天晚上八点"/"下午3点"/"8月8日"/"2024年1月1日"/"昨天15:00"解析为 (start_date, start_time)。

        - start_date: 年月日（Date 精度）
        - start_time: 含时刻的完整 datetime；仅当文本含时间信息时非 None
        """
        t = (time_text or "").strip()
        if not t:
            return None, None
        now = datetime.now()
        base = now.date()
        if "前天" in t:
            base = now.date() - timedelta(days=2)
        elif "昨天" in t:
            base = now.date() - timedelta(days=1)
        elif "今天" in t or "今晚" in t:
            base = now.date()

        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", t)
        if m:
            try:
                base = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                pass
        else:
            m = re.search(r"(\d{1,2})月(\d{1,2})日", t)
            if m:
                try:
                    base = date(now.year, int(m.group(1)), int(m.group(2)))
                except ValueError:
                    pass

        period = None
        for kw, cfg in DrugRecordTool._PERIOD.items():
            if kw in t:
                period = cfg
                break

        hour: Optional[int] = None
        minute = 0
        m = re.search(r"(\d{1,2})\s*[:：点时]\s*(\d{1,2})?", t)
        if m:
            hour = int(m.group(1))
            if m.group(2):
                minute = int(m.group(2))
        else:
            m = re.search(r"([零一二三四五六七八九十]+)\s*点", t)
            if m:
                hour = DrugRecordTool._CN_HOUR.get(m.group(1))

        if hour is None and period:
            hour = period[0]
        elif hour is not None and period and period[1] and hour < 12:
            hour += 12  # 下午/晚上"八点" → 20 点

        if hour is None:
            return base, None
        try:
            return base, datetime.combine(base, dtime(hour % 24, minute or 0))
        except ValueError:
            return base, None

    async def add_record(
        self,
        *,
        user_id: str,
        drug_name: str,
        dosage: str = "",
        frequency: str = "",
        time_text: str = "",
        start_date: Optional[date] = None,
    ) -> dict:
        if not user_id or not drug_name:
            return {"ok": False, "message": "缺少必要参数"}

        parsed_date = start_date
        parsed_time: Optional[datetime] = None
        if time_text:
            d, tm = self._parse_intake_time(time_text)
            if parsed_date is None:
                parsed_date = d
            parsed_time = tm

        dedup = await DrugRecordDeduplicator.check_duplicate(
            user_id=user_id,
            drug_name=drug_name,
            dosage=dosage,
            frequency=frequency,
            start_date=parsed_date,
        )
        if dedup["is_duplicate"]:
            logger.info(
                "drug record dedup skipped user_id=%s drug=%s reason=%s",
                user_id, drug_name, dedup["reason"],
            )
            return {"ok": True, "created": False, "message": dedup["reason"]}

        idempotent_key = DrugRecordDeduplicator.compute_idempotent_key(
            user_id, drug_name, dosage, frequency, parsed_date,
        )

        async_session = get_sessionmaker()
        async with async_session() as session:
            record = UserDrugRecord(
                user_id=user_id,
                drug_name=drug_name,
                dosage=dosage or "未指定",
                frequency=frequency or "未指定",
                start_date=parsed_date,
                start_time=parsed_time,
                intake_time_text=(time_text or "").strip() or None,
                end_date=None,
                idempotent_key=idempotent_key,
                remark=f"用户描述时间: {time_text or '未提供'}",
            )
            session.add(record)
            await session.commit()
            logger.info(
                "drug record created user_id=%s drug=%s start_time=%s key=%s",
                user_id, drug_name, parsed_time, idempotent_key,
            )
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
                    "start_time": str(r.start_time) if r.start_time else None,
                    "intake_time_text": r.intake_time_text,
                    "remark": r.remark,
                }
                for r in rows
            ]

    async def update_by_name(
        self,
        *,
        user_id: str,
        drug_name: str,
        dosage: str | None = None,
        frequency: str | None = None,
        time_text: str | None = None,
    ) -> dict:
        """更新用药记录：默认最近一条同药记录；若 time_text 指定日期则匹配对应日期的记录。

        只更新提供的字段（剂量/频次/时间），不新建记录。
        """
        if not user_id or not drug_name:
            return {"ok": False, "message": "缺少必要参数"}

        target_date = None
        if time_text:
            d, _tm = self._parse_intake_time(time_text)
            target_date = d

        async_session = get_sessionmaker()
        async with async_session() as session:
            stmt = select(UserDrugRecord).where(
                UserDrugRecord.user_id == user_id,
                UserDrugRecord.drug_name == drug_name,
                UserDrugRecord.is_deleted == 0,
            )
            if target_date:
                stmt = stmt.where(UserDrugRecord.start_date == target_date)
            stmt = stmt.order_by(UserDrugRecord.drug_record_id.desc()).limit(1)
            row = (await session.execute(stmt)).scalars().first()
            if not row:
                return {"ok": False, "message": f"未找到{drug_name}的用药记录，可先告诉我'记录一下'。"}

            changed: list[str] = []
            if dosage and dosage != "未指定":
                row.dosage = dosage
                changed.append("剂量")
            if frequency and frequency != "未指定":
                row.frequency = frequency
                changed.append("频次")
            if time_text:
                d2, tm2 = self._parse_intake_time(time_text)
                if d2:
                    row.start_date = d2
                if tm2:
                    row.start_time = tm2
                row.intake_time_text = time_text or row.intake_time_text
                changed.append("时间")

            await session.commit()
            if not changed:
                return {"ok": True, "message": f"已定位到{drug_name}的记录，但未检测到需要更新的字段。"}
            return {"ok": True, "message": f"已更新{drug_name}的记录（{'、'.join(changed)}）。"}

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
            return {"ok": True, "message": f"已删除最近一条'{drug_name}'记录"}
