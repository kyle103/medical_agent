from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import Select, select

from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord, UserLabReportItem, UserLabReportRecord


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

    async def query(self, *, user_id: str, query_type: str, query_conditions: dict) -> dict:
        """查询档案信息
        
        Args:
            user_id: 用户ID
            query_type: 查询类型，包括：general, visit_records, drug_records, lab_records, basic_info
            query_conditions: 查询条件
            
        Returns:
            查询结果
        """
        if query_type == "drug_records" or query_type == "general":
            # 对于用药记录查询，调用已有的query_recent_drugs方法
            return await self.query_recent_drugs(user_id=user_id, days=30, limit=20)
        elif query_type == "lab_records":
            # 处理化验记录查询
            async_session = get_sessionmaker()
            async with async_session() as session:
                # 查询用户的化验记录
                stmt = (
                    select(UserLabReportRecord)
                    .where(
                        UserLabReportRecord.user_id == user_id,
                        UserLabReportRecord.is_deleted == 0,
                    )
                    .order_by(UserLabReportRecord.report_id.desc())
                    .limit(10)
                )
                reports = list((await session.execute(stmt)).scalars().all())
                
                if not reports:
                    return {
                        "final_desc": "我没有在你的档案中查询到化验记录。你可以先通过档案接口录入化验记录，之后我才能帮你查询。",
                        "items": [],
                    }
                
                # 构建化验记录结果
                items = []
                report_names = []
                for report in reports:
                    report_names.append(report.report_name)
                    # 查询该化验报告的具体项目
                    item_stmt = (
                        select(UserLabReportItem)
                        .where(
                            UserLabReportItem.report_id == report.report_id,
                            UserLabReportItem.is_deleted == 0,
                        )
                    )
                    report_items = list((await session.execute(item_stmt)).scalars().all())
                    
                    for item in report_items:
                        items.append({
                            "report_id": report.report_id,
                            "report_name": report.report_name,
                            "test_time": str(report.test_time) if report.test_time else None,
                            "item_id": item.item_id,
                            "item_name": item.item_name,
                            "test_value": item.test_value,
                            "unit": item.unit,
                            "reference_range": item.reference_range,
                            "abnormal_flag": item.abnormal_flag,
                        })
                
                return {
                    "final_desc": "根据你的档案，最近的化验记录包括：" + "、".join(report_names[:5]) + "。",
                    "items": items,
                }
        else:
            # 其他查询类型，返回通用信息
            return {
                "final_desc": "目前仅支持用药记录和化验记录查询。你可以通过档案接口录入相关信息，之后我才能帮你查询。",
                "items": [],
            }
