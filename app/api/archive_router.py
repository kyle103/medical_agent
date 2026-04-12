from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.tools.archive_query_tool import ArchiveQueryTool
from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord, UserLabReportItem, UserLabReportRecord
from app.schema.base import APIResponse

router = APIRouter()


class ArchiveListResponse(BaseModel):
    archive_type: Literal["drug", "lab"]
    items: list[dict]


class ArchiveSearchRequest(BaseModel):
    q: str = Field(default="", description="搜索关键词（药名/化验项目名）")
    archive_type: Optional[Literal["drug", "lab", "all"]] = Field(default="all", description="检索范围")
    limit: int = Field(default=20, ge=1, le=100, description="返回条数")


class DrugAddPayload(BaseModel):
    drug_name: str = Field(..., min_length=1, max_length=128, description="药品名称")
    drug_alias: Optional[str] = Field(default=None, max_length=256, description="别名")
    dosage: Optional[str] = Field(default=None, max_length=64, description="剂量")
    frequency: Optional[str] = Field(default=None, max_length=64, description="频次")
    start_date: Optional[date] = Field(default=None, description="开始日期 YYYY-MM-DD")
    end_date: Optional[date] = Field(default=None, description="结束日期 YYYY-MM-DD")
    prescribe_hospital: Optional[str] = Field(default=None, max_length=128, description="开方医院")
    remark: Optional[str] = Field(default=None, description="备注")


class LabItemAddPayload(BaseModel):
    item_name: str = Field(..., min_length=1, max_length=64, description="检验项目名称")
    item_en_name: Optional[str] = Field(default=None, max_length=64, description="英文名")
    test_value: str = Field(..., min_length=1, max_length=32, description="结果值")
    unit: Optional[str] = Field(default=None, max_length=32, description="单位")
    reference_range: Optional[str] = Field(default=None, max_length=64, description="参考范围")
    abnormal_flag: Optional[str] = Field(default=None, max_length=8, description="异常标记，如 H/L")


class LabReportAddPayload(BaseModel):
    report_name: str = Field(..., min_length=1, max_length=128, description="报告名称")
    test_time: date = Field(..., description="检验日期 YYYY-MM-DD")
    test_organization: Optional[str] = Field(default=None, max_length=128, description="检验机构")
    report_content: Optional[str] = Field(default=None, description="报告原文/摘要")
    remark: Optional[str] = Field(default=None, description="备注")
    items: list[LabItemAddPayload] = Field(default_factory=list, description="报告条目")


class ArchiveAddRequest(BaseModel):
    drug: Optional[DrugAddPayload] = Field(default=None, description="添加 drug 时填写")
    lab: Optional[LabReportAddPayload] = Field(default=None, description="添加 lab 时填写")


class ArchiveAddResponse(BaseModel):
    archive_type: Literal["drug", "lab"]
    id: int = Field(..., description="drug_record_id 或 report_id")


@router.post("/{archive_type}/add", response_model=APIResponse[ArchiveAddResponse])
async def add_archive(
    archive_type: Literal["drug", "lab"], payload: ArchiveAddRequest, request: Request
):
    user_id = getattr(request.state, "user_id", "")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    async_session = get_sessionmaker()
    async with async_session() as session:
        async with session.begin():
            if archive_type == "drug":
                if payload.drug is None:
                    raise HTTPException(status_code=422, detail="Missing field: drug")

                r = UserDrugRecord(
                    user_id=user_id,
                    drug_name=payload.drug.drug_name.strip(),
                    drug_alias=(payload.drug.drug_alias.strip() if payload.drug.drug_alias else None),
                    dosage=payload.drug.dosage,
                    frequency=payload.drug.frequency,
                    start_date=payload.drug.start_date,
                    end_date=payload.drug.end_date,
                    prescribe_hospital=payload.drug.prescribe_hospital,
                    remark=payload.drug.remark,
                    is_deleted=0,
                )
                session.add(r)
                await session.flush()
                new_id = int(r.drug_record_id)
                resp = ArchiveAddResponse(archive_type="drug", id=new_id)

            else:
                if payload.lab is None:
                    raise HTTPException(status_code=422, detail="Missing field: lab")

                rep = UserLabReportRecord(
                    user_id=user_id,
                    report_name=payload.lab.report_name.strip(),
                    test_time=payload.lab.test_time,
                    test_organization=payload.lab.test_organization,
                    report_content=payload.lab.report_content,
                    remark=payload.lab.remark,
                    is_deleted=0,
                )
                session.add(rep)
                await session.flush()

                report_id = int(rep.report_id)
                for it in payload.lab.items:
                    item = UserLabReportItem(
                        report_id=report_id,
                        user_id=user_id,
                        item_name=it.item_name.strip(),
                        item_en_name=it.item_en_name,
                        test_value=it.test_value,
                        unit=it.unit,
                        reference_range=it.reference_range,
                        abnormal_flag=it.abnormal_flag,
                        is_deleted=0,
                    )
                    session.add(item)

                resp = ArchiveAddResponse(archive_type="lab", id=report_id)

    return APIResponse(data=resp, request_id=getattr(request.state, "request_id", ""))


@router.get("/{archive_type}/list", response_model=APIResponse[ArchiveListResponse])
async def list_archive(archive_type: Literal["drug", "lab"], request: Request):
    user_id = getattr(request.state, "user_id", "")

    async_session = get_sessionmaker()
    async with async_session() as session:
        if archive_type == "drug":
            stmt = (
                select(UserDrugRecord)
                .where(UserDrugRecord.user_id == user_id, UserDrugRecord.is_deleted == 0)
                .order_by(UserDrugRecord.drug_record_id.desc())
                .limit(50)
            )
            rows = list((await session.execute(stmt)).scalars().all())
            items = [
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
            ]
        else:
            stmt = (
                select(UserLabReportRecord)
                .where(UserLabReportRecord.user_id == user_id, UserLabReportRecord.is_deleted == 0)
                .order_by(UserLabReportRecord.report_id.desc())
                .limit(20)
            )
            reports = list((await session.execute(stmt)).scalars().all())

            items = []
            for rep in reports:
                it_stmt = (
                    select(UserLabReportItem)
                    .where(UserLabReportItem.user_id == user_id, UserLabReportItem.report_id == rep.report_id)
                    .order_by(UserLabReportItem.item_id.asc())
                )
                rep_items = list((await session.execute(it_stmt)).scalars().all())
                items.append(
                    {
                        "report_id": rep.report_id,
                        "report_name": rep.report_name,
                        "test_time": str(rep.test_time) if rep.test_time else None,
                        "create_time": str(rep.create_time),
                        "items": [
                            {
                                "item_id": it.item_id,
                                "item_name": it.item_name,
                                "test_value": it.test_value,
                                "unit": it.unit,
                                "reference_range": it.reference_range,
                                "abnormal_flag": it.abnormal_flag,
                            }
                            for it in rep_items
                        ],
                    }
                )

    return APIResponse(
        data=ArchiveListResponse(archive_type=archive_type, items=items),
        request_id=getattr(request.state, "request_id", ""),
    )


@router.post("/search", response_model=APIResponse[dict])
async def search_archive(payload: ArchiveSearchRequest, request: Request):
    user_id = getattr(request.state, "user_id", "")
    q = (payload.q or "").strip()

    async_session = get_sessionmaker()
    async with async_session() as session:
        out: dict = {"q": q, "drug": [], "lab": []}

        if payload.archive_type in ("all", "drug"):
            if q:
                stmt = (
                    select(UserDrugRecord)
                    .where(
                        UserDrugRecord.user_id == user_id,
                        UserDrugRecord.is_deleted == 0,
                        UserDrugRecord.drug_name.like(f"%{q}%"),
                    )
                    .order_by(UserDrugRecord.drug_record_id.desc())
                    .limit(payload.limit)
                )
                rows = list((await session.execute(stmt)).scalars().all())
                out["drug"] = [
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
                ]
            else:
                tool = ArchiveQueryTool()
                tool_res = await tool.query_recent_drugs(user_id=user_id, days=7, limit=min(payload.limit, 20))
                out["drug"] = tool_res.get("items") or []

        if payload.archive_type in ("all", "lab") and q:
            it_stmt = (
                select(UserLabReportItem)
                .where(UserLabReportItem.user_id == user_id, UserLabReportItem.item_name.like(f"%{q}%"))
                .order_by(UserLabReportItem.item_id.desc())
                .limit(payload.limit)
            )
            items = list((await session.execute(it_stmt)).scalars().all())

            for it in items:
                rep_stmt = (
                    select(UserLabReportRecord)
                    .where(
                        UserLabReportRecord.user_id == user_id,
                        UserLabReportRecord.report_id == it.report_id,
                        UserLabReportRecord.is_deleted == 0,
                    )
                    .limit(1)
                )
                rep = (await session.execute(rep_stmt)).scalars().first()
                out["lab"].append(
                    {
                        "report_id": it.report_id,
                        "report_name": rep.report_name if rep else None,
                        "test_time": str(rep.test_time) if rep and rep.test_time else None,
                        "item_id": it.item_id,
                        "item_name": it.item_name,
                        "test_value": it.test_value,
                        "unit": it.unit,
                        "reference_range": it.reference_range,
                        "abnormal_flag": it.abnormal_flag,
                    }
                )

    return APIResponse(data=out, request_id=getattr(request.state, "request_id", ""))
