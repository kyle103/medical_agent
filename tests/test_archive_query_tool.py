import uuid

import pytest

from sqlalchemy import delete

from app.db.database import get_engine, get_sessionmaker
from app.db.init_db import init_schema
from app.db.models import UserDrugRecord, UserInfo
from app.core.tools.archive_query_tool import ArchiveQueryTool


@pytest.mark.asyncio
async def test_archive_query_recent_drugs_returns_items():
    engine = get_engine()
    await init_schema(engine)

    async_session = get_sessionmaker()
    user_id = f"u_{uuid.uuid4().hex}"

    # Ensure clean rows for this user (in case a shared sqlite file is used)
    async with async_session() as session:
        await session.execute(delete(UserDrugRecord).where(UserDrugRecord.user_id == user_id))
        await session.execute(delete(UserInfo).where(UserInfo.user_id == user_id))
        session.add(UserInfo(user_id=user_id, user_nickname="t"))
        session.add(
            UserDrugRecord(
                user_id=user_id,
                drug_name="阿司匹林",
                dosage="100mg",
                frequency="qd",
            )
        )
        session.add(
            UserDrugRecord(
                user_id=user_id,
                drug_name="布洛芬",
                dosage="200mg",
                frequency="bid",
            )
        )
        await session.commit()

    tool = ArchiveQueryTool()
    out = await tool.query_recent_drugs(user_id=user_id, days=7, limit=20)

    assert "final_desc" in out
    assert "items" in out
    assert len(out["items"]) >= 2
    assert any(i.get("drug_name") == "阿司匹林" for i in out["items"])
