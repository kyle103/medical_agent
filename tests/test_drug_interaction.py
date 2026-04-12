import pytest

from app.db.init_db import ensure_min_csv, import_min_kb, init_schema
from app.db.database import get_engine
from app.core.tools.drug_interaction_tool import DrugInteractionTool


@pytest.mark.asyncio
async def test_drug_interaction_from_kb_only():
    ensure_min_csv()
    engine = get_engine()
    await init_schema(engine)
    await import_min_kb(engine)

    tool = DrugInteractionTool()
    out = await tool.check_interactions(user_id="u1", drug_name_list=["布洛芬", "阿司匹林"], sync_to_archive=False)
    assert "interaction_result" in out
