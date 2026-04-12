import pytest

from app.db.init_db import ensure_min_csv, import_min_kb, init_schema
from app.db.database import get_engine
from app.core.tools.lab_report_tool import LabReportTool


@pytest.mark.asyncio
async def test_lab_report_range_compare():
    ensure_min_csv()
    engine = get_engine()
    await init_schema(engine)
    await import_min_kb(engine)

    tool = LabReportTool()
    out = await tool.interpret(
        user_id="u1",
        lab_item_list=[{"item_name": "血糖", "test_value": "7.0", "unit": "mmol/L"}],
        sync_to_archive=False,
    )
    assert out["item_list"][0]["abnormal_flag"] == "H"
