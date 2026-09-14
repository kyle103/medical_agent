"""化验参考库与判定逻辑的契约测试。

覆盖三块容易被改坏的地方：

1. **参考范围解析** —— 双侧 / 仅上限 / 仅下限 / 不识别（后者必须"不做判定"，不能猜）；
2. **别名匹配** —— 文本解析器输出短名（白细胞 / 转氨酶），化验单上印正式名（白细胞计数 /
   丙氨酸氨基转移酶）。没有别名这一路，抽出来的名字一个都匹配不上；
3. **端到端契约** —— 解析器 → 参考库串起来测。只测单边会出现"两边各自绿、接口对不上"，
   这正是本仓库反复踩过的"测试断言的不是它声称的契约"。
"""

import pytest

from app.core.tools.lab_report_tool import LabReportTool, _parse_reference_range
from app.db.database import get_engine, get_sessionmaker
from app.db.init_db import ensure_min_csv, import_min_kb, init_schema
from app.db.models import LabItemReferenceBase


async def _prepare() -> None:
    ensure_min_csv()
    engine = get_engine()
    await init_schema(engine)
    await import_min_kb(engine)


def test_parse_reference_range_forms():
    """三种可判定形态；其余一律 `(None, None)`，绝不猜。"""
    assert _parse_reference_range("3.9-6.1") == (3.9, 6.1)
    assert _parse_reference_range(" 3.9 - 6.1 ") == (3.9, 6.1)
    assert _parse_reference_range("<5.2") == (None, 5.2)
    assert _parse_reference_range("≤5.2") == (None, 5.2)
    assert _parse_reference_range("<=5.2") == (None, 5.2)
    assert _parse_reference_range(">1.0") == (1.0, None)
    assert _parse_reference_range("≥1.0") == (1.0, None)
    assert _parse_reference_range(">=1.0") == (1.0, None)

    # 不可识别的写法必须返回 (None, None)：宁可"不做判定"，不可猜
    for bad in ("阴性", "男：130-175 女：115-150", "", "5.2", "3.9~6.1", "3.9—6.1"):
        assert _parse_reference_range(bad) == (None, None), bad


@pytest.mark.asyncio
async def test_lab_report_range_compare():
    """回归：双侧区间仍能判出 H。"""
    await _prepare()
    out = await LabReportTool().interpret(
        user_id="u1",
        lab_item_list=[{"item_name": "血糖", "test_value": "7.0", "unit": "mmol/L"}],
        sync_to_archive=False,
    )
    assert out["item_list"][0]["abnormal_flag"] == "H"


@pytest.mark.asyncio
async def test_onesided_range_produces_flag():
    """仅上限 / 仅下限的指标必须能判出 H / L —— 血脂四项在化验单上就是这么印的。

    若只支持 `low-high`，甘油三酯（`<1.7`）与高密度脂蛋白（`>1.0`）会走到
    "数值或参考范围格式无法解析"分支，等于收了数据却给不出判定。
    """
    await _prepare()
    out = await LabReportTool().interpret(
        user_id="u1",
        lab_item_list=[
            {"item_name": "甘油三酯", "test_value": "2.5", "unit": "mmol/L"},
            {"item_name": "高密度脂蛋白胆固醇", "test_value": "0.8", "unit": "mmol/L"},
            {"item_name": "总胆固醇", "test_value": "4.0", "unit": "mmol/L"},
        ],
        sync_to_archive=False,
    )
    flags = {i["item_name"]: i["abnormal_flag"] for i in out["item_list"]}
    assert flags == {
        "甘油三酯": "H",
        "高密度脂蛋白胆固醇": "L",
        "总胆固醇": "N",
    }


@pytest.mark.asyncio
async def test_alias_and_english_match():
    """别名 / 英文缩写必须能命中参考库，且大小写不敏感。"""
    from app.core.rag.lab_reference_service import LabReferenceService

    await _prepare()
    got: dict[str, str | None] = {}
    for r in await LabReferenceService().match_items(
        ["白细胞", "WBC", "wbc", "转氨酶", "ALT", "胆固醇", "高密度脂蛋白", "不存在的指标"]
    ):
        got[r["query"]] = (r["match"] or {}).get("item_name")

    assert got["白细胞"] == "白细胞计数"
    assert got["WBC"] == "白细胞计数"
    assert got["wbc"] == "白细胞计数"
    assert got["转氨酶"] == "丙氨酸氨基转移酶"
    assert got["ALT"] == "丙氨酸氨基转移酶"
    assert got["胆固醇"] == "总胆固醇"
    assert got["高密度脂蛋白"] == "高密度脂蛋白胆固醇"
    assert got["不存在的指标"] is None


@pytest.mark.asyncio
async def test_lab_parser_output_matches_reference_library():
    """端到端：解析器真能抽出指标 **且** 抽出的名字真能匹配上参考库。

    单独测解析、单独测匹配，两边都可能各自绿而接口对不上 —— 所以串起来测。
    这里用解析器白名单里的短名（白细胞 / 血小板 / 转氨酶），
    它们与化验单上的正式名不同，是别名机制存在的理由。
    """
    from app.core.rag.lab_reference_service import LabReferenceService
    from app.core.tools.lab_item_parser import parse_lab_items

    await _prepare()
    items = parse_lab_items("血常规：白细胞11.2，血小板150；转氨酶45")
    assert items, "解析器没抽出任何条目"
    matches = await LabReferenceService().match_items([i["item_name"] for i in items])
    missed = [m["query"] for m in matches if not m["match"]]
    assert not missed, f"解析器抽出的名字在参考库里匹配不上：{missed}"


@pytest.mark.asyncio
async def test_single_char_lab_alias_not_in_entity_dictionary():
    """参考库的**单字别名不进实体词典**，但参考库自己仍接受它。

    实体词典是在**原文**上做前向最大匹配，而单字化学元素名恰好是药品名的组成部分：
    实测加入「钙」之后，「钙片要吃吗」被解析成化验项「总钙」，「氯化钾」也会命中「钾」。
    而 `LabReferenceService.match_items` 的入参是**已经抽好的指标名**，
    不存在在自由文本里乱匹配的问题，所以那里保留单字别名。

    **两个口径不同是有意的**：改动任一侧前先想清楚过滤发生在哪一层。
    """
    from app.core.rag.entity_dictionary import reload_entity_dictionary
    from app.core.rag.lab_reference_service import LabReferenceService

    await _prepare()
    d = await reload_entity_dictionary(log=False)
    assert d.resolve("钙片要吃吗") == []
    assert d.resolve("氯化钾缓释片") == []
    # 正常指标名/短名仍要能解析
    assert d.resolve("血钾偏低") == ["钾"]
    assert d.resolve("我血红蛋白有点低") == ["血红蛋白"]

    # 参考库侧不受影响
    got = await LabReferenceService().match_items(["钙", "血钙", "K"])
    assert [r["match"]["item_name"] for r in got] == ["总钙", "总钙", "钾"]


@pytest.mark.asyncio
async def test_import_is_idempotent():
    """重复导入不产生重复行。"""
    from sqlalchemy import func, select

    await _prepare()

    async def _count() -> int:
        async with get_sessionmaker()() as s:
            return (
                await s.execute(select(func.count()).select_from(LabItemReferenceBase))
            ).scalar()

    before = await _count()
    await import_min_kb(get_engine())
    assert await _count() == before
    assert before > 20, "参考库条目数过少，扩表可能没生效"


@pytest.mark.asyncio
async def test_import_upserts_existing_rows():
    """upsert 语义：CSV 是事实源，重跑导入必须能把值写回已有行。

    早先的"已存在就跳过"会让新增列永远回填不到老行（`item_alias` 就是第一例），
    并且改 CSV 之后重跑导入**静默无效** —— 看着像更新了、库里没变。
    """
    from sqlalchemy import select

    await _prepare()

    async def _alias() -> str | None:
        async with get_sessionmaker()() as s:
            row = (
                await s.execute(
                    select(LabItemReferenceBase).where(LabItemReferenceBase.item_name == "血糖")
                )
            ).scalars().first()
            return row.item_alias

    assert await _alias(), "CSV 里血糖的别名应当非空"

    # 人为清掉，再跑一次导入 —— 只有 upsert 才会把它写回来
    async with get_sessionmaker()() as s:
        row = (
            await s.execute(
                select(LabItemReferenceBase).where(LabItemReferenceBase.item_name == "血糖")
            )
        ).scalars().first()
        row.item_alias = None
        await s.commit()
    assert await _alias() is None

    await import_min_kb(get_engine())
    assert await _alias(), "重跑导入没有回填已有行 —— upsert 退回了 insert-if-absent"


# --------------------------------------------------------------------------
# 异常标记：识别形态与判定优先级（2026-09-14 新增）
# --------------------------------------------------------------------------


def test_flag_extraction_is_conservative_about_unit_l():
    """`L` 只在**紧贴数值的固定位置**才算标记，否则 `1.2 g/L` 会被判成偏低。

    单位里到处是 `L`（`g/L`、`mmol/L`、`10^9/L`），所以 H/L 的识别形态必须与
    `lab_report_vision._strip_marks` 保持一致：只在数值**前面**或**空格之后**的末尾。
    箭头没有这个歧义，认到就算。
    """
    from app.core.tools.lab_report_vision import _extract_flag, _normalize_flag

    assert _extract_flag("6.5↑") == "H"
    assert _extract_flag("↓6.5") == "L"
    assert _extract_flag("6.5 H") == "H"
    assert _extract_flag("L 6.5") == "L"
    assert _extract_flag("6.5") == ""
    assert _extract_flag("1.2 g/L") == "", "单位里的 L 不是异常标记"
    assert _extract_flag("阴性") == ""

    assert _normalize_flag("↑") == "H"
    assert _normalize_flag("偏高") == "H"
    assert _normalize_flag("LOW") == "L"
    assert _normalize_flag("正常") == "", "不认识的词不做语义猜测"
    assert _normalize_flag("", "6.5↓") == "L", "模型不填时从 test_value 兜底"
    assert _normalize_flag("H", "6.5↓") == "H", "模型填了就以模型为准"


@pytest.mark.asyncio
async def test_image_flag_wins_over_general_range():
    """图上标了 H，即使按库内区间算出来是正常，也以图为准。

    理由：单子上的标记是这家医院**按该患者**给出的结论，口径比通用区间准。
    """
    await _prepare()
    out = await LabReportTool().interpret(
        user_id="u1",
        lab_item_list=[{"item_name": "血糖", "test_value": "5.0"}],
        sync_to_archive=False,
        image_items={"血糖": {"flag": "H", "reference_range": "3.9-6.1"}},
    )
    item = out["item_list"][0]
    assert item["abnormal_flag"] == "H"
    assert item["judge_source"] == "image_flag"


@pytest.mark.asyncio
async def test_image_blank_means_normal():
    """图上空白 = 正常 —— 即使按库内区间该值算偏高。

    这是本次改动里**语义最强的一条**：空白不是"没读到"，是"医院没标"。
    医院的报告规范是只给异常项打标记，所以空白本身就是信息。
    """
    await _prepare()
    out = await LabReportTool().interpret(
        user_id="u1",
        lab_item_list=[{"item_name": "血糖", "test_value": "7.0"}],
        sync_to_archive=False,
        image_items={"血糖": {"flag": "", "reference_range": "3.9-6.1"}},
    )
    item = out["item_list"][0]
    assert item["abnormal_flag"] == "N"
    assert item["judge_source"] == "image_blank"


@pytest.mark.asyncio
async def test_text_path_still_compares_range():
    """没有图的项（用户手打）只能比库内区间 —— 这条路不能因为改成图标注优先而断掉。

    这一条是本次改动**唯一必须保住的既有能力**：没有它，用户手打
    「血红蛋白 110 偏低吗」就再也得不到"偏低"这个结论。
    """
    await _prepare()
    out = await LabReportTool().interpret(
        user_id="u1",
        lab_item_list=[{"item_name": "血糖", "test_value": "7.0"}],
        sync_to_archive=False,
    )
    item = out["item_list"][0]
    assert item["abnormal_flag"] == "H"
    assert item["judge_source"] == "general_range"


@pytest.mark.asyncio
async def test_image_flag_judges_item_missing_from_reference_base():
    """库内没有这一项 + 图上有标记 → 仍然给结论，只是没有解释文本。

    旧实现遇到"不在参考库"一律回「无法提供解读服务」，但图上明明标着偏高 ——
    说"无法解读"是错的，我们只是说不出原因和建议。
    """
    await _prepare()
    out = await LabReportTool().interpret(
        user_id="u1",
        lab_item_list=[{"item_name": "某医院自建指标", "test_value": "1.2"}],
        sync_to_archive=False,
        image_items={"某医院自建指标": {"flag": "H", "reference_range": "0-1.0"}},
    )
    item = out["item_list"][0]
    assert item["abnormal_flag"] == "H"
    assert item["judge_source"] == "image_flag"
    assert "未纳入参考库" in item["meaning"]


@pytest.mark.asyncio
async def test_advice_attached_for_abnormal_and_rendered_in_final_desc():
    """偏高项要带上「可能原因 / 建议 / 何时就医」，并且必须出现在 `final_desc` 里。

    为什么必须落在 `final_desc`：单步化验解读走 `final_response` **短路分支、不过 LLM**，
    `final_desc` 就是用户直接看到的那段文本。建议只挂在结构化字段上等于没写。
    """
    await _prepare()
    out = await LabReportTool().interpret(
        user_id="u1",
        lab_item_list=[
            {"item_name": "血糖", "test_value": "7.0"},
            {"item_name": "总胆固醇", "test_value": "4.0"},
        ],
        sync_to_archive=False,
    )
    by_name = {i["item_name"]: i for i in out["item_list"]}

    adv = by_name["血糖"].get("advice") or {}
    assert adv.get("causes"), "偏高必须有『可能原因』"
    assert adv.get("advice"), "偏高必须有『建议』"
    assert adv.get("when_to_see_doctor"), "偏高必须有『何时就医』"
    assert "医生" in (adv.get("disclaimer") or ""), "免责说明必须提医生"

    desc = out["final_desc"]
    assert "可能原因" in desc and "建议" in desc and "何时就医" in desc

    # 正常项不该硬凑建议
    assert not by_name["总胆固醇"].get("advice")
