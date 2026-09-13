"""化验指标解析与路由守卫的契约测试。

背景（2026-09-13 用户实测）：用户问「帮我解读一下血常规化验单，我直接发单子给你吗」，
系统判定为「化验 95%」并回「未识别到有效的检验指标。请提供指标名称和数值…」，
用户再问一次，收到**同一句**。

排查出三层问题，本文件把后两层的修复固化下来：

1. **解析层（安全相关，最严重）**：旧实现把 `是`/`为` 当键值分隔符，并有「匹配到指标名后
   抓整句第一个数字挂上去」的兜底 → `"血糖不高，我今年35岁"` 被解析成 `血糖 = 35`，
   下游 `LabReportTool` 据此输出 **H/L 异常判定** —— 即系统会告诉用户"血糖偏高"，
   只因为他说了自己 35 岁。**这是错误的医疗判断，不是体验问题。**
2. **路由层**：`intent=lab` 或命中「化验/血常规」关键词就路由到 `lab_report` 工具，
   不检查消息里是否真有数值。
3. **生成层**：工具零抽取（`item_list` 为空）被当作"工具正常返回"，
   生成模式判定为事实型 `llm_format`，把「请提供指标名称和数值」当成解读结论输出。
"""

import pytest

from app.core.agent.planner_agent import _route_by_intent_and_text
from app.core.tools.lab_item_parser import has_lab_values, parse_lab_items


# ───────────────────────────── A. 解析层：宁可漏抽，不可错抽


#: 这些句子里**没有**可解读的检验数据。必须产出空列表。
#: 每一条都对应一类会被旧实现当成数值的非检验量。
NO_VALUE_CASES = [
    # 非检验量（年龄 / 体重 / 体温 / 天数 / 饮酒量）
    ("血糖不高，我今年35岁", "年龄"),
    ("我血糖正常，但体重70公斤", "体重"),
    ("血常规里白细胞高，我发烧38度5", "体温"),
    ("转氨酶高，我喝了2两酒", "饮酒量"),
    ("尿酸高，脚疼了3天", "天数"),
    ("我血压高，今年60岁", "年龄"),
    # 非数值（旧实现把「是怎么回事」当成了血糖值）
    ("血糖偏高是怎么回事", "追问原因"),
    # 归属不明：同一句话里两个指标共用一个数字
    ("血糖和血压都查了，结果是6.5和120", "跨指标串值"),
    # 流程 / 能力咨询 —— 用户报的这个 case
    ("帮我解读一下血常规化验单，我直接发单子给你吗", "问能否发图片"),
    ("直接提供文本还是直接发化验单指标给你？", "问输入方式"),
    ("血常规化验单怎么看", "问怎么读"),
    ("化验单上的箭头是什么意思", "问符号含义"),
    ("我该多久查一次血常规", "问频次"),
    ("帮我解读一下化验单", "只给意图未给数据"),
]


@pytest.mark.parametrize("text,label", NO_VALUE_CASES)
def test_no_value_texts_produce_nothing(text, label):
    got = parse_lab_items(text)
    assert got == [], (
        f"[{label}] 不得产出化验条目 —— 错抽会让下游输出错误的 H/L 异常判定。"
        f"输入={text!r} 实际={got!r}"
    )
    assert has_lab_values(text) is False


#: 这些句子里**有**可解读的数据。必须抽出来，否则真正的解读请求会被降级成闲聊。
VALUE_CASES = [
    ("血糖6.5", [("血糖", "6.5")]),
    ("血糖 6.5 mmol/L", [("血糖", "6.5")]),
    ("血糖：6.5", [("血糖", "6.5")]),
    ("我的血压是120/80", [("血压", "120/80")]),
    ("血压150高吗", [("血压", "150")]),
    ("血糖6.5，血压120/80", [("血糖", "6.5"), ("血压", "120/80")]),
    ("空腹血糖 6.5 mmol/L，尿酸 480", [("血糖", "6.5"), ("尿酸", "480")]),
    # 任意指标名（超出白名单）+ 显式分隔符：参考库里没有也要能识别出"这是数据"
    ("血红蛋白：130", [("血红蛋白", "130")]),
]


@pytest.mark.parametrize("text,expected", VALUE_CASES)
def test_value_texts_are_extracted(text, expected):
    got = [(i["item_name"], i["test_value"]) for i in parse_lab_items(text)]
    assert got == expected, f"输入={text!r} 期望={expected!r} 实际={got!r}"
    assert has_lab_values(text) is True


def test_extraction_never_carries_a_non_numeric_value():
    """固化不变量：**任何**条目的 test_value 都必须能解析成数值或数值区间。

    这条比逐例断言更根本 —— 旧实现正是产出了 `"怎么回事"` 这种"值"。
    """
    import re

    numeric = re.compile(r"^\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?$")
    probes = [t for t, _ in NO_VALUE_CASES] + [t for t, _ in VALUE_CASES]
    probes += [
        "血糖高，我今年35岁，体重70公斤，发烧38度5",
        "化验单：血糖6.5；血压 120/80；年龄 35 岁",
    ]
    for text in probes:
        for item in parse_lab_items(text):
            assert numeric.match(item["test_value"]), (
                f"test_value {item['test_value']!r} 不是数值/数值区间 ← {text!r}"
            )


def test_whitelist_item_name_is_canonical_not_raw_text():
    """指标名必须是**规范名**，不能是原文片段。

    旧实现把分隔符左边整段当指标名（`"我的血压是120/80"` → `"我的血压"`），
    而 `LabReferenceService.match_items` 是**精确等值匹配** → 永远匹配不上参考库，
    于是所有指标都退化显示为"暂未纳入参考库"。
    """
    got = parse_lab_items("我的血压是120/80")
    assert got and got[0]["item_name"] == "血压", f"实际={got!r}"


# ───────────────────────────── B. 路由层：有数据才进化验工具


def _route(text: str, intent: str) -> dict:
    return _route_by_intent_and_text(
        {"intent": intent, "user_input": text, "intent_confidence": 0.95}
    )


@pytest.mark.parametrize("text", [t for t, _ in NO_VALUE_CASES])
def test_lab_intent_without_values_falls_back_to_general(text):
    """LLM 判成 `intent=lab`（用户报的 case 是 lab/95%）但没有数值 → 必须退回通用问答。

    退回后 `intent_type=general`，于是：① 生成模式走 llm_chat，LLM 直接回答用户的问题；
    ② `knowledge_retrieve` 的 `need_public` 为真，知识库检索不再被跳过。
    """
    route = _route(text, "lab")
    assert route["target_name"] == "main_qa_agent", f"输入={text!r} 实际={route!r}"
    assert route["intent_type"] == "general"


def test_lab_intent_with_values_still_routes_to_tool():
    """有数值时不得误伤 —— 真正的解读请求仍要进化验工具。"""
    for text, _ in VALUE_CASES:
        route = _route(text, "lab")
        assert route["target_name"] == "lab_report", f"输入={text!r} 实际={route!r}"
        assert route["intent_type"] == "lab_report"


@pytest.mark.parametrize(
    "text",
    ["血常规化验单怎么看", "帮我解读一下化验单", "化验单上的箭头是什么意思"],
)
def test_lab_keyword_fallback_routing_also_requires_values(text):
    """关键词兜底路径也要有数值。**必须传空 intent 才能真正打到该分支** ——
    `intent` 为 lab/archive/general/drug 时函数会提前 return，走不到关键词规则。"""
    route = _route(text, "")
    assert route["target_name"] == "main_qa_agent", f"输入={text!r} 实际={route!r}"
    assert route["intent_type"] == "general"


def test_lab_keyword_fallback_still_routes_when_values_present():
    """关键词命中且有数值 → 仍走化验工具。

    注意用「血常规」而不是「血糖」：关键词表是 化验/检验/血常规/尿常规/指标，
    **不含**「血糖」，所以「血糖6.5」在这条分支上不会被命中（它靠 intent=lab 路由）。
    """
    route = _route("血常规：白细胞11.2", "")
    assert route["target_name"] == "lab_report", f"实际={route!r}"
    assert route["intent_type"] == "lab_report"


def test_lab_keyword_fallback_is_actually_reachable():
    """固化前提：空 intent 时确实会命中「化验」关键词规则。

    防止将来有人把关键词规则删掉/挪位置后，上面两条测试变成"因错误原因通过"。
    """
    route = _route("我这份血常规怎么看", "")
    assert "no readable values" in route["reason"] or route["target_name"] == "main_qa_agent"
    assert route["reason"].startswith("route by text: lab"), f"实际={route!r}"


# ───────────────────────────── C. 生成层：零抽取不得当结论输出（无 LLM）


def test_no_data_tool_result_downgrades_generation_mode():
    """工具零抽取 → 生成模式必须回到 `llm_chat`，不得走事实型 `llm_format`。

    否则那句「请提供指标名称和数值」会被当成化验结论输出 —— 用户实测到的答非所问。
    """
    from app.core.agent.nodes import _decide_response_mode

    empty = {
        "intent": "lab",
        "intent_type": "lab_report",
        "tool_name": "lab_report",
        "tool_result": {"item_list": [], "no_data": True, "final_desc": "…"},
    }
    assert _decide_response_mode(empty) == "llm_chat", (
        "零抽取仍被判成事实型 —— 「请提供指标名称和数值」会被当成解读结论"
    )


def test_real_tool_result_still_uses_fact_mode():
    """有真实结果时仍必须是事实型，不得因为这条守卫放宽约束。"""
    from app.core.agent.nodes import _decide_response_mode

    real = {
        "intent": "lab",
        "intent_type": "lab_report",
        "tool_name": "lab_report",
        "tool_result": {"item_list": [{"item_name": "血糖", "test_value": "7.0"}]},
    }
    assert _decide_response_mode(real) == "llm_format"


def test_multi_step_with_one_empty_step_is_not_downgraded():
    """多步场景：只要有任何一步拿到真数据，整轮仍按事实型约束输出。"""
    from app.core.agent.nodes import _decide_response_mode

    mixed = {
        "intent": "general",
        "tool_result": {"item_list": [], "no_data": True},
        "plan_step_results": {
            "s1": {"tool_result": {"item_list": [{"item_name": "血糖", "test_value": "7.0"}]}},
            "s2": {"tool_result": {"item_list": [], "no_data": True}},
        },
    }
    assert _decide_response_mode(mixed) == "llm_format", (
        "另一步空手而归不应让整轮退回对话模式，否则真数据会丢掉事实约束"
    )


def test_plain_question_without_tool_result_is_unchanged():
    """无工具结果时不触发守卫（回归保护）。"""
    from app.core.agent.nodes import _decide_response_mode

    assert _decide_response_mode({"intent": "general", "user_input": "你好"}) == "llm_chat"


@pytest.mark.asyncio
async def test_reconcile_does_not_turn_no_data_into_final_response():
    """**关键**：零抽取的说明文本不得被写进 `final_response`。

    这是修完前两处后仍然漏掉的路径 —— 而且是最要命的一处：LLM planner 可以直接给出
    `target_name=lab_report`（不经过 `_route_by_intent_and_text`），于是路由守卫不生效；
    `reconcile_node` 再把这句说明写进 `final_response`，
    `build_generation_prompt` 的 `final_response` 短路分支会把整个生成阶段跳过，
    连 `_decide_response_mode` 都不会被问到 —— 用户收到的永远是同一句模板。

    实测：`"帮我解读一下血常规化验单，我直接发单子给你吗"` 走的就是这条路径。
    """
    from app.core.agent.nodes import reconcile_node

    state = {
        "user_input": "帮我解读一下血常规化验单，我直接发单子给你吗",
        "execution_plan": {
            "steps": [
                {
                    "step_id": "s1",
                    "query": "帮我解读一下血常规化验单",
                    "target_type": "tool",
                    "target_name": "lab_report",
                    "intent_type": "lab_report",
                }
            ]
        },
        "plan_step_results": {
            "s1": {
                "tool_result": {
                    "item_list": [],
                    "no_data": True,
                    "final_desc": "未识别到有效的检验指标。请提供指标名称和数值。",
                },
                "intent_type": "lab_report",
            }
        },
    }
    out = await reconcile_node(state)
    assert not out.get("final_response"), (
        f"零抽取的说明被写进 final_response → 短路分支会跳过 LLM，答案永远是同一句。"
        f"实际={out.get('final_response')!r}"
    )


@pytest.mark.asyncio
async def test_reconcile_still_writes_real_tool_result():
    """有真实工具结果时仍要写 final_response（不得为了防止短路而丢内容）。"""
    from app.core.agent.nodes import reconcile_node

    state = {
        "user_input": "血糖6.5",
        "execution_plan": {
            "steps": [
                {
                    "step_id": "s1",
                    "query": "血糖6.5",
                    "target_type": "tool",
                    "target_name": "lab_report",
                    "intent_type": "lab_report",
                }
            ]
        },
        "plan_step_results": {
            "s1": {
                "tool_result": {
                    "item_list": [{"item_name": "血糖", "test_value": "6.5"}],
                    "final_desc": "化验指标通用解读：血糖 6.5（参考 3.9-6.1）状态：H",
                },
                "intent_type": "lab_report",
            }
        },
    }
    out = await reconcile_node(state)
    assert "血糖" in (out.get("final_response") or ""), f"实际={out.get('final_response')!r}"
