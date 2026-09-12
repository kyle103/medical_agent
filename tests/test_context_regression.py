"""上下文 / 技能 / 能力注册表回归。

本文件替代原 `test_router_and_context_regression.py`。原文件测的 6 个方法都长在
`SmartAgentRouter` 上，而该 router 已退化为薄兼容层——编排逻辑迁到了 LangGraph 工作流
与 `planner_agent`，下列方法在源码里已不存在：

    _extract_drug_candidates / _route_and_execute_single / _route_from_state_intent
    _normalize_notice_once / get_agent_capabilities / route_and_execute

本文件保留其中**断言意图仍然成立**的部分，并把调用点迁到当前实现：

| 原测试 | 现在的落点 |
|---|---|
| `SmartAgentRouter._split_user_queries` | `planner_agent._split_user_queries` |
| `SmartAgentRouter.get_agent_capabilities` | `planner_agent.CAPABILITY_REGISTRY` |
| `_extract_drug_candidates` | 已由 `tests/test_entity_dictionary.py` 覆盖，不重复 |
| 其余 4 项 | 对应 API 已删除且无等价实现，随原文件移除 |

`SessionManager` 复用与 `MedicationConfirmationSkill` 文案两项原本就是通过状态，原样保留。
"""
import pytest

from app.core.agent import planner_agent
from app.core.session.session_manager import SessionManager
from app.core.skills.medication_confirmation_skill import MedicationConfirmationSkill


@pytest.mark.asyncio
async def test_split_user_queries_multi():
    """一句多意图输入应被拆成多个子查询。

    LLM 不可用时该函数回落为规则结果（可能只有一段），因此这里不硬断言段数，
    而是断言**关键信息不丢**——两种路径下都必须成立。
    """
    text = "阿司匹林和布洛芬一起吃有冲突吗？帮我记录一下我昨天晚上吃了两片感康。我是否有高血压病史"
    parts = await planner_agent._split_user_queries(text)
    assert parts
    joined = "".join(parts)
    for key in ("阿司匹林", "布洛芬", "感康", "高血压病史"):
        assert key in joined, f"拆分后丢失关键信息：{key}"


def test_session_manager_reuse_latest_session_for_same_user():
    mgr = SessionManager()
    sid1 = mgr.get_or_create_session(user_id="u_test")
    sid2 = mgr.get_or_create_session(user_id="u_test")
    assert sid1 == sid2


def test_medication_confirmation_skill_message():
    skill = MedicationConfirmationSkill()
    msg = skill.build_confirmation_message(
        [{"drug_name": "布洛芬", "full_text": "我昨天吃了布洛芬"}]
    )
    assert "布洛芬" in msg
    assert "加入用药档案" in msg


def test_capability_registry_exposes_all_targets():
    """能力注册表是「能力集封闭」的唯一事实源：规划只能路由到注册过的 target。"""
    registry = planner_agent.CAPABILITY_REGISTRY
    assert registry

    names = {c["name"] for c in registry}
    assert {"drug_interaction", "drug_record_agent", "main_qa_agent", "lab_report"} <= names

    for card in registry:
        assert card.get("type") in ("agent", "tool"), card
        assert card.get("description"), card
        assert card.get("when_to_use"), card

    # 注册表里的 agent 必须与 AGENT_TARGETS 一致，否则 planner 会把步骤派给不存在的 agent
    assert {c["name"] for c in registry if c["type"] == "agent"} == set(planner_agent.AGENT_TARGETS)
