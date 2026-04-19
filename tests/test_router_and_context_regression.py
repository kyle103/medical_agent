from app.core.agent.smart_agent_router import SmartAgentRouter
from app.core.session.session_manager import SessionManager
import pytest


def test_extract_drug_candidates_for_conflict_question():
    names = SmartAgentRouter._extract_drug_candidates("阿司匹林和布洛芬一起吃有冲突吗？")
    assert "阿司匹林" in names
    assert "布洛芬" in names


def test_session_manager_reuse_latest_session_for_same_user():
    mgr = SessionManager()
    sid1 = mgr.get_or_create_session(user_id="u_test")
    sid2 = mgr.get_or_create_session(user_id="u_test")
    assert sid1 == sid2


def test_split_user_queries_multi():
    text = "阿司匹林和布洛芬一起吃有冲突吗？帮我记录一下我昨天晚上吃了两片感康。我是否有高血压病史"
    parts = SmartAgentRouter._split_user_queries(text)
    assert len(parts) >= 3
    assert any("阿司匹林" in p for p in parts)
    assert any("感康" in p for p in parts)
    assert any("高血压病史" in p for p in parts)


@pytest.mark.asyncio
async def test_route_and_execute_multi_aggregates(monkeypatch):
    router = SmartAgentRouter()

    async def _fake_single(state):
        q = state.get("user_input", "")
        return {
            **state,
            "final_response": f"已处理:{q}",
            "intent_type": "general",
            "intent": "general",
        }

    monkeypatch.setattr(router, "_route_and_execute_single", _fake_single)
    state = {"user_input": "问题一？问题二。问题三", "history": []}
    out = await router.route_and_execute(state)
    content = out.get("final_response", "")
    assert "我分条为你处理如下" in content
    assert "已处理:问题一" in content
    assert "已处理:问题二" in content
    assert "已处理:问题三" in content


@pytest.mark.asyncio
async def test_single_route_sets_pending_confirmation(monkeypatch):
    router = SmartAgentRouter()

    async def _fake_analyze(user_input, conversation_history=None):
        return {
            "target_agent": "drug_record_agent",
            "confidence": 0.9,
            "reason": "need confirm",
            "intent_type": "drug_record",
            "needs_confirmation": True,
        }

    monkeypatch.setattr(router, "_route_from_state_intent", lambda state: None)
    monkeypatch.setattr(router, "analyze_intent_with_llm", _fake_analyze)

    out = await router._route_and_execute_single({"user_input": "帮我记录一下吃药", "history": []})
    assert out.get("needs_confirmation") is True
    pending = out.get("pending_confirmation") or {}
    assert pending.get("type") == "agent_confirmation"
    assert isinstance(pending.get("payload"), dict)
    assert pending.get("payload", {}).get("target_agent") == "drug_record_agent"
