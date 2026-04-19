from app.core.agent.smart_agent_router import SmartAgentRouter
from app.core.session.session_manager import SessionManager


def test_extract_drug_candidates_for_conflict_question():
    names = SmartAgentRouter._extract_drug_candidates("阿司匹林和布洛芬一起吃有冲突吗？")
    assert "阿司匹林" in names
    assert "布洛芬" in names


def test_session_manager_reuse_latest_session_for_same_user():
    mgr = SessionManager()
    sid1 = mgr.get_or_create_session(user_id="u_test")
    sid2 = mgr.get_or_create_session(user_id="u_test")
    assert sid1 == sid2
