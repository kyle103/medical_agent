from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.common.logger import get_logger
from app.core.agent.registry import build_agent_registry
from app.core.agent.smart_agent_router import SmartAgentRouter
from app.core.session.session_manager import session_manager

router = APIRouter()
logger = get_logger(__name__)


class IntentAnalysisRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@router.post("/test-route")
async def test_route(request: IntentAnalysisRequest):
    try:
        smart_router = SmartAgentRouter()

        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session(request.user_id)

        session = session_manager.get_session(session_id)
        if not session:
            session_id = session_manager.create_session(request.user_id)
            session = session_manager.get_session(session_id)

        conversation_history = session_manager.get_conversation_history(session_id)

        state = {
            "user_input": request.user_input,
            "history": conversation_history,
            "user_id": session["user_id"],
            "session_id": session_id,
            "stream": False,
            "enable_archive_link": False,
        }

        result_state = await smart_router.route_and_execute(state)

        session_manager.add_message_to_session(session_id, "user", request.user_input)
        if result_state.get("final_response"):
            session_manager.add_message_to_session(session_id, "assistant", result_state.get("final_response"))

        return {
            "session_id": session_id,
            "intent": result_state.get("intent"),
            "error_msg": result_state.get("error_msg"),
            "final_response": result_state.get("final_response"),
            "conversation_history": session_manager.get_conversation_history(session_id),
        }

    except Exception as e:
        logger.error("test_route failed: %s", e)
        raise HTTPException(status_code=500, detail=f"路由测试失败: {str(e)}")


@router.get("/agent-cards")
async def agent_cards():
    try:
        agents = build_agent_registry()
        cards = [agent.get_agent_card() for agent in agents.values()]
        return {"items": [c.dict() if hasattr(c, "dict") else c for c in cards]}
    except Exception as e:
        logger.error("agent_cards failed: %s", e)
        raise HTTPException(status_code=500, detail=f"获取 Agent Card 失败: {str(e)}")
