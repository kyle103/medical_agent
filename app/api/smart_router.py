"""智能路由测试API"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from app.core.agent.smart_agent_router import SmartAgentRouter
from app.core.session.session_manager import session_manager

router = APIRouter()


class IntentAnalysisRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@router.post("/test-route")
async def test_route(request: IntentAnalysisRequest):
    """测试完整路由流程 - 自动管理对话历史"""
    try:
        router = SmartAgentRouter()
        
        # 获取或创建会话
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session(request.user_id)
        
        session = session_manager.get_session(session_id)
        if not session:
            session_id = session_manager.create_session(request.user_id)
            session = session_manager.get_session(session_id)
        
        # 获取对话历史
        conversation_history = session_manager.get_conversation_history(session_id)
        
        # 构建state对象，包含长期记忆
        state = {
            "user_input": request.user_input,
            "history": conversation_history,
            "user_id": session["user_id"],
            "session_id": session_id,
            "long_memory_text": _build_long_memory_text(session)
        }
        
        # 执行路由
        result_state = await router.route_and_execute(state)
        
        # 将当前对话添加到会话历史
        session_manager.add_message_to_session(session_id, "user", request.user_input)
        if result_state.get("final_response"):
            session_manager.add_message_to_session(session_id, "assistant", result_state.get("final_response"))
        
        # 更新长期记忆（如果有新信息）
        if result_state.get("update_memory"):
            _update_long_term_memory(session_id, result_state.get("update_memory"))
        
        # 返回结果
        return {
            "session_id": session_id,
            "intent_analysis": result_state.get("intent_analysis"),
            "target_agent": result_state.get("target_agent"),
            "intent": result_state.get("intent"),
            "needs_confirmation": result_state.get("needs_confirmation", False),
            "confirmation_message": result_state.get("confirmation_message"),
            "error_msg": result_state.get("error_msg"),
            "final_response": result_state.get("final_response"),
            "conversation_history": session_manager.get_conversation_history(session_id)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"路由测试失败: {str(e)}")
    
def _build_long_memory_text(session: Dict[str, Any]) -> str:
    """构建长期记忆文本"""
    long_term_memory = session.get("long_term_memory", {})
    if not long_term_memory:
        return ""
    
    memory_text = "长期记忆信息：\n"
    for key, value_info in long_term_memory.items():
        value = value_info.get("value", "")
        memory_text += f"- {key}: {value}\n"
    
    return memory_text

def _update_long_term_memory(session_id: str, memory_updates: Dict[str, Any]):
    """更新长期记忆"""
    for key, value in memory_updates.items():
        session_manager.update_long_term_memory(session_id, key, value)