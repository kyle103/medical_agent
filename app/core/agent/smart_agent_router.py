from __future__ import annotations

import logging
import time
from typing import Any

from app.core.agent.workflow import MedicalAgent
from app.core.session.agent_state_store import AgentStateStore

logger = logging.getLogger(__name__)


class SmartAgentRouter:
    """API适配层：所有请求统一委托给 MedicalAgent 工作流处理。
    
    原有的独立路由逻辑（多意图拆分、LLM意图分析、pending_confirmation等）
    已全部迁移至 MedicalAgent 的 plan_node / execute_node / reconcile_node 流程。
    本类仅保留为 API 入口兼容层。
    """

    def __init__(self):
        self._agent = MedicalAgent()

    async def route_and_execute(self, state: dict[str, Any]) -> dict[str, Any]:
        user_id = state.get("user_id")
        session_id = state.get("session_id")
        user_input = state.get("user_input", "")

        logger.info("SmartAgentRouter delegate to MedicalAgent user_id=%s session_id=%s input_len=%s", user_id, session_id, len(user_input))

        try:
            result = await self._agent.run(
                user_id=user_id or "",
                session_id=session_id or "",
                user_input=user_input,
                stream=bool(state.get("stream", False)),
                enable_archive_link=bool(state.get("enable_archive_link", False)),
            )

            state["final_response"] = result.get("assistant_output", "")
            state["intent"] = result.get("intent", "general")
            return state
        except Exception as e:
            logger.error("SmartAgentRouter delegate failed: %s", e)
            state["error_msg"] = f"处理请求时出现错误: {str(e)}"
            state["final_response"] = "抱歉，处理请求时出现错误，请稍后重试。"
            return state
