from __future__ import annotations

from datetime import datetime

from langgraph.graph import END, StateGraph

from app.common.exceptions import UserAuthException
from app.core.agent.nodes import (
    entity_extraction,
    error_finalize,
    input_check,
    intent_recognition,
    llm_generate,
    memory_load,
    memory_update,
    output_check_and_disclaimer,
    response_plan,
    tool_execute,
    tool_route,
)
from app.core.agent.state import AgentState


class MedicalAgent:
    def __init__(self):
        self.graph = self._build()

    def _build(self):
        g = StateGraph(AgentState)

        g.add_node("input_check", input_check)
        g.add_node("mem_load", memory_load)
        g.add_node("intent_node", intent_recognition)
        g.add_node("entities", entity_extraction)
        g.add_node("route", tool_route)
        g.add_node("tool", tool_execute)
        g.add_node("plan", response_plan)
        g.add_node("llm", llm_generate)
        g.add_node("out", output_check_and_disclaimer)
        g.add_node("mem", memory_update)
        g.add_node("err", error_finalize)

        g.set_entry_point("input_check")

        def _need_error(state: dict) -> str:
            return "err" if state.get("error_msg") else "mem_load"

        g.add_conditional_edges("input_check", _need_error, {"err": "err", "mem_load": "mem_load"})

        g.add_edge("mem_load", "intent_node")
        g.add_edge("intent_node", "entities")
        g.add_edge("entities", "route")
        g.add_edge("route", "tool")
        g.add_edge("tool", "plan")
        g.add_edge("plan", "llm")
        g.add_edge("llm", "out")

        def _need_error2(state: dict) -> str:
            return "err" if state.get("error_msg") else "mem"

        g.add_conditional_edges("out", _need_error2, {"err": "err", "mem": "mem"})

        g.add_edge("mem", END)
        g.add_edge("err", END)

        return g.compile()

    async def run(
        self,
        *,
        user_id: str,
        session_id: str,
        user_input: str,
        stream: bool,
        enable_archive_link: bool,
    ) -> dict:
        if not user_id:
            raise UserAuthException("未授权")

        state: dict = {
            "user_id": user_id,
            "session_id": session_id,
            "user_input": user_input,
            "stream": stream,
            "enable_archive_link": enable_archive_link,
        }
        out = await self.graph.ainvoke(state)

        return {
            "session_id": session_id,
            "user_input": user_input,
            "assistant_output": out.get("final_response", ""),
            "intent": out.get("intent", "general"),
            "create_time": datetime.now().isoformat(timespec="seconds"),
        }
