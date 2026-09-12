from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncGenerator
from datetime import datetime

from langgraph.graph import END, StateGraph

from app.common.exceptions import UserAuthException
from app.common.logger import get_logger, log_node_execution
from app.core.agent.nodes import (
    build_generation_prompt,
    commit_gate,
    error_finalize,
    execute_node,
    fact_check,
    input_check,
    intent_recognition,
    knowledge_retrieve,
    llm_generate,
    memory_load,
    memory_update,
    output_check_and_disclaimer,
    plan_node,
    reconcile_node,
)
from app.core.agent.state import AgentState
from app.core.llm.llm_service import LLMService, begin_usage_tracking, end_usage_tracking
from app.core.skills.medication_confirmation_skill import MedicationConfirmationSkill

logger = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r'([。！？\n])')


def _intent_event(state: dict) -> str:
    return json.dumps({"type": "intent", "intent": state.get("intent", "general")}, ensure_ascii=False) + "\n"


def _chunk_event(content: str) -> str:
    return json.dumps({"type": "chunk", "content": content}, ensure_ascii=False) + "\n"


def _log_stream_node(branch: str, t0: float, **detail) -> None:
    """流式路径的节点级打点（原先整条流式链路都没有 log_node_execution）。"""
    log_node_execution(
        node_name="llm_generate_stream",
        latency_ms=int((time.perf_counter() - t0) * 1000),
        branch=branch,
        **detail,
    )


class MedicalAgent:
    def __init__(self):
        self.graph = self._build()

    def _build(self):
        g = StateGraph(AgentState)

        g.add_node("input_check", input_check)
        g.add_node("mem_load", memory_load)
        g.add_node("intent_node", intent_recognition)
        g.add_node("knowledge", knowledge_retrieve)
        g.add_node("plan", plan_node)
        g.add_node("execute", execute_node)
        g.add_node("reconcile", reconcile_node)
        g.add_node("llm", llm_generate)
        g.add_node("fact_check", fact_check)
        g.add_node("out", output_check_and_disclaimer)
        g.add_node("commit", commit_gate)
        g.add_node("mem", memory_update)
        g.add_node("err", error_finalize)

        g.set_entry_point("input_check")

        def _need_error(state: dict) -> str:
            return "err" if state.get("error_msg") else "mem_load"

        g.add_conditional_edges("input_check", _need_error, {"err": "err", "mem_load": "mem_load"})

        g.add_edge("mem_load", "intent_node")
        g.add_edge("intent_node", "knowledge")
        g.add_edge("knowledge", "plan")
        g.add_edge("plan", "execute")

        def _after_execute(state: dict) -> str:
            if state.get("needs_replan"):
                return "plan"
            return "reconcile"

        g.add_conditional_edges("execute", _after_execute, {"plan": "plan", "reconcile": "reconcile"})

        # response_plan 已降级为 build_generation_prompt 内部按需调用的纯函数，不再是图节点
        g.add_edge("reconcile", "llm")
        g.add_edge("llm", "fact_check")
        g.add_edge("fact_check", "out")

        def _need_error2(state: dict) -> str:
            return "err" if state.get("error_msg") else "commit"

        g.add_conditional_edges("out", _need_error2, {"err": "err", "commit": "commit"})
        g.add_edge("commit", "mem")

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
        begin_usage_tracking()
        out = await self.graph.ainvoke(state, config={"callbacks": None})
        cache_stats = end_usage_tracking()

        intent_analysis_raw = out.get("intent_analysis") or {}
        intent_analysis = None
        if intent_analysis_raw:
            intent_analysis = {
                "intent_type": intent_analysis_raw.get("intent_type", out.get("intent_type", "")),
                "confidence": intent_analysis_raw.get("confidence", out.get("intent_confidence", 0.0)),
                "reason": intent_analysis_raw.get("reason", out.get("intent_reason", "")),
                "target_name": intent_analysis_raw.get("target_name", out.get("target_agent", "")),
            }

        history = out.get("history") or []

        return {
            "session_id": session_id,
            "user_input": user_input,
            "assistant_output": out.get("final_response", ""),
            "intent": out.get("intent", "general"),
            "create_time": datetime.now().isoformat(timespec="seconds"),
            "intent_analysis": intent_analysis,
            "target_agent": out.get("target_agent", ""),
            "needs_confirmation": bool(out.get("needs_confirmation")),
            "conversation_turns": len(history) // 2 if history else 0,
            "cache_stats": cache_stats,
        }

    async def run_stream(
        self,
        *,
        user_id: str,
        session_id: str,
        user_input: str,
        enable_archive_link: bool,
    ) -> AsyncGenerator[str, None]:
        if not user_id:
            raise UserAuthException("未授权")

        state: dict = {
            "user_id": user_id,
            "session_id": session_id,
            "user_input": user_input,
            "stream": True,
            "enable_archive_link": enable_archive_link,
        }
        begin_usage_tracking()

        pre_llm_nodes = [
            ("input_check", input_check),
            ("mem_load", memory_load),
            ("intent_node", intent_recognition),
            ("knowledge", knowledge_retrieve),
            ("plan", plan_node),
        ]

        for node_name, node_fn in pre_llm_nodes:
            yield json.dumps({"type": "progress", "node": node_name}, ensure_ascii=False) + "\n"
            state = await node_fn(state)
            if state.get("error_msg"):
                yield json.dumps({"type": "error", "content": state.get("error_msg", "处理失败")}, ensure_ascii=False) + "\n"
                return
            if node_name == "intent_node":
                ia = state.get("intent_analysis") or {}
                yield json.dumps({
                    "type": "intent",
                    "intent": state.get("intent", "general"),
                    "intent_analysis": {
                        "intent_type": ia.get("intent_type", state.get("intent_type", "")),
                        "confidence": ia.get("confidence", state.get("intent_confidence", 0.0)),
                        "reason": ia.get("reason", state.get("intent_reason", "")),
                        "target_name": ia.get("target_name", state.get("target_agent", "")),
                    },
                    "target_agent": state.get("target_agent", ""),
                }, ensure_ascii=False) + "\n"

        max_replan = 2
        replan_count = 0
        while True:
            state = await execute_node(state)
            if state.get("needs_replan") and replan_count < max_replan:
                state = await plan_node(state)
                replan_count += 1
                continue
            break

        state = await reconcile_node(state)

        _t_llm = time.perf_counter()
        plan = build_generation_prompt(state)
        branch = plan["branch"]

        if branch == "multi_intent":
            yield _intent_event(state)
            full_response = ""
            try:
                async for chunk in LLMService().chat_completion_stream(
                    prompt=plan["user_prompt"],
                    system_prompt=plan["system_prompt"],
                    timeout_s=15.0,
                    max_tokens=1200,
                ):
                    full_response += chunk
                    yield _chunk_event(chunk)
            except Exception as e:
                logger.error("stream multi-intent llm_generate failed: %s", e)
                full_response = "\n\n".join([f"## {s}" for s in (state.get("reconciled_sections") or [])])
                yield _chunk_event(full_response)

            state["llm_output"] = full_response
            state["final_response"] = full_response
            _log_stream_node("multi_intent", _t_llm, section_count=len(state.get("reconciled_sections") or []))

        elif branch == "final_response":
            state["llm_output"] = state["final_response"]
            yield _intent_event(state)
            # 上游已产出完整文本，按句切分模拟流式，避免一次性吐出一大段
            sentence_buf = ""
            for part in _SENTENCE_SPLIT_RE.split(state["final_response"]):
                sentence_buf += part
                if len(sentence_buf) >= 12 or part in ("。", "！", "？", "\n"):
                    if sentence_buf.strip():
                        yield _chunk_event(sentence_buf)
                    sentence_buf = ""
            if sentence_buf.strip():
                yield _chunk_event(sentence_buf)
            _log_stream_node("final_response", _t_llm, shortcut="final_response")

        elif branch == "confirmation":
            # 与改造前一致：该分支不发 intent 事件
            state["llm_output"] = state["confirmation_message"]
            yield _chunk_event(state["confirmation_message"])
            _log_stream_node("confirmation", _t_llm, shortcut="confirmation")

        elif branch == "drug_confirmation":
            # 与改造前一致：该分支不发 intent 事件
            state.setdefault("skill_ctx", {})
            state["skill_ctx"]["medication_confirmation"] = {"candidate_events": state["candidate_drug_events"]}
            confirm_msg = MedicationConfirmationSkill().build_confirmation_message(state["candidate_drug_events"])
            state["llm_output"] = confirm_msg
            yield _chunk_event(confirm_msg)
            _log_stream_node("drug_confirmation", _t_llm, shortcut="drug_confirmation")

        else:
            yield _intent_event(state)
            full_response = ""
            try:
                async for chunk in LLMService().chat_completion_stream(
                    prompt=plan["user_prompt"],
                    system_prompt=plan["system_prompt"],
                    timeout_s=15.0,
                    max_tokens=900,
                ):
                    full_response += chunk
                    yield _chunk_event(chunk)
            except Exception as e:
                logger.error("stream llm_generate failed: %s", e)
                full_response = plan["content"]
                yield _chunk_event(full_response)

            state["llm_output"] = full_response
            _log_stream_node("normal", _t_llm, mode=plan["mode"])

        state = await fact_check(state)
        state = await output_check_and_disclaimer(state)
        state = await commit_gate(state)

        disclaimer_text = state.get("final_response", "")
        llm_output = state.get("llm_output", "")
        if disclaimer_text and llm_output and disclaimer_text != llm_output:
            added = disclaimer_text[len(llm_output):]
            if added.strip():
                yield json.dumps({"type": "chunk", "content": added}, ensure_ascii=False) + "\n"

        # 本轮同步 LLM 调用的缓存命中统计（异步记忆提取不计入，它在 done 之后运行）
        cache_stats = end_usage_tracking()

        asyncio.create_task(self._async_memory_update(state))

        history = state.get("history") or []
        yield json.dumps({
            "type": "done",
            "session_id": session_id,
            "intent": state.get("intent", "general"),
            "needs_confirmation": bool(state.get("needs_confirmation")),
            "conversation_turns": len(history) // 2 if history else 0,
            "cache": cache_stats,
        }, ensure_ascii=False) + "\n"

    @staticmethod
    async def _async_memory_update(state: dict):
        try:
            await memory_update(state)
        except Exception as e:
            logger.error("async memory_update failed: %s", e)
