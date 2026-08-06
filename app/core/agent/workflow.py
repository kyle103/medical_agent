from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime

from langgraph.graph import END, StateGraph

from app.common.exceptions import UserAuthException
from app.common.logger import get_logger
from app.core.agent.nodes import (
    commit_gate,
    entity_extraction,
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
    response_plan,
)
from app.core.agent.state import AgentState
from app.core.llm.llm_service import LLMService

logger = get_logger(__name__)


class MedicalAgent:
    def __init__(self):
        self.graph = self._build()

    def _build(self):
        g = StateGraph(AgentState)

        g.add_node("input_check", input_check)
        g.add_node("mem_load", memory_load)
        g.add_node("intent_node", intent_recognition)
        g.add_node("entities", entity_extraction)
        g.add_node("knowledge", knowledge_retrieve)
        g.add_node("plan", plan_node)
        g.add_node("execute", execute_node)
        g.add_node("reconcile", reconcile_node)
        g.add_node("response_plan", response_plan)
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
        g.add_edge("intent_node", "entities")
        g.add_edge("entities", "knowledge")
        g.add_edge("knowledge", "plan")
        g.add_edge("plan", "execute")

        def _after_execute(state: dict) -> str:
            if state.get("needs_replan"):
                return "plan"
            return "reconcile"

        g.add_conditional_edges("execute", _after_execute, {"plan": "plan", "reconcile": "reconcile"})

        g.add_edge("reconcile", "response_plan")

        g.add_edge("response_plan", "llm")
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
        out = await self.graph.ainvoke(state, config={"callbacks": None})

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

        pre_llm_nodes = [
            ("input_check", input_check),
            ("mem_load", memory_load),
            ("intent_node", intent_recognition),
            ("entities", entity_extraction),
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
        state = await response_plan(state)

        reconciled_sections = state.get("reconciled_sections")
        if reconciled_sections and len(reconciled_sections) > 1:
            yield json.dumps({"type": "intent", "intent": state.get("intent", "general")}, ensure_ascii=False) + "\n"

            system_prompt = (
                "你是医疗问答助手，需要将多个子问题的回答整合为一个清晰、自然的回复。\n"
                "原则：\n"
                "1) 必须保留每个子问题的回答要点，不得遗漏或添加原文未提及的医学事实。\n"
                "2) 每个子问题用二级标题（##）分隔，标题即为子问题本身。\n"
                "3) 每个子问题的回答要简洁精炼，去除重复和冗余内容。\n"
                "4) 使用**加粗**标记关键信息（如药名、症状、注意事项）。\n"
                "5) 使用项目符号或编号列表组织多条信息，每条之间空一行。\n"
                "6) 语言自然亲切，像一位耐心的家庭医生在和你聊天。\n"
                "7) 如果某个子问题无法回答（如工具查询失败），用简短一句话说明，不要输出原始错误信息或凭空补充。\n"
                "8) 整体回复结尾用一句温馨提示收束。\n"
            )
            sections_text = ""
            for i, section in enumerate(reconciled_sections):
                sections_text += f"\n\n--- 子问题 {i + 1} ---\n{section}"
            user_prompt = f"用户原始问题：{state.get('user_input', '')}\n\n以下是各子问题的回答：{sections_text}"

            llm = LLMService()
            full_response = ""
            try:
                async for chunk in llm.chat_completion_stream(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    timeout_s=15.0,
                    max_tokens=1200,
                ):
                    full_response += chunk
                    yield json.dumps({"type": "chunk", "content": chunk}, ensure_ascii=False) + "\n"
            except Exception as e:
                logger.error("stream multi-intent llm_generate failed: %s", e)
                full_response = "\n\n".join([f"## {s}" for s in reconciled_sections])
                yield json.dumps({"type": "chunk", "content": full_response}, ensure_ascii=False) + "\n"

            state["llm_output"] = full_response
            state["final_response"] = full_response
        elif state.get("final_response"):
            state["llm_output"] = state["final_response"]
            yield json.dumps({"type": "intent", "intent": state.get("intent", "general")}, ensure_ascii=False) + "\n"
            resp_text = state["final_response"]
            import re as _re
            chunks = _re.split(r'([。！？\n])', resp_text)
            sentence_buf = ""
            for part in chunks:
                sentence_buf += part
                if len(sentence_buf) >= 12 or part in ("。", "！", "？", "\n"):
                    if sentence_buf.strip():
                        yield json.dumps({"type": "chunk", "content": sentence_buf}, ensure_ascii=False) + "\n"
                    sentence_buf = ""
            if sentence_buf.strip():
                yield json.dumps({"type": "chunk", "content": sentence_buf}, ensure_ascii=False) + "\n"
        elif state.get("needs_confirmation") and state.get("confirmation_message"):
            state["llm_output"] = state["confirmation_message"]
            yield json.dumps({"type": "chunk", "content": state["confirmation_message"]}, ensure_ascii=False) + "\n"
        else:
            content = (state.get("tool_result") or {}).get("final_desc") or ""
            if not content:
                content = "当前未获取到有效工具结果。"

            mode = state.get("response_mode") or "llm_chat"
            inject_memory = bool(state.get("inject_memory"))

            mem_summary = (state.get("memory_summary") or "").strip()
            if not mem_summary and inject_memory:
                from app.core.agent.nodes import _short_window_history
                mem_summary = _short_window_history(state.get("history") or [], max_turns=4)

            long_mem = (state.get("long_memory_text") or "").strip()
            retrieved_knowledge = state.get("retrieved_knowledge") or {}

            candidate_drug_events = state.get("candidate_drug_events")
            if candidate_drug_events:
                from app.core.skills.medication_confirmation_skill import MedicationConfirmationSkill
                skill = MedicationConfirmationSkill()
                confirm_msg = skill.build_confirmation_message(candidate_drug_events)
                state["llm_output"] = confirm_msg
                yield json.dumps({"type": "chunk", "content": confirm_msg}, ensure_ascii=False) + "\n"
            else:
                if mode == "llm_format":
                    system_prompt = (
                        "你是医疗问答助手，任务是把\"工具/数据库查询结果\"用清晰、自然、结构化的中文表达出来。\n"
                        "核心要求：\n"
                        "1) 严格基于提供的工具/检索结果输出，不得添加任何结果中未提及的医学事实、数据或结论。\n"
                        "2) 如果结果中某项信息缺失或无法确定，直接说明\"该信息未在查询结果中体现\"，不要自行补充。\n"
                        "3) 输出尽量简洁，分点呈现，必要时补充就医建议边界。\n"
                        "4) 禁止给出诊断结论或处方/调整用药建议。\n"
                        "5) 回复格式要求：\n"
                        "   - 使用**加粗**标记关键信息（如药名、指标名）\n"
                        "   - 使用编号列表或项目符号组织多条信息\n"
                        "   - 每个要点之间用空行分隔，保持视觉清晰\n"
                        "   - 语言自然亲切，像医生对患者解释一样，避免生硬的罗列\n"
                        "   - 结尾用一段简短的温馨提示收束\n"
                    )
                    user_prompt = f"用户问题：{state.get('user_input', '')}\n\n工具结果：\n{content}\n"
                    if long_mem:
                        user_prompt = f"长期记忆（用户历史偏好/事实，供参考，可能与本轮有关）：\n{long_mem}\n\n" + user_prompt
                    if inject_memory and mem_summary:
                        user_prompt = f"会话记忆（可能与本轮有关）：\n{mem_summary}\n\n" + user_prompt
                    if retrieved_knowledge:
                        user_prompt = f"医疗知识库检索结果（供参考）：\n{json.dumps(retrieved_knowledge, ensure_ascii=False)}\n\n" + user_prompt
                else:
                    system_prompt = (
                        "你是医疗问答助手，需要用自然的对话方式回答用户。\n"
                        "核心原则：\n"
                        "1) 如果提供了知识库检索结果、工具查询结果、或会话/长期记忆，你的回答必须基于这些信息。\n"
                        "   知识库未收录的内容应明确说\"目前知识库中未查到相关信息\"，不得凭训练数据编造医学事实。\n"
                        "2) 不得编造不存在的个人信息/检查结果/用药记录/药物数据。\n"
                        "3) 禁止诊断与处方/调整用药建议；可以给出通用科普与就医指引。\n"
                        "4) 对于纯闲聊或非医疗问题（如\"你好\"），正常友好回复即可，不需要强行关联医学内容。\n"
                        "5) 回复格式要求：\n"
                        "   - 语气亲切自然，像一位耐心的家庭医生在和你聊天\n"
                        "   - 使用**加粗**标记关键信息（如药名、症状、注意事项）\n"
                        "   - 多条信息用编号列表或项目符号组织，每条之间空一行\n"
                        "   - 先给简短总结，再展开说明，避免一上来就堆砌大量文字\n"
                        "   - 结尾用一句温馨提示收束（如：如有不适请及时就医）\n"
                    )
                    parts = []
                    if long_mem:
                        parts.append("长期记忆（用户历史偏好/事实）：\n" + long_mem)
                    if inject_memory and mem_summary:
                        parts.append("会话记忆：\n" + mem_summary)
                    if content:
                        parts.append("工具/检索结果：\n" + content)
                    if retrieved_knowledge:
                        parts.append("医疗知识库检索结果（供参考）：\n" + json.dumps(retrieved_knowledge, ensure_ascii=False))
                    parts.append("用户输入：\n" + (state.get("user_input", "") or ""))
                    user_prompt = "\n\n".join(parts)

                yield json.dumps({"type": "intent", "intent": state.get("intent", "general")}, ensure_ascii=False) + "\n"

                llm = LLMService()
                full_response = ""
                try:
                    async for chunk in llm.chat_completion_stream(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        timeout_s=15.0,
                        max_tokens=900,
                    ):
                        full_response += chunk
                        yield json.dumps({"type": "chunk", "content": chunk}, ensure_ascii=False) + "\n"
                except Exception as e:
                    logger.error("stream llm_generate failed: %s", e)
                    full_response = content
                    yield json.dumps({"type": "chunk", "content": content}, ensure_ascii=False) + "\n"

                state["llm_output"] = full_response

        state = await fact_check(state)
        state = await output_check_and_disclaimer(state)
        state = await commit_gate(state)

        disclaimer_text = state.get("final_response", "")
        llm_output = state.get("llm_output", "")
        if disclaimer_text and llm_output and disclaimer_text != llm_output:
            added = disclaimer_text[len(llm_output):]
            if added.strip():
                yield json.dumps({"type": "chunk", "content": added}, ensure_ascii=False) + "\n"

        asyncio.create_task(self._async_memory_update(state))

        history = state.get("history") or []
        yield json.dumps({
            "type": "done",
            "session_id": session_id,
            "intent": state.get("intent", "general"),
            "needs_confirmation": bool(state.get("needs_confirmation")),
            "conversation_turns": len(history) // 2 if history else 0,
        }, ensure_ascii=False) + "\n"

    @staticmethod
    async def _async_memory_update(state: dict):
        try:
            await memory_update(state)
        except Exception as e:
            logger.error("async memory_update failed: %s", e)
