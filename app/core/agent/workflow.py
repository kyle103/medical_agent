from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app.common.exceptions import UserAuthException
from app.core.agent.nodes import (
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
    turn_reset,
)
from app.core.agent.state import AgentState
from app.core.agent.stream_events import (
    PROGRESS_EXCLUDED_NODES,
    chunk_payload,
    done_payload,
    error_payload,
    intent_payload,
    progress_payload,
    serialize,
)
from app.core.llm.llm_service import begin_usage_tracking, end_usage_tracking


def _thread_id(user_id: str, session_id: str) -> str:
    """组合 `thread_id` = f"{user_id}:{session_id}"——治 BUG-2 跨用户串话。

    背景：langgraph checkpointer 按 `thread_id` 索引状态；`session_id` 是客户端可控的
    （chat_router.py:25 仅在空时生成 uuid），同一 `session_id` 在两个不同 `user_id`
    下若直接用作 `thread_id`，会让 user-B 的运行读到 user-A 留下的快照——
    表现为上轮的 `final_response` / `error_msg` 跨用户残留（严重情形还会因
    `plan_step_results` 自引用触发 msgpack 递归深度爆炸）。

    `run()` / `run_stream()` 已要求非空 user_id（`UserAuthException`）；
    缺失时退到 `anon:` 前缀，避免空键混淆。
    """
    prefix = user_id if user_id else "anon"
    return f"{prefix}:{session_id}"


class MedicalAgent:
    def __init__(self, *, checkpointer: BaseCheckpointSaver | None = None):
        """默认挂 `InMemorySaver`；显式传 `None` 表示不加 checkpointer（仅供特殊测试场景）。

        选 InMemorySaver 作为 Step 4 的冒烟载体，理由：
          1. 零新增依赖；
          2. 不写盘，与现有 `AgentStateStore`（DB 行）职责不交叠；
          3. 接口形态与 Step 5 的 `langgraph-checkpoint-sqlite` 完全一致，换持久化只需改构造函数参数。

        checkpointer 的职责被刻意收窄为「执行进度」（见方案 §3.3）：
          - checkpointer：图执行位置、节点间传递的小标量、控制流
          - `AgentStateStore`：跨轮业务态（`pending_confirmation` / `last_decision`）
          - `MemoryService`：对话历史与长期记忆
        互不重叠；`AgentState` 上 23 个 `UntrackedValue` 字段（Step 3.5）保证业务态与大对象不进快照。
        """
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = self._build()

    def _build(self):
        g = StateGraph(AgentState)

        g.add_node("turn_reset", turn_reset)
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

        # turn_reset 是图入口：每轮先重置语义单轮字段，
        # 再走 input_check 的合规检查。详见 nodes.py::turn_reset 的注释。
        g.set_entry_point("turn_reset")
        g.add_edge("turn_reset", "input_check")

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

        return g.compile(checkpointer=self.checkpointer)

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
        out = await self.graph.ainvoke(
            state,
            config={"callbacks": None, "configurable": {"thread_id": _thread_id(user_id, session_id)}},
        )
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
        """流式执行。

        **唯一执行定义就是 `_build()` 那张图。**（Step 3 合并前，本方法手写了一份节点
        序列 `pre_llm_nodes` 和一份自建的 replan while 循环，与图定义并存：加节点要改两处，
        `MAX_REPLAN` 维护在三处，且流式路径不走 `error_finalize`。）

        两个通道：
          - `updates` → 节点完成 → progress / 完整版 intent / error
          - `custom`  → `llm_generate` 节点内经 writer 推来的 chunk / 精简 intent

        事件契约见 `app/core/agent/stream_events.py`；
        前端消费点 `frontend/app.js::handleSSEEvent`。
        """
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

        final_state: dict = dict(state)
        # 首个 chunk 之后就不再发 progress：前端对 progress 是直接覆写气泡 innerHTML，
        # 在回答已经开始渲染后再发会把回答正文盖成"正在…"（已有 chunk 仍累积在 JS 侧，
        # 下一次 paint 会恢复，但视觉上会闪一下）。
        answer_started = False
        had_error = False
        # commit 之后立刻收口用量统计，见下方注释；用于兜住错误/异常路径的泄漏
        usage_closed = False
        cache_stats = None

        try:
            async for mode, payload in self.graph.astream(
                state,
                config={"callbacks": None, "configurable": {"thread_id": _thread_id(user_id, session_id)}},
                stream_mode=["updates", "custom"],
            ):
                # ---------- 节点内 writer 推来的事件 ----------
                if mode == "custom":
                    if isinstance(payload, dict):
                        if payload.get("type") == "chunk":
                            answer_started = True
                        yield serialize(payload)
                    continue

                # ---------- 节点完成事件 ----------
                if not isinstance(payload, dict):
                    continue

                for node_name, node_output in payload.items():
                    if node_name == "__interrupt__":
                        continue
                    if isinstance(node_output, dict):
                        final_state.update(node_output)

                    if node_name not in PROGRESS_EXCLUDED_NODES and not answer_started:
                        yield serialize(progress_payload(node_name))

                    # 意图节点完成 → 完整版 intent（带 intent_analysis / target_agent）。
                    # 开始生成前 `llm_generate` 内还会再发一次精简版，这是改造前的既有行为。
                    if node_name == "intent_node":
                        yield serialize(intent_payload(final_state, full=True))

                    # 输出闸门完成 → 补发免责声明等"新增尾段"。
                    # 改造前是拿 final_response 减去 llm_output 求增量，此处保持一致。
                    if node_name == "out":
                        disclaimer_text = final_state.get("final_response", "")
                        llm_output = final_state.get("llm_output", "")
                        if disclaimer_text and llm_output and disclaimer_text != llm_output:
                            added = disclaimer_text[len(llm_output):]
                            if added.strip():
                                yield serialize(chunk_payload(added))

                    # commit 完成 → 收口本轮 LLM 用量统计。
                    # **必须在 mem 之前取**：mem 会 create_task 触发长期记忆写入（内含 LLM 调用），
                    # 晚取会把那次调用的 token 算进本轮的缓存命中统计。
                    if node_name == "commit":
                        cache_stats = end_usage_tracking()
                        usage_closed = True

                    # `err` 节点运行 == 图判定的**终态**错误。用图自己的信号，而不是到处查
                    # error_msg：execute 内部也会写 error_msg，但图并不会因此短路（照常
                    # 走 reconcile → llm 出答案），那种情况不该给用户报错。
                    if node_name == "err":
                        yield serialize(error_payload(final_state.get("error_msg") or "处理失败"))
                        had_error = True
        finally:
            # 错误/异常/consumer 提前关闭时也要收口，否则用量计数会泄漏到下一个请求
            if not usage_closed:
                end_usage_tracking()

        if had_error:
            # 与改造前一致：错误路径不发 done
            return

        yield serialize(done_payload(session_id=session_id, state=final_state, cache_stats=cache_stats))
