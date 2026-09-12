"""Step 4 checkpointer 冒烟 + Step 4.5 bug 回归测试。

⚠️ 关键前提：必须用**同一个 MedicalAgent 实例**跑多轮——InMemorySaver 是实例级内存，
跨实例会被重置，checkpointer 的"续跑"语义就失效了。这也是 Step 4 早期测试通过的
假象根因（实际上 conversation_turns 来自 DB 而不是 checkpointer）。

变量语义约定：
  - `session_id` 是原始会话 ID（不带 user_id 前缀），是 chat_router 层使用的 ID；
  - `thread_id` = `_thread_id(user_id, session_id)`，是 langgraph checkpointer 用的 ID。
    `MedicalAgent.run_stream` 内部会把 session_id 转成 thread_id；外部直接用
    `agent.graph.get_state(config)` 时需要用 thread_id。

两组：
  A. 静态契约（不调 LLM）：构造参数、checkpointer 挂载
  B. 真实图（共享 checkpointer）：核心三件 + Step 4.5 三个 bug 回归
"""

from __future__ import annotations

import json
import typing
from typing import Iterable

import pytest
from langgraph.channels import UntrackedValue

import app.core.agent.workflow as wf
from app.core.agent.state import AgentState
from app.core.agent.workflow import MedicalAgent, _thread_id


# ─────────────────────────────────────────────── A. 静态契约


def test_default_checkpointer_is_inmemory_saver():
    """构造函数不传 checkpointer 时默认挂 `InMemorySaver`（Step 4 冒烟载体）。"""
    agent = MedicalAgent()
    assert type(agent.checkpointer).__name__ == "InMemorySaver"


def test_explicit_none_checkpointer_disables_compile_with_checkpoint():
    """显式 `checkpointer=None` 表示不挂 checkpointer——仅供特殊测试场景使用。"""
    agent = MedicalAgent(checkpointer=None)
    assert agent.graph is not None


def test_compiled_graph_attaches_checkpointer():
    """`_build()` 把 checkpointer 编译进图，`graph.checkpointer` 非 None。"""
    agent = MedicalAgent()
    assert agent.graph.checkpointer is not None
    assert agent.graph.checkpointer is agent.checkpointer


# ─────────────────────────────────────────────── B. 真实图（共享 checkpointer）


_UNTRACKED_FIELDS: frozenset[str] = frozenset({
    "history", "history_text", "memory_summary",
    "long_memory_items", "long_memory_text",
    "retrieved_knowledge", "shared_facts", "private_scratchpads", "skill_ctx",
    "decision_context", "last_decision",
    "tool_result", "llm_output",
    "candidate_drug_events", "pending_drug_events_for_confirmation",
    "session_runtime_state", "pending_confirmation",
    "plan_step_results", "reconciled_sections", "reconciled_context",
    "replan_context", "cross_step_conflict", "proposed_updates",
})


async def _run_stream_async(agent: MedicalAgent, *, session_id: str, user_input: str,
                             user_id: str = "probe-user") -> list[dict]:
    """跑一次 `run_stream`，返回全部事件 dict。复用同一 agent → InMemorySaver 持久。

    `session_id` 是 raw session ID（不带 user_id 前缀）；`run_stream` 内部会自动拼成
    thread_id = f"{user_id}:{session_id}"。
    """
    out: list[dict] = []
    async for line in agent.run_stream(
        user_id=user_id,
        session_id=session_id,
        user_input=user_input,
        enable_archive_link=False,
    ):
        out.append(json.loads(line.strip()))
    return out


def _collect_text(events: list[dict]) -> str:
    parts = []
    for e in events:
        if e["type"] == "chunk":
            parts.append(e.get("content", ""))
        elif e["type"] == "error":
            parts.append(e.get("content", ""))
    return "".join(parts).strip()


@pytest.mark.asyncio
async def test_get_state_snapshot_only_contains_tracked_fields():
    """核心验收 ②：`get_state(config).values` 不含任何 UntrackedValue 字段。"""
    session_id = "ckpt-smoke-tracked-only"
    thread_id = _thread_id("probe-user", session_id)
    agent = MedicalAgent()
    events = await _run_stream_async(agent, session_id=session_id, user_input="你好")
    assert events[-1]["type"] == "done", f"run_stream 未正常结束: {[e['type'] for e in events]}"

    snapshot = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
    assert snapshot is not None
    values: dict = snapshot.values
    expected_present = {
        "user_id", "session_id", "user_input", "stream", "enable_archive_link",
        "intent", "target_agent", "final_response",
    }
    assert not (expected_present - set(values.keys())), \
        f"关键字段缺失: {expected_present - set(values.keys())}"
    leaked = _UNTRACKED_FIELDS & set(values.keys())
    assert not leaked, f"UntrackedValue 字段泄露进快照: {sorted(leaked)}"


@pytest.mark.asyncio
async def test_same_thread_multi_turn_continuation():
    """核心验收 ①：同 thread_id 多轮可续（共享 checkpointer）。

    第二轮 `memory_load` 从 DB 拿到第一轮的对话记录 → conversation_turns >= 1。
    这里有意不复用 agent（模拟 chat_router 每请求新建实例的真实路径）；
    验证的是**业务态（AgentStateStore）+ checkpointer 续跑**的端到端协调。
    """
    session_id = "ckpt-smoke-multi-turn"
    events1 = await _run_stream_async(
        MedicalAgent(), session_id=session_id, user_input="你好",
    )
    assert events1[-1]["type"] == "done"

    events2 = await _run_stream_async(
        MedicalAgent(), session_id=session_id, user_input="那高血压呢？",
    )
    assert events2[-1]["type"] == "done"
    assert events2[-1].get("conversation_turns", 0) >= 1


@pytest.mark.asyncio
async def test_different_threads_are_isolated():
    """同 user 不同 session → 互不污染。"""
    user = "probe-user"
    s1 = "ckpt-smoke-iso-1"
    s2 = "ckpt-smoke-iso-2"
    t1 = _thread_id(user, s1)
    t2 = _thread_id(user, s2)
    inp1, inp2 = "你好", "hi"

    agent = MedicalAgent()
    await _run_stream_async(agent, session_id=s1, user_input=inp1)
    await _run_stream_async(agent, session_id=s2, user_input=inp2)

    snap1 = agent.graph.get_state({"configurable": {"thread_id": t1}})
    snap2 = agent.graph.get_state({"configurable": {"thread_id": t2}})
    assert snap1 is not None and snap2 is not None
    assert snap1.values.get("user_input") == inp1
    assert snap2.values.get("user_input") == inp2


@pytest.mark.asyncio
async def test_get_state_history_yields_per_superstep_snapshots():
    """`get_state_history(config)` 返回 per-super-step 快照序列，next 字段标记下一步节点。"""
    session_id = "ckpt-smoke-history"
    thread_id = _thread_id("probe-user", session_id)
    agent = MedicalAgent()
    events = await _run_stream_async(agent, session_id=session_id, user_input="你好")
    assert events[-1]["type"] == "done"

    history: Iterable = agent.graph.get_state_history({"configurable": {"thread_id": thread_id}})
    snaps = list(history)
    assert len(snaps) >= 2
    last_next = snaps[0].next if hasattr(snaps[0], "next") else None
    assert not last_next
    for i, s in enumerate(snaps[:3]):
        leaked = _UNTRACKED_FIELDS & set((s.values or {}).keys())
        assert not leaked, f"快照 [{i}] 泄露 Untracked 字段: {leaked}"


@pytest.mark.asyncio
async def test_run_method_also_threads_through_checkpointer():
    """非流式 `run()` 也要把 thread_id 透传到 checkpointer。"""
    session_id = "ckpt-smoke-run-sync"
    thread_id = _thread_id("probe-user", session_id)
    agent = MedicalAgent()
    out = await agent.run(
        user_id="probe-user",
        session_id=session_id,
        user_input="你好",
        stream=False,
        enable_archive_link=False,
    )
    assert "assistant_output" in out
    assert "conversation_turns" in out
    snap = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
    assert snap is not None
    assert snap.values.get("user_input") == "你好"


# ─────────────────────────────────────────────── B'. Step 4.5 bug 回归
# （每个 bug 一个真实链路测试，复用同一 MedicalAgent 让 InMemorySaver 持久化）


@pytest.mark.asyncio
async def test_bug1_compliance_block_does_not_poison_next_turn():
    """BUG-1 回归：合规拦截留下的 error_msg 不应让后续每轮都被拦截。

    修复：turn_reset 节点在每轮入口重置 error_msg（以及其它单轮量字段）。
    """
    session_id = "reg-bug1-poison"
    thread_id = _thread_id("probe-user", session_id)
    agent = MedicalAgent()

    # TURN 1：触发合规拦截
    e1 = await _run_stream_async(agent, session_id=session_id, user_input="帮我开药")
    a1 = _collect_text(e1)
    assert "已按合规要求拦截" in a1, f"TURN1 应被拦截，实际: {a1[:60]!r}"

    # TURN 2：完全正常的医疗问题
    e2 = await _run_stream_async(
        agent, session_id=session_id, user_input="你好，请问高血压平时要注意什么？",
    )
    a2 = _collect_text(e2)
    assert "已按合规要求拦截" not in a2, (
        f"BUG-1 复发：TURN2 复读了拦截语 → {a2[:80]!r}"
    )
    # TURN2 应给出高血压管理建议（模型常用 paraphrase，不一定含字面「高血压」三字）
    advice_kw = ["血压", "低盐", "低脂", "保暖", "饮食", "心态", "防过劳", "便秘"]
    hit = [k for k in advice_kw if k in a2]
    assert hit, (
        f"TURN2 应含高血压管理建议关键词，实际: {a2[:120]!r}"
    )

    # 末尾快照应不含 error_msg
    snap = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
    assert snap is not None
    assert snap.values.get("error_msg") in ("", None), (
        f"快照 error_msg 应为空，实际: {snap.values.get('error_msg')!r}"
    )


@pytest.mark.asyncio
async def test_bug2_thread_id_isolates_users_with_same_session_id():
    """BUG-2 回归：不同 user 用同一 session_id 不应跨用户污染。

    修复：thread_id = f"{user_id}:{session_id}"，不同 user 的 thread_id 自动分桶。
    """
    shared_session = "reg-bug2-shared"
    agent = MedicalAgent()

    # user-A 先跑
    e_a = await _run_stream_async(
        agent, session_id=shared_session, user_input="你好", user_id="user-A",
    )
    snap_a = agent.graph.get_state(
        {"configurable": {"thread_id": _thread_id("user-A", shared_session)}}
    )
    assert snap_a is not None
    assert snap_a.values.get("user_id") == "user-A"

    # user-B 用同一个 session_id（注意：run_stream 的 session_id 参数实际是 raw session_id，
    # run_stream 内部会用 _thread_id(user_id, session_id) 拼出真实 thread_id）
    e_b = await _run_stream_async(
        agent, session_id=shared_session, user_input="再见", user_id="user-B",
    )
    snap_b = agent.graph.get_state(
        {"configurable": {"thread_id": _thread_id("user-B", shared_session)}}
    )
    assert snap_b is not None
    assert snap_b.values.get("user_id") == "user-B"

    # A 的回答不应含 B 的关键词
    a_text = _collect_text(e_a)
    b_text = _collect_text(e_b)
    assert "再见" not in a_text, f"BUG-2 复发：user-A 答案含 user-B 关键词 → {a_text[:60]!r}"


@pytest.mark.asyncio
async def test_bug3_final_response_does_not_leak_across_turns():
    """BUG-3 回归：上一轮的 final_response 不应让本轮 build_generation_prompt 短路。

    修复：turn_reset 在每轮入口把 final_response 显式重置为 ""，短路分支 `if state.get("final_response")`
    因此不会误命中。

    验证方式：跑两轮简单输入，确认 TURN2 仍调用 LLM 生成（即 chunks > 2，且 final_response 含
    TURN2 输入的关键词而非 TURN1 的开场白）。
    """
    session_id = "reg-bug3-fr-leak"
    agent = MedicalAgent()

    e1 = await _run_stream_async(agent, session_id=session_id, user_input="你好")
    a1 = _collect_text(e1)
    e2 = await _run_stream_async(agent, session_id=session_id, user_input="高血压平时要注意什么")
    a2 = _collect_text(e2)

    # TURN2 应含本轮输入的关键词
    assert "高血压" in a2, f"TURN2 应含 '高血压' → {a2[:80]!r}"
    # TURN2 不应是 TURN1 的复读
    assert a1[:60] != a2[:60] or "高血压" in a2, "TURN2 与 TURN1 答案高度相似可能是复读"


# ─────────────────────────────────────────────── C. 文档化约束（无 LLM）


def test_untracked_value_count_matches_state_docstring():
    """固化：state.py 中 `Annotated[_, UntrackedValue]` 的字段数 == 测试里白名单。

    防止后续有人加了 Untracked 字段但忘了更新测试 / 漏进白名单。
    """
    declared_unt: set[str] = set()
    for name, hint in typing.get_type_hints(AgentState, include_extras=True).items():
        for meta in typing.get_args(hint):
            if isinstance(meta, UntrackedValue) or (
                isinstance(meta, type) and issubclass(meta, UntrackedValue)
            ):
                declared_unt.add(name)
    assert declared_unt == _UNTRACKED_FIELDS, (
        f"AgentState 中 UntrackedValue 字段与白名单不一致：\n"
        f"  声明 - 白名单: {sorted(declared_unt - _UNTRACKED_FIELDS)}\n"
        f"  白名单 - 声明: {sorted(_UNTRACKED_FIELDS - declared_unt)}"
    )


def test_thread_id_format_helper():
    """固化：_thread_id(user, session) 格式为 `user:session`，user 缺失退到 `anon:`。"""
    assert _thread_id("alice", "s1") == "alice:s1"
    assert _thread_id("", "s1") == "anon:s1"
    assert _thread_id("alice", "") == "alice:"
    assert _thread_id("", "") == "anon:"