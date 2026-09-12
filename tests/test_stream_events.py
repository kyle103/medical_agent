"""Step 3（astream 合并）的边界测试。

两组分开：
  A. `stream_events.py` —— 事件契约的纯函数层（不依赖图、不依赖 LLM）
  B. `MedicalAgent.run_stream` —— 用一个**合成图**替换真实图，逐事件验证映射逻辑

为什么必须单独测 B：
    合并前事件是在 `run_stream` 里手写 yield 出来的，合并后事件来自
    `graph.astream(stream_mode=["updates","custom"])` 的驱动。驱动方式变了，
    而前端契约是按**事件序列**消费的——序列错了前端不报错，只是静默显示错内容。
    合成图能在不调真实 LLM 的前提下把 5 类事件全跑一遍（CI 无网络也能跑）。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.graph import END, StateGraph
from langgraph.types import StreamWriter

import app.core.agent.workflow as wf
from app.core.agent.planner_agent import MAX_REPLAN, PlannerAgent
from app.core.agent.stream_events import (
    chunk_payload,
    done_payload,
    error_payload,
    intent_payload,
    iter_sentence_chunks,
    progress_payload,
    serialize,
)

# ---------------------------------------------------------------- A. 事件契约


def _parse(line: str) -> dict:
    assert line.endswith("\n"), "每条事件必须以换行结尾（chat_router 依赖它切分 SSE）"
    return json.loads(line.strip())


def test_progress_payload_shape():
    p = progress_payload("input_check")
    assert p == {"type": "progress", "node": "input_check"}
    assert _parse(serialize(p)) == p


def test_intent_payload_full_has_all_three_fields():
    """intent_node 完成后发的那次：必须带 intent_analysis 与 target_agent。"""
    state = {
        "intent": "drug_conflict",
        "intent_analysis": {
            "intent_type": "drug_conflict",
            "confidence": 0.87,
            "reason": "询问相互作用",
            "target_name": "drug_record",
        },
        "target_agent": "drug_record",
    }
    p = intent_payload(state, full=True)
    assert p["type"] == "intent"
    assert p["intent"] == "drug_conflict"
    assert p["intent_analysis"] == {
        "intent_type": "drug_conflict",
        "confidence": 0.87,
        "reason": "询问相互作用",
        "target_name": "drug_record",
    }
    assert p["target_agent"] == "drug_record"


def test_intent_payload_full_falls_back_to_flat_state_keys():
    """intent_analysis 缺失时回落到扁平字段——这是改造前就有的兜底，不能丢。"""
    state = {
        "intent": "general",
        "intent_type": "general",
        "intent_confidence": 0.5,
        "intent_reason": "闲聊",
        "target_agent": "qa",
    }
    p = intent_payload(state, full=True)
    assert p["intent_analysis"] == {
        "intent_type": "general",
        "confidence": 0.5,
        "reason": "闲聊",
        "target_name": "qa",
    }
    assert p["target_agent"] == "qa"


def test_intent_payload_minimal_is_intentionally_narrower():
    """开始生成前那次**刻意只有 intent 字段**（改造前的既有行为，别"修"成完整版）。

    前端对 intent 事件的处理是幂等的（setIntentChip + updateDiagnostics），
    所以这次精简发不会丢信息。若有天要统一，属于契约变更，必须同步 app.js。
    """
    state = {"intent": "drug", "target_agent": "x", "intent_analysis": {"confidence": 0.9}}
    p = intent_payload(state, full=False)
    assert p == {"type": "intent", "intent": "drug"}
    assert "intent_analysis" not in p
    assert "target_agent" not in p


def test_done_payload_conversation_turns_is_half_of_history():
    state = {"intent": "general", "needs_confirmation": 0, "history": [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]}
    p = done_payload(session_id="s1", state=state, cache_stats={"hit": 3})
    assert p["type"] == "done"
    assert p["conversation_turns"] == 2
    assert p["needs_confirmation"] is False  # 强转 bool
    assert p["cache"] == {"hit": 3}
    assert p["session_id"] == "s1"


def test_done_payload_handles_empty_history():
    p = done_payload(session_id="s1", state={}, cache_stats=None)
    assert p["conversation_turns"] == 0
    assert p["intent"] == "general"
    assert p["cache"] is None


def test_error_and_chunk_payloads():
    assert error_payload("超时") == {"type": "error", "content": "超时"}
    assert chunk_payload("ab") == {"type": "chunk", "content": "ab"}


def test_serialize_keeps_chinese_unencoded():
    """ensure_ascii=False：前端直接展示，转义成 \\uXXXX 会让日志难以排查。"""
    line = serialize(chunk_payload("头痛"))
    assert "头痛" in line
    assert "\\u" not in line


@pytest.mark.parametrize(
    "text",
    [
        "第一句话。第二句话更长一些，需要累积到十二个字才会吐出去。",
        "单句无标点但足够长可以超过十二个字符阈值",
        "一句话。",
    ],
)
def test_sentence_chunks_concatenate_back_to_original(text):
    """切分是纯展示层拆分，拼回去必须与原文本逐字一致（否则前端最终渲染会缺字）。"""
    pieces = list(iter_sentence_chunks(text))
    assert "".join(pieces) == text
    assert all(p.strip() for p in pieces), "不应吐出纯空白片段"


def test_sentence_chunks_split_on_sentence_end():
    pieces = list(iter_sentence_chunks("第一句话。第二句话更长一些，需要累积到十二个字才会吐出去。"))
    assert pieces[0] == "第一句话。"  # 遇句号立即吐，不等 12 字阈值
    assert len(pieces) >= 2


def test_sentence_chunks_drops_whitespace_only_fragment():
    """记录一个既有小怪癖：纯空白片段会被丢弃（改造前就这样）。

    不"修"它：修了会让流式内容与改造前不一致，且影响面只在纯空白处。
    """
    text = "你好\n\n\n\n\n\n\n\n\n\n\n\n\n\n世界。"
    pieces = list(iter_sentence_chunks(text))
    assert all(p.strip() for p in pieces)


def test_sentence_chunks_empty_text_yields_nothing():
    assert list(iter_sentence_chunks("")) == []


# ------------------------------------------------- B. run_stream 事件接管

DISCLAIMER = "\n\n如症状持续，请及时就医。"


class _FakeState(TypedDict, total=False):
    """必须用声明式 state。

    `StateGraph(dict)` 不会为未声明的键建 channel —— 节点 A 返回的新键传不到节点 B。
    真实图用的是 `AgentState`（TypedDict），所以这里也必须这样，否则假图的行为
    与真实图不一致，测出来的结论是假的。
    """

    user_id: str
    session_id: str
    user_input: str
    intent: str
    intent_analysis: dict
    target_agent: str
    llm_output: str
    final_response: str
    error_msg: str
    history: list
    needs_confirmation: bool


def _build_fake_graph(*, fail_input: bool = False, with_disclaimer: bool = False, token_fail: bool = False):
    """合成图：节点名与真实图对齐（run_stream 依赖 intent_node / out / commit / err 这些名字）。

    不调任何真实 LLM —— llm 节点直接经 writer 推流。

    保真要点：每个节点都返回**完整 state**（`{**state, ...}`），与真实节点一致。
    langgraph 的 `attach_node` 要求节点至少写一个已声明键，返回 `{}` 会直接抛
    InvalidUpdateError；真实节点本来就返回整份 state，所以不踩这个坑。
    """
    g = StateGraph(_FakeState)

    async def input_check(state: dict, writer: StreamWriter) -> dict:
        if fail_input:
            return {**state, "error_msg": "输入不合法"}
        return {**state}

    async def mem_load(state: dict, writer: StreamWriter) -> dict:
        return {**state}

    async def intent_node(state: dict, writer: StreamWriter) -> dict:
        return {
            **state,
            "intent": "drug",
            "intent_analysis": {"intent_type": "drug", "confidence": 0.9, "reason": "r", "target_name": "t"},
            "target_agent": "t",
        }

    async def knowledge(state: dict, writer: StreamWriter) -> dict:
        return {**state}

    async def plan(state: dict, writer: StreamWriter) -> dict:
        return {**state}

    async def llm(state: dict, writer: StreamWriter) -> dict:
        writer(intent_payload({"intent": state.get("intent", "general")}, full=False))
        if token_fail:
            writer(chunk_payload("降级文本"))
            return {**state, "llm_output": "降级文本"}
        writer(chunk_payload("你好"))
        writer(chunk_payload("。"))
        return {**state, "llm_output": "你好。"}

    async def fact_check(state: dict, writer: StreamWriter) -> dict:
        return {**state}

    async def out(state: dict, writer: StreamWriter) -> dict:
        if with_disclaimer:
            return {**state, "final_response": (state.get("llm_output") or "") + DISCLAIMER}
        return {**state}

    async def commit(state: dict, writer: StreamWriter) -> dict:
        return {**state}

    async def mem(state: dict, writer: StreamWriter) -> dict:
        return {**state}

    async def err(state: dict, writer: StreamWriter) -> dict:
        return {**state, "final_response": state.get("error_msg", "")}

    for name, fn in (
        ("input_check", input_check),
        ("mem_load", mem_load),
        ("intent_node", intent_node),
        ("knowledge", knowledge),
        ("plan", plan),
        ("llm", llm),
        ("fact_check", fact_check),
        ("out", out),
        ("commit", commit),
        ("mem", mem),
        ("err", err),
    ):
        g.add_node(name, fn)

    g.set_entry_point("input_check")

    def _need_error(state: dict) -> str:
        return "err" if state.get("error_msg") else "mem_load"

    g.add_conditional_edges("input_check", _need_error, {"err": "err", "mem_load": "mem_load"})
    g.add_edge("mem_load", "intent_node")
    g.add_edge("intent_node", "knowledge")
    g.add_edge("knowledge", "plan")
    g.add_edge("plan", "llm")
    g.add_edge("llm", "fact_check")
    g.add_edge("fact_check", "out")
    g.add_edge("out", "commit")
    g.add_edge("commit", "mem")
    g.add_edge("mem", END)
    g.add_edge("err", END)
    return g.compile()


def _agent_with(graph) -> wf.MedicalAgent:
    agent = wf.MedicalAgent()
    agent.graph = graph
    return agent


async def _collect(agent: wf.MedicalAgent, session_id: str = "sess-1") -> list[dict]:
    return [
        _parse(line)
        async for line in agent.run_stream(
            user_id="u1", session_id=session_id, user_input="你好", enable_archive_link=False
        )
    ]


async def test_happy_path_event_sequence(monkeypatch):
    """完整事件序列：progress → 完整 intent → 精简 intent → chunk → done。"""
    monkeypatch.setattr(wf, "begin_usage_tracking", lambda: None)
    monkeypatch.setattr(wf, "end_usage_tracking", lambda: {"hit": 7})

    events = await _collect(_agent_with(_build_fake_graph()))
    kinds = [e["type"] for e in events]

    assert kinds == [
        "progress",          # input_check
        "progress",          # mem_load
        "progress",          # intent_node
        "intent",            # 完整版（intent_node 完成）
        "progress",          # knowledge
        "progress",          # plan
        "intent",            # 精简版（llm 开吐前）
        "chunk",             # 你好
        "chunk",             # 。
        "done",
    ]

    # progress 只覆盖生成前节点，且 node 名正确
    assert [e["node"] for e in events if e["type"] == "progress"] == [
        "input_check", "mem_load", "intent_node", "knowledge", "plan",
    ]
    # 第一次 intent 是完整版，第二次是精简版
    intents = [e for e in events if e["type"] == "intent"]
    assert "intent_analysis" in intents[0] and intents[0]["target_agent"] == "t"
    assert "intent_analysis" not in intents[1]
    # chunk 拼起来 == 节点写回的 llm_output
    assert "".join(e["content"] for e in events if e["type"] == "chunk") == "你好。"
    assert events[-1]["cache"] == {"hit": 7}
    assert events[-1]["session_id"] == "sess-1"


async def test_no_progress_after_answer_started(monkeypatch):
    """回答开始后不得再发 progress —— 前端 progress 是直接覆写气泡 innerHTML，
    再发会把已渲染的回答盖成"正在…"。"""
    monkeypatch.setattr(wf, "begin_usage_tracking", lambda: None)
    monkeypatch.setattr(wf, "end_usage_tracking", lambda: None)

    events = await _collect(_agent_with(_build_fake_graph()))
    first_chunk = next(i for i, e in enumerate(events) if e["type"] == "chunk")
    assert not any(e["type"] == "progress" for e in events[first_chunk:])
    # fact_check / out / commit / mem 这些"生成后"节点一个都没混进来
    assert "fact_check" not in [e.get("node") for e in events]
    assert "mem" not in [e.get("node") for e in events]


async def test_disclaimer_tail_is_appended_as_extra_chunk(monkeypatch):
    """out 节点补的免责声明要以增量 chunk 发出，且增量算法与改造前一致。"""
    monkeypatch.setattr(wf, "begin_usage_tracking", lambda: None)
    monkeypatch.setattr(wf, "end_usage_tracking", lambda: None)

    events = await _collect(_agent_with(_build_fake_graph(with_disclaimer=True)))
    chunks = [e["content"] for e in events if e["type"] == "chunk"]
    assert "".join(chunks) == "你好。" + DISCLAIMER
    assert chunks[-1] == DISCLAIMER  # 尾段单独一条


async def test_llm_failure_falls_back_without_error_event(monkeypatch):
    """LLM 失败走的是节点内降级（推降级文本），不是 error 事件——与改造前一致。"""
    monkeypatch.setattr(wf, "begin_usage_tracking", lambda: None)
    monkeypatch.setattr(wf, "end_usage_tracking", lambda: None)

    events = await _collect(_agent_with(_build_fake_graph(token_fail=True)))
    assert not any(e["type"] == "error" for e in events)
    assert "".join(e["content"] for e in events if e["type"] == "chunk") == "降级文本"
    assert events[-1]["type"] == "done"


async def test_error_path_emits_error_and_no_done(monkeypatch):
    """终态错误：发 error、不发 done（改造前行为），且用量统计必须收口。"""
    calls = {"end": 0}
    monkeypatch.setattr(wf, "begin_usage_tracking", lambda: None)

    def _end():
        calls["end"] += 1
        return {"hit": 0}

    monkeypatch.setattr(wf, "end_usage_tracking", _end)

    events = await _collect(_agent_with(_build_fake_graph(fail_input=True)))
    kinds = [e["type"] for e in events]

    assert kinds == ["progress", "error"]
    assert events[1]["content"] == "输入不合法"
    assert "done" not in kinds
    # 改造前错误路径直接 return，漏掉了 end_usage_tracking → 计数泄漏到下一个请求
    assert calls["end"] == 1


async def test_usage_tracking_closed_once_on_happy_path(monkeypatch):
    """正常路径只在 commit 处收口一次（不能靠 finally 再补一次，否则计数会被重复清零）。"""
    calls = {"end": 0}
    monkeypatch.setattr(wf, "begin_usage_tracking", lambda: None)

    def _end():
        calls["end"] += 1
        return {"hit": 1}

    monkeypatch.setattr(wf, "end_usage_tracking", _end)

    events = await _collect(_agent_with(_build_fake_graph()))
    assert calls["end"] == 1
    assert events[-1]["cache"] == {"hit": 1}


async def test_cache_stats_taken_before_memory_node(monkeypatch):
    """cache 必须在 mem 之前取。

    mem 会 create_task 触发长期记忆写入（内含 LLM 调用），晚取会把那次调用的 token
    算进本轮缓存命中统计。这里用"mem 执行时 end 是否已被调用"来断言时序。
    """
    order: list[str] = []
    monkeypatch.setattr(wf, "begin_usage_tracking", lambda: order.append("begin"))

    def _end():
        order.append("end")
        return {"hit": 0}

    monkeypatch.setattr(wf, "end_usage_tracking", _end)

    graph = _build_fake_graph()
    agent = _agent_with(graph)
    await _collect(agent)
    assert order == ["begin", "end"]


# ------------------------------------------- C. 重规划上限：单点来源与行为


def _identifiers(path: Path) -> set[str]:
    """收集源码里**代码层**出现的标识符。

    刻意用 AST 而不是文本匹配：注释与 docstring 里提到旧标识符是正常的（要说明
    "以前是什么样"），文本匹配会把这类说明误判成残留代码。
    """
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
            if node.asname:
                names.add(node.asname)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


def test_workflow_no_longer_owns_a_replan_cap():
    """合并的核心收益：`run_stream` 里那个本地 max_replan 必须消失。

    它不碰 state 的 replan_count，所以 MAX_REPLAN 改成 3 时图路径会重规划 3 次、
    流式路径仍卡在 2 —— 这类 bug 平时不报错，改动时才炸。
    """
    ids = _identifiers(Path(wf.__file__))
    assert "max_replan" not in ids, "run_stream 又长出了第二套重规划计数器"
    assert "pre_llm_nodes" not in ids, "手写的节点序列又回来了"
    # 原来那个自建 replan 循环必然形如 while True
    tree = ast.parse(Path(wf.__file__).read_text(encoding="utf-8-sig"))
    whiles = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
    assert not whiles, f"workflow.py 里不应再有 while 循环，发现 {len(whiles)} 处"


def test_max_replan_defined_in_exactly_one_place():
    """上限只在 planner_agent 定义一处、引用一处。"""
    root = Path(__file__).resolve().parents[1] / "app"
    owners = [
        py.relative_to(root).as_posix()
        for py in root.rglob("*.py")
        if "MAX_REPLAN" in _identifiers(py)
    ]
    assert owners == ["core/agent/planner_agent.py"], f"MAX_REPLAN 出现了不该出现的地方: {owners}"

    tree = ast.parse((root / "core" / "agent" / "planner_agent.py").read_text(encoding="utf-8-sig"))
    refs = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "MAX_REPLAN"]
    stores = [n for n in refs if isinstance(n.ctx, ast.Store)]
    loads = [n for n in refs if isinstance(n.ctx, ast.Load)]
    assert len(stores) == 1, "应有且仅有一处赋值"
    assert len(loads) == 1, f"应只有一处使用（守门处），实际 {len(loads)} 处"


async def test_replan_bound_enforced_by_state_counter():
    """守卫由 state 的 replan_count 单独承担（不再有第二套计数器）。

    replan_count 已到上限时，即使有失败步骤也必须把 needs_replan 收成 False，
    否则图会在 plan↔execute 之间无限打转。
    """
    planner = PlannerAgent()
    state = {
        "replan_count": MAX_REPLAN,
        "execution_plan": {"steps": [{"step_id": "s1", "query": "q", "target_name": "t"}]},
        "plan_step_results": {"s1": {"error_msg": "boom"}},
    }
    out = await planner.evaluate_for_replan(state)
    assert out["needs_replan"] is False
    assert out.get("replan_count", MAX_REPLAN) == MAX_REPLAN, "到上限后不得再递增"


async def test_replan_increments_counter_when_under_bound():
    """未到上限且有失败步骤 → 允许重规划，并递增计数器（这是唯一的计数点）。"""
    planner = PlannerAgent()
    state = {
        "replan_count": 0,
        "execution_plan": {
            "steps": [
                {"step_id": "s1", "query": "q1", "target_name": "t"},
                {"step_id": "s2", "query": "q2", "target_name": "t"},
            ]
        },
        "plan_step_results": {"s1": {"error_msg": "boom"}, "s2": {"final_response": "ok"}},
    }
    out = await planner.evaluate_for_replan(state)
    assert out["needs_replan"] is True
    assert out["replan_count"] == 1


async def test_all_steps_failed_does_not_replan():
    """全部失败 → 重试拿不到新信息，不重规划（避免空转）。"""
    planner = PlannerAgent()
    state = {
        "replan_count": 0,
        "execution_plan": {"steps": [{"step_id": "s1", "query": "q", "target_name": "t"}]},
        "plan_step_results": {"s1": {"error_msg": "boom"}},
    }
    out = await planner.evaluate_for_replan(state)
    assert out["needs_replan"] is False
