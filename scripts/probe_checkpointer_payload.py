"""实验：checkpointer 究竟会把哪些字段写进快照？能否排除大字段？

目的：回答「加了 checkpointer 会不会造成冗余数据保存」。
用最小图复刻主链路的状态结构，直接打印 checkpoint 的内容。

用法：./.venv/Scripts/python.exe scripts/probe_checkpointer_payload.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.channels import EphemeralValue, UntrackedValue  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

print("=== 通道文档说明（用于确认语义） ===")
print("UntrackedValue.__doc__ :", (UntrackedValue.__doc__ or "").strip()[:400])
print()
print("EphemeralValue.__doc__ :", (EphemeralValue.__doc__ or "").strip()[:400])
print()


# ---------- 场景 1：普通 TypedDict（= 项目当前 AgentState 的写法） ----------
class StatePlain(dict):
    pass


from typing import TypedDict  # noqa: E402


class PlainState(TypedDict, total=False):
    user_input: str
    history: list                 # 模拟：最多 12 条消息
    retrieved_knowledge: dict     # 模拟：RAG 召回片段（大对象）
    session_runtime_state: dict   # 模拟：数据库业务态整行副本
    final_response: str


def node_a(state: dict) -> dict:
    return {
        "history": [{"role": "user", "content": "x" * 50} for _ in range(12)],
        "retrieved_knowledge": {"chunks": ["y" * 200] * 20},
        "session_runtime_state": {"pending_confirmation": {"drug": "阿司匹林"}},
    }


def node_b(state: dict) -> dict:
    return {"final_response": "done"}


def build(state_schema: Any):
    g = StateGraph(state_schema)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile(checkpointer=InMemorySaver())


def inspect(app: Any, label: str) -> None:
    cfg = {"configurable": {"thread_id": "t-" + label}}
    app.invoke({"user_input": "hi"}, config=cfg)
    snap = app.get_state(cfg)
    values = dict(snap.values)

    print(f"--- {label} ---")
    print(f"  最终状态字段: {sorted(values.keys())}")

    # 直接读底层 checkpoint 里存了什么
    total = 0
    for cp in app.get_state_history(cfg):
        ch = getattr(cp, "channel_values", None) or {}
        keys = sorted(ch.keys())
        size = sum(len(str(v)) for v in ch.values())
        total = max(total, size)
        print(f"  快照 step={cp.metadata.get('step')} 通道={keys}")
        print(f"    估算体量≈{len(str(ch))} 字符")
        break  # 只看最近一次
    print(f"  历史快照数量: {sum(1 for _ in app.get_state_history(cfg))}")
    print()


print("=== 场景 1：PlainState（等价于项目当前写法） ===")
inspect(build(PlainState), "plain")


# ---------- 场景 2：把大字段标为不持久化通道 ----------
class SlimState(TypedDict, total=False):
    user_input: str
    final_response: str
    history: Annotated[list, UntrackedValue]
    retrieved_knowledge: Annotated[dict, UntrackedValue]
    session_runtime_state: Annotated[dict, UntrackedValue]


print("=== 场景 2：SlimState（大字段用 UntrackedValue 标注） ===")
try:
    inspect(build(SlimState), "slim")
except Exception as e:
    print(f"  Annotated[..., UntrackedValue] 方式失败: {type(e).__name__}: {str(e)[:300]}")
    print()
