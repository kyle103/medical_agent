"""量化：checkpointer 每轮会写多少字节？把大字段排除后能省多少？

用法：./.venv/Scripts/python.exe scripts/probe_checkpointer_size.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.channels import UntrackedValue  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

# 模拟真实体量：history 最多 12 条；RAG 召回 20 段、每段约 200 字
HISTORY = [{"role": "user", "content": "x" * 60} for _ in range(12)]
CHUNKS = {"chunks": ["y" * 200 for _ in range(20)]}
RUNTIME = {"pending_confirmation": {"drug": "阿司匹林", "dose": "100mg"}}


def make_nodes():
    def node_a(state: dict) -> dict:
        return {
            "history": HISTORY,
            "retrieved_knowledge": CHUNKS,
            "session_runtime_state": RUNTIME,
            "intent": "drug",
        }

    def node_b(state: dict) -> dict:
        return {"plan_step_results": {"s1": "z" * 300}, "reconcile_note": "ok"}

    def node_c(state: dict) -> dict:
        return {"final_response": "done"}

    return node_a, node_b, node_c


def build(state_schema: Any):
    a, b, c = make_nodes()
    g = StateGraph(state_schema)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_node("c", c)
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    return g.compile(checkpointer=InMemorySaver())


class Plain(TypedDict, total=False):
    user_input: str
    history: list
    retrieved_knowledge: dict
    session_runtime_state: dict
    intent: str
    plan_step_results: dict
    reconcile_note: str
    final_response: str


class Slim(TypedDict, total=False):
    user_input: str
    intent: str
    final_response: str
    # 只在单次运行内需要、体量大 → 不进快照
    history: Annotated[list, UntrackedValue]
    retrieved_knowledge: Annotated[dict, UntrackedValue]
    session_runtime_state: Annotated[dict, UntrackedValue]
    plan_step_results: Annotated[dict, UntrackedValue]
    reconcile_note: Annotated[str, UntrackedValue]


def measure(app: Any, label: str, turns: int = 3) -> dict[str, int]:
    cfg = {"configurable": {"thread_id": "t-" + label}}
    payloads: list[int] = []
    for i in range(turns):
        app.invoke({"user_input": f"turn{i}"}, config=cfg)
        snap = app.get_state(cfg)
        payloads.append(len(json.dumps(dict(snap.values), ensure_ascii=False, default=str).encode("utf-8")))
    n_snapshots = sum(1 for _ in app.get_state_history(cfg))
    last = payloads[-1]
    print(f"{label}")
    print(f"  快照总数（{turns} 轮）           : {n_snapshots}")
    print(f"  末次快照 body                  : {last:,} 字节")
    print(f"  估算累计写入                   : {last * n_snapshots:,} 字节  （末次体量 × 快照数，上界估计）")
    return {"snapshots": n_snapshots, "last_bytes": last, "est_total_bytes": last * n_snapshots}


print("量化对比：大字段进快照 vs 用 UntrackedValue 排除\n")
plain = measure(build(Plain), "Plain（当前写法）")
print()
slim = measure(build(Slim), "Slim（UntrackedValue 排除大字段）")

print("\n" + "=" * 62)
print("结论")
print("=" * 62)
ratio = plain["est_total_bytes"] / slim["est_total_bytes"] if slim["est_total_bytes"] else 0
print(f"  末次快照体量: {plain['last_bytes']:,} → {slim['last_bytes']:,} 字节"
      f"（降低 {100 * (1 - slim['last_bytes'] / plain['last_bytes']):.1f}%）")
print(f"  估算累计写入: {plain['est_total_bytes']:,} → {slim['est_total_bytes']:,} 字节"
      f"（降低 {100 * (1 - slim['est_total_bytes'] / plain['est_total_bytes']):.1f}%）")
print(f"  倍数关系    : 约 {ratio:.1f}x")
with open("probe_checkpointer_size_result.json", "w", encoding="utf-8") as f:
    json.dump({"plain": plain, "slim": slim, "ratio": ratio}, f, ensure_ascii=False, indent=2)
print("  报告已写入: probe_checkpointer_size_result.json")
