"""Step 3 / 2.1 冒烟：验证 langgraph 0.2.33 的 stream writer 可达性。

设计文档要求"落地前先做 5 行冒烟"：起最小图，节点内推一条消息，
astream(stream_mode='custom') 能收到即通过。此脚本把三种可能的写法一次验完，
以便确定 llm_generate 的改造形态。

结论用途：
  A. writer: StreamWriter 参数注入  -> 若通过，直接改 llm_generate 签名，无需助手模块
  B. writer 无注解                   -> 兼容写法
  C. contextvar 手工取 writer        -> 兜底方案（若 A/B 不通过）

退出码 0 = A 或 B 至少一种可用（即参数注入方案成立）。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import StreamWriter

results: dict[str, Any] = {}


# --- A: 显式注解 StreamWriter ---
async def node_a(state: dict, writer: StreamWriter) -> dict:
    writer({"type": "chunk", "content": "from_A", "probe": "A"})
    return {**state, "a": 1}


# --- B: 不写注解 ---
async def node_b(state: dict, writer) -> dict:  # noqa: ANN001
    writer({"type": "chunk", "content": "from_B", "probe": "B"})
    return {**state, "b": 1}


# --- C: contextvar 手工取 ---
async def node_c(state: dict) -> dict:
    try:
        from langchain_core.runnables.config import var_child_runnable_config
        from langgraph.constants import CONF, CONFIG_KEY_STREAM_WRITER

        cfg = var_child_runnable_config.get()
        w = (cfg or {}).get(CONF, {}).get(CONFIG_KEY_STREAM_WRITER)
        results["C_writer_found"] = w is not None
        if w:
            w({"type": "chunk", "content": "from_C", "probe": "C"})
    except Exception as e:  # noqa: BLE001
        results["C_error"] = f"{type(e).__name__}: {e}"
    return {**state, "c": 1}


def build(node_fn) -> Any:
    g = StateGraph(dict)
    g.add_node("n", node_fn)
    g.set_entry_point("n")
    g.add_edge("n", END)
    return g.compile()


def _normalize(ev: Any) -> tuple[str, Any]:
    """0.2.33 实测：stream_mode 传 list 时产出 2 元组 (mode, payload)；
    subgraphs=True 时才是 3 元组 (namespace, mode, payload)；单 mode 时直接是 payload。"""
    if isinstance(ev, tuple):
        if len(ev) == 2:
            return str(ev[0]), ev[1]
        if len(ev) == 3:
            return str(ev[1]), ev[2]
    return "custom", ev


async def collect(node_fn) -> list[tuple[str, Any]]:
    """返回 astream(stream_mode=['updates','custom']) 归一化后的 (mode, payload) 列表。"""
    out = []
    async for ev in build(node_fn).astream({}, stream_mode=["updates", "custom"]):
        out.append(_normalize(ev))
    return out


async def main() -> int:
    ok = True

    for label, fn in (("A", node_a), ("B", node_b), ("C", node_c)):
        events = await collect(fn)
        customs = [e for e in events if e[0] == "custom"]
        updates = [e for e in events if e[0] == "updates"]
        results[f"{label}_custom_events"] = customs
        results[f"{label}_update_events"] = len(updates)

        hit = any(isinstance(c[1], dict) and c[1].get("probe") == label for c in customs)
        results[f"{label}_ok"] = hit
        print(f"[{label}] custom={len(customs)} updates={len(updates)} writer_reachable={hit}")
        if not hit:
            ok = False

    # 反向验证：不经图、直接调用带 writer 的节点函数 -> 应报缺参数（说明注入只发生在图执行内）
    try:
        await node_a({})  # type: ignore[call-arg]
        results["direct_call"] = "no-error (unexpected)"
    except TypeError as e:
        results["direct_call"] = f"TypeError: {e}"
    except Exception as e:  # noqa: BLE001
        results["direct_call"] = f"{type(e).__name__}: {e}"
    print(f"[direct] 直接调用 node_a(无 writer) -> {results['direct_call']}")

    print("\n=== 结论 ===")
    print(f"A(注解注入) = {results.get('A_ok')}")
    print(f"B(无注解)   = {results.get('B_ok')}")
    print(f"C(contextvar 找到 writer) = {results.get('C_writer_found')}")
    verdict = "参数注入方案成立，无需助手模块" if (results.get("A_ok") or results.get("B_ok")) else "参数注入不可用，需走 contextvar"
    print(f"判定：{verdict}")

    import json

    with open("scripts/smoke_stream_writer_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
