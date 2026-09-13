"""BUG-3 复现：上一轮的 final_response 会残留并被复读（共享 InMemorySaver）。

⚠️ 必须用同一个 MedicalAgent 实例跑两轮，否则 InMemorySaver 在新实例里被重置，
bug 不会复现。

机制：
- final_response 是 state.py:113 未标注字段 → 进 checkpointer 快照
- 输入侧 input_check 不重置它
- build_generation_prompt (nodes.py:1253) 是短路分支：
    if state.get("final_response"):
        return {"branch": "final_response", ...}    # 不调 LLM
- llm_generate (nodes.py:1414) 看到 branch="final_response" → 把现有 final_response 当答案吐出去

正确性靠 reconcile_node 覆盖 final_response 才不出事（nodes.py:1090）。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.agent.workflow import MedicalAgent  # noqa: E402


def _collect_text(events: list[dict]) -> str:
    parts = []
    for e in events:
        if e["type"] == "chunk":
            parts.append(e.get("content", ""))
        elif e["type"] == "error":
            parts.append(e.get("content", ""))
    return "".join(parts).strip()


def _count_chunks(events: list[dict]) -> int:
    return sum(1 for e in events if e["type"] == "chunk")


async def _run_on(agent: MedicalAgent, *, thread_id: str, user_input: str) -> list[dict]:
    out: list[dict] = []
    async for line in agent.run_stream(
        user_id="probe-user",
        session_id=thread_id,
        user_input=user_input,
        enable_archive_link=False,
    ):
        out.append(json.loads(line.strip()))
    return out


async def main() -> int:
    thread_id = "bug3-repro-fr-leak"
    agent = MedicalAgent()  # 共用 saver

    # TURN 1：问候语（会通过 reconcile_node 设置 final_response）
    print("=" * 78)
    print("[TURN1] user_input='你好'")
    print("=" * 78)
    events1 = await _run_on(agent, thread_id=thread_id, user_input="你好")
    n_chunks_1 = _count_chunks(events1)
    answer1 = _collect_text(events1)
    print(f"  chunks={n_chunks_1}")
    print(f"  answer: {answer1[:80]}")

    # TURN 2：完全不同的问题
    print()
    print("=" * 78)
    print("[TURN2] user_input='高血压有什么要注意的吗'")
    print("=" * 78)
    events2 = await _run_on(agent, thread_id=thread_id, user_input="高血压有什么要注意的吗")
    n_chunks_2 = _count_chunks(events2)
    answer2 = _collect_text(events2)
    print(f"  chunks={n_chunks_2}")
    print(f"  answer: {answer2[:160]}")

    # 看快照里的 final_response
    print()
    print("=" * 78)
    print("[快照残留检查]")
    print("=" * 78)
    snap = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
    if snap is not None:
        print(f"  snapshot.final_response (前 60) = {snap.values.get('final_response', '')[:60]!r}")

    print()
    print("=" * 78)
    print("[诊断]")
    print("=" * 78)
    issues = []
    # 1) TURN2 答案应含 "高血压"
    if "高血压" not in answer2 and len(answer2) > 20:
        issues.append(f"TURN2 答案未含 '高血压' 关键词 → 复读了上轮")
    # 2) 答案不应与 TURN1 高度相似
    if answer1 and answer1[:60] == answer2[:60] and len(answer1) > 60:
        issues.append(f"TURN1 与 TURN2 前 60 字完全相同 → 复读")
    # 3) chunk 数 ≤ 2 通常是短路（正常 LLM 流式会有多个 chunk）
    if n_chunks_2 <= 2 and len(answer1) > 0 and answer1[:30] != answer2[:30]:
        issues.append(f"TURN2 chunks={n_chunks_2}（疑似短路），但 answer2 ≠ answer1 → final_response 残留但被另一条路径部分覆盖")
    print(f"  chunks: TURN1={n_chunks_1} TURN2={n_chunks_2}")
    print(f"  answer1[:30]: {answer1[:30]!r}")
    print(f"  answer2[:30]: {answer2[:30]!r}")
    print(f"  answer2 含 '高血压'? {'高血压' in answer2}")

    if issues:
        print(f"\n❌ BUG-3 复现成功（{len(issues)} 项）:")
        for s in issues:
            print(f"   - {s}")
        return 1
    else:
        print("\n✅ 未复现：TURN2 答案包含输入关键词、与 TURN1 不一致")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))