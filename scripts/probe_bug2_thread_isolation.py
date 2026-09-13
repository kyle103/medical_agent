"""BUG-2 复现：thread_id 没绑定 user_id → 跨用户串话（共享 InMemorySaver）。

⚠️ 必须用同一个 MedicalAgent 实例跑两个 user，否则 InMemorySaver 在新实例里被重置，
bug 不会复现。

机制：
- chat_router.py:25 session_id = req.session_id（客户端控制，仅空时才生成 uuid）
- workflow.py:131, 206 thread_id = session_id（无 user_id 前缀）
- 同一 session_id 被两个不同 user 复用时，InMemorySaver 按 thread_id 索引
  → user-B 的运行从 user-A 留下的快照里读出 state

实测关注点：
1) thread_id 隔离：用同一 agent（同一 saver），user-A 和 user-B 用同一个 session_id，
   B 跑完后查 snapshot.user_id 应该 == "user-B"（不是 "user-A"）
2) final_response 残留：user-B 的 final_response 不应包含 user-A 留下的内容
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


async def _run_on(agent: MedicalAgent, *, user_id: str, session_id: str, user_input: str) -> list[dict]:
    out: list[dict] = []
    async for line in agent.run_stream(
        user_id=user_id,
        session_id=session_id,
        user_input=user_input,
        enable_archive_link=False,
    ):
        out.append(json.loads(line.strip()))
    return out


async def main() -> int:
    shared_session = "bug2-repro-shared"
    agent = MedicalAgent()  # 共用 saver

    # user-A 先跑
    print("=" * 78)
    print(f"[user-A] session='{shared_session}' input='你好'")
    print("=" * 78)
    events_a = await _run_on(agent, user_id="user-A", session_id=shared_session, user_input="你好")
    answer_a = _collect_text(events_a)
    print(f"  answer: {answer_a[:80]}")

    # 看快照
    snap_a = agent.graph.get_state({"configurable": {"thread_id": shared_session}})
    if snap_a is not None:
        print(f"  snapshot.user_id = {snap_a.values.get('user_id')!r}")
        print(f"  snapshot.final_response (前 40) = {snap_a.values.get('final_response', '')[:40]!r}")

    # user-B 用同一个 session_id 跑一个完全不同的问题
    print()
    print("=" * 78)
    print(f"[user-B] session='{shared_session}' (与 A 同) input='布洛芬有什么副作用？'")
    print("=" * 78)
    events_b = await _run_on(agent, user_id="user-B", session_id=shared_session, user_input="布洛芬有什么副作用？")
    answer_b = _collect_text(events_b)
    print(f"  answer: {answer_b[:120]}")

    # 看 B 的快照
    snap_b = agent.graph.get_state({"configurable": {"thread_id": shared_session}})
    if snap_b is not None:
        print(f"  snapshot.user_id (after B) = {snap_b.values.get('user_id')!r}")

    print()
    print("=" * 78)
    print("[诊断]")
    print("=" * 78)
    issues = []
    # 1) 同一 thread 的 snapshot.user_id 应当随最新一次写入而更新（不严格，但至少不应是 A 的内容当成 B 的 final_response）
    if snap_b is not None:
        if snap_b.values.get("user_id") != "user-B":
            issues.append(f"snapshot.user_id 应为 'user-B'，实际 {snap_b.values.get('user_id')!r}")
    # 2) B 的答案不应包含 A 留下的内容（关键词交叉检查）
    if answer_a and "你好" in answer_a[:60] and answer_b.startswith(answer_a[:60]):
        issues.append("user-B 的回答前缀与 user-A 相同 → 可能是 A 的 final_response 被复用")

    if issues:
        print(f"\n❌ BUG-2 复现成功（{len(issues)} 项）:")
        for s in issues:
            print(f"   - {s}")
        return 1
    else:
        print("\n✅ 未复现：同一 thread 隔离了 user 维度（或 B 的答案确实是针对 B 的输入）")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))