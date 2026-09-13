"""BUG-1 复现：合规拦截会把整个 session 永久毒化（共享 InMemorySaver）。

⚠️ 必须用同一个 MedicalAgent 实例跑两轮，否则 InMemorySaver 在新实例里被重置，
bug 不会复现。这正好揭示了 Step 4 的一个隐藏问题：当前生产 chat_router.py 每请求
新建 MedicalAgent，导致 InMemorySaver 等价于无——但只要换成持久 saver（Step 5）
或单例化 chat_router 里的 agent，Bug-1 就会立刻暴露。

机制：
- error_msg 是 state.py:114 未标注的字段 → 进 checkpointer 快照
- 全仓只有 orchestrator.py:39 在子状态里 pop 它，主图状态从不重置
- _need_error 是 input_check 之后的第一个条件边：
    return "err" if state.get("error_msg") else "mem_load"
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


def _is_blocked(text: str) -> bool:
    return "已按合规要求拦截" in text or "涉及诊疗或用药决策" in text


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
    thread_id = "bug1-repro-poison"
    agent = MedicalAgent()  # 共用同一实例 → InMemorySaver 持久

    # TURN 1：触发合规拦截
    print("=" * 78)
    print(f"[TURN1] user_input='帮我开药'")
    print("=" * 78)
    events1 = await _run_on(agent, thread_id=thread_id, user_input="帮我开药")
    answer1 = _collect_text(events1)
    print(f"  answer: {answer1[:80]}")
    print(f"  blocked: {_is_blocked(answer1)}")

    # TURN 2：完全正常的医疗问题
    print()
    print("=" * 78)
    print(f"[TURN2] user_input='你好，请问高血压平时要注意什么？'")
    print("=" * 78)
    events2 = await _run_on(agent, thread_id=thread_id, user_input="你好，请问高血压平时要注意什么？")
    answer2 = _collect_text(events2)
    print(f"  answer: {answer2[:120]}")
    print(f"  blocked: {_is_blocked(answer2)}")

    # 验证快照里残留的 error_msg
    print()
    print("=" * 78)
    print("[快照残留检查]")
    print("=" * 78)
    snap = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
    if snap is not None:
        print(f"  snapshot.error_msg = {snap.values.get('error_msg')!r}")
    else:
        print("  get_state 返回 None")

    print()
    print("=" * 78)
    print("[诊断]")
    print("=" * 78)
    if _is_blocked(answer2):
        print(f"❌ BUG-1 复现成功：TURN2 复读了拦截语")
        print(f"   TURN1 answer = {answer1[:60]!r}")
        print(f"   TURN2 answer = {answer2[:60]!r}")
        return 1
    elif answer1 == answer2 and len(answer1) > 0:
        print(f"⚠️  TURN1 与 TURN2 完全相同（可能是另一类毒化）")
        return 1
    else:
        print(f"✅ 未复现：TURN2 给出正常回答")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))