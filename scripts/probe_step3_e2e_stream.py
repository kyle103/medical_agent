"""真实图 run_stream 端到端：确认合并后整条链路的事件序列与收尾都成立。

合成图只验证了"事件映射逻辑"；这里用真实 MedicalAgent（13 节点 + 条件边 +
真实的 execute/reconcile）跑一遍，确认没有只在真实图里才暴露的问题。

依赖 DB / Milvus，若不可用会在事件序列里直接体现出来（正是要看的东西）。
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


async def main() -> int:
    agent = MedicalAgent()
    print("图节点:", sorted(agent.graph.nodes.keys()))

    events: list[dict] = []
    async for line in agent.run_stream(
        user_id="probe-user",
        session_id="probe-e2e-session",
        user_input="你好",
        enable_archive_link=False,
    ):
        evt = json.loads(line.strip())
        events.append(evt)
        label = evt.get("node") or evt.get("intent") or (evt.get("content") or "")[:30]
        print(f"  {evt['type']:<9} {label}")

    kinds = [e["type"] for e in events]
    print("\n--- 断言 ---")
    checks = [
        ("有 progress 事件", any(k == "progress" for k in kinds)),
        ("至少一次 intent", kinds.count("intent") >= 1),
        ("有 chunk 事件", any(k == "chunk" for k in kinds)),
        ("以 done 结尾", kinds[-1] == "done"),
        ("无 error 事件", "error" not in kinds),
        ("chunk 内容非空", bool("".join(e.get("content", "") for e in events if e["type"] == "chunk").strip())),
    ]
    ok = True
    for name, cond in checks:
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")
        ok &= cond

    done = events[-1] if events else {}
    print(f"\ndone: conversation_turns={done.get('conversation_turns')} cache={done.get('cache')}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
