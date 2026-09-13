"""A/B：同一份代码，只切换 checkpointer 有无，判定 'Recursion limit reached' 的引入方。"""
import asyncio
import sys

sys.path.insert(0, ".")

from app.core.agent.workflow import MedicalAgent

QUERY = "我在吃阿司匹林和华法林，还想加布洛芬，一起吃有风险吗？"


def build(with_checkpointer: bool) -> MedicalAgent:
    agent = MedicalAgent.__new__(MedicalAgent)
    agent.checkpointer = MedicalAgent.__init__.__defaults__  # placeholder, replaced below
    if with_checkpointer:
        from langgraph.checkpoint.memory import InMemorySaver
        agent.checkpointer = InMemorySaver()
    else:
        agent.checkpointer = None
    agent.graph = agent._build()
    return agent


async def run_one(label: str, with_ckpt: bool, sid: str):
    agent = build(with_ckpt)
    try:
        r = await agent.run(
            user_id="ab", session_id=sid, user_input=QUERY, stream=False, enable_archive_link=False
        )
        print("  [%s] ✓ 完成，回答长度=%d" % (label, len(r.get("assistant_output") or "")))
        print("      前80字:", (r.get("assistant_output") or "")[:80].replace("\n", " "))
    except Exception as e:  # noqa: BLE001
        print("  [%s] ✗ 异常: %s: %s" % (label, type(e).__name__, str(e)[:120]))


async def main():
    print("=" * 72)
    print("A/B：同一查询、同一份代码，只切换 checkpointer")
    print("查询:", QUERY)
    print("=" * 72)
    await run_one("无 checkpointer", False, "ab-no-1")
    await run_one("有 checkpointer", True, "ab-yes-1")


asyncio.run(main())
