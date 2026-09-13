"""抓 plan<->execute 循环的真实节点序列，判定 Recursion limit 是「有界但超限」还是「真死循环」。"""
import asyncio
import sys

sys.path.insert(0, ".")

from app.core.agent.workflow import MedicalAgent

QUERY = "我在吃阿司匹林和华法林，还想加布洛芬，一起吃有风险吗？"


async def main():
    agent = MedicalAgent()
    state = {
        "user_id": "loop-probe",
        "session_id": "loop-probe-1",
        "user_input": QUERY,
        "stream": False,
        "enable_archive_link": False,
    }
    seq = []
    try:
        async for ev in agent.graph.astream(
            state,
            config={"configurable": {"thread_id": "loop-probe:loop-probe-1"}, "recursion_limit": 60},
            stream_mode="updates",
        ):
            for node in ev:
                seq.append(node)
                rep = ""
                if node in ("execute_node", "plan_node"):
                    snap = ev[node] or {}
                    rep = " needs_replan=%r replan_count=%r" % (
                        snap.get("needs_replan"), snap.get("replan_count"),
                    )
                print("  ->", node, rep)
    except Exception as e:  # noqa: BLE001
        print("  异常:", type(e).__name__, e)

    print()
    print("节点总数:", len(seq))
    print("plan 次数:", seq.count("plan_node"), " execute 次数:", seq.count("execute_node"))
    print("完整序列:", seq)


asyncio.run(main())
