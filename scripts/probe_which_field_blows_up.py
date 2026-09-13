"""定位：checkpointer 序列化时到底哪个字段/载荷把 ormsgpack 撑爆。

做法：不带 checkpointer 跑完同一查询，插桩 mem 节点抓取**原始图状态**，
逐字段单独 ormsgpack.packb，找出抛 'Recursion limit reached' 的字段并测嵌套深度。
"""
import asyncio
import sys

sys.path.insert(0, ".")

import ormsgpack

from app.core.agent import nodes

_captured: dict = {}
_orig_mem = nodes.memory_update


async def _spy_mem(state):
    _captured.clear()
    _captured.update(state)
    return await _orig_mem(state)


nodes.memory_update = _spy_mem

from app.core.agent.workflow import MedicalAgent  # noqa: E402

QUERY = "我在吃阿司匹林和华法林，还想加布洛芬，一起吃有风险吗？"


def depth(obj, seen=None, cur=1):
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return -1
    if isinstance(obj, dict):
        if not obj:
            return cur
        seen = seen | {id(obj)}
        return max((depth(v, seen, cur + 1) for v in obj.values()), default=cur)
    if isinstance(obj, (list, tuple)):
        if not obj:
            return cur
        seen = seen | {id(obj)}
        return max((depth(v, seen, cur + 1) for v in obj), default=cur)
    return cur


async def main():
    agent = MedicalAgent.__new__(MedicalAgent)
    agent.checkpointer = None
    agent.graph = agent._build()

    await agent.run(
        user_id="wf", session_id="wf-1", user_input=QUERY, stream=False, enable_archive_link=False
    )
    print("无 checkpointer 跑通。mem 节点收到的原始状态 %d 个键\n" % len(_captured))

    print("%-28s %-24s %-8s %-8s %s" % ("字段", "packb", "深度", "自引用", "类型"))
    print("-" * 88)
    bad = []
    for k, v in sorted(_captured.items()):
        d = depth(v)
        try:
            ormsgpack.packb(v, default=lambda o: None)
            ok = "OK"
        except Exception as e:  # noqa: BLE001
            ok = type(e).__name__
            bad.append(k)
        print("%-28s %-24s %-8s %-8s %s" % (k, ok, d, "是" if d == -1 else "", type(v).__name__))

    print()
    print("整份状态一次性 packb（≈ checkpointer 的实际行为）：")
    try:
        ormsgpack.packb(_captured, default=lambda o: None)
        print("  OK")
    except Exception as e:  # noqa: BLE001
        print("  %s: %s" % (type(e).__name__, e))
    print()
    print("序列化会炸的字段:", bad or "无")


asyncio.run(main())
