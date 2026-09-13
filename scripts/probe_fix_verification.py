"""验证 turn_reset + _thread_id 修复是否真的消除 BUG-1/2/3，以及是否引入 None 回归。

四个观测点：
  A. 第 2 轮入口状态是否仍携带第 1 轮的 turn-local 字段（BUG-1/3 的根因）
  B. 第 1 轮被合规拦截后，第 2 轮能否给出真实回答（BUG-1）
  C. 两用户共享 session_id 时是否串状态（BUG-2）
  D. 第 1 轮触发过重规划时，第 2 轮 replan_count 是否被置 None 导致 TypeError（新引入）
"""
import asyncio
import sys
import traceback

sys.path.insert(0, ".")

from app.core.agent import nodes
from app.core.agent.nodes import _TURN_LOCAL_FIELDS

_entry_states: list[dict] = []
_orig_input_check = nodes.input_check


async def _spy_input_check(state):
    _entry_states.append({k: type(v).__name__ for k, v in state.items()})
    return await _orig_input_check(state)


nodes.input_check = _spy_input_check

from app.core.agent.workflow import MedicalAgent  # noqa: E402

TURN_LOCAL = set(_TURN_LOCAL_FIELDS)


async def turn(agent, uid, sid, text):
    try:
        r = await agent.run(user_id=uid, session_id=sid, user_input=text, stream=False, enable_archive_link=False)
        return r, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


async def main():
    agent = MedicalAgent()

    print("=" * 72)
    print("B. BUG-1 合规投毒复现")
    print("=" * 72)
    sid = "probe-fix-b1"
    _entry_states.clear()
    r1, e1 = await turn(agent, "u1", sid, "帮我开药")
    print("  第1轮 '帮我开药'      ->", (r1 or {}).get("assistant_output", e1)[:60])
    r2, e2 = await turn(agent, "u1", sid, "你好，请问高血压平时要注意什么？")
    out2 = (r2 or {}).get("assistant_output", e2) or e2
    print("  第2轮 '高血压注意什么' ->", out2[:120])
    if e1 or e2:
        print("  ✗ 抛异常:", e1 or e2)
    stuck = "开药" in out2 and "无法" in out2
    print("  →", "✗ 仍被第1轮拦截消息毒化" if stuck else "✓ 第2轮给出真实回答（BUG-1 已修复）")

    print()
    print("C. BUG-2 跨用户串状态")
    sid_shared = "probe-fix-b2-shared"
    await turn(agent, "userA", sid_shared, "帮我开药")
    rb, eb = await turn(agent, "userB", sid_shared, "你好，请问高血压平时要注意什么？")
    outb = (rb or {}).get("assistant_output", eb) or eb
    print("  userB 回答 ->", str(outb)[:120])
    if eb:
        print("  ✗ 抛异常:", eb)
    print("  →", "✗ userB 继承了 userA 的拦截状态" if "开药" in str(outb) and "无法" in str(outb) else "✓ 隔离（BUG-2 已修复）")

    print()
    print("D. 新风险：replan_count 被置 None")
    print("=" * 72)
    sid3 = "probe-fix-b3"
    _entry_states.clear()
    r3, e3 = await turn(agent, "u3", sid3, "我在吃阿司匹林和华法林，还想加布洛芬，一起吃有风险吗？")
    print("  第1轮 多药查询 ->", str((r3 or {}).get("assistant_output", e3))[:180])
    if e3:
        print("  第1轮异常:", e3)
    r4, e4 = await turn(agent, "u3", sid3, "你好，请问高血压平时要注意什么？")
    print("  第2轮 ->", str((r4 or {}).get("assistant_output", e4))[:180])
    if e4:
        print("  ✗✗ 第2轮抛异常:", e4)
        traceback.print_exc()
    else:
        print("  → 第2轮正常")

    print()
    print("A. 第 2 轮入口状态里的 turn-local 字段（truthy 值即为泄漏）")
    print("=" * 72)
    for i, snap in enumerate(_entry_states):
        print("  入口 #%d: %s" % (i, snap))
    leaks = []
    for snap in _entry_states[1:]:
        for k in TURN_LOCAL:
            if k in snap:
                leaks.append(k)
    print("  第2+轮入口仍出现的 turn-local 字段:", sorted(set(leaks)) or "无")


asyncio.run(main())
