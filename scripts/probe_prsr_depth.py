"""精确测量 plan_step_results 的最大嵌套深度 + 是否存在真循环。"""
import asyncio, sys
sys.path.insert(0, ".")
from app.core.agent import nodes
_cap = {}; _orig = nodes.memory_update
async def _spy(state):
    _cap.clear(); _cap.update(state); return await _orig(state)
nodes.memory_update = _spy
from app.core.agent.workflow import MedicalAgent
QUERY = "我在吃阿司匹林和华法林，还想加布洛芬，一起吃有风险吗？"

def measure(root):
    """返回 (最大深度, 是否真循环). 用祖先集合判环。"""
    max_d = 0; cycle = False
    stack = [(root, 1, frozenset())]
    while stack:
        node, d, anc = stack.pop()
        max_d = max(max_d, d)
        if isinstance(node, (dict, list, tuple)):
            if id(node) in anc:
                cycle = True; continue
            anc2 = anc | {id(node)}
            vals = node.values() if isinstance(node, dict) else node
            for v in vals:
                if isinstance(v, (dict, list, tuple)):
                    stack.append((v, d + 1, anc2))
                else:
                    max_d = max(max_d, d + 1)
    return max_d, cycle

async def main():
    agent = MedicalAgent.__new__(MedicalAgent); agent.checkpointer = None; agent.graph = agent._build()
    await agent.run(user_id="dp", session_id="dp-1", user_input=QUERY, stream=False, enable_archive_link=False)
    psr = _cap["plan_step_results"]
    d, cyc = measure(psr)
    print("plan_step_results: 最大嵌套深度 =", d, " 真循环 =", cyc)
    d2, cyc2 = measure(_cap)
    print("整个 state      : 最大嵌套深度 =", d2, " 真循环 =", cyc2)
    # 逐层剥开 plan_step_results 链
    cur, lvl = psr, 0
    while isinstance(cur, dict):
        nxt = None
        for sid, res in cur.items():
            if isinstance(res, dict) and isinstance(res.get("plan_step_results"), dict):
                nxt = res["plan_step_results"]; break
        if nxt is None: break
        lvl += 1; cur = nxt
    print("plan_step_results 自身链式嵌套层数 =", lvl)
    # 同一层内 fan-out 统计
    for sid, res in psr.items():
        if isinstance(res, dict):
            print(f"  [{sid}] 体积估计={len(str(res))} 字符")
asyncio.run(main())
