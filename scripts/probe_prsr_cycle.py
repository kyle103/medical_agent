"""精确判定 plan_step_results 是否为自引用（循环）结构，并给出引用路径。"""
import asyncio, sys
sys.path.insert(0, ".")
from app.core.agent import nodes
_cap = {}
_orig = nodes.memory_update
async def _spy(state):
    _cap.clear(); _cap.update(state)
    return await _orig(state)
nodes.memory_update = _spy
from app.core.agent.workflow import MedicalAgent

QUERY = "我在吃阿司匹林和华法林，还想加布洛芬，一起吃有风险吗？"

def find_ref(root, target_id, path="$", seen=None, out=None):
    """找 root 中引用 target_id 的路径。"""
    out = out if out is not None else []
    seen = seen if seen is not None else set()
    if id(root) in seen: return out
    if isinstance(root, dict):
        if id(root) == target_id and path != "$": out.append(path); return out
        seen = seen | {id(root)}
        for k, v in root.items(): find_ref(v, target_id, f"{path}.{k}", seen, out)
    elif isinstance(root, (list, tuple)):
        if id(root) == target_id and path != "$": out.append(path); return out
        seen = seen | {id(root)}
        for i, v in enumerate(root): find_ref(v, target_id, f"{path}[{i}]", seen, out)
    return out

async def main():
    agent = MedicalAgent.__new__(MedicalAgent); agent.checkpointer = None; agent.graph = agent._build()
    await agent.run(user_id="cy", session_id="cy-1", user_input=QUERY, stream=False, enable_archive_link=False)
    psr = _cap.get("plan_step_results")
    print("plan_step_results 类型:", type(psr).__name__, " 步骤:", list(psr.keys()) if isinstance(psr, dict) else None)
    refs = find_ref(psr, id(psr))
    print("自引用路径数:", len(refs))
    for r in refs[:8]: print("   ", r[:220])
    # 子结果里还引用了 state 里的哪些大对象
    for sid, res in (psr or {}).items():
        if not isinstance(res, dict): continue
        print(f"\n  [{sid}] 键: {sorted(res.keys())}")
        for k, v in res.items():
            if isinstance(v, (dict, list)) and len(str(type(v))) < 60:
                inner = find_ref(v, id(psr))
                if inner: print(f"      {k} 内部引用了 plan_step_results -> {inner[:2]}")
asyncio.run(main())
