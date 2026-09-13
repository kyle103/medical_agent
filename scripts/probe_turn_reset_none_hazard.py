"""验证 turn_reset 的重置值是否满足「类型正确」与「.get(k, 默认值) 语义不被破坏」。

核心不变量：`None` 与「键不存在」对 `state.get(k, 默认值)` **不等价**。
reset 写 None 之后键仍存在，`.get(k, 默认值)` 会拿到 None 而不是默认值，
后续的比较 / 算术 / 属性访问即崩。

本脚本只做确定性单元级断言，不依赖 LLM。
"""
import asyncio
import sys

sys.path.insert(0, ".")

from app.core.agent.nodes import turn_reset, _TURN_LOCAL_FIELDS, _TURN_LOCAL_DEFAULTS
from app.core.agent import state as state_mod

# 真实读取点：(字段, 断言「reset 后该读取点仍能安全执行且语义与缺失一致」)
# 每个断言都刻意写得和真实读取点一致——用 `.get(k, 默认值)` 取，而不是直接用 full[k]。
READ_SITES = [
    ("replan_count",
     lambda s: (s.get("replan_count", 0) >= 2) is False,
     "planner_agent.py:700→702 `state.get('replan_count', 0)` 紧接 `>= MAX_REPLAN`"),
    ("execution_plan",
     lambda s: isinstance(s.get("execution_plan", {}).get("steps", []), list),
     "nodes.py:847 `state.get('execution_plan', {})` 紧接 `.get('steps')`"),
    ("intent_confidence",
     lambda s: s.get("intent_confidence", 0.0) == 0.0,
     "stream_events.py:84 / workflow.py:161 `state.get('intent_confidence', 0.0)`"),
    ("needs_confirmation",
     lambda s: not (s.get("needs_confirmation") and "x"),
     "nodes.py:1331 `if state.get('needs_confirmation') and state.get('confirmation_message')`"),
    ("intent_analysis",
     lambda s: isinstance(s.get("intent_analysis") or {}, dict),
     "stream_events.py:81 `state.get('intent_analysis') or {}`"),
    ("tool_result",
     lambda s: isinstance(s.get("tool_result") or {}, dict),
     "reconcile_node `(single.get('tool_result') or {})`"),
]

print("=" * 78)
print("重置清单 %d 个字段 / 默认值 %d 项" % (len(_TURN_LOCAL_FIELDS), len(_TURN_LOCAL_DEFAULTS)))
print("=" * 78)

# --- 1) 构造「第 1 轮**全部**字段都产生过真值」的状态，过一遍 turn_reset ---
# 必须覆盖全部 27 个字段：turn_reset 只重置「存在于 state 的」键，
# 漏掉某个字段会让它保持缺失，从而掩盖该字段默认值是否有问题。
def _truthy_like(default):
    if isinstance(default, bool):
        return True
    if isinstance(default, int) and not isinstance(default, bool):
        return 1
    if isinstance(default, float):
        return 0.93
    if isinstance(default, dict):
        return {"sentinel": 1}
    if isinstance(default, list):
        return [{"sentinel": 1}]
    return "sentinel" if not default else "executing"


print("\n[1] 满值状态 → turn_reset，检查每个字段的类型正确性")
full = {k: _truthy_like(d) for k, d in _TURN_LOCAL_DEFAULTS.items()}
assert set(full) == set(_TURN_LOCAL_FIELDS), "覆盖不全"
asyncio.run(turn_reset(full))

bad_none = [k for k in _TURN_LOCAL_FIELDS if full.get(k) is None]
wrong = [k for k, d in _TURN_LOCAL_DEFAULTS.items() if full.get(k) != d]
print("  仍为 None 的字段:", bad_none or "无 ✓")
print("  与默认值不符的字段:", wrong or "无 ✓")

# --- 2) 类型正确性：reset 后的值必须与声明类型一致 ---
print("\n[2] reset 后的值（对照 state.py 声明）")
for k in _TURN_LOCAL_FIELDS:
    v = full.get(k)
    print("    %-28s = %-12r (%s)" % (k, v, type(v).__name__))

# --- 3) 真实读取点：reset 之后仍能安全执行 ---
print("\n[3] 真实读取点验证")
all_ok = True
for field, check, where in READ_SITES:
    try:
        passed = check(full)
    except Exception as e:  # noqa: BLE001
        passed = False
        print("    ✗ %-20s 抛 %s: %s" % (field, type(e).__name__, e))
    else:
        print("    %s %-20s %s" % ("✓" if passed else "✗", field, where))
    all_ok = all_ok and passed

# --- 4) 关键反例：.get(k, 默认值) 必须等价于「键不存在」 ---
print("\n[4] 语义等价性：`in state` 与 `.get(k, 默认值)`")
for field, default in [("replan_count", 0), ("execution_plan", {}), ("intent_confidence", 0.0)]:
    present = full.get(field, default)
    absent = {}.get(field, default)
    same = present == absent
    print("    %-20s 有键=%r  无键=%r  %s" % (field, present, absent, "等价 ✓" if same else "不等价 ✗"))
    all_ok = all_ok and same

print()
print("结论:", "全部通过 ✓" if (all_ok and not bad_none) else "存在失败 ✗")
