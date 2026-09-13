"""真实图上实测：状态里哪些字段真的占体量（Step 3.5 加 `UntrackedValue` 标注的**判据来源**）。

为什么不用最小图复刻：最小图只能证明"`UntrackedValue` 机制成立"（已由
`probe_checkpointer_payload.py` / `probe_checkpointer_size.py` 完成）；
**该给哪些字段加标注**这件事必须在真实图上量，否则就是拍脑袋。

做法：用与 `run_stream` 完全相同的取态方式（`astream(stream_mode=["updates","custom"])`
+ 逐节点 `final_state.update`）跑一次真实图，拿到合并末态；然后
  1. 逐字段量 JSON 字节数（降序），暴露真正的"大字段"；
  2. 按 `AgentState` 的 `Annotated[..., UntrackedValue]` 标注把字段分成
     "会进快照 / 不进快照"两桶，给出加标注前后的快照体量对比。

依赖 DB / 向量库；跑一次会真实调用 LLM（一次，问候语，成本可忽略）。
"""

from __future__ import annotations

import asyncio
import json
import sys
import typing
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.channels import UntrackedValue  # noqa: E402

from app.core.agent.state import AgentState  # noqa: E402
from app.core.agent.workflow import MedicalAgent  # noqa: E402


def _byte_size(value: object) -> tuple[int, str]:
    """返回 (字节数, 错误说明)。序列化失败返回 (-1, 原因)。

    失败本身就是**加 checkpointer 前必须解决的信号**：快照序列化器（JSON/msgpack）
    对循环引用、不可序列化对象同样会失败。
    """
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")), ""
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def _describe(value: object, depth: int = 0, max_depth: int = 3) -> list[str]:
    """浅描对象结构（类型分布），用于定位序列化失败的来源。"""
    pad = "  " * depth
    if depth > max_depth:
        return [f"{pad}..."]
    if isinstance(value, dict):
        lines = [f"{pad}dict({len(value)}) keys={list(value)[:12]}"]
        for k, v in list(value.items())[:12]:
            lines.append(f"{pad}  [{k}] -> {type(v).__name__}")
            if isinstance(v, (dict, list, tuple)):
                lines.extend(_describe(v, depth + 2, max_depth))
        return lines
    if isinstance(value, (list, tuple)):
        lines = [f"{pad}{type(value).__name__}({len(value)})"]
        for i, v in enumerate(list(value)[:5]):
            lines.append(f"{pad}  #{i} -> {type(v).__name__}")
            if isinstance(v, (dict, list, tuple)):
                lines.extend(_describe(v, depth + 2, max_depth))
        return lines
    lines = [f"{pad}{type(value).__name__} = {repr(value)[:120]}"]
    if hasattr(value, "__dict__") and depth < max_depth:
        for k, v in list(vars(value).items())[:8]:
            lines.append(f"{pad}  .{k} -> {type(v).__name__}")
    return lines


def _untracked_fields() -> set[str]:
    """从 AgentState 标注里读出哪些字段被排除出快照。"""
    hints = typing.get_type_hints(AgentState, include_extras=True)
    out: set[str] = set()
    for name, hint in hints.items():
        for meta in typing.get_args(hint):
            if isinstance(meta, UntrackedValue) or (
                isinstance(meta, type) and issubclass(meta, UntrackedValue)
            ):
                out.add(name)
    return out


async def _run_graph(user_id: str, session_id: str, user_input: str) -> dict:
    agent = MedicalAgent()
    state: dict = {
        "user_id": user_id,
        "session_id": session_id,
        "user_input": user_input,
        "stream": True,
        "enable_archive_link": False,
    }
    final_state: dict = dict(state)
    async for mode, payload in agent.graph.astream(
        state, config={"callbacks": None}, stream_mode=["updates", "custom"]
    ):
        if mode != "updates" or not isinstance(payload, dict):
            continue
        for node_name, node_output in payload.items():
            if node_name == "__interrupt__":
                continue
            if isinstance(node_output, dict):
                final_state.update(node_output)
    return final_state


async def main() -> int:
    user_id = sys.argv[1] if len(sys.argv) > 1 else "probe-user"
    session_id = sys.argv[2] if len(sys.argv) > 2 else "probe-step35-snapshot"
    user_input = sys.argv[3] if len(sys.argv) > 3 else "你好"

    print("=" * 78)
    print(f"真实图末态逐字段体量（JSON 字节）  user={user_id} session={session_id}")
    print(f"输入：{user_input}")
    print("=" * 78)

    final_state = await _run_graph(user_id, session_id, user_input)
    untracked = _untracked_fields()

    declared = set(typing.get_type_hints(AgentState, include_extras=True).keys())
    extra_keys = sorted(set(final_state) - declared)

    rows = sorted(
        ((k, _byte_size(v)) for k, v in final_state.items()),
        key=lambda r: r[1][0],
        reverse=True,
    )

    print(f"{'字段':<42}{'字节':>10}  快照")
    print("-" * 78)
    total_all = 0
    total_tracked = 0
    total_untracked = 0
    failed: list[str] = []
    for key, (size, err) in rows:
        if size < 0:
            failed.append(key)
            mark = "排除" if key in untracked else "写入"
            print(f"{key:<42}{'序列化失败':>10}  {mark}")
            continue
        total_all += size
        mark = "排除" if key in untracked else "写入"
        if key in untracked:
            total_untracked += size
        else:
            total_tracked += size
        flag = "" if key in declared else "  (未在 AgentState 声明)"
        print(f"{key:<42}{size:>10}  {mark}{flag}")

    print("-" * 78)
    print(f"字段数：{len(rows)}（其中未声明 {len(extra_keys)}：{extra_keys or '无'}）")
    print(f"快照会写入的字段数：{len([k for k, _ in rows if k not in untracked])}")
    print(f"被排除的字段数    ：{len([k for k, _ in rows if k in untracked])}")
    print()
    print(f"不处理（全部进快照）  ：{total_all:>8} 字节")
    print(f"加 UntrackedValue 标注：{total_tracked:>8} 字节")
    print(f"被排除掉的体量        ：{total_untracked:>8} 字节")
    if total_tracked > 0:
        print(f"→ 单次快照下降约 {total_all / total_tracked:.1f} 倍")
    print()

    if failed:
        print("=" * 78)
        print(f"!! 以下字段无法 JSON 序列化：{failed}")
        print("（上表合计未计入这些字段的体量）")
        print("=" * 78)
        for key in failed:
            print(f"\n--- {key} 结构 ---")
            for line in _describe(final_state[key]):
                print(line)
        print()

    if not untracked:
        print("[提示] 当前 AgentState 尚无任何 UntrackedValue 标注 → 上面就是「改造前」基线。")
    else:
        print(f"[提示] 已标注 {len(untracked)} 个字段：{sorted(untracked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
