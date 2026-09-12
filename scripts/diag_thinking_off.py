"""量化「关闭思考」对决策链路的影响。

已证实现象：本模型是推理型，`max_tokens` 同时覆盖推理与答案。
推理吃满预算时 content=''（finish_reason='length'、out_tok 顶格）——
split_route_deps 的提示词最长，实测连续两次尝试全部空响应。
调大预算 → 又撞 12s 超时。两头堵。

假设：对**结构化决策**这类任务，关掉思考即可同时解决"空响应"与"慢"。
验证方式：只在传输层注入 extra_body={"enable_thinking": False}，
提示词/schema/业务代码全部不动，对比开关前后的延迟、成功率、空响应数。

用法：./.venv/Scripts/python.exe scripts/diag_thinking_off.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.agent.llm_decision_service import LLMDecisionService  # noqa: E402
from app.core.llm.llm_service import get_shared_client  # noqa: E402

TEXT = "我的血糖正常吗？布洛芬能和它一起吃吗？另外帮我记录一下今天吃了阿莫西林"
ROUNDS = 3


async def run(label: str, thinking_off: bool) -> None:
    client = await get_shared_client()
    original = client.chat.completions.create
    seen: list[dict] = []

    async def spy(**kwargs):
        if thinking_off:
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}), "enable_thinking": False}
        t = time.perf_counter()
        resp = await original(**kwargs)
        dt = time.perf_counter() - t
        try:
            ch = resp.choices[0]
            content = ch.message.content or ""
            fr = getattr(ch, "finish_reason", None)
        except Exception:  # noqa: BLE001
            content, fr = "", None
        usage = getattr(resp, "usage", None)
        seen.append(
            {
                "dt": dt,
                "fr": fr,
                "len": len(content),
                "out_tok": getattr(usage, "completion_tokens", None),
            }
        )
        return resp

    client.chat.completions.create = spy  # type: ignore[method-assign]
    try:
        print(f"\n===== {label} =====")
        ok_n = 0
        for i in range(ROUNDS):
            seen.clear()
            t = time.perf_counter()
            out = await LLMDecisionService().split_route_deps(TEXT)
            wall = time.perf_counter() - t
            ok = out[0] is not None
            ok_n += ok
            dropped = sum(1 for r in (out[1] or []) if r is None) if ok else "-"
            lat = " ".join(f"{s['dt']:.1f}s" for s in seen) or "-"
            toks = " ".join(str(s["out_tok"]) for s in seen) or "-"
            lens = " ".join(str(s["len"]) for s in seen) or "-"
            print(
                f"  第{i + 1}次 {'成功' if ok else '失败':<4} 墙钟={wall:5.1f}s "
                f"单次延迟=[{lat}] out_tok=[{toks}] len=[{lens}] 丢弃路由={dropped}"
            )
        print(f"  成功 {ok_n}/{ROUNDS}")
    finally:
        client.chat.completions.create = original  # type: ignore[method-assign]


async def main() -> None:
    await run("思考开启（现状）", thinking_off=False)
    await run("思考关闭 enable_thinking=False", thinking_off=True)


if __name__ == "__main__":
    asyncio.run(main())
