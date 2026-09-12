"""Step 3 真实链路冒烟：验证 `llm_generate` 真的会经 writer 推流，且四个分支行为正确。

为什么必须真实跑一次（不能只信 mock）：
    本次改造的**唯一实质变化**是「token 不再由 run_stream 直接调 stream API，而是
    由图节点内 writer 推出」。mock 只能证明"writer 被调用时事件映射正确"，
    证明不了「节点真的拿到了 writer」——而 writer 注入是 langgraph 按
    「参数名 == writer 且注解 == StreamWriter」做的，**名字/注解写错不报错，
    只会静默退化成非流式**（回答照样出，只是不再逐字流出来，最难查的一类故障）。

覆盖：
    normal        —— 真实 LLM，验证逐 token 推流 + 首字延迟 + 拼接一致
    final_response—— 不走 LLM，验证按句切分
    confirmation  —— 不走 LLM，验证单 chunk + 不发 intent
    drug_confirmation —— 不走 LLM，验证单 chunk + 不发 intent

用法（在 medical_agent 目录下执行）：
    ./.venv/Scripts/python.exe scripts/probe_step3_stream_smoke.py

会真实调用 LLM（约 1 次请求）。不打印、不落盘任何密钥。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.agent import nodes  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


class _Capture:
    """冒充 langgraph 注入的 writer，把节点推来的事件收下来。"""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.first_chunk_at: float | None = None
        self._t0 = time.perf_counter()

    def __call__(self, payload: Any) -> None:
        if self.first_chunk_at is None and isinstance(payload, dict) and payload.get("type") == "chunk":
            self.first_chunk_at = time.perf_counter() - self._t0
        self.events.append(payload)

    @property
    def kinds(self) -> list[str]:
        return [e.get("type") for e in self.events]

    @property
    def chunks(self) -> list[str]:
        return [e.get("content", "") for e in self.events if e.get("type") == "chunk"]


async def _branch_normal() -> None:
    cap = _Capture()
    state = {
        "user_id": "probe-user",
        "session_id": "probe-session",
        "user_input": "高血压患者在饮食上要注意什么？请分点简要说明。",
        "history": [],
        "memory_summary": "",
        "long_memory_text": "",
        "retrieved_knowledge": {},
    }
    out = await nodes.llm_generate(state, cap)

    text = "".join(cap.chunks)
    ok = (
        cap.kinds[:1] == ["intent"]            # 开吐前先发精简 intent
        and len(cap.chunks) > 1                # 真的是逐 token，而不是一整段
        and text == out.get("llm_output")
        and bool(text.strip())
    )
    _record(
        "normal 分支逐 token 推流",
        ok,
        f"chunks={len(cap.chunks)} 首字={cap.first_chunk_at:.2f}s 长度={len(text)}",
    )
    if not ok:
        print("     kinds=", cap.kinds[:6], " chunk0=", (cap.chunks[0][:40] if cap.chunks else None))
    print(f"     答案预览: {text[:60]!r}")


async def _branch_final_response() -> None:
    cap = _Capture()
    text = "第一句话。第二句话更长一些，需要累积到十二个字才会吐出去。"
    state = {"final_response": text, "intent": "general", "user_input": "q"}
    out = await nodes.llm_generate(state, cap)
    # 切分规则：遇句末标点立即吐、否则累积满 12 字吐。
    # 于是 "...。" 结尾会单独成段（"。" 自己一段）——这是改造前的既有行为。
    ok = (
        cap.kinds[:1] == ["intent"]
        and set(cap.kinds[1:]) == {"chunk"}
        and len(cap.chunks) >= 2
        and "".join(cap.chunks) == text
        and out["llm_output"] == text
    )
    _record(
        "final_response 分支按句切分",
        ok,
        f"chunks={len(cap.chunks)} 段长={[len(c) for c in cap.chunks]}",
    )


async def _branch_confirmation() -> None:
    cap = _Capture()
    msg = "检测到您想删除用药记录，确认执行吗？"
    state = {
        "needs_confirmation": True,
        "confirmation_message": msg,
        "intent": "general",
        "user_input": "q",
    }
    out = await nodes.llm_generate(state, cap)
    ok = cap.kinds == ["chunk"] and cap.chunks == [msg] and out["llm_output"] == msg
    _record("confirmation 分支单 chunk 且不发 intent", ok, f"kinds={cap.kinds}")


async def _branch_drug_confirmation() -> None:
    cap = _Capture()
    events = [
        {"drug_name": "阿司匹林", "action": "add", "confidence": 0.9},
    ]
    state = {
        "candidate_drug_events": events,
        "intent": "drug_record",
        "user_input": "q",
    }
    out = await nodes.llm_generate(state, cap)
    ok = (
        cap.kinds == ["chunk"]
        and len(cap.chunks) == 1
        and bool(cap.chunks[0].strip())
        and out["llm_output"] == cap.chunks[0]
        and out.get("skill_ctx", {}).get("medication_confirmation", {}).get("candidate_events") == events
    )
    _record("drug_confirmation 分支单 chunk + 写回 skill_ctx", ok, f"kinds={cap.kinds}")


async def _no_writer_default() -> None:
    """不经图直接调用时，writer 必须走兜底 no-op 而不是抛错。"""
    state = {"final_response": "兜底。", "intent": "general", "user_input": "q"}
    out = await nodes.llm_generate(state)
    _record("不传 writer 走 no-op 兜底", out.get("llm_output") == "兜底。")


async def main() -> int:
    print("=== Step 3 流式接管真实链路冒烟 ===\n")
    await _no_writer_default()
    await _branch_final_response()
    await _branch_confirmation()
    await _branch_drug_confirmation()
    await _branch_normal()

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n=== 汇总 ===")
    print(f"通过 {len(RESULTS) - len(failed)}/{len(RESULTS)}")
    if failed:
        print("失败项: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
