"""量 `run_stream` 的 chunk 事件到达节奏 —— 判定用户看到的是否"真流式"。

背景：前端 `app.js` 是逐事件 `reader.read()` 增量渲染的，协议层没问题。
但 SSE 事件**产生得太密**时（同一事件循环轮次内全部产出、中间没有 await），
HTTP 会在一两个 TCP 包内全部送达，浏览器在一次 paint 里渲染完 →
用户看到的是"一大坨文字一次性出现"。

本脚本不猜，直接量：记录每个 chunk 事件到达的墙钟时间，打印
「chunk 数 / 首字耗时 / 末字耗时 / 总跨度 / 相邻最大间隔」。

判据：
    - 总跨度 < 50ms 且 chunk 数 > 3  → 假流式（全部同一瞬间产生）
    - 总跨度 >> 首字耗时            → 真流式（逐 token 流入）

用法（在 medical_agent 目录下执行）：
    ./.venv/Scripts/python.exe scripts/probe_stream_timing.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.agent.workflow import MedicalAgent  # noqa: E402

CASES = [
    ("① 通用问答（预期走 LLM 生成）", "高血压平时要注意什么"),
    ("② 药物相互作用（预期走工具）", "布洛芬和对乙酰氨基酚能一起吃吗"),
    ("③ 短闲聊", "你好"),
]


async def measure(label: str, user_input: str, idx: int) -> None:
    agent = MedicalAgent()
    session_id = f"timing-{idx}"
    t_start = time.perf_counter()
    chunk_at: list[float] = []
    kinds: list[str] = []
    first_byte_at: float | None = None

    async for line in agent.run_stream(
        user_id="timing-user",
        session_id=session_id,
        user_input=user_input,
        enable_archive_link=False,
    ):
        now = time.perf_counter() - t_start
        if first_byte_at is None:
            first_byte_at = now
        evt = json.loads(line.strip())
        kinds.append(evt["type"])
        if evt["type"] == "chunk":
            chunk_at.append(now)

    print("=" * 74)
    print(f"{label}   输入：{user_input!r}")
    print(f"  首个事件到达        : {first_byte_at * 1000:.0f} ms" if first_byte_at is not None else "  无事件")
    print(f"  事件序列（前 12）    : {kinds[:12]}")
    print(f"  chunk 事件数        : {len(chunk_at)}")
    if chunk_at:
        span = (chunk_at[-1] - chunk_at[0]) * 1000
        first_chunk = chunk_at[0] * 1000
        last_chunk = chunk_at[-1] * 1000
        gaps = [(b - a) * 1000 for a, b in zip(chunk_at, chunk_at[1:])]
        max_gap = max(gaps) if gaps else 0.0
        print(f"  首 chunk 于         : {first_chunk:.0f} ms")
        print(f"  末 chunk 于         : {last_chunk:.0f} ms")
        print(f"  chunk 总跨度        : {span:.0f} ms")
        print(f"  相邻最大间隔        : {max_gap:.0f} ms")
        verdict = "假流式（同一瞬间全部产生）" if span < 50 and len(chunk_at) > 3 else "真流式"
        print(f"  判定                : {verdict}")
    else:
        print("  判定                : 无 chunk（走 error/done 路径）")


async def main() -> None:
    # 预热：首跑要建 RAG/DB/模型连接，会把第一个用例的耗时抬高一截，
    # 不预热会把冷启动时间误读成"这个分支慢"。预热结果不参与判定。
    print(">>> 预热中（不参与判定）...")
    try:
        await measure("⓪ 预热", "你好", 99)
    except Exception:  # noqa: BLE001
        pass

    for i, (label, inp) in enumerate(CASES):
        try:
            await measure(label, inp, i)
        except Exception as e:  # noqa: BLE001
            print("=" * 74)
            print(f"{label}  异常：{type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
