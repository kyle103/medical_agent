from __future__ import annotations

import asyncio
import time

from app.core.agent.workflow import MedicalAgent


async def main() -> None:
    t0 = time.perf_counter()
    out = await MedicalAgent().run(
        user_id="u_perf",
        session_id="s_perf",
        user_input="你好",
        stream=False,
        enable_archive_link=True,
    )
    dt = time.perf_counter() - t0
    print("elapsed", round(dt, 3))
    print("intent", out.get("intent"))
    print("assistant_head", (out.get("assistant_output") or "")[:120])


if __name__ == "__main__":
    asyncio.run(main())
