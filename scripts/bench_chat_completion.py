from __future__ import annotations

import argparse
import asyncio
import time

from app.core.agent.workflow import MedicalAgent


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", default="u_perf")
    p.add_argument("--session-id", default="s_perf")
    p.add_argument("--text", default="我昨天吃的什么药")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--stream", action="store_true")
    args = p.parse_args()

    agent = MedicalAgent()

    for i in range(args.n):
        t0 = time.perf_counter()
        out = await agent.run(
            user_id=args.user_id,
            session_id=args.session_id,
            user_input=args.text,
            stream=bool(args.stream),
            enable_archive_link=True,
        )
        dt = time.perf_counter() - t0
        print(f"#{i+1} elapsed={dt:.3f}s intent={out.get('intent')} head={(out.get('assistant_output') or '')[:60]!r}")


if __name__ == "__main__":
    asyncio.run(main())
