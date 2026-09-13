"""抓 'Recursion limit reached' 的完整栈，确认抛出点。"""
import asyncio
import sys
import traceback

sys.path.insert(0, ".")

from app.core.agent.workflow import MedicalAgent

QUERY = "我在吃阿司匹林和华法林，还想加布洛芬，一起吃有风险吗？"


async def main():
    agent = MedicalAgent()
    try:
        await agent.run(
            user_id="tb-probe", session_id="tb-probe-1", user_input=QUERY,
            stream=False, enable_archive_link=False,
        )
        print("无异常")
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb)


asyncio.run(main())
