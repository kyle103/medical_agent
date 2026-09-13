"""化验「流程咨询」被当成解读请求的端到端复现 / 验收。

用户实测（2026-09-13）：
    用户：帮我解读一下血常规化验单，我直接发单子给你吗
    助手：化验 · 95% → 未识别到有效的检验指标。请提供指标名称和数值，例如：血糖6.5，血压120/80。
    用户：直接提供文本还是直接发化验单指标给你？
    助手：化验 · 95% → （同一句）

三层问题与对应修复：
1. 解析层（安全）—— `app/core/tools/lab_item_parser.py` 重写。
   旧实现把 `是`/`为` 当分隔符并兜底"抓整句第一个数字"，会把年龄/体重/体温/天数
   当成检验值并输出 H/L 异常判定。
2. 路由层 —— `planner_agent._route_by_intent_and_text` 的 lab 分支与关键词兜底
   都改为「有可读数值才路由到 lab_report」。
3. 生成层 —— 工具零抽取时标 `no_data`，`_decide_response_mode` 据此回到 llm_chat，
   不再把「请提供指标名称和数值」当成化验结论输出。

用法：
    python scripts/probe_lab_meta_question.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.agent.planner_agent import _route_by_intent_and_text  # noqa: E402
from app.core.agent.workflow import MedicalAgent  # noqa: E402
from app.core.tools.lab_item_parser import parse_lab_items  # noqa: E402


#: 旧版那句固定回复，用于判定"是否复读模板"
_OLD_CANNED = "未识别到有效的检验指标"

#: (用户输入, 期望路由 target, 说明)
CASES: list[tuple[str, str, str]] = [
    ("帮我解读一下血常规化验单，我直接发单子给你吗", "main_qa_agent", "用户报的第 1 条：问能不能发图片"),
    ("直接提供文本还是直接发化验单指标给你？", "main_qa_agent", "用户报的第 2 条：问输入方式"),
    ("化验单上的箭头是什么意思", "main_qa_agent", "问符号含义"),
    ("血糖偏高是怎么回事", "main_qa_agent", "追问原因（旧版会抽出 血糖=怎么回事）"),
    ("血糖不高，我今年35岁", "main_qa_agent", "旧版会抽出 血糖=35 并判为偏高"),
    ("血糖6.5", "lab_report", "真数据：必须仍进化验工具"),
    ("血常规：白细胞11.2，血小板150", "lab_report", "真数据（超白名单）：仍进化验工具"),
]


def _collect_text(events: list[dict]) -> str:
    parts = []
    for e in events:
        if e["type"] in ("chunk", "error"):
            parts.append(e.get("content", ""))
    return "".join(parts).strip()


async def _route_check() -> int:
    """A 档：纯规则，无 LLM —— 逐条核对路由与解析结果。"""
    print("=" * 88)
    print("A 档｜路由 + 解析（无 LLM，决定性）")
    print("=" * 88)
    bad = 0
    for text, expect_target, note in CASES:
        items = parse_lab_items(text)
        route = _route_by_intent_and_text(
            {"intent": "lab", "user_input": text, "intent_confidence": 0.95}
        )
        ok = route["target_name"] == expect_target
        if not ok:
            bad += 1
        print(f"  [{'OK' if ok else 'XX'}] {note}")
        print(f"       输入   : {text}")
        print(f"       抽出   : {[(i['item_name'], i['test_value']) for i in items] or '（无）'}")
        print(f"       路由   : {route['target_name']}  ({route['reason']})")
    print()
    return bad


async def _e2e_check() -> int:
    """B 档：真实图端到端 —— 确认答案不再是那句模板。"""
    print("=" * 88)
    print("B 档｜真实图端到端（含 LLM）")
    print("=" * 88)
    agent = MedicalAgent()
    bad = 0
    for i, (text, expect_target, note) in enumerate(CASES, start=1):
        events = []
        async for line in agent.run_stream(
            user_id="probe-lab",
            session_id=f"probe-lab-{i}",   # 每例独立 thread，避免 checkpointer 跨轮干扰
            user_input=text,
            enable_archive_link=False,
        ):
            events.append(json.loads(line.strip()))
        answer = _collect_text(events)
        repeated = _OLD_CANNED in answer
        if expect_target == "main_qa_agent" and repeated:
            bad += 1
        print(f"  [{'OK' if not repeated else 'XX'}] {note}")
        print(f"       输入 : {text}")
        print(f"       回答 : {answer[:150].replace(chr(10), ' / ')}")
        print()
    return bad


async def main() -> int:
    bad = await _route_check()
    bad += await _e2e_check()
    print("=" * 88)
    print(f"结论：{'全部通过' if bad == 0 else f'{bad} 项不符合预期'}")
    print("=" * 88)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
