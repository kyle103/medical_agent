"""决定性实验：response_format(json_schema) 的 schema 究竟有没有被下发并强制？

背景：v2 探针显示接口不报错、模型输出 JSON，但字段名始终是模型自己的先验（intent/executor），
不是我们 schema 里的字段名（intent_type/target_name）。但 v2 的「平凡 schema」测试里
prompt 直接写出了字段名，属混淆变量，无法区分「schema 生效」与「模型照抄 prompt」。

本实验用**无提示的独有字段名**做判据：
  若返回里出现 zz_unique_marker 这个字段名 → schema 确实被下发并强制
  若返回的是模型自己的字段          → schema 未生效（被静默忽略）
另含一项非法 response_format 诊断：用于确认提供方是「严格校验」还是「静默忽略未知参数」。

用法：./.venv/Scripts/python.exe scripts/probe_schema_enforcement.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.settings import settings

MARKER = "zz_unique_marker"

SYS = "你只输出一个 JSON 对象，不要输出任何解释文字或 markdown 代码块。"
USER = "用一句话说明阿司匹林和布洛芬能否同时服用。"
# 单字段 schema，字段名是独有标记，prompt 里绝不出现它
SCHEMA = {
    "type": "object",
    "properties": {MARKER: {"type": "string", "description": "答案"}},
    "required": [MARKER],
    "additionalProperties": False,
}


async def call(client: Any, label: str, response_format: dict | None, user: str = USER) -> dict[str, Any]:
    print(f"\n{'-' * 70}\n[{label}]")
    print(f"  response_format = {json.dumps(response_format, ensure_ascii=False)[:150] if response_format else 'None'}")
    out: dict[str, Any] = {"label": label, "raw": "", "keys": [], "marker_present": False, "error": ""}
    try:
        resp = await client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=300,
            **({"response_format": response_format} if response_format is not None else {}),
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:260]}"
        print(f"  接口调用 : 失败 -> {out['error']}")
        return out

    raw = resp.choices[0].message.content or ""
    out["raw"] = raw
    print(f"  接口调用 : OK")
    print(f"  原始返回 : {raw[:300]!r}")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            out["keys"] = list(data.keys())
            out["marker_present"] = MARKER in data
            print(f"  返回字段 : {out['keys']}")
            print(f"  独有标记字段存在: {'是 → schema 生效' if out['marker_present'] else '否 → schema 未生效'}")
        else:
            print(f"  返回顶层类型: {type(data).__name__}（非 object）")
    except Exception as e:
        print(f"  解析失败 : {type(e).__name__}: {str(e)[:120]}")
    return out


async def main() -> int:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_API_BASE, timeout=90.0)

    print("schema 强制力决定性实验")
    print(f"  model: {settings.LLM_MODEL_NAME}   判据字段: {MARKER}")

    results = []
    # T1：标准 OpenAI strict 形式
    results.append(await call(client, "T1-strict=true（OpenAI 标准形式）", {
        "type": "json_schema",
        "json_schema": {"name": "t1", "strict": True, "schema": SCHEMA},
    }))
    # T2：不带 strict（部分提供方只认这一种）
    results.append(await call(client, "T2-不带 strict 字段", {
        "type": "json_schema",
        "json_schema": {"name": "t2", "schema": SCHEMA},
    }))
    # T3：诊断——故意给残缺的 response_format，看是报错还是静默忽略
    results.append(await call(client, "T3-诊断-残缺参数（缺 json_schema 字段）", {"type": "json_schema"}))
    # T4：json_object + prompt 内描述结构（现实可行的替代路线）
    results.append(await call(
        client, "T4-json_object + prompt 描述结构", {"type": "json_object"},
        user=f"用一句话说明阿司匹林和布洛芬能否同时服用，结果放入字段 {MARKER}。",
    ))

    print(f"\n{'=' * 70}\n结论\n{'=' * 70}")
    t1 = results[0].get("marker_present")
    t2 = results[1].get("marker_present")
    t1_err = bool(results[0].get("error"))
    t2_err = bool(results[1].get("error"))
    t3_err = bool(results[2].get("error"))
    t4 = results[3].get("marker_present")

    print(f"  T1 strict=true 时 schema 字段被强制 : {'是' if t1 else '否'}")
    print(f"  T2 不带 strict 时 schema 字段被强制 : {'是' if t2 else '否'}")
    print(f"  T3 残缺参数是否被参数校验拦住       : {'被拦住（说明认识该参数）' if t3_err else '未拦住'}")
    print(f"  T4 json_object + prompt 描述结构     : {'字段可对齐' if t4 else '字段不可对齐'}")

    if t1 or t2:
        verdict = "schema 真实生效"
        advice = "可直接走 json_schema 路线，模型侧强约束可用。"
    elif not (t1_err or t2_err) and t3_err:
        # 关键判据：T1/T2 不报错但字段名不遵守，同时 T3 证明参数确实被校验
        # → 提供方「接受该参数但不履行其约束」，比忽略未知参数更隐蔽（无任何报错信号）
        verdict = "接受参数但不履行约束（静默不生效）"
        advice = (
            "提供方认识并校验 response_format 参数格式（残缺会 400），但不下发 schema 给模型，"
            "字段名仍由模型先验决定。\n"
            "         → Step 1 走 B/C 档混合路线：\n"
            "           ① 加 response_format={'type':'json_object'} 保证顶层是合法 JSON（无损兜底）；\n"
            "           ② 字段名与枚举必须在 prompt 里显式逐条列出（T4 证明这样能对齐）；\n"
            "           ③ 用 Pydantic 校验 + 失败时把 ValidationError 作为反馈重试一次；\n"
            "           ④ 失败计数要打日志（这是改造前完全拿不到的指标）。\n"
            "         可删：4 处正则捞取（json_object 已保证是 JSON）。\n"
            "         不可指望：提供方做类型/枚举/required 约束。"
        )
    else:
        verdict = "需看报错细节"
        advice = "查看 T1/T2 的报错信息再判断：报错说明参数被拒，不报错但不遵守说明静默不生效。"

    print(f"\n  判定：{verdict}")
    print(f"  建议：{advice}")

    with open("probe_schema_enforcement_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": settings.LLM_MODEL_NAME,
            "marker": MARKER,
            "verdict": verdict,
            "advice": advice,
            "checks": {"t1_strict": t1, "t2_no_strict": t2, "t3_error": t3_err, "t4_prompt_described": t4},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print("  报告已写入: probe_schema_enforcement_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
