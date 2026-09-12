"""多模型矩阵探针：schema 强制力究竟取决于「模型」还是「提供方」？

背景：probe_schema_enforcement.py 用无提示的独有字段名判定出
deepseek-v4-flash-0731 所在通道「接受 response_format 参数但不履行约束」。
本轮要回答三个问题：
  1. 换成 qwen3.8-flash 是否就生效？（用户假设）
  2. 换档位（flash → max）是否有变化？→ 排除「Flash 档不支持」
  3. 换厂商（deepseek → qwen）是否有变化？→ 排除「DeepSeek 全系不支持」

判据（沿用上一轮，必须是无提示的独有字段名）：
  A 档 strict json_schema：prompt 里绝不出现 MARKER
      → 返回里出现 MARKER = schema 真被下发并强制
  B 档 json_object + prompt 显式描述字段名
      → 返回里出现 MARKER = 字段名可经 prompt 对齐（现实兜底路线）
  C 档 无 response_format，prompt 要求 JSON
      → 对照组：验证「JSON 输出」本身是不是 prompt 在起作用

用法：
  ./.venv/Scripts/python.exe scripts/probe_model_matrix.py
  ./.venv/Scripts/python.exe scripts/probe_model_matrix.py qwen3.8-flash qwen3.5-flash
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

DEFAULT_MODELS = [
    "deepseek-v4-flash-0731",  # 现状基线
    "qwen3.8-flash",           # 用户假设：同档位换厂商
    "qwen3.8-max",             # 同厂商升档 → 排除「Flash 档不行」
    "deepseek-v4-pro",         # 同厂商升档 → 排除「DeepSeek 全系不行」
]

SYS = "你只输出一个 JSON 对象，不要输出任何解释文字，不要用 markdown 代码块包裹。"
USER_PLAIN = "用一句话说明阿司匹林和布洛芬能否同时服用。"
USER_DESCRIBED = f"用一句话说明阿司匹林和布洛芬能否同时服用，把答案放进字段 {MARKER}。"

# 单字段 schema，字段名是独有标记，prompt 里绝不出现它
SCHEMA = {
    "type": "object",
    "properties": {MARKER: {"type": "string", "description": "答案"}},
    "required": [MARKER],
    "additionalProperties": False,
}


async def call(client: Any, model: str, label: str, response_format: dict | None, user: str) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "raw": "", "keys": [], "marker_present": False, "error": ""}
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=1024,
            **({"response_format": response_format} if response_format is not None else {}),
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return out

    # 兼容思考型模型：正文可能在 reasoning_content 之外
    msg = resp.choices[0].message
    raw = (msg.content or "").strip()
    out["raw"] = raw
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            out["keys"] = list(data.keys())
            out["marker_present"] = MARKER in data
    except Exception:
        # 尝试从包裹文本里捞一层
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                if isinstance(data, dict):
                    out["keys"] = list(data.keys())
                    out["marker_present"] = MARKER in data
            except Exception:
                pass
    return out


async def probe_model(client: Any, model: str) -> dict[str, Any]:
    print(f"\n{'=' * 74}\n模型: {model}\n{'=' * 74}")
    row: dict[str, Any] = {"model": model}

    a = await call(client, model, "A-strict json_schema（字段名无提示）", {
        "type": "json_schema",
        "json_schema": {"name": "a", "strict": True, "schema": SCHEMA},
    }, USER_PLAIN)
    row["A_strict"] = a

    b = await call(client, model, "B-json_object（prompt 描述字段名）", {"type": "json_object"}, USER_DESCRIBED)
    row["B_json_object"] = b

    c = await call(client, model, "C-无 response_format（对照组）", None, USER_DESCRIBED)
    row["C_none"] = c

    for k, r in (("A", a), ("B", b), ("C", c)):
        if r["error"]:
            print(f"  [{k}] 失败 : {r['error']}")
        else:
            print(f"  [{k}] 字段={r['keys']}  MARKER={'有' if r['marker_present'] else '无'}  原文={r['raw'][:110]!r}")

    if any(r["error"] for r in (a, b, c)):
        verdict = "存在调用失败，见 error"
    elif a["marker_present"]:
        verdict = "schema 真实生效（可直接走 json_schema）"
    elif b["marker_present"]:
        verdict = "schema 静默不生效；但字段名可经 prompt 对齐（走 json_object 路线）"
    elif c["marker_present"]:
        verdict = "完全靠 prompt 输出 JSON，response_format 无额外收益"
    else:
        verdict = "连 prompt 都无法对齐字段名"
    row["verdict"] = verdict
    print(f"  → 判定: {verdict}")
    return row


async def main() -> int:
    from openai import AsyncOpenAI

    models = sys.argv[1:] or DEFAULT_MODELS
    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_API_BASE, timeout=120.0)

    print("多模型 schema 强制力矩阵探针")
    print(f"  base_url : {settings.LLM_API_BASE}")
    print(f"  判据字段 : {MARKER}（A 档 prompt 中绝不出现）")
    print(f"  待测模型 : {', '.join(models)}")

    rows = []
    for m in models:
        try:
            rows.append(await probe_model(client, m))
        except Exception as e:
            print(f"\n模型 {m} 整体失败: {type(e).__name__}: {str(e)[:200]}")
            rows.append({"model": m, "verdict": f"整体失败: {type(e).__name__}"})

    print(f"\n{'=' * 74}\n汇总\n{'=' * 74}")
    print(f"{'模型':<26}{'A strict':<12}{'B json_obj':<13}{'C 无参数':<11}判定")
    for r in rows:
        a = r.get("A_strict", {})
        b = r.get("B_json_object", {})
        c = r.get("C_none", {})

        def mark(r_: dict) -> str:
            if r_.get("error"):
                return "报错"
            return "有" if r_.get("marker_present") else "无"

        print(f"{r['model']:<26}{mark(a):<12}{mark(b):<13}{mark(c):<11}{r.get('verdict', '')}")

    print("\n判读指南：")
    print("  A 列出现「有」→ 该模型/通道支持原生 schema 约束，Step 1 可直接用 json_schema。")
    print("  A 列全「无」而 B 列「有」→ schema 在兼容层被丢弃，但字段名可经 prompt 对齐。")
    print("  若换上 qwen 后 A 变「有」→ 说明瓶颈在「提供方对 DeepSeek 系模型的兼容层」，换模型可解。")
    print("  若升到 max 档 A 仍「无」→ 说明与模型档位无关，属通道特性，换档位无收益。")

    out_path = _ROOT.parent / "probe_model_matrix_result.json"
    if not out_path.parent.exists():
        out_path = _ROOT / "probe_model_matrix_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"base_url": settings.LLM_API_BASE, "marker": MARKER, "models": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
