"""schema 严格性冲突实验：证明 schema 是「真下发」还是「碰巧猜中」。

上一轮 probe_model_matrix 发现：qwen3.8-flash / qwen3.8-max / deepseek-v4-pro
在 strict json_schema 下返回了 prompt 中从未出现的字段名 zz_unique_marker。
这已是很强的证据，但仍需排除两种残余解释：
  (a) 模型"猜中"了字段名（虽然极不可能）
  (b) 模型只是顺从最后一个看到的标识符，而非真正读取 schema

做法：制造 prompt 与 schema 的**直接冲突**，看谁赢。schema 赢 = 真下发。
  E1 字段名冲突：schema 要 zz_schema_field，prompt 要 aa_prompt_field
  E2 枚举冲突  ：schema 的 enum 只允许 zz_allowed_a / zz_allowed_b，prompt 要求 zz_forbidden_c
  E3 类型冲突  ：schema 要求 integer，prompt 要求输出中文句子
  E4 稳定性    ：A 档连打 3 次，看是否稳定（排除偶发）
  E5 多余字段  ：additionalProperties=false 时，是否还能夹带额外字段

用法：
  ./.venv/Scripts/python.exe scripts/probe_schema_strictness.py
  ./.venv/Scripts/python.exe scripts/probe_schema_strictness.py qwen3.8-flash
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

SCHEMA_FIELD = "zz_schema_field"
PROMPT_FIELD = "aa_prompt_field"
ENUM_FIELD = "zz_route"
ENUM_OK = ["zz_allowed_a", "zz_allowed_b"]
ENUM_BAD = "zz_forbidden_c"

DEFAULT_MODELS = ["qwen3.8-flash", "deepseek-v4-pro", "deepseek-v4-flash-0731"]

SYS = "你只输出一个 JSON 对象，不要输出任何解释文字，不要用 markdown 代码块包裹。"
Q = "用一句话说明阿司匹林和布洛芬能否同时服用。"


def _strict(name: str, schema: dict) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


async def raw_call(client: Any, model: str, rf: dict, user: str, sys_: str = SYS) -> tuple[str, str]:
    """返回 (原始文本, 错误)。"""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_}, {"role": "user", "content": user}],
            temperature=0.1,
            max_tokens=1024,
            response_format=rf,
        )
    except Exception as e:
        return "", f"{type(e).__name__}: {str(e)[:220]}"
    return (resp.choices[0].message.content or "").strip(), ""


def parse(raw: str) -> dict:
    try:
        d = json.loads(raw)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e > s:
        try:
            d = json.loads(raw[s : e + 1])
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


async def probe(client: Any, model: str) -> dict[str, Any]:
    print(f"\n{'=' * 76}\n模型: {model}\n{'=' * 76}")
    row: dict[str, Any] = {"model": model, "steps": {}}

    # E1 字段名冲突
    e1_raw, e1_err = await raw_call(
        client, model,
        _strict("e1", {
            "type": "object",
            "properties": {SCHEMA_FIELD: {"type": "string", "description": "答案"}},
            "required": [SCHEMA_FIELD],
            "additionalProperties": False,
        }),
        f"{Q}把答案放进字段 {PROMPT_FIELD}。",
    )
    e1 = parse(e1_raw)
    if e1_err:
        e1_verdict = f"调用失败: {e1_err}"
    elif SCHEMA_FIELD in e1:
        e1_verdict = "schema 赢 → 真下发"
    elif PROMPT_FIELD in e1:
        e1_verdict = "prompt 赢 → schema 未下发"
    else:
        e1_verdict = f"都不是（字段={list(e1.keys())}）"
    row["steps"]["E1_字段名冲突"] = {"raw": e1_raw, "keys": list(e1.keys()), "verdict": e1_verdict, "error": e1_err}
    print(f"  [E1] 字段名冲突   : {e1_verdict}")
    print(f"       返回字段={list(e1.keys())}  原文={e1_raw[:110]!r}")

    # E2 枚举冲突（prompt 指定一个非法枚举值，看模型敢不敢给）
    e2_raw, e2_err = await raw_call(
        client, model,
        _strict("e2", {
            "type": "object",
            "properties": {
                SCHEMA_FIELD: {"type": "string", "description": "答案"},
                ENUM_FIELD: {"type": "string", "enum": ENUM_OK, "description": "路由"},
            },
            "required": [SCHEMA_FIELD, ENUM_FIELD],
            "additionalProperties": False,
        }),
        f"{Q}答案放进 {SCHEMA_FIELD}，并把字段 {ENUM_FIELD} 设为 {ENUM_BAD}。",
    )
    e2 = parse(e2_raw)
    got_enum = e2.get(ENUM_FIELD)
    if e2_err:
        e2_verdict = f"调用失败: {e2_err}"
    elif got_enum in ENUM_OK:
        e2_verdict = f"enum 被强制（输出 {got_enum}，拒绝了 prompt 要求的 {ENUM_BAD}）"
    elif got_enum == ENUM_BAD:
        e2_verdict = "enum 未强制（顺从了 prompt 的非法值）"
    else:
        e2_verdict = f"无法判定（{ENUM_FIELD}={got_enum!r}）"
    row["steps"]["E2_枚举冲突"] = {"raw": e2_raw, "enum_value": got_enum, "verdict": e2_verdict, "error": e2_err}
    print(f"  [E2] 枚举冲突     : {e2_verdict}")

    # E3 类型冲突（schema 要 integer，prompt 要中文）
    e3_raw, e3_err = await raw_call(
        client, model,
        _strict("e3", {
            "type": "object",
            "properties": {"zz_risk_score": {"type": "integer", "description": "风险分"}},
            "required": ["zz_risk_score"],
            "additionalProperties": False,
        }),
        "把阿司匹林和布洛芬同服的风险用一句话说清楚，并放进字段 zz_risk_score。",
    )
    e3 = parse(e3_raw)
    v3 = e3.get("zz_risk_score")
    if e3_err:
        e3_verdict = f"调用失败: {e3_err}"
    elif isinstance(v3, int) and not isinstance(v3, bool):
        e3_verdict = f"类型被强制（得到 int {v3}）"
    elif isinstance(v3, str):
        e3_verdict = f"类型未强制（得到字符串 {v3[:40]!r}）"
    else:
        e3_verdict = f"无法判定（{v3!r}）"
    row["steps"]["E3_类型冲突"] = {"raw": e3_raw, "value": v3, "verdict": e3_verdict, "error": e3_err}
    print(f"  [E3] 类型冲突     : {e3_verdict}")

    # E4 稳定性：A 档连打 3 次
    hits = 0
    reps = []
    for _ in range(3):
        r, err = await raw_call(
            client, model,
            _strict("e4", {
                "type": "object",
                "properties": {SCHEMA_FIELD: {"type": "string", "description": "答案"}},
                "required": [SCHEMA_FIELD],
                "additionalProperties": False,
            }),
            Q,
        )
        d = parse(r)
        ok = SCHEMA_FIELD in d
        hits += 1 if ok else 0
        reps.append({"ok": ok, "keys": list(d.keys()), "error": err})
    e4_verdict = f"{hits}/3 次字段名遵守 schema" + ("（稳定）" if hits == 3 else "（不稳定！）")
    row["steps"]["E4_稳定性"] = {"repeats": reps, "verdict": e4_verdict}
    print(f"  [E4] 稳定性       : {e4_verdict}")

    # E5 多余字段
    e5_raw, e5_err = await raw_call(
        client, model,
        _strict("e5", {
            "type": "object",
            "properties": {SCHEMA_FIELD: {"type": "string", "description": "答案"}},
            "required": [SCHEMA_FIELD],
            "additionalProperties": False,
        }),
        f"{Q}放进 {SCHEMA_FIELD}，另外**必须**再加一个字段 aa_extra 写'额外信息'。",
    )
    e5 = parse(e5_raw)
    extra = [k for k in e5.keys() if k != SCHEMA_FIELD]
    if e5_err:
        e5_verdict = f"调用失败: {e5_err}"
    elif not extra:
        e5_verdict = "additionalProperties=false 生效（无多余字段）"
    else:
        e5_verdict = f"未生效，夹带了多余字段 {extra}"
    row["steps"]["E5_多余字段"] = {"raw": e5_raw, "extra": extra, "verdict": e5_verdict, "error": e5_err}
    print(f"  [E5] 多余字段     : {e5_verdict}")

    passed = sum(
        1 for k, v in row["steps"].items()
        if any(p in v.get("verdict", "") for p in ("真下发", "被强制", "生效", "3/3"))
    )
    row["score"] = f"{passed}/5 项通过"
    print(f"  → 小计: {row['score']}")
    return row


async def main() -> int:
    from openai import AsyncOpenAI

    models = sys.argv[1:] or DEFAULT_MODELS
    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_API_BASE, timeout=120.0)

    print("schema 严格性冲突实验（schema 与 prompt 直接对抗）")
    print(f"  base_url: {settings.LLM_API_BASE}")
    print(f"  冲突设计: schema 字段={SCHEMA_FIELD} vs prompt 字段={PROMPT_FIELD}")
    print(f"            schema enum={ENUM_OK} vs prompt 要求={ENUM_BAD}")

    rows = [await probe(client, m) for m in models]

    print(f"\n{'=' * 76}\n汇总（'schema 赢' 的数量 = 原生约束可信度）\n{'=' * 76}")
    for r in rows:
        s = r.get("steps", {})
        print(f"\n{r['model']}   小计 {r.get('score', '')}")
        for k, v in s.items():
            print(f"   {k:<16}: {v.get('verdict', '')}")

    out = _ROOT / "probe_schema_strictness_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"base_url": settings.LLM_API_BASE, "models": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
