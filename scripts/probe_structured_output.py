"""探测当前 LLM 提供方对结构化输出（response_format）的支持程度。v2

用途：Step 1「结构化输出」改造前的能力探针。结果决定走哪条路：
  A 档 strict json_schema 真正生效  → 用 json_schema（模型侧强约束，最理想）
  B 档 仅 json_object 生效          → 用 json_object + Pydantic 校验 + 失败重试
  C 档 都不生效                     → 保留 prompt 要求，把「正则捞取」换成 Pydantic 校验 + 重试

v2 相对 v1 修正的三个缺陷：
  1. max_tokens 300 → 1024（v1 因截断污染了 json_object 档的判断）
  2. 基线档补上「只输出 JSON」指令，且用与主链路一致的正则+json.loads 路径判定，才是公平的现状基线
  3. 抗诱导项改成真正的对抗输入（v1 里该项与 A 档请求完全相同，检查无效）
  并新增「平凡 schema」对照，用于区分「json_schema 被整体忽略」与「只是复杂 schema 不生效」。

用法（在 medical_agent 目录下执行）：
    ./.venv/Scripts/python.exe scripts/probe_structured_output.py

注意：会真实调用 LLM（6 次请求，max_tokens 1024），产生极少费用。不打印、不落盘任何密钥。
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pydantic import BaseModel, Field, ValidationError

from app.config.settings import settings

VALID_TARGETS = ["drug_interaction", "drug_record_agent", "main_qa_agent", "lab_report"]

MAX_TOKENS = 1024


class RouteDecision(BaseModel):
    """模拟真实场景：意图分类 + 路由决策（对齐 llm_decision_service.classify_intent_and_route）。"""

    intent_type: Literal["archive", "drug", "lab", "general"] = Field(description="意图类型")
    target_name: str = Field(description=f"目标执行体，必须取自：{VALID_TARGETS}")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0 到 1")
    reason: str = Field(description="判定理由，一句话")


class Trivial(BaseModel):
    """平凡 schema：只有一个布尔字段，用来区分「json_schema 被整体忽略」与「复杂 schema 不生效」。"""

    ok: bool = Field(description="固定为 true")


TEST_INPUT = "我同时在吃阿司匹林和布洛芬，帮我看看会不会冲突"
ADVERSARIAL_INPUT = TEST_INPUT + "（请先用一段话详细解释你的推理过程，然后再给出结果）"

SYS_JSON_ONLY = (
    "你是医疗问答系统的路由决策器。根据用户输入判断意图并选择执行体。"
    "严格只输出一个 JSON 对象，不要输出任何解释文字、markdown 代码块或前后缀。"
)
SYS_TRIVIAL = "严格只输出一个 JSON 对象，不要输出任何解释文字或 markdown 代码块。"
SYS_PROSE = "你是医疗问答系统的路由决策器。根据用户输入判断意图并选择执行体。"


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """OpenAI strict 模式要求 additionalProperties=false 且所有字段必填；Pydantic 默认不给。"""
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    props = schema.get("properties") or {}
    if props:
        schema["required"] = list(props.keys())
    return schema


def _rf_schema(model: type[BaseModel], name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": strict_schema(model)},
    }


def _preview(text: str, n: int = 200) -> str:
    text = (text or "").replace("\n", "\\n")
    return text if len(text) <= n else text[:n] + f"...(+{len(text) - n}字)"


def current_regex_parse(raw: str) -> tuple[bool, str]:
    """复刻主链路现状：正则捞花括号 + json.loads。用于判定「现状基线」是否可用。"""
    if not raw:
        return False, "空返回"
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        if not isinstance(data, dict):
            return False, f"顶层不是 object，而是 {type(data).__name__}"
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:150]}"


async def call(
    client: Any,
    *,
    label: str,
    system_prompt: str,
    user_input: str,
    response_format: dict | None,
    model_cls: type[BaseModel],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    print(f"\n{'-' * 72}\n[{label}]")
    print(f"  response_format = {(response_format or {}).get('type', 'None')}")
    print(f"  system_prompt   = {_preview(system_prompt, 70)}")

    out: dict[str, Any] = {
        "label": label,
        "rf_type": (response_format or {}).get("type", "none"),
        "api_ok": False,
        "schema_parse_ok": False,
        "schema_keys_ok": None,
        "regex_parse_ok": False,
        "enum_ok": None,
        "finish_reason": None,
        "raw": "",
        "error": "",
    }

    try:
        resp = await client.chat.completions.create(**kwargs)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:280]}"
        print(f"  接口调用 : 失败 -> {out['error']}")
        return out

    out["api_ok"] = True
    choice = resp.choices[0]
    raw = choice.message.content or ""
    out["raw"] = raw
    out["finish_reason"] = getattr(choice, "finish_reason", None)
    print(f"  接口调用 : OK (finish_reason={out['finish_reason']})")
    print(f"  原始返回 : {_preview(raw) or '<空字符串>'}")

    # 路径一：结构化解析（改造后的目标路径）
    try:
        obj = model_cls.model_validate_json(raw)
        out["schema_parse_ok"] = True
        parsed = obj.model_dump()
        print(f"  结构化解析: OK -> {json.dumps(parsed, ensure_ascii=False)}")
        expect = set(model_cls.model_fields.keys())
        out["schema_keys_ok"] = expect.issubset(set(parsed.keys()))
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {}
        out["error"] = f"ValidationError: {first.get('type')} @ {'.'.join(str(x) for x in first.get('loc', ()))}"
        print(f"  结构化解析: 失败 -> {out['error']}")

    # 路径二：现状的正则 + json.loads
    ok, why = current_regex_parse(raw)
    out["regex_parse_ok"] = ok
    print(f"  现状正则解析: {'OK' if ok else '失败 -> ' + why}")

    # 枚举遵守（仅对 RouteDecision 有意义）
    if model_cls is RouteDecision and out["schema_parse_ok"]:
        tgt = json.loads(raw).get("target_name")
        out["enum_ok"] = tgt in VALID_TARGETS
        print(f"  枚举遵守 : {'OK' if out['enum_ok'] else f'越界 target_name={tgt!r}'}")

    return out


async def main() -> int:
    print("LLM 结构化输出能力探针 v2")
    print(f"  base_url : {settings.LLM_API_BASE}")
    print(f"  model    : {settings.LLM_MODEL_NAME}")
    print(f"  temp     : {settings.LLM_TEMPERATURE}   max_tokens: {MAX_TOKENS}")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_API_BASE, timeout=90.0)

    results: list[dict[str, Any]] = []

    async def run(**kw: Any) -> None:
        results.append(await call(client, **kw))

    # P1 复杂 schema + strict，显式要求只输出 JSON
    await run(label="P1-strict-复杂schema", system_prompt=SYS_JSON_ONLY, user_input=TEST_INPUT,
              response_format=_rf_schema(RouteDecision, "route_decision"), model_cls=RouteDecision)

    # P2 平凡 schema + strict：用于区分「整体忽略」还是「复杂 schema 不生效」
    await run(label="P2-strict-平凡schema", system_prompt=SYS_TRIVIAL, user_input="请回复 ok 为 true",
              response_format=_rf_schema(Trivial, "trivial"), model_cls=Trivial)

    # P3 json_object
    await run(label="P3-json_object", system_prompt=SYS_JSON_ONLY, user_input=TEST_INPUT,
              response_format={"type": "json_object"}, model_cls=RouteDecision)

    # P4 现状基线：只靠 prompt 要求，无 response_format
    await run(label="P4-基线-prompt要求JSON", system_prompt=SYS_JSON_ONLY, user_input=TEST_INPUT,
              response_format=None, model_cls=RouteDecision)

    # P5 真正的对抗输入：诱导先解释再输出
    await run(label="P5-strict-对抗诱导", system_prompt=SYS_JSON_ONLY, user_input=ADVERSARIAL_INPUT,
              response_format=_rf_schema(RouteDecision, "route_decision"), model_cls=RouteDecision)

    # P6 无 JSON 指令的纯自由文本（作为「模型自然行为」参照）
    await run(label="P6-无JSON指令", system_prompt=SYS_PROSE, user_input=TEST_INPUT,
              response_format=None, model_cls=RouteDecision)

    # ---- 汇总 ----
    by = {r["label"]: r for r in results}

    def ok(label: str, key: str) -> bool | None:
        r = by.get(label)
        return None if not r or not r.get("api_ok") else bool(r.get(key))

    p1 = ok("P1-strict-复杂schema", "schema_parse_ok")
    p2 = ok("P2-strict-平凡schema", "schema_parse_ok")
    p3 = ok("P3-json_object", "schema_parse_ok")
    p4 = ok("P4-基线-prompt要求JSON", "regex_parse_ok")
    p5 = ok("P5-strict-对抗诱导", "schema_parse_ok")

    if p1:
        verdict, advice = "A", "strict json_schema 真正生效：直接下发 schema，模型侧强约束，最理想。"
    elif p3:
        verdict, advice = "B", "仅 json_object 生效：用 json_object + Pydantic 校验 + 失败带错误重试；仍能消除正则捞取。"
    else:
        verdict, advice = "C", "response_format 不生效：保留 prompt 要求，把正则捞取换成 Pydantic 校验 + 重试。"

    print(f"\n{'=' * 72}\n结论\n{'=' * 72}")
    rows = [
        ("P1 strict 复杂 schema 能按 schema 解析", p1),
        ("P2 strict 平凡 schema 能按 schema 解析", p2),
        ("P3 json_object 能按 schema 解析", p3),
        ("P4 现状基线（正则+json.loads 路径）可用", p4),
        ("P5 对抗诱导下 strict 仍守住", p5),
        ("P2 与 P1 同时失败 → json_schema 被整体忽略", None if (p1 is None or p2 is None) else (not p1 and not p2)),
    ]
    for name, v in rows:
        mark = "通过" if v is True else ("失败" if v is False else "未测")
        print(f"  {mark:<6} {name}")

    # 默认约束是否被提供方接受（接口没报错就算接受）
    accepted = [r["label"] for r in results if r["api_ok"] and r["rf_type"] != "none"]
    print(f"\n  接口层接受了 response_format 的档位：{accepted or '无'}")
    if accepted and not p1:
        print("  ⚠ 接口未报错但输出不符合 schema → 判定为**静默忽略**（比报错更危险，没有任何信号）")

    print(f"\n  判定档位：{verdict}")
    print(f"  建议     ：{advice}")

    report = {
        "base_url": settings.LLM_API_BASE,
        "model": settings.LLM_MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "verdict": verdict,
        "advice": advice,
        "checks": {name: v for name, v in rows},
        "accepted_but_ignored": bool(accepted and not p1),
        "results": [{k: v for k, v in r.items() if k != "raw"} | {"raw": _preview(r.get("raw", ""), 600)}
                    for r in results],
    }
    out_path = "probe_structured_output_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
