"""结构化输出改造的边界测试。

全程 **mock LLM**，不发真实请求，因此可在 CI 无网络环境下运行。

分三层覆盖：
  1. `app/core/llm/structured_output.py` —— 能力层纯粹逻辑（提取、严格化、档位解析、探针）
  2. `LLMService.chat_completion_json` —— 校验/重试/可观测/异常分层
  3. `LLMDecisionService` 的 P0 三处 —— 迁移后契约不变 + 逐项容错不退化

重点覆盖那些"看起来能过、实际会静默错"的边界：
  - 探针被 max_tokens 截断（正文不完整但字段名已出现）
  - 探针偶发空返回（无结论 ≠ 不支持）
  - 模型夹带 ``` 围栏 / 前置解释
  - 非法枚举、缺字段、confidence 越界、target_type 与注册表不一致
  - 批量路由中**单条**不合法只影响该条（不能被整批失败吃掉）
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.common.exceptions import LLMCallException
from app.config.settings import settings
from app.core.agent import llm_decision_service as lds
from app.core.agent import schemas as S
from app.core.agent.capabilities import CAPABILITY_REGISTRY
from app.core.agent.llm_decision_service import LLMDecisionService
from app.core.agent.schemas import BatchRouteDecision, RouteDecision
from app.core.llm import structured_output as so
from app.core.llm.llm_service import (
    LLMService,
    reset_structured_output_stats,
    structured_output_stats,
)

# --------------------------------------------------------------------------------------
# 测试替身：一个可编排的假 OpenAI 客户端
# --------------------------------------------------------------------------------------


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Usage:
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18
    prompt_tokens_details = None


class _Resp:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage()


class _FakeCompletions:
    """按脚本依次返回。

    脚本元素可以是：
      - `str`                    → content，finish_reason="stop"
      - `(content, finish_reason)` → 指定 finish_reason（用于模拟截断）
      - `BaseException` 实例      → 抛出
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("假客户端脚本已用尽，测试用例的响应次数不够")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, tuple):
            return _Resp(*item)
        return _Resp(item)


class _FakeClient:
    def __init__(self, script: list):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(script)})()

    @property
    def completions(self) -> _FakeCompletions:
        return self.chat.completions


def make_service(script: list) -> tuple[LLMService, _FakeCompletions]:
    """构造注入了假客户端的 LLMService（_get_client 会直接返回 self._client）。"""
    svc = LLMService()
    fake = _FakeClient(script)
    svc._client = fake  # noqa: SLF001 —— 测试注入点
    return svc, fake.completions


def j(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False)


class DemoSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: int = 0
    kind: str = ""


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """每个用例重置能力缓存、计数，并固定档位（避免触发真实探针）。"""
    so.reset_capability_cache()
    reset_structured_output_stats()
    monkeypatch.setattr(settings, "STRUCTURED_OUTPUT_MODE", "json_schema")
    yield
    so.reset_capability_cache()
    reset_structured_output_stats()


# --------------------------------------------------------------------------------------
# 1. 能力层：提取
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expect_text", "expect_fallback"),
    [
        ('{"a": 1}', '{"a": 1}', False),
        ("[1, 2]", "[1, 2]", False),
        ('```json\n{"a": 1}\n```', '{"a": 1}', True),
        ('```\n{"a": 1}\n```', '{"a": 1}', True),
        ('好的，结果如下：{"a": 1}\n以上。', '{"a": 1}', True),
        ("", "", False),
        ("   ", "", False),
    ],
)
def test_extract_json_candidate(raw, expect_text, expect_fallback):
    text, fallback = so.extract_json_candidate(raw)
    assert text == expect_text
    assert fallback is expect_fallback


def test_extract_json_candidate_truncated_keeps_text():
    """截断的 JSON 无法解析，但也要把文本交出去，由上层决定是否失败（不能直接吞掉）。"""
    raw = '{"zz_schema_field": "阿司匹'
    text, fallback = so.extract_json_candidate(raw)
    assert text == raw
    assert fallback is True


# --------------------------------------------------------------------------------------
# 2. 能力层：严格化
# --------------------------------------------------------------------------------------


def test_strictify_fills_required_and_closes_additional_properties():
    schema = so.strictify(DemoSchema.model_json_schema())
    assert schema["additionalProperties"] is False
    # Pydantic 会漏掉带默认值的字段，strict 模式要求全部进 required
    assert set(schema["required"]) == {"name", "score", "kind"}


def test_strictify_recurses_into_defs():
    class Sub(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: str = ""

    class Outer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        subs: list[Sub] = []

    schema = so.strictify(Outer.model_json_schema())
    assert schema["$defs"]["Sub"]["additionalProperties"] is False
    assert schema["$defs"]["Sub"]["required"] == ["a"]


def test_strictify_leaves_free_form_object_untouched():
    """没有 properties 的自由形态对象不能被锁死，否则 dict[str, Any] 类字段会全被拒。"""
    raw = {"type": "object", "additionalProperties": True}
    assert so.strictify(raw)["additionalProperties"] is True


def test_strictify_does_not_mutate_input():
    raw = {"type": "object", "properties": {"a": {"type": "string"}}}
    so.strictify(raw)
    assert "required" not in raw and "additionalProperties" not in raw


# --------------------------------------------------------------------------------------
# 3. 能力层：response_format 构造与档位解析
# --------------------------------------------------------------------------------------


def test_build_response_format_schema_vs_object():
    fmt = so.build_response_format(DemoSchema, "json_schema")
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "DemoSchema"
    assert set(fmt["json_schema"]["schema"]["required"]) == {"name", "score", "kind"}

    assert so.build_response_format(DemoSchema, "json_object") == {"type": "json_object"}


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("json_schema", "json_schema"), ("json_object", "json_object"), ("auto", "auto"), ("AUTO", "auto"), ("bogus", "auto")],
)
def test_configured_mode_normalises(monkeypatch, configured, expected):
    monkeypatch.setattr(settings, "STRUCTURED_OUTPUT_MODE", configured)
    assert so.configured_mode() == expected


@pytest.mark.asyncio
async def test_resolve_mode_explicit_config_skips_probe(monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURED_OUTPUT_MODE", "json_object")
    _svc, comps = make_service([])  # 空脚本：一旦发起探针就会 AssertionError
    assert await so.resolve_mode(_svc._client, "m") == "json_object"  # noqa: SLF001
    assert comps.calls == []


# --------------------------------------------------------------------------------------
# 4. 能力层：探针（含两个实测得来的反直觉边界）
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_true_when_marker_present():
    svc, _ = make_service([j(**{so._PROBE_FIELD: "你好"})])  # noqa: SLF001
    assert await so.probe_schema_support(svc._client, "m") is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_probe_true_when_truncated_after_marker():
    """回归测试：截断让 JSON 不合法，但字段名已完整出现 → 必须判为支持。

    这是真实踩过的坑：用"能否解析成合法 JSON"做判据，会把支持 schema 的模型误判为不支持。
    """
    svc, _ = make_service([('{"zz_schema_capability_probe": "阿司匹', "length")])
    assert await so.probe_schema_support(svc._client, "m") is True  # noqa: SLF001


@pytest.mark.asyncio
async def test_probe_truncated_before_marker_is_inconclusive_not_false():
    """回归测试（更狠的一种截断）：字段名被从中间切断。

    实测就是这个形态——`max_tokens` 偏小时输出停在 `{"zz_schema_capability_`，
    文本非空但不含完整判据字段。若按"否"处理就会误判；正确做法是当作无结论重试。
    """
    svc, comps = make_service([
        ('{"zz_schema_capability_', "length"),
        ('{"zz_schema_capability_', "length"),
        ('{"zz_schema_capability_probe": "你好"}', "stop"),
    ])
    assert await so.probe_schema_support(svc._client, "m", attempts=3) is True  # noqa: SLF001
    assert len(comps.calls) == 3  # 前两次截断被当作无结论，没有提前判否


@pytest.mark.asyncio
async def test_probe_all_attempts_truncated_is_inconclusive():
    """持续截断时无法判定 → None（不可误判为"不支持"）。"""
    svc, comps = make_service([('{"zz_schema_capability_', "length")] * 2)
    assert await so.probe_schema_support(svc._client, "m") is None  # noqa: SLF001
    assert len(comps.calls) == 2


@pytest.mark.asyncio
async def test_probe_false_when_content_lacks_marker():
    svc, comps = make_service([j(answer="你好")])
    assert await so.probe_schema_support(svc._client, "m") is False  # noqa: SLF001
    assert len(comps.calls) == 1  # 决定性结论，不需要重试


@pytest.mark.asyncio
async def test_probe_retries_on_empty_content_then_succeeds():
    """回归测试：偶发空返回属"无结论"，重试即可，不能当成"不支持"。"""
    svc, comps = make_service([None, "", j(**{so._PROBE_FIELD: "你好"})])  # noqa: SLF001
    assert await so.probe_schema_support(svc._client, "m", attempts=3) is True  # noqa: SLF001
    assert len(comps.calls) == 3


@pytest.mark.asyncio
async def test_probe_returns_none_when_all_attempts_inconclusive():
    """「无结论」必须与「否」区分开：返回 None，不能返回 False。"""
    svc, comps = make_service([None, None])
    assert await so.probe_schema_support(svc._client, "m") is None  # noqa: SLF001
    assert len(comps.calls) == 2


@pytest.mark.asyncio
async def test_probe_returns_none_on_exception():
    svc, _ = make_service([RuntimeError("500 upstream"), RuntimeError("x")])
    assert await so.probe_schema_support(svc._client, "m") is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_probe_recovers_from_exception_then_succeeds():
    svc, comps = make_service([asyncio.TimeoutError(), j(**{so._PROBE_FIELD: "ok"})])  # noqa: SLF001
    assert await so.probe_schema_support(svc._client, "m") is True  # noqa: SLF001
    assert len(comps.calls) == 2


@pytest.mark.asyncio
async def test_resolve_mode_auto_probes_once_then_caches(monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURED_OUTPUT_MODE", "auto")
    svc, comps = make_service([j(**{so._PROBE_FIELD: "ok"})])
    assert await so.resolve_mode(svc._client, "m") == "json_schema"  # noqa: SLF001
    assert await so.resolve_mode(svc._client, "m") == "json_schema"  # noqa: SLF001
    assert len(comps.calls) == 1  # 第二次命中缓存


@pytest.mark.asyncio
async def test_resolve_mode_auto_downgrades_on_decisive_false(monkeypatch):
    """决定性「否」（内容完整但字段缺席）应被缓存。"""
    monkeypatch.setattr(settings, "STRUCTURED_OUTPUT_MODE", "auto")
    svc, comps = make_service([j(answer="你好")])
    assert await so.resolve_mode(svc._client, "m") == "json_object"  # noqa: SLF001
    assert await so.resolve_mode(svc._client, "m") == "json_object"  # noqa: SLF001
    assert len(comps.calls) == 1  # 命中缓存


@pytest.mark.asyncio
async def test_resolve_mode_inconclusive_is_not_cached(monkeypatch):
    """回归测试：无结论时本次退 json_object，但**不得写缓存**。

    否则一次网络抖动会把整个进程永久钉在低档位——又一种静默降级。
    这里第 1 轮探针无结论（异常），第 2 轮给出决定性 True，验证能被纠正。
    """
    monkeypatch.setattr(settings, "STRUCTURED_OUTPUT_MODE", "auto")
    svc, comps = make_service([
        RuntimeError("临时抖动"), RuntimeError("临时抖动"),  # 第 1 轮探针：无结论
        j(**{so._PROBE_FIELD: "你好"}),                      # 第 2 轮探针：决定性支持
    ])
    assert await so.resolve_mode(svc._client, "m") == "json_object"  # noqa: SLF001
    assert await so.resolve_mode(svc._client, "m") == "json_schema"  # noqa: SLF001
    assert len(comps.calls) == 3


# --------------------------------------------------------------------------------------
# 5. chat_completion_json：正常路径
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_json_schema_mode_passes_strict_response_format():
    svc, comps = make_service([j(name="x", score=1, kind="k")])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert obj == DemoSchema(name="x", score=1, kind="k")
    rf = comps.calls[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert comps.calls[0]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_json_object_mode_uses_object_format(monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURED_OUTPUT_MODE", "json_object")
    svc, comps = make_service([j(name="x")])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert obj is not None and obj.name == "x"
    assert comps.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_thinking_is_disabled_for_structured_calls_by_default():
    """结构化决策默认关闭思考。实测（2026-09-12）这是 0/3→3/3、墙钟 20~27s→2.9~4.5s、
    并把「intent 串味导致路由被静默丢弃」一并消除的关键，见 settings.LLM_DISABLE_THINKING。

    注意本字段的**存在性**也要断言：若哪天有人把它删掉，行为会静默退回推理模式，
    而症状只是"变慢 + 偶发空响应"，很难追。"""
    svc, comps = make_service([j(name="x")])
    await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert comps.calls[0]["extra_body"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_thinking_flag_can_be_turned_off_for_incompatible_gateways(monkeypatch):
    """换到不接受 enable_thinking 的模型/网关时，置 false 必须能整体回退（不留残余参数）。"""
    monkeypatch.setattr(settings, "LLM_DISABLE_THINKING", False)
    svc, comps = make_service([j(name="x")])
    await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert "extra_body" not in comps.calls[0]


@pytest.mark.asyncio
async def test_clean_json_does_not_count_as_fallback():
    svc, _ = make_service([j(name="x")])
    await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert structured_output_stats() == {
        "extract_fallback": 0,
        "validation_failure": 0,
        "empty_response": 0,
        "truncated": 0,
    }


@pytest.mark.asyncio
async def test_fenced_and_prose_wrapped_outputs_are_recovered():
    svc, _ = make_service(['```json\n{"name": "a"}\n```', '结果：{"name": "b"} 完毕'])
    assert (await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)).name == "a"
    assert (await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)).name == "b"
    assert structured_output_stats()["extract_fallback"] == 2


# --------------------------------------------------------------------------------------
# 6. chat_completion_json：校验失败 / 重试 / 可观测
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_succeeds_and_feeds_error_back_to_model():
    svc, comps = make_service(['{"name": "x", "score": "not-an-int", "kind": "k"}', j(name="x", score=3)])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert obj is not None and obj.score == 3
    assert structured_output_stats()["validation_failure"] == 1
    # 第二次请求必须带上"上一轮原文 + 修正要求"
    assert len(comps.calls) == 2
    msgs = comps.calls[1]["messages"]
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-1]["role"] == "user"
    assert "name" in msgs[-1]["content"] and "score" in msgs[-1]["content"]


@pytest.mark.asyncio
async def test_retry_exhausted_returns_none_and_logs():
    svc, comps = make_service(['{"name": "x", "score": "bad"}', '{"name": "x", "score": "still-bad"}'])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema, max_retry=1)
    assert obj is None
    assert len(comps.calls) == 2
    assert structured_output_stats()["validation_failure"] == 2


@pytest.mark.asyncio
async def test_max_retry_zero_makes_single_attempt():
    svc, comps = make_service(['{"nope": 1}'])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema, max_retry=0)
    assert obj is None
    assert len(comps.calls) == 1


@pytest.mark.asyncio
async def test_empty_content_is_failure_not_crash():
    """空响应仍是失败，但**不计入 schema 校验失败**——
    它和「JSON 不合规」是两类问题，混在一起会把排查引向错误方向。
    （本条在 Step 2.5 真实链路验证后由 `validation_failure == 2` 收紧为分类计数。）
    """
    svc, _ = make_service([None, ""])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert obj is None
    stats = structured_output_stats()
    assert stats["empty_response"] == 2
    assert stats["validation_failure"] == 0


@pytest.mark.asyncio
async def test_empty_response_retry_asks_for_output_not_schema_repair():
    """空响应的重试话术必须是「你没输出内容」，而不是回灌校验错误——
    模型本来就没输出，回灌字段错误没有任何信息量。"""
    svc, comps = make_service([None, j(name="ok")])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert obj is not None and obj.name == "ok"
    assert structured_output_stats()["empty_response"] == 1

    assert len(comps.calls) == 2
    retry_msgs = comps.calls[1]["messages"]
    assert retry_msgs[-1]["role"] == "user"
    assert "没有返回任何可用内容" in retry_msgs[-1]["content"]
    # 不能出现「校验错误」这类 schema 反馈——那是另一类失败的修法
    assert "校验错误" not in retry_msgs[-1]["content"]
    assert "必须包含且仅包含这些字段" not in retry_msgs[-1]["content"]


@pytest.mark.asyncio
async def test_empty_response_exhausted_returns_none():
    svc, comps = make_service([None, "", ""])
    assert await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema) is None
    assert structured_output_stats()["empty_response"] == 2


@pytest.mark.asyncio
async def test_truncated_response_counted_separately_and_retry_asks_for_brevity():
    """finish_reason=length 且内容非空 → 截断：单独计数，重试要求**精简**而不是报字段错。"""
    svc, comps = make_service([('{"name": "x", "score":', "length"), j(name="ok")])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert obj is not None and obj.name == "ok"

    stats = structured_output_stats()
    assert stats["truncated"] == 1
    assert stats["validation_failure"] == 1  # 截断内容确实也过不了校验
    assert stats["empty_response"] == 0

    retry_msgs = comps.calls[1]["messages"]
    assert "被截断" in retry_msgs[-1]["content"]
    assert "必须包含且仅包含这些字段" not in retry_msgs[-1]["content"]


@pytest.mark.asyncio
async def test_truncated_with_salvageable_content_does_not_waste_retry():
    """截断但内容恰好是合法 JSON → 直接用，不因 finish_reason 就判失败。"""
    svc, comps = make_service([(j(name="ok"), "length")])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert obj is not None and obj.name == "ok"
    assert len(comps.calls) == 1
    # 仍会记一次截断计数（用于暴露 max_tokens 偏紧），但不影响结果
    assert structured_output_stats()["truncated"] == 1


@pytest.mark.asyncio
async def test_missing_required_field_triggers_retry():
    svc, _ = make_service(['{"score": 1}', j(name="ok")])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert obj is not None and obj.name == "ok"


@pytest.mark.asyncio
async def test_array_instead_of_object_is_failure():
    svc, _ = make_service(["[1,2,3]", "[4,5,6]"])
    assert await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema) is None


@pytest.mark.asyncio
async def test_extra_field_rejected_by_extra_forbid():
    svc, _ = make_service(['{"name": "x", "surprise": 1}', j(name="y")])
    obj = await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)
    assert obj is not None and obj.name == "y"


# --------------------------------------------------------------------------------------
# 7. chat_completion_json：异常分层（传输层异常必须上抛，与既有 chat_completion 一致）
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transport_error_raises_llm_call_exception():
    svc, _ = make_service([RuntimeError("connection reset")])
    with pytest.raises(LLMCallException):
        await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema)


@pytest.mark.asyncio
async def test_timeout_raises_llm_call_exception():
    svc, _ = make_service([asyncio.TimeoutError()])
    with pytest.raises(LLMCallException):
        await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema, timeout_s=1.0)


@pytest.mark.asyncio
async def test_transport_error_is_not_retried():
    """传输层故障重试无意义（且会放大延迟），应直接上抛。"""
    svc, comps = make_service([RuntimeError("boom"), j(name="never")])
    with pytest.raises(LLMCallException):
        await svc.chat_completion_json(prompt="p", system_prompt="s", schema=DemoSchema, max_retry=3)
    assert len(comps.calls) == 1


# --------------------------------------------------------------------------------------
# 8. DecisionService：契约不变
# --------------------------------------------------------------------------------------


def make_decision_service(script: list) -> LLMDecisionService:
    svc = LLMDecisionService()
    fake = _FakeClient(script)
    svc.llm._client = fake  # noqa: SLF001
    return svc


VALID = {"intent": "drug", "intent_type": "drug_conflict", "target_type": "tool",
         "target_name": "drug_interaction", "confidence": 0.9, "reason": "r"}


@pytest.mark.asyncio
async def test_classify_returns_expected_contract():
    svc = make_decision_service([j(**VALID)])
    out = await svc.classify_intent_and_route("布洛芬和阿司匹林能一起吃吗")
    assert out == {
        "intent": "drug",
        "intent_type": "drug_conflict",
        "target_type": "tool",
        "target_name": "drug_interaction",
        "confidence": 0.9,
        "reason": "r",
    }


@pytest.mark.asyncio
async def test_confidence_out_of_range_is_clamped_not_rejected():
    """既有行为是收敛而非丢弃：1.8 应变成 1.0，而不是让整条决策失败。"""
    svc = make_decision_service([j(**{**VALID, "confidence": 1.8})])
    out = await svc.classify_intent_and_route("x")
    assert out is not None and out["confidence"] == 1.0

    svc2 = make_decision_service([j(**{**VALID, "confidence": -3})])
    out2 = await svc2.classify_intent_and_route("x")
    assert out2 is not None and out2["confidence"] == 0.0


@pytest.mark.asyncio
async def test_empty_intent_type_falls_back_to_intent():
    """改造前 data.get("intent_type", intent) 在"字段存在但为空"时会传空串给下游；
    现收紧为回退到 intent——下游多处按 .get("intent_type", "general") 取值，空串会被当成有效值。"""
    svc = make_decision_service([j(**{**VALID, "intent_type": ""})])
    out = await svc.classify_intent_and_route("x")
    assert out is not None and out["intent_type"] == "drug"


@pytest.mark.asyncio
async def test_target_type_is_corrected_from_registry():
    """模型把 drug_interaction 标成 agent 时应被纠正为 tool，而不是丢弃整条决策。"""
    svc = make_decision_service([j(**{**VALID, "target_type": "agent"})])
    out = await svc.classify_intent_and_route("x")
    assert out is not None and out["target_type"] == "tool"


@pytest.mark.asyncio
async def test_invalid_target_name_exhausts_retry_and_returns_none():
    bad = {**VALID, "target_name": "not_registered"}
    svc = make_decision_service([j(**bad), j(**bad)])
    assert await svc.classify_intent_and_route("x") is None


@pytest.mark.asyncio
async def test_invalid_intent_enum_returns_none():
    bad = {**VALID, "intent": "unknown_intent"}
    svc = make_decision_service([j(**bad), j(**bad)])
    assert await svc.classify_intent_and_route("x") is None


@pytest.mark.asyncio
async def test_llm_disabled_short_circuits(monkeypatch):
    monkeypatch.setattr(settings, "LLM_MODEL_NAME", "{{未配置}}")
    svc = make_decision_service([])  # 空脚本：一旦发起调用就会 AssertionError
    assert await svc.classify_intent_and_route("x") is None
    assert await svc.batch_route_queries(["a", "b"]) == [None, None]


# --------------------------------------------------------------------------------------
# 9. DecisionService：实体组装
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lab_entities_are_reassembled():
    svc = make_decision_service([j(
        intent="lab", intent_type="lab_report", target_type="tool", target_name="lab_report",
        confidence=0.9, reason="r", is_multi_intent=False,
        lab_items=[{"item_name": "白细胞", "test_value": "11.2", "unit": "10^9/L"}],
    )])
    out = await svc.classify_route_and_extract("看化验单")
    assert out is not None
    assert out["entities"] == {"lab_items": [{"item_name": "白细胞", "test_value": "11.2", "unit": "10^9/L"}]}
    assert out["is_multi_intent"] is False


@pytest.mark.asyncio
async def test_drug_entities_strip_blank_names():
    svc = make_decision_service([j(
        intent="drug", intent_type="drug_record", target_type="agent", target_name="drug_record_agent",
        confidence=0.8, reason="r", drug_name_list=["阿莫西林", "  ", " 布洛芬 "],
        dosage="0.5g", frequency="一天三次", start_date_text="昨天", purpose="",
    )])
    out = await svc.classify_route_and_extract("我吃了阿莫西林")
    assert out is not None
    assert out["entities"]["drug_name_list"] == ["阿莫西林", "布洛芬"]
    assert out["entities"]["dosage"] == "0.5g"


@pytest.mark.asyncio
async def test_general_intent_entities_is_empty_object():
    svc = make_decision_service([j(
        intent="general", intent_type="general", target_type="agent", target_name="main_qa_agent",
        confidence=0.7, reason="r",
    )])
    out = await svc.classify_route_and_extract("感冒怎么办")
    assert out is not None and out["entities"] == {}


# --------------------------------------------------------------------------------------
# 10. DecisionService：批量路由的逐项容错（迁移中最容易弄坏的语义）
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_keeps_good_items_when_one_is_invalid():
    """单条不合法只能影响该条：其余必须照常返回（与改造前的逐项容错一致）。"""
    svc = make_decision_service([j(decisions=[
        {**VALID, "intent": "lab", "target_name": "lab_report", "target_type": "tool"},
        {**VALID, "target_name": "not_registered"},  # 该条应只把自己置 None
        {**VALID, "target_name": "main_qa_agent", "target_type": "agent"},
    ])])
    out = await svc.batch_route_queries(["q1", "q2", "q3"])
    assert len(out) == 3
    assert out[0] is not None and out[0]["target_name"] == "lab_report"
    assert out[1] is None
    assert out[2] is not None and out[2]["target_name"] == "main_qa_agent"


@pytest.mark.asyncio
async def test_batch_pads_and_truncates_to_query_count():
    svc = make_decision_service([j(decisions=[VALID, VALID, VALID])])
    out = await svc.batch_route_queries(["q1", "q2"])
    assert len(out) == 2 and all(x is not None for x in out)

    svc2 = make_decision_service([j(decisions=[VALID])])
    out2 = await svc2.batch_route_queries(["q1", "q2", "q3"])
    assert len(out2) == 3
    assert out2[0] is not None and out2[1] is None and out2[2] is None


@pytest.mark.asyncio
async def test_batch_wrapper_shape_is_required():
    """根节点是数组（旧提示词的产物）时必须判失败，而不是猜着解析。"""
    svc = make_decision_service(["[{\"intent\": \"drug\"}]", "[{\"intent\": \"drug\"}]"])
    assert await svc.batch_route_queries(["q1"]) == [None]


@pytest.mark.asyncio
async def test_batch_single_query_delegates_to_classify():
    svc = make_decision_service([j(**VALID)])
    out = await svc.batch_route_queries(["只有一个子查询"])
    assert len(out) == 1 and out[0] is not None and out[0]["target_name"] == "drug_interaction"


@pytest.mark.asyncio
async def test_batch_empty_queries_returns_empty_list():
    svc = make_decision_service([])
    assert await svc.batch_route_queries([]) == []


@pytest.mark.asyncio
async def test_batch_ignores_extra_items_beyond_query_count():
    """模型多给条目时截断到查询数，不能多返回（注意单查询会走委托路径，故用 2 条）。"""
    svc = make_decision_service([j(decisions=[VALID, VALID, VALID])])
    out = await svc.batch_route_queries(["q1", "q2"])
    assert len(out) == 2 and all(x is not None for x in out)


# --------------------------------------------------------------------------------------
# 11. schema 自身的静态校验
# --------------------------------------------------------------------------------------


def test_registry_driven_validation_matches_registry():
    names = {c["name"] for c in CAPABILITY_REGISTRY}
    assert names  # 注册表非空
    for name in names:
        assert RouteDecision.model_validate({**VALID, "target_name": name}).target_name == name


def test_batch_schema_relaxes_enums_on_purpose():
    """批量元素用裸 str 接住非法枚举（否则会整批失败），
    真正的枚举校验由 service 层逐条 `RouteDecision.model_validate()` 执行——
    这正是"单条不合法只影响该条"的实现机制。"""
    item = BatchRouteDecision.model_validate(
        {"decisions": [{"intent": "NOT_AN_INTENT", "target_type": "tool", "target_name": "x"}]}
    ).decisions[0]
    assert item.intent == "NOT_AN_INTENT"  # 批量层放行（只校验结构）

    with pytest.raises(ValidationError) as ei:
        RouteDecision.model_validate(item.model_dump())
    assert "NOT_AN_INTENT" in str(ei.value)  # 严格层接管并给出可回灌的错误


def test_route_decision_json_schema_has_enums_and_no_free_form_object():
    schema = so.strictify(RouteDecision.model_json_schema())
    props = schema["properties"]
    assert props["intent"]["enum"] == ["archive", "drug", "lab", "general"]
    assert props["target_type"]["enum"] == ["agent", "tool"]
    assert schema["additionalProperties"] is False


# --------------------------------------------------------------------------------------
# 12. Step 2.5：P1/P2 七处 json.loads 迁移后的契约与边界
#
# 这一批与 P0 三处不同，踩的是两类**新**坑，因此单独一组用例钉住：
#   (a) 输出形态变了：以步骤 id 为键的字典 → 带显式 step_id 的数组；
#       数组根 → 包一层对象。形态变化必须让"旧形态被拒"也被测到，
#       否则模型退回旧格式时会静默产出空结果。
#   (b) 字典键变字段后衍生出"重复 / 越界 / 畸形 id"这类**改造前不存在**的输入，
#       必须显式收敛，不能静默写到错误位置。
# --------------------------------------------------------------------------------------

ROUTE = {
    "intent": "drug",
    "intent_type": "drug_conflict",
    "target_type": "tool",
    "target_name": "drug_interaction",
    "confidence": 0.9,
    "reason": "r",
}


def route(**kw) -> dict:
    """一份合法的路由元素（默认 drug_interaction），按用例覆盖字段。"""
    return {**ROUTE, **kw}


# --- (b) step_id 解析：数组化带来的新输入 ---------------------------------------------


def test_step_id_to_index_parsing():
    assert lds._step_id_to_index("s1", 3) == 0  # noqa: SLF001
    assert lds._step_id_to_index("s3", 3) == 2  # noqa: SLF001
    assert lds._step_id_to_index(" s2 ", 3) == 1  # noqa: SLF001
    # 越界 / 畸形一律 -1（改造前这些情况不可能出现——它们是字典的键）
    for bad in ("s0", "s4", "s", "sx", "", "1", "S1"):
        assert lds._step_id_to_index(bad, 3) == -1, bad  # noqa: SLF001


# --- batch_route_with_deps ------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_route_with_deps_parses_arrays():
    """改造前 routes/deps 是 {"s1": {...}}；现在是带 step_id 的数组。"""
    svc = make_decision_service([
        j(
            routes=[
                route(step_id="s1"),
                route(
                    step_id="s2",
                    intent="lab",
                    intent_type="lab_report",
                    target_type="tool",
                    target_name="lab_report",
                ),
            ],
            deps=[{"step_id": "s2", "depends_on": ["s1"]}],
        )
    ])
    routes, deps = await svc.batch_route_with_deps(["布洛芬和阿司匹林", "那它的数值呢"])
    assert routes[0]["target_name"] == "drug_interaction"
    assert routes[1]["intent"] == "lab"
    assert deps == [[], ["s1"]]


@pytest.mark.asyncio
async def test_batch_route_with_deps_ignores_bad_step_ids():
    """越界/畸形 step_id 必须被丢弃，否则会写到错误的下标上（静默错位）。"""
    svc = make_decision_service([
        j(
            routes=[
                route(step_id="s1"),
                route(step_id="s9", target_name="lab_report"),  # 越界
                route(step_id="x", target_name="lab_report"),  # 畸形
            ],
            deps=[
                {"step_id": "s9", "depends_on": ["s1"]},
                {"step_id": "s2", "depends_on": ["s1"]},
            ],
        )
    ])
    routes, deps = await svc.batch_route_with_deps(["a", "b"])
    assert routes[0]["target_name"] == "drug_interaction"
    assert routes[1] is None  # 没有被 s9/x 误写
    assert deps == [[], ["s1"]]


@pytest.mark.asyncio
async def test_batch_route_with_deps_one_invalid_route_does_not_kill_batch():
    """逐项容错不能退化：单条非法只置 None，其余照常返回。"""
    svc = make_decision_service([
        j(routes=[route(step_id="s1"), route(step_id="s2", target_name="not_registered")], deps=[])
    ])
    routes, _ = await svc.batch_route_with_deps(["a", "b"])
    assert routes[0] is not None
    assert routes[1] is None


@pytest.mark.asyncio
async def test_batch_route_with_deps_keeps_self_dependency():
    """batch_route_with_deps 改造前**不**排除自依赖（split_route_deps 排除）。
    这是既有不一致，这里把两边各自的行为钉住，防止日后被顺手"统一"。"""
    svc = make_decision_service([
        j(routes=[route(step_id="s1")], deps=[{"step_id": "s1", "depends_on": ["s1"]}])
    ])
    _, deps = await svc.batch_route_with_deps(["a", "b"])
    assert deps == [["s1"], []]


@pytest.mark.asyncio
async def test_batch_route_with_deps_drops_unknown_dep_targets():
    svc = make_decision_service([
        j(routes=[route(step_id="s1")], deps=[{"step_id": "s2", "depends_on": ["s1", "s7"]}])
    ])
    _, deps = await svc.batch_route_with_deps(["a", "b"])
    assert deps == [[], ["s1"]]  # s7 不存在 → 被滤掉


@pytest.mark.asyncio
async def test_batch_route_with_deps_duplicate_step_id_last_wins():
    """字典键不会重复，数组元素会。约定为简单赋值的"后写覆盖先写"。"""
    svc = make_decision_service([
        j(
            routes=[
                route(
                    step_id="s1",
                    intent="general",
                    intent_type="general",
                    target_type="agent",
                    target_name="main_qa_agent",
                ),
                route(step_id="s1"),
            ],
            deps=[],
        )
    ])
    routes, _ = await svc.batch_route_with_deps(["a", "b"])
    assert routes[0]["target_name"] == "drug_interaction"


@pytest.mark.asyncio
async def test_batch_route_with_deps_duplicate_step_id_can_hide_the_valid_one():
    """反向用例：重复时若**先**非法后合法，前者被覆盖——语义与字典不同，记录在案。"""
    svc = make_decision_service([
        j(
            routes=[route(step_id="s1", target_name="not_registered"), route(step_id="s1")],
            deps=[],
        )
    ])
    routes, _ = await svc.batch_route_with_deps(["a", "b"])
    assert routes[0]["target_name"] == "drug_interaction"


@pytest.mark.asyncio
async def test_batch_route_with_deps_tolerates_null_optional_fields():
    """容错类型：模型把可选字段写成 null 时不该让**整批**校验失败。"""
    svc = make_decision_service([
        j(
            routes=[
                {
                    "step_id": "s1",
                    "intent": "drug",
                    "intent_type": None,
                    "target_type": "tool",
                    "target_name": "drug_interaction",
                    "confidence": None,
                    "reason": None,
                }
            ],
            deps=[],
        )
    ])
    routes, _ = await svc.batch_route_with_deps(["a", "b"])
    assert routes[0] is not None
    assert routes[0]["confidence"] == 0.0  # null → 0.0
    assert routes[0]["intent_type"] == "drug"  # null → "" → 回退 intent


@pytest.mark.asyncio
async def test_batch_route_with_deps_deps_null_is_empty_not_failure():
    svc = make_decision_service([
        j(routes=[route(step_id="s1")], deps=[{"step_id": "s2", "depends_on": None}])
    ])
    _, deps = await svc.batch_route_with_deps(["a", "b"])
    assert deps == [[], []]


@pytest.mark.asyncio
async def test_batch_route_with_deps_single_query_delegates():
    svc = make_decision_service([j(**VALID)])
    routes, deps = await svc.batch_route_with_deps(["a"])
    assert deps == [[]]
    assert routes[0]["target_name"] == "drug_interaction"


# --- split_route_deps -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_route_deps_parses_arrays():
    svc = make_decision_service([
        j(
            sub_queries=["我的血糖正常吗", "布洛芬能和它一起吃吗"],
            routes=[
                route(
                    step_id="s1",
                    intent="lab",
                    intent_type="lab_report",
                    target_type="tool",
                    target_name="lab_report",
                ),
                route(step_id="s2"),
            ],
            deps=[{"step_id": "s2", "depends_on": ["s1"]}],
        )
    ])
    queries, routes, deps = await svc.split_route_deps("我的血糖正常吗？布洛芬能和它一起吃吗")
    assert queries == ["我的血糖正常吗", "布洛芬能和它一起吃吗"]
    assert routes[0]["intent"] == "lab"
    assert routes[1]["target_name"] == "drug_interaction"
    assert deps == [[], ["s1"]]


@pytest.mark.asyncio
async def test_split_route_deps_excludes_self_dependency():
    """与 batch_route_with_deps 相反：这里**排除**自依赖（改造前就如此）。"""
    svc = make_decision_service([
        j(
            sub_queries=["a", "b"],
            routes=[route(step_id="s1")],
            deps=[{"step_id": "s1", "depends_on": ["s1"]}],
        )
    ])
    _, _, deps = await svc.split_route_deps("x")
    assert deps == [[], []]


@pytest.mark.asyncio
async def test_split_route_deps_empty_sub_queries_returns_none_triple():
    """拆分结果为空 → 三步全 None，由调用方回落旧链路（改造前语义）。"""
    svc = make_decision_service([j(sub_queries=[], routes=[], deps=[])])
    assert await svc.split_route_deps("x") == (None, None, None)


@pytest.mark.asyncio
async def test_split_route_deps_drops_blank_sub_queries_and_keeps_alignment():
    """空串子查询被剔除后，n 变小——routes/deps 长度必须跟着 n，不能按原长度对齐。"""
    svc = make_decision_service([
        j(
            sub_queries=["  ", "真正的问题", ""],
            routes=[route(step_id="s1")],
            deps=[{"step_id": "s1", "depends_on": ["s1"]}],
        )
    ])
    queries, routes, deps = await svc.split_route_deps("x")
    assert queries == ["真正的问题"]
    assert len(routes) == 1 and len(deps) == 1
    assert routes[0] is not None


@pytest.mark.asyncio
async def test_split_route_deps_step_id_beyond_sub_query_count_ignored():
    svc = make_decision_service([
        j(
            sub_queries=["只有一个"],
            routes=[route(step_id="s1"), route(step_id="s2", target_name="lab_report")],
            deps=[],
        )
    ])
    queries, routes, _ = await svc.split_route_deps("x")
    assert queries == ["只有一个"]
    assert len(routes) == 1


@pytest.mark.asyncio
async def test_split_route_deps_transport_error_returns_none_triple():
    svc = make_decision_service([LLMCallException("boom")])
    assert await svc.split_route_deps("x") == (None, None, None)


# --- replan_failed_steps --------------------------------------------------------------

FAILED_STEPS = [
    {"step_id": "s2", "query": "q2", "target_name": "lab_report", "error_msg": "指标缺失"}
]


@pytest.mark.asyncio
async def test_replan_filters_invalid_actions_per_item():
    """逐条容错：非法 action / 不在失败清单里的 step_id 被丢弃，其余动作保留。"""
    svc = make_decision_service([
        j(
            actions=[
                {"step_id": "s2", "action": "rewrite", "query": "补上数值", "target_name": "", "reason": "缺信息"},
                {"step_id": "s2", "action": "explode", "query": "x", "target_name": "", "reason": ""},
                {"step_id": "s9", "action": "retry", "query": "y", "target_name": "", "reason": ""},
                {"step_id": "s2", "action": "drop", "query": "", "target_name": "", "reason": "拿不到"},
            ]
        )
    ])
    out = await svc.replan_failed_steps("原始问题", FAILED_STEPS, [])
    assert out is not None
    assert [a["action"] for a in out] == ["rewrite", "drop"]
    assert out[0]["query"] == "补上数值"


@pytest.mark.asyncio
async def test_replan_reroute_with_unregistered_target_degrades_to_drop():
    """reroute 到不存在的执行体 → 退化为 drop（改造前行为），而不是原样转发。"""
    svc = make_decision_service([
        j(actions=[{"step_id": "s2", "action": "reroute", "query": "q", "target_name": "ghost_tool", "reason": "r"}])
    ])
    out = await svc.replan_failed_steps("原始问题", FAILED_STEPS, [])
    assert out[0]["action"] == "drop"
    assert out[0]["target_name"] == ""


@pytest.mark.asyncio
async def test_replan_reroute_with_registered_target_is_kept():
    svc = make_decision_service([
        j(actions=[{"step_id": "s2", "action": "reroute", "query": "q", "target_name": "lab_report", "reason": "r"}])
    ])
    out = await svc.replan_failed_steps("原始问题", FAILED_STEPS, [])
    assert out[0]["action"] == "reroute"
    assert out[0]["target_name"] == "lab_report"


@pytest.mark.asyncio
async def test_replan_non_reroute_action_clears_unregistered_target():
    """非 reroute 动作也统一清空未注册的 target_name（改造前行为，别丢）。"""
    svc = make_decision_service([
        j(actions=[{"step_id": "s2", "action": "retry", "query": "q", "target_name": "ghost_tool", "reason": "r"}])
    ])
    out = await svc.replan_failed_steps("原始问题", FAILED_STEPS, [])
    assert out[0]["action"] == "retry"
    assert out[0]["target_name"] == ""


@pytest.mark.asyncio
async def test_replan_empty_actions_returns_none():
    svc = make_decision_service([j(actions=[])])
    assert await svc.replan_failed_steps("原始问题", FAILED_STEPS, []) is None


@pytest.mark.asyncio
async def test_replan_no_failed_steps_short_circuits_without_llm_call():
    svc = make_decision_service([])  # 空脚本：一旦发起调用就会 AssertionError
    assert await svc.replan_failed_steps("原始问题", [], []) is None


# --- split_queries --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_split_queries_uses_object_wrapper():
    """改造前是数组根 ["q1","q2"]；strict 要求根是 object → 包一层 queries。"""
    svc = make_decision_service([j(queries=["我的血糖正常吗", "布洛芬能和它一起吃吗"])])
    assert await svc.split_queries("x") == ["我的血糖正常吗", "布洛芬能和它一起吃吗"]


@pytest.mark.asyncio
async def test_split_queries_strips_and_drops_blanks():
    svc = make_decision_service([j(queries=["  ", " 有内容的 ", ""])])
    assert await svc.split_queries("x") == ["有内容的"]


@pytest.mark.asyncio
async def test_split_queries_empty_list_returns_none():
    svc = make_decision_service([j(queries=[])])
    assert await svc.split_queries("x") is None


@pytest.mark.asyncio
async def test_split_queries_old_array_root_is_now_rejected():
    """回归护栏：模型若退回改造前的数组根形态，必须判失败（重试后仍失败 → None），
    而不是悄悄产出空结果。"""
    svc = make_decision_service(['["a", "b"]', '["a", "b"]'])
    assert await svc.split_queries("x") is None


# --- 实体抽取三件套 --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_entities_drug_requires_at_least_one_name():
    """无药名 = 抽取失败 → None，让调用方走兜底（改造前语义）。"""
    svc = make_decision_service(
        [j(drug_name_list=[], dosage="100mg", frequency="", start_date_text="", purpose="")]
    )
    assert await svc.extract_entities("x", "drug") is None


@pytest.mark.asyncio
async def test_extract_entities_drug_strips_blank_names():
    svc = make_decision_service(
        [j(drug_name_list=["阿司匹林", "  ", " 布洛芬 "], dosage="100mg",
           frequency="qd", start_date_text="今天", purpose="抗凝")]
    )
    out = await svc.extract_entities("x", "drug")
    assert out is not None
    assert out["drug_name_list"] == ["阿司匹林", "布洛芬"]
    assert out["dosage"] == "100mg"
    assert out["purpose"] == "抗凝"


@pytest.mark.asyncio
async def test_extract_entities_lab_keeps_raw_and_normalises_items():
    """`raw` 是既有契约的一部分，必须保留；元素统一为三字段（缺 unit 补空串）。"""
    svc = make_decision_service([j(lab_items=[{"item_name": "血糖", "test_value": "6.5"}])])
    out = await svc.extract_entities("血糖6.5", "lab")
    assert out is not None
    assert out["raw"] == "血糖6.5"
    assert out["lab_items"] == [{"item_name": "血糖", "test_value": "6.5", "unit": ""}]


@pytest.mark.asyncio
async def test_extract_entities_lab_filters_non_dict_items():
    """非 dict 元素在改造前会一路透传到下游才炸；现在在校验层滤掉。"""
    svc = make_decision_service(
        [j(lab_items=[{"item_name": "血糖", "test_value": "6.5", "unit": "mmol/L"}, "垃圾", None])]
    )
    out = await svc.extract_entities("x", "lab")
    assert len(out["lab_items"]) == 1


@pytest.mark.asyncio
async def test_extract_entities_lab_empty_items_returns_none():
    svc = make_decision_service([j(lab_items=[])])
    assert await svc.extract_entities("x", "lab") is None


@pytest.mark.asyncio
async def test_extract_entities_lab_extra_field_is_rejected_then_retries():
    """夹带未声明字段 → 校验失败（extra=forbid），重试后仍失败 → None。"""
    bad = j(lab_items=[{"item_name": "血糖", "test_value": "6.5", "unit": "", "杂字段": 1}])
    svc = make_decision_service([bad, bad])
    assert await svc.extract_entities("x", "lab") is None


@pytest.mark.asyncio
async def test_extract_entities_unsupported_intent_short_circuits():
    svc = make_decision_service([])  # 空脚本：不应发起调用
    assert await svc.extract_entities("x", "general") is None
    assert await svc.extract_entities("x", "archive") is None


@pytest.mark.asyncio
async def test_extract_drug_info_requires_drug_name():
    svc = make_decision_service(
        [j(drug_name="", dosage="200mg", frequency="", start_date_text="", purpose="")]
    )
    assert await svc.extract_drug_info("x") is None


@pytest.mark.asyncio
async def test_extract_drug_info_happy_path_strips_name():
    svc = make_decision_service(
        [j(drug_name=" 布洛芬 ", dosage="200mg", frequency="bid",
           start_date_text="昨晚", purpose="退烧")]
    )
    out = await svc.extract_drug_info("x")
    assert out == {
        "drug_name": "布洛芬",
        "dosage": "200mg",
        "frequency": "bid",
        "start_date_text": "昨晚",
        "purpose": "退烧",
    }


# --- 13. 新增契约的静态约束 ------------------------------------------------------------


def _strict_violations(node, path="root", bad=None):
    """递归找出不满足 strict 模式两条硬要求的位置。"""
    if bad is None:
        bad = []
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict) and props:
            if node.get("additionalProperties") is not False:
                bad.append(f"{path}: additionalProperties 未关闭")
            if set(node.get("required") or []) != set(props.keys()):
                bad.append(f"{path}: required 未覆盖全部 properties")
        for k, v in node.items():
            _strict_violations(v, f"{path}.{k}", bad)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _strict_violations(v, f"{path}[{i}]", bad)
    return bad


def test_step25_schemas_are_strict_safe():
    """新增契约都必须满足 strict：每个 object 关 additionalProperties 且 required 全覆盖。"""
    classes = [
        S.LabItem,
        S.BatchRouteItem,
        S.RouteAndEntities,
        S.StepRouteItem,
        S.DepsItem,
        S.BatchRouteWithDeps,
        S.SplitRouteDeps,
        S.ReplanAction,
        S.ReplanResult,
        S.QuerySplit,
        S.DrugEntities,
        S.LabEntities,
        S.DrugRecordInfo,
    ]
    for cls in classes:
        violations = _strict_violations(so.strictify(cls.model_json_schema()))
        assert not violations, f"{cls.__name__}: {violations}"


def test_tolerant_types_do_not_create_free_form_objects():
    """容错类型（Text/Number/TextList）不能把 schema 变成自由形态对象——
    否则 strict 模式表达不了，整条链路只能退回 json_object。"""
    props = S.DrugEntities.model_json_schema()["properties"]
    assert props["drug_name_list"]["type"] == "array"
    assert props["drug_name_list"]["items"] == {"type": "string"}
    assert props["dosage"]["type"] == "string"
    assert S.RouteDecision.model_json_schema()["properties"]["confidence"]["type"] == "number"


def test_tolerant_types_accept_null():
    d = S.DrugEntities.model_validate(
        {
            "drug_name_list": None,
            "dosage": None,
            "frequency": None,
            "start_date_text": None,
            "purpose": None,
        }
    )
    assert d.drug_name_list == [] and d.dosage == "" and d.frequency == ""

    item = S.StepRouteItem.model_validate(
        {
            "step_id": None,
            "intent": "drug",
            "target_type": "tool",
            "target_name": "drug_interaction",
        }
    )
    assert item.step_id == "" and item.confidence == 0.0 and item.reason == ""


def test_tolerant_str_list_keeps_old_non_list_semantics():
    """裸字符串**不**被收成单元素列表——改造前 `isinstance(x, list)` 不成立即视为没有。

    若在这里"顺手兼容"单字符串，`drug_name_list` 的含义就会从"模型给的列表"
    变成"模型给的任意东西"，下游按列表处理的地方会静默错。
    """
    assert S.DrugEntities.model_validate({"drug_name_list": "阿司匹林"}).drug_name_list == []
    assert S.DrugEntities.model_validate({"drug_name_list": None}).drug_name_list == []
    # 列表元素的 str() 转换保留（对齐旧代码 str(n).strip()）
    assert S.DrugEntities.model_validate({"drug_name_list": ["a", 1]}).drug_name_list == ["a", "1"]


def test_batch_route_with_deps_schema_has_no_dict_keyed_route():
    """形态护栏：routes/deps 必须是数组——回归到字典键就说明约束 1 被破坏。"""
    props = S.BatchRouteWithDeps.model_json_schema()["properties"]
    assert props["routes"]["type"] == "array"
    assert props["deps"]["type"] == "array"
    # 数组元素必须带 step_id（字典键提升成的字段）
    assert "step_id" in S.StepRouteItem.model_json_schema()["properties"]
