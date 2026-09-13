from __future__ import annotations

import asyncio
import base64
import contextvars
from collections.abc import AsyncGenerator

import httpx

from pydantic import BaseModel, ValidationError

from app.common.exceptions import LLMCallException
from app.common.langfuse_helper import elapsed_ms, time_block, track_llm_call
from app.common.logger import get_logger, log_llm_call
from app.config.settings import settings
from app.core.llm.structured_output import (
    build_response_format,
    extract_json_candidate,
    resolve_mode,
)

logger = get_logger(__name__)

# 结构化输出路径的可观测计数（便于判断 response_format 是否真的在起作用）
_extract_fallback_count = 0
_validation_failure_count = 0
# 空响应 / 截断**单独计数**：它们不是「schema 不匹配」。
# 混在 validation_failure 里会让「模型压根没输出」被读成「模型的 JSON 不合规」，
# 排查方向直接错掉（2026-09-12 实测踩过：空响应被记成 "Invalid JSON: EOF"）。
_empty_response_count = 0
_truncated_count = 0


def structured_output_stats() -> dict[str, int]:
    """返回结构化输出路径的计数快照。"""
    return {
        "extract_fallback": _extract_fallback_count,
        "validation_failure": _validation_failure_count,
        "empty_response": _empty_response_count,
        "truncated": _truncated_count,
    }


def reset_structured_output_stats() -> None:
    """清零计数（测试用）。"""
    global _extract_fallback_count, _validation_failure_count
    global _empty_response_count, _truncated_count
    _extract_fallback_count = 0
    _validation_failure_count = 0
    _empty_response_count = 0
    _truncated_count = 0


def _validation_error_brief(err: ValidationError, limit: int = 8) -> str:
    """把 ValidationError 压成"字段路径: 原因"的短列表，便于回灌给模型和打日志。"""
    lines: list[str] = []
    for e in err.errors()[:limit]:
        loc = ".".join(str(p) for p in (e.get("loc") or ())) or "(根)"
        lines.append(f"{loc}: {e.get('msg', '')}")
    if len(err.errors()) > limit:
        lines.append(f"...（其余 {len(err.errors()) - limit} 条略）")
    return "; ".join(lines)


def _build_repair_prompt(schema: type[BaseModel], err: ValidationError) -> str:
    """构造"修正重试"消息：给出字段清单 + 具体校验错误。

    这是 json_object 档位下**唯一真正能对齐字段/枚举的手段**；在 json_schema 档位下
    则是兜底（服务端约束已能挡掉大部分结构问题，但语义错误仍需反馈）。
    """
    fields = ", ".join(schema.model_fields.keys())
    return (
        "你上一次的输出未通过 JSON 校验。请修正后**只输出一个 JSON 对象**，"
        "不要解释文字，不要 markdown 代码块。\n"
        f"必须包含且仅包含这些字段：{fields}\n"
        f"校验错误：{_validation_error_brief(err)}"
    )


def _build_no_output_prompt(reason: str) -> str:
    """空响应后的重试提示。

    与 `_build_repair_prompt` 的差别是本质的：校验失败时该把**具体字段错误**回灌，
    而空响应意味着模型压根没产出内容——回灌"校验错误"没有任何信息量，
    只会让它继续瞎猜。这里直接点明事实并要求补全。
    """
    return (
        f"你上一次没有返回任何可用内容（{reason}）。"
        "请只输出一个完整的 JSON 对象：不要解释文字，不要 markdown 代码块，"
        "不要省略任何字段。"
    )


def _build_truncated_prompt() -> str:
    """截断后的重试提示：要求精简但仍字段齐全。

    不能沿用 `_build_repair_prompt`——截断场景下的字段错误只是"后半段没写完"的
    副产品，回灌一堆校验错误会误导模型去改本来正确的字段。
    """
    return (
        "你上一次的输出被截断了，JSON 没有写完。请**精简**后重发："
        "仍然必须包含全部字段，但 reason 之类的说明性字段请尽量短（20 字以内），"
        "不要重复或展开解释。"
    )

_global_client: AsyncOpenAI | None = None
_http2_available: bool = False
_client_lock = asyncio.Lock()

# 每请求 LLM usage 累加器（ContextVar，随请求上下文传播）
_usage_accum_var: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "llm_usage_accum", default=None
)


def begin_usage_tracking() -> None:
    """开启本轮请求的 LLM usage 聚合（在 run_stream / run 开头调用）。"""
    _usage_accum_var.set([])


def record_usage(prompt_tokens: int | None, cached_tokens: int | None) -> None:
    """每次成功 LLM 调用后追加一条 usage。"""
    accum = _usage_accum_var.get()
    if accum is None or prompt_tokens is None:
        return
    accum.append((int(prompt_tokens), int(cached_tokens or 0)))


def end_usage_tracking() -> dict | None:
    """汇总本轮请求的缓存命中统计；无调用返回 None。"""
    accum = _usage_accum_var.get()
    if not accum:
        return None
    prompt = sum(p for p, _ in accum)
    cached = sum(c for _, c in accum)
    return {
        "calls": len(accum),
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "hit_rate": round(cached / prompt, 4) if prompt else 0.0,
    }

try:
    import h2  # noqa: F401
    _http2_available = True
except ImportError:
    pass


def _build_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=120,
        ),
        timeout=httpx.Timeout(
            connect=10.0,
            read=60.0,
            write=30.0,
            pool=10.0,
        ),
        http2=_http2_available,
    )


def _extract_cache_tokens(usage, input_tokens: int | None):
    """从 usage.prompt_tokens_details 提取缓存命中/未命中 token 数。

    OpenAI 兼容格式：usage.prompt_tokens_details.cached_tokens 为缓存命中数；
    未命中数由 prompt_tokens - cached_tokens 推导（仅当两者都有时）。
    """
    cached = None
    if usage is not None:
        ptd = usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
        if isinstance(ptd, dict):
            cached = ptd.get("cached_tokens")
        elif ptd is not None:
            cached = getattr(ptd, "cached_tokens", None)
        if cached is None:
            cached = 0
    miss = None
    if cached is not None and input_tokens is not None:
        miss = int(input_tokens) - int(cached)
    return cached, miss


async def get_shared_client() -> AsyncOpenAI:
    global _global_client
    if _global_client is not None:
        return _global_client
    async with _client_lock:
        if _global_client is not None:
            return _global_client
        from openai import AsyncOpenAI

        _global_client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE,
            http_client=_build_http_client(),
        )
        logger.info(
            "LLM shared client created: base_url=%s max_conn=20 keepalive=10 http2=%s",
            settings.LLM_API_BASE,
            _http2_available,
        )
        return _global_client


class LLMService:
    def __init__(self):
        self._client = None

    async def _get_client(self):
        if self._client is not None:
            return self._client
        self._client = await get_shared_client()
        return self._client

    async def chat_completion_stream(
        self,
        *,
        prompt: str,
        system_prompt: str,
        timeout_s: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        model = settings.LLM_MODEL_NAME
        start = time_block()
        total_content = ""
        try:
            client = await self._get_client()
            coro = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
            )
            stream_resp = await (asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else coro)
            usage = None
            async for chunk in stream_resp:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    total_content += content
                    yield content
            latency_ms = elapsed_ms(start)
            input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            cached_tokens, cache_miss_tokens = _extract_cache_tokens(usage, input_tokens)
            record_usage(input_tokens, cached_tokens)
            log_llm_call(
                model=model,
                prompt_len=len(prompt),
                system_prompt_len=len(system_prompt),
                response_len=len(total_content),
                input_tokens=input_tokens,
                cached_tokens=cached_tokens,
                cache_miss_tokens=cache_miss_tokens,
                latency_ms=latency_ms,
                success=True,
            )
        except asyncio.TimeoutError:
            latency_ms = elapsed_ms(start)
            logger.warning("LLM流式调用超时(%.2fs)", float(timeout_s or 0))
            track_llm_call(model=model, latency_ms=latency_ms, success=False, error="timeout")
            raise LLMCallException("大模型流式调用超时")
        except Exception as e:
            latency_ms = elapsed_ms(start)
            logger.error("LLM流式调用失败: %s", str(e))
            track_llm_call(model=model, latency_ms=latency_ms, success=False, error="call_failed")
            raise LLMCallException("大模型流式调用失败")

    async def chat_completion(
        self,
        *,
        prompt: str,
        system_prompt: str,
        stream: bool = False,
        timeout_s: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        start = time_block()
        model = settings.LLM_MODEL_NAME
        try:
            client = await self._get_client()
            coro = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
                stream=stream,
            )

            resp = await (asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else coro)

            if stream:
                raise LLMCallException("当前调用不支持 stream=True")
            usage = getattr(resp, "usage", None)
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
            else:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

            cached_tokens, cache_miss_tokens = _extract_cache_tokens(usage, input_tokens)
            record_usage(input_tokens, cached_tokens)

            content = resp.choices[0].message.content or ""
            latency_ms = elapsed_ms(start)

            track_llm_call(
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                cache_miss_tokens=cache_miss_tokens,
                success=True,
            )

            log_llm_call(
                model=model,
                prompt_len=len(prompt),
                system_prompt_len=len(system_prompt),
                response_len=len(content),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                cache_miss_tokens=cache_miss_tokens,
                latency_ms=latency_ms,
                success=True,
            )

            return content
        except asyncio.TimeoutError as e:
            latency_ms = elapsed_ms(start)
            logger.warning("LLM调用超时(%.2fs)", float(timeout_s or 0))
            track_llm_call(
                model=model,
                latency_ms=latency_ms,
                success=False,
                error="timeout",
            )
            log_llm_call(
                model=model,
                prompt_len=len(prompt),
                system_prompt_len=len(system_prompt),
                latency_ms=latency_ms,
                success=False,
                error="timeout",
            )
            raise LLMCallException("大模型调用超时") from e
        except Exception as e:
            latency_ms = elapsed_ms(start)
            logger.error("LLM调用失败: %s", str(e))
            track_llm_call(
                model=model,
                latency_ms=latency_ms,
                success=False,
                error="call_failed",
            )
            log_llm_call(
                model=model,
                prompt_len=len(prompt),
                system_prompt_len=len(system_prompt),
                latency_ms=latency_ms,
                success=False,
                error=str(e)[:100],
            )
            raise LLMCallException("大模型调用失败") from e

    async def chat_completion_json(
        self,
        *,
        prompt: str,
        system_prompt: str,
        schema: type[BaseModel],
        timeout_s: float | None = None,
        max_tokens: int | None = None,
        max_retry: int = 1,
        schema_name: str | None = None,
    ) -> BaseModel | None:
        """结构化输出调用：返回已校验的 Pydantic 对象；无法满足 schema 时返回 None。

        与 `chat_completion` 的差别就是本方法的全部价值：
        - **输出受约束**：档位支持时下发 strict `json_schema`，由服务端约束解码
        - **失败可观测**：校验失败会打出带字段路径的日志并计数，不再是"静默 return None"
        - **失败可自愈**：把校验错误回灌给模型，重试 `max_retry` 次

        分层职责（重要）：
        - `response_format` = **加速器**。能不能额外拿到一层服务端约束，取决于模型快照，
          由 `structured_output.resolve_mode` 在运行期确定（见该模块文档）。
        - 客户端 `model_validate_json` + 重试 = **不变量**。两个档位都必须执行——
          因为"模型支持某能力"是部署属性，不是能写进代码的契约。

        返回：校验通过的模型实例；重试耗尽仍不合法则返回 None（调用方沿用原有兜底语义）。
        传输层错误（超时/网络）抛 `LLMCallException`，与 `chat_completion` 保持一致。
        """
        global _extract_fallback_count, _validation_failure_count
        global _empty_response_count, _truncated_count

        model = settings.LLM_MODEL_NAME
        start = time_block()
        client = await self._get_client()
        mode = await resolve_mode(client, model)
        response_format = build_response_format(schema, mode, name=schema_name)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        attempts = max(1, max_retry + 1)

        for attempt in range(attempts):
            try:
                coro = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
                    response_format=response_format,
                    # 关闭「思考」：见 settings.LLM_DISABLE_THINKING 的实测数据。
                    # 只作用于结构化决策路径；答案生成走 chat_completion，不受影响。
                    **(
                        {"extra_body": {"enable_thinking": False}}
                        if settings.LLM_DISABLE_THINKING
                        else {}
                    ),
                )
                resp = await (asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else coro)
            except asyncio.TimeoutError as e:
                latency_ms = elapsed_ms(start)
                logger.warning("LLM结构化调用超时(%.2fs)", float(timeout_s or 0))
                track_llm_call(model=model, latency_ms=latency_ms, success=False, error="timeout")
                log_llm_call(
                    model=model,
                    prompt_len=len(prompt),
                    system_prompt_len=len(system_prompt),
                    latency_ms=latency_ms,
                    success=False,
                    error="timeout",
                    caller="chat_completion_json",
                )
                raise LLMCallException("大模型结构化调用超时") from e
            except Exception as e:
                latency_ms = elapsed_ms(start)
                logger.error("LLM结构化调用失败: %s", str(e))
                track_llm_call(model=model, latency_ms=latency_ms, success=False, error="call_failed")
                log_llm_call(
                    model=model,
                    prompt_len=len(prompt),
                    system_prompt_len=len(system_prompt),
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e)[:100],
                    caller="chat_completion_json",
                )
                raise LLMCallException("大模型结构化调用失败") from e

            usage = getattr(resp, "usage", None)
            if isinstance(usage, dict):
                input_tokens = usage.get("prompt_tokens")
                output_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")
            else:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
            cached_tokens, cache_miss_tokens = _extract_cache_tokens(usage, input_tokens)
            record_usage(input_tokens, cached_tokens)

            choice = resp.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)

            # --- 空响应：先于 JSON 校验判定，且**不计入** schema 校验失败 ---
            # 实测（2026-09-12，deepseek-v4-flash）：偶发 content='' 且
            # finish_reason='length'、completion_tokens 正好顶到 max_tokens。
            # 该模型的 max_tokens 同时覆盖**推理**与答案，推理吃满预算时答案侧
            # 一个 token 都拿不到。若当成 JSON 语法错误处理，会白烧一次往返，
            # 并留下 "Invalid JSON: EOF" 这种把排查引向错误方向的日志。
            if not raw.strip():
                _empty_response_count += 1
                logger.warning(
                    "[structured] 空响应(mode=%s attempt=%d/%d finish_reason=%s out_tok=%s)"
                    "→ 按「无输出」重试；不计入 schema 校验失败",
                    mode,
                    attempt + 1,
                    attempts,
                    finish_reason,
                    output_tokens,
                )
                if attempt + 1 < attempts:
                    messages.append(
                        {
                            "role": "user",
                            "content": _build_no_output_prompt(
                                f"上一条回复为空，finish_reason={finish_reason}"
                            ),
                        }
                    )
                    continue
                break

            # --- 截断：内容非空但没写完，同样不是 JSON 语法问题 ---
            if finish_reason == "length":
                _truncated_count += 1
                logger.warning(
                    "[structured] 响应被截断(mode=%s attempt=%d/%d len=%d out_tok=%s)："
                    "strict schema 要求字段全部出现，输出比「缺字段也行」的形态更长，"
                    "调用方应提高 max_tokens",
                    mode,
                    attempt + 1,
                    attempts,
                    len(raw),
                    output_tokens,
                )

            text, used_fallback = extract_json_candidate(raw)
            if used_fallback:
                _extract_fallback_count += 1
                logger.info(
                    "[structured] 响应需兜底提取(mode=%s attempt=%d) raw=%r",
                    mode,
                    attempt + 1,
                    raw[:120],
                )

            try:
                obj = schema.model_validate_json(text)
            except ValidationError as e:
                _validation_failure_count += 1
                logger.warning(
                    "[structured] schema 校验失败(mode=%s attempt=%d/%d): %s",
                    mode,
                    attempt + 1,
                    attempts,
                    _validation_error_brief(e),
                )
                if attempt + 1 < attempts:
                    messages.append({"role": "assistant", "content": raw})
                    # 截断导致的不完整：字段错误只是"没写完"的副产品，
                    # 应要求精简而不是回灌一堆校验错误去误导模型。
                    repair = (
                        _build_truncated_prompt()
                        if finish_reason == "length"
                        else _build_repair_prompt(schema, e)
                    )
                    messages.append({"role": "user", "content": repair})
                    continue
                break

            latency_ms = elapsed_ms(start)
            track_llm_call(
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                cache_miss_tokens=cache_miss_tokens,
                success=True,
            )
            log_llm_call(
                model=model,
                prompt_len=len(prompt),
                system_prompt_len=len(system_prompt),
                response_len=len(raw),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                cache_miss_tokens=cache_miss_tokens,
                latency_ms=latency_ms,
                success=True,
                caller="chat_completion_json",
            )
            return obj

        # 重试耗尽：可观测降级。改造前这里是"静默 return None"，拿不到任何指标。
        latency_ms = elapsed_ms(start)
        log_llm_call(
            model=model,
            prompt_len=len(prompt),
            system_prompt_len=len(system_prompt),
            latency_ms=latency_ms,
            success=False,
            error="schema_validation",
            caller="chat_completion_json",
        )
        return None

    async def chat_completion_vision(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        mime: str,
        schema: type[BaseModel],
        system_prompt: str = "",
        timeout_s: float | None = None,
        max_tokens: int | None = None,
        max_retry: int = 1,
        schema_name: str | None = None,
    ) -> BaseModel | None:
        """图像 + 文本 → 受 schema 约束的结构化结果。

        **为什么单独成方法，而不是给 `chat_completion_json` 加个参数**：
        后者的 `messages` 是 `list[dict[str, str]]`，塞不进图片 content block
        （图片块是 `list[dict]`）；而它承载的是**文本主链路**，签名一改就把风险引到主链路上。
        这里只新增入口，主链路一行不动。

        与 `chat_completion_json` **有意复用**的部分：

        - 档位判定 `resolve_mode(client, model)` —— 缓存按 `(base_url, model)` 分键，
          视觉模型是**独立的一次探测**，不会蹭文本模型的结论
        - 失败三分类计数：`empty_response` / `truncated` / `validation_failure`
        - 重试时回灌校验错误（`_build_repair_prompt` / `_build_truncated_prompt`）
        - 客户端 `model_validate_json` **始终执行** —— `response_format` 只是加速器

        **不同**的部分：

        - 模型取 `settings.LLM_VISION_MODEL_NAME`（独立配置，留空视为未开启）
        - `enable_thinking` 由 `LLM_VISION_DISABLE_THINKING` 单独控制，默认**不下发** ——
          该参数是提供方耦合点，换模型/换提供方必须重测
          （`scripts/probe_lab_vision_params.py`）

        返回：校验通过的模型实例；重试耗尽仍不合法返回 `None`。
        传输层错误抛 `LLMCallException`，与另外两个方法一致。
        """
        global _extract_fallback_count, _validation_failure_count
        global _empty_response_count, _truncated_count

        model = (settings.LLM_VISION_MODEL_NAME or "").strip()
        if not model:
            raise LLMCallException("未配置视觉模型（LLM_VISION_MODEL_NAME 为空）")

        start = time_block()
        client = await self._get_client()
        mode = await resolve_mode(client, model)
        response_format = build_response_format(schema, mode, name=schema_name)

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )
        attempts = max(1, max_retry + 1)

        for attempt in range(attempts):
            try:
                coro = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=max_tokens
                    if max_tokens is not None
                    else settings.LLM_VISION_MAX_TOKENS,
                    response_format=response_format,
                    **(
                        {"extra_body": {"enable_thinking": False}}
                        if settings.LLM_VISION_DISABLE_THINKING
                        else {}
                    ),
                )
                resp = await (asyncio.wait_for(coro, timeout=timeout_s) if timeout_s else coro)
            except asyncio.TimeoutError as e:
                latency_ms = elapsed_ms(start)
                logger.warning("LLM视觉调用超时(%.2fs)", float(timeout_s or 0))
                track_llm_call(model=model, latency_ms=latency_ms, success=False, error="timeout")
                log_llm_call(
                    model=model,
                    prompt_len=len(prompt),
                    latency_ms=latency_ms,
                    success=False,
                    error="timeout",
                    caller="chat_completion_vision",
                )
                raise LLMCallException("视觉模型调用超时") from e
            except Exception as e:
                latency_ms = elapsed_ms(start)
                logger.error("LLM视觉调用失败: %s", str(e))
                track_llm_call(model=model, latency_ms=latency_ms, success=False, error="call_failed")
                log_llm_call(
                    model=model,
                    prompt_len=len(prompt),
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e)[:100],
                    caller="chat_completion_vision",
                )
                raise LLMCallException("视觉模型调用失败") from e

            usage = getattr(resp, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
            total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
            cached_tokens, cache_miss_tokens = _extract_cache_tokens(usage, input_tokens)
            record_usage(input_tokens, cached_tokens)

            choice = resp.choices[0]
            raw = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)

            # --- 空响应：先于 JSON 校验判定，且不计入 schema 校验失败 ---
            # 视觉模型（尤其推理型）的 max_tokens 可能同时覆盖推理与答案，
            # 推理吃满预算时答案侧拿不到 token。当成 JSON 语法错误处理会把排查带偏。
            if not raw.strip():
                _empty_response_count += 1
                logger.warning(
                    "[vision] 空响应(mode=%s attempt=%d/%d finish_reason=%s out_tok=%s)"
                    "→ 按「无输出」重试；不计入 schema 校验失败",
                    mode,
                    attempt + 1,
                    attempts,
                    finish_reason,
                    output_tokens,
                )
                if attempt + 1 < attempts:
                    messages.append(
                        {
                            "role": "user",
                            "content": _build_no_output_prompt(
                                f"上一条回复为空，finish_reason={finish_reason}"
                            ),
                        }
                    )
                    continue
                break

            if finish_reason == "length":
                _truncated_count += 1
                logger.warning(
                    "[vision] 响应被截断(mode=%s attempt=%d/%d len=%d out_tok=%s)："
                    "strict schema 要求字段全部出现，应提高 LLM_VISION_MAX_TOKENS",
                    mode,
                    attempt + 1,
                    attempts,
                    len(raw),
                    output_tokens,
                )

            text, used_fallback = extract_json_candidate(raw)
            if used_fallback:
                _extract_fallback_count += 1
                logger.info(
                    "[vision] 响应需兜底提取(mode=%s attempt=%d) raw=%r",
                    mode,
                    attempt + 1,
                    raw[:120],
                )

            try:
                obj = schema.model_validate_json(text)
            except ValidationError as e:
                _validation_failure_count += 1
                logger.warning(
                    "[vision] schema 校验失败(mode=%s attempt=%d/%d): %s",
                    mode,
                    attempt + 1,
                    attempts,
                    _validation_error_brief(e),
                )
                if attempt + 1 < attempts:
                    messages.append({"role": "assistant", "content": raw})
                    repair = (
                        _build_truncated_prompt()
                        if finish_reason == "length"
                        else _build_repair_prompt(schema, e)
                    )
                    messages.append({"role": "user", "content": repair})
                    continue
                break

            latency_ms = elapsed_ms(start)
            track_llm_call(
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                cache_miss_tokens=cache_miss_tokens,
                success=True,
            )
            log_llm_call(
                model=model,
                prompt_len=len(prompt),
                response_len=len(raw),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                cache_miss_tokens=cache_miss_tokens,
                latency_ms=latency_ms,
                success=True,
                caller="chat_completion_vision",
            )
            return obj

        latency_ms = elapsed_ms(start)
        log_llm_call(
            model=model,
            prompt_len=len(prompt),
            latency_ms=latency_ms,
            success=False,
            error="schema_validation",
            caller="chat_completion_vision",
        )
        return None
