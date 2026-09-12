from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncGenerator

import httpx

from app.common.exceptions import LLMCallException
from app.common.langfuse_helper import elapsed_ms, time_block, track_llm_call
from app.common.logger import get_logger, log_llm_call
from app.config.settings import settings

logger = get_logger(__name__)

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
