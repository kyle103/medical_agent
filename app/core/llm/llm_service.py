from __future__ import annotations

import asyncio
import logging

from app.common.exceptions import LLMCallException
from app.common.langfuse_helper import elapsed_ms, time_block, track_llm_call
from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            # lazy import：避免模块导入阶段加载 openai SDK（会显著拖慢启动/pytest collecting）
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_API_BASE)
        return self._client

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
        try:
            client = self._get_client()
            coro = client.chat.completions.create(
                model=settings.LLM_MODEL_NAME,
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

            track_llm_call(
                model=settings.LLM_MODEL_NAME,
                latency_ms=elapsed_ms(start),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                success=True,
            )

            return resp.choices[0].message.content or ""
        except asyncio.TimeoutError as e:
            logger.warning("LLM调用超时(%.2fs)", float(timeout_s or 0))
            track_llm_call(
                model=settings.LLM_MODEL_NAME,
                latency_ms=elapsed_ms(start),
                success=False,
                error="timeout",
            )
            raise LLMCallException("大模型调用超时") from e
        except Exception as e:
            logger.error("LLM调用失败: %s", str(e))
            track_llm_call(
                model=settings.LLM_MODEL_NAME,
                latency_ms=elapsed_ms(start),
                success=False,
                error="call_failed",
            )
            raise LLMCallException("大模型调用失败") from e
