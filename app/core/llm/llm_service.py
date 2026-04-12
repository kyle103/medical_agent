from __future__ import annotations

import asyncio
import logging

from app.common.exceptions import LLMCallException
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

            return resp.choices[0].message.content or ""
        except asyncio.TimeoutError as e:
            logger.warning("LLM调用超时(%.2fs)", float(timeout_s or 0))
            raise LLMCallException("大模型调用超时") from e
        except Exception as e:
            logger.error("LLM调用失败: %s", str(e))
            raise LLMCallException("大模型调用失败") from e
