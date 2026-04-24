from __future__ import annotations

import os
import time
from typing import Any, Optional

from app.common.logger import get_logger

logger = get_logger(__name__)

_langfuse_client = None


def _langfuse_enabled() -> bool:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return False
    return True


def _get_langfuse_client():
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    if not _langfuse_enabled():
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )
        return _langfuse_client
    except Exception:
        logger.exception("langfuse init failed")
        return None


def track_llm_call(
    *,
    model: str,
    latency_ms: int,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    client = _get_langfuse_client()
    if client is None:
        return

    try:
        trace = client.trace(
            name="llm_call",
            metadata={
                "model": model,
                "latency_ms": latency_ms,
                "success": success,
            },
        )
        trace.generation(
            name="chat_completion",
            model=model,
            input={"redacted": True},
            output={"redacted": True},
            metadata={
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "success": success,
                "error": error,
            },
        )
    except Exception:
        logger.exception("langfuse track_llm_call failed")


def track_rag_retrieval(
    *,
    source: str,
    count: int,
    latency_ms: int,
    success: bool = True,
    error: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    client = _get_langfuse_client()
    if client is None:
        return

    try:
        trace = client.trace(
            name="rag_retrieval",
            metadata={"source": source, "latency_ms": latency_ms, "success": success},
        )
        trace.event(
            name="retrieval",
            metadata={
                "source": source,
                "count": count,
                "latency_ms": latency_ms,
                "success": success,
                "error": error,
                "extra": extra or {},
            },
        )
    except Exception:
        logger.exception("langfuse track_rag_retrieval failed")


def time_block() -> float:
    return time.perf_counter()


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
