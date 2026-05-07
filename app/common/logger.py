import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

from app.common.log_context import get_request_id, get_trace_id


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        record.request_id = get_request_id()
        return True


def _resolve_log_level() -> int:
    env = os.getenv("APP_ENV", "local").lower()
    default_level = "INFO" if env == "prod" else "DEBUG"
    level_str = os.getenv("LOG_LEVEL", default_level).upper()
    return getattr(logging, level_str, logging.INFO)


def _current_log_filename() -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    return f"logs/medical_agent_{date_str}.log"


def setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger()
    if logger.handlers:
        return

    logger.setLevel(_resolve_log_level())

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | trace_id=%(trace_id)s | request_id=%(request_id)s | %(name)s | %(message)s"
    )

    context_filter = RequestContextFilter()

    file_handler = RotatingFileHandler(
        filename=_current_log_filename(),
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(context_filter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    stream_handler.addFilter(context_filter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name or "medical_agent")


def log_llm_call(
    *,
    model: str,
    prompt_len: int = 0,
    system_prompt_len: int = 0,
    response_len: int = 0,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: int = 0,
    success: bool = True,
    error: Optional[str] = None,
    caller: str = "",
) -> None:
    _logger = get_logger("medical_agent.llm")
    token_info = ""
    if input_tokens is not None:
        token_info += f" in_tok={input_tokens}"
    if output_tokens is not None:
        token_info += f" out_tok={output_tokens}"
    if total_tokens is not None:
        token_info += f" total_tok={total_tokens}"
    status = "OK" if success else f"FAIL({error})"
    _logger.info(
        "[LLM] %s | model=%s prompt_len=%d sys_prompt_len=%d resp_len=%d%s | latency=%dms | %s",
        caller or "chat_completion",
        model,
        prompt_len,
        system_prompt_len,
        response_len,
        token_info,
        latency_ms,
        status,
    )


def log_rag_retrieval(
    *,
    source: str,
    query: str = "",
    method: str = "",
    collection: str = "",
    top_k: int = 0,
    result_count: int = 0,
    latency_ms: int = 0,
    success: bool = True,
    error: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    _logger = get_logger("medical_agent.rag")
    status = "OK" if success else f"FAIL({error})"
    query_preview = (query[:50] + "...") if len(query) > 50 else query
    extra_str = ""
    if extra:
        extra_str = " " + " ".join(f"{k}={v}" for k, v in extra.items())
    _logger.info(
        "[RAG] %s | query=\"%s\" method=%s collection=%s top_k=%d results=%d | latency=%dms | %s%s",
        source,
        query_preview,
        method,
        collection,
        top_k,
        result_count,
        latency_ms,
        status,
        extra_str,
    )


def log_node_execution(
    *,
    node_name: str,
    latency_ms: int = 0,
    success: bool = True,
    error: Optional[str] = None,
    **detail: Any,
) -> None:
    _logger = get_logger("medical_agent.workflow")
    status = "OK" if success else f"FAIL({error})"
    detail_str = ""
    if detail:
        detail_str = " " + " ".join(f"{k}={v}" for k, v in detail.items())
    _logger.info(
        "[NODE] %s | latency=%dms | %s%s",
        node_name,
        latency_ms,
        status,
        detail_str,
    )


def log_step_execution(
    *,
    step_id: str,
    target: str = "",
    query: str = "",
    latency_ms: int = 0,
    has_error: bool = False,
    depends_on: Optional[list] = None,
    **detail: Any,
) -> None:
    _logger = get_logger("medical_agent.workflow")
    query_preview = (query[:40] + "...") if len(query) > 40 else query
    dep_str = ""
    if depends_on:
        dep_str = f" depends_on={depends_on}"
    detail_str = ""
    if detail:
        detail_str = " " + " ".join(f"{k}={v}" for k, v in detail.items())
    _logger.info(
        "[STEP] %s | target=%s query=\"%s\" | latency=%dms | has_error=%s%s%s",
        step_id,
        target,
        query_preview,
        latency_ms,
        has_error,
        dep_str,
        detail_str,
    )


class NodeTimer:
    def __init__(self, node_name: str, **detail: Any):
        self.node_name = node_name
        self.detail = detail
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = int((time.perf_counter() - self.start) * 1000)
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        log_node_execution(
            node_name=self.node_name,
            latency_ms=latency_ms,
            success=success,
            error=error,
            **self.detail,
        )
        return False
