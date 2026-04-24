import logging
import os
from logging.handlers import RotatingFileHandler

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
        filename="logs/app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
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
    """获取项目统一 logger（依赖全局 setup_logging 初始化）"""
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name or "medical_agent")