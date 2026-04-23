import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger()
    if logger.handlers:
        return

    logger.setLevel(log_level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        filename="logs/app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)



def get_logger(name: str | None = None) -> logging.Logger:
    """获取项目统一 logger（不会重复添加 handler）"""
    logger_name = name or "medical_agent"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # 控制台输出
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 文件输出（如果需要）
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = logger_name.replace(".", "_")
    fh = logging.FileHandler(log_dir / f"{safe_name}.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger