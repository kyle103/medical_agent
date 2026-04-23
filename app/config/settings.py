from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_env() -> None:
    env = os.getenv("APP_ENV", "local")
    root = Path(__file__).resolve().parents[2]

    # 优先加载显式环境文件；不存在则忽略
    if env == "prod":
        load_dotenv(root / ".env.prod", override=False)
    else:
        load_dotenv(root / ".env.local", override=False)


_load_env()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # LLM
    LLM_API_BASE: str = Field(default="{{LLM_API地址}}")
    LLM_API_KEY: str = Field(default="{{LLM_API密钥}}")
    LLM_MODEL_NAME: str = Field(default="{{LLM模型名称}}")
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    # 对低意图/不确定意图场景的额外 LLM 增强（意图识别专用）
    INTENT_LLM_ENABLED: bool = True
    INTENT_LLM_TIMEOUT_S: float = 10.2
    INTENT_LLM_MAX_TOKENS: int = 512

    # Embedding
    EMBEDDING_TYPE: str = Field(default="{{local/api}}")
    EMBEDDING_MODEL_PATH: str = Field(default="{{本地Embedding模型路径}}")
    EMBEDDING_API_BASE: str = Field(default="{{Embedding API地址}}")
    EMBEDDING_API_KEY: str = Field(default="{{Embedding API密钥}}")
    EMBEDDING_MODEL_NAME: str = Field(default="{{Embedding模型名称}}")
    EMBEDDING_API_MAX_BATCH: int = Field(default=10)

    # DB
    DB_TYPE: str = Field(default="{{sqlite/mysql}}")
    SQLITE_DB_PATH: str = Field(default="{{本地SQLite文件路径}}")
    MYSQL_HOST: str = Field(default="{{MySQL地址}}")
    MYSQL_PORT: int = Field(default=3306)
    MYSQL_USER: str = Field(default="{{MySQL用户名}}")
    MYSQL_PASSWORD: str = Field(default="{{MySQL密码}}")
    MYSQL_DATABASE: str = Field(default="{{MySQL数据库名}}")

    # Vector store
    VECTOR_STORE_TYPE: str = Field(default="{{chroma_file/chroma_server/qdrant}}")
    CHROMA_PERSIST_DIRECTORY: str = Field(default="{{本地Chroma存储路径}}")
    CHROMA_SERVER_HOST: str = Field(default="{{Chroma服务地址}}")
    CHROMA_SERVER_PORT: int = Field(default=8000)
    QDRANT_HOST: str = Field(default="{{Qdrant地址}}")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_API_KEY: str = Field(default="{{Qdrant API密钥}}")

    # Public KB
    PUBLIC_KB_COLLECTION: str = Field(default="kb_general")
    PUBLIC_KB_TOP_K: int = Field(default=5)
    PUBLIC_KB_EXPAND_WINDOW: int = Field(default=1)
    PUBLIC_KB_BM25_TOP_K: int = Field(default=20)
    PUBLIC_KB_RRF_K: int = Field(default=60)
    PUBLIC_KB_BM25_CACHE_DIR: str = Field(default="data/bm25_cache")

    # Redis (prod)
    REDIS_HOST: str = Field(default="{{Redis地址}}")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: str = Field(default="{{Redis密码}}")
    REDIS_DB: int = 0

    # Security
    SECRET_KEY: str = Field(default="{{JWT加密密钥}}")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ALLOWED_HOSTS: str = Field(default="{{允许的跨域域名列表}}")

    # Compliance
    FORCE_DISCLAIMER: bool = False
    ENABLE_INPUT_CHECK: bool = False
    ENABLE_OUTPUT_CHECK: bool = False

    # Misc
    APP_ENV: str = Field(default="local")


settings = Settings()
