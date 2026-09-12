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
    # 结构化输出档位：auto / json_schema / json_object
    # auto = 启动时用探针判定模型是否真的下发 schema，并缓存结果（见 app/core/llm/structured_output.py）
    # 背景：response_format 的支持度**逐模型快照不同**（如 deepseek-v4-flash 支持、deepseek-v4-flash-0731 不支持），
    #       写死任一种都会在换模型时静默失效。客户端 Pydantic 校验始终是不变量。
    STRUCTURED_OUTPUT_MODE: str = Field(default="auto")
    # 结构化决策调用是否关闭「思考」（下发给 extra_body.enable_thinking=False）。
    # 本模型的 max_tokens **同时覆盖推理与答案**：推理吃满预算时 content 返回空串
    # （finish_reason='length'、completion_tokens 顶格），调用方只能看到"失败"。
    # 实测（2026-09-12，deepseek-v4-flash，split_route_deps，各 3 次）：
    #   思考开启 → 0/3 成功、墙钟 20~27s、且模型会把 intent_type 的取值写进 intent，
    #              导致大量路由被业务层判非法后**静默丢弃**；
    #   思考关闭 → 3/3 成功、墙钟 2.9~4.5s、0 条丢弃。
    # 结构化决策是"照 schema 填字段"的任务，推理只带来延迟与字段串味。
    # 只作用于 chat_completion_json（决策/抽取），不影响 chat_completion 的答案生成——
    # 后者仍保留推理能力。
    # 若换到不接受该参数的模型/网关，置 false 即可整体回退。
    LLM_DISABLE_THINKING: bool = True
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
    MILVUS_URI: str = Field(default="{{Milvus地址}}")
    MILVUS_TOKEN: str = Field(default="{{Milvus Token}}")
    MILVUS_PUBLIC_KB_COLLECTION: str = Field(default="kb_general")
    MILVUS_LONG_MEMORY_COLLECTION: str = Field(default="user_long_memory")

    # Public KB
    PUBLIC_KB_COLLECTION: str = Field(default="kb_general")
    PUBLIC_KB_TOP_K: int = Field(default=5)
    PUBLIC_KB_EXPAND_WINDOW: int = Field(default=1)
    PUBLIC_KB_BM25_TOP_K: int = Field(default=20)
    PUBLIC_KB_RRF_K: int = Field(default=60)
    PUBLIC_KB_BM25_CACHE_DIR: str = Field(default="data/bm25_cache")

    # Public KB keyword (BM25-like, 云端 Milvus 仅倒排索引 -> LIKE 模拟)
    PUBLIC_KB_KEYWORD_TOP_K: int = Field(default=8)          # 抽取关键词上限
    PUBLIC_KB_KEYWORD_LLM_ENABLED: bool = Field(default=False)  # 长查询是否启用 LLM 抽取（默认关，省延迟）
    PUBLIC_KB_KEYWORD_MODE: str = Field(default="or")         # and(严) / or(宽召回，靠打分排优)
    PUBLIC_KB_KEYWORD_VECTOR_WEIGHT: float = Field(default=0.7)  # RRF dense 权重
    PUBLIC_KB_KEYWORD_WEIGHT: float = Field(default=0.3)      # RRF keyword 权重

    # Rerank
    RERANK_API_BASE: str = Field(default="")
    RERANK_API_KEY: str = Field(default="")
    RERANK_MODEL_NAME: str = Field(default="qwen3-rerank")

    # Redis (prod)
    REDIS_HOST: str = Field(default="{{Redis地址}}")
    REDIS_PORT: int = Field(default=6379)
    REDIS_PASSWORD: str = Field(default="{{Redis密码}}")
    REDIS_DB: int = 0

    # Security
    SECRET_KEY: str = Field(default="{{JWT加密密钥}}")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ALLOWED_HOSTS: str = Field(default="{{允许的跨域域名列表}}")

    # Compliance（免责声明由前端界面永久展示，不再逐条回复追加）
    FORCE_DISCLAIMER: bool = False
    ENABLE_INPUT_CHECK: bool = False
    ENABLE_OUTPUT_CHECK: bool = False

    # Selective RAG & Fact Check
    ENABLE_SELECTIVE_RAG: bool = True
    ENABLE_FACT_CHECK: bool = True

    # Misc
    APP_ENV: str = Field(default="local")
    LANGFUSE_PUBLIC_KEY: str = Field(default="")
    LANGFUSE_SECRET_KEY: str = Field(default="")
    LANGFUSE_HOST: str = Field(default="")
    LANGSMITH_TRACING: bool = False


settings = Settings()
