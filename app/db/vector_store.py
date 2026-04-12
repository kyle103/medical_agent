from __future__ import annotations

import os
from typing import Any

from app.common.exceptions import ServiceUnavailableException
from app.config.settings import settings


def get_vector_store_persist_dir() -> str:
    persist_dir = settings.CHROMA_PERSIST_DIRECTORY
    if not persist_dir or persist_dir.startswith("{{"):
        persist_dir = "./data/chroma"
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


def init_chroma_collection(*, collection_name: str):
    """初始化本地 Chroma（文件持久化）集合。

    说明：
    - 本项目的向量库仅用于“用户档案语义检索/补充信息”，不用于生成核心医疗结论。
    - 生产环境可替换为 chroma_server/qdrant，仅需改配置。
    """

    if settings.VECTOR_STORE_TYPE not in ("chroma_file", "{{chroma_file/chroma_server/qdrant}}"):
        raise ServiceUnavailableException("当前 VECTOR_STORE_TYPE 未启用 chroma_file")

    # 关闭 Chroma 的遥测（避免本地开发环境报 posthog capture 参数不兼容的噪音日志）
    # 需在 chromadb import/初始化前设置。
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    try:
        import chromadb

        client = chromadb.PersistentClient(path=get_vector_store_persist_dir())
        return client.get_or_create_collection(name=collection_name)
    except Exception as e:
        raise ServiceUnavailableException(f"向量库初始化失败: {str(e)}") from e
