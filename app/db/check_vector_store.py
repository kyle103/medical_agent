# 测试向量库（Chroma）连通性，可删除。

from __future__ import annotations

from app.common.logger import get_logger
from app.config.settings import settings
from app.db.chroma_store import get_chroma_client

logger = get_logger(__name__)


def main() -> None:
    logger.info("CHROMA_PERSIST_DIRECTORY=%s", settings.CHROMA_PERSIST_DIRECTORY)

    client = get_chroma_client()
    collections = client.list_collections()
    print("VECTOR_STORE_OK")
    print("chroma_collections=", collections)


if __name__ == "__main__":
    main()
