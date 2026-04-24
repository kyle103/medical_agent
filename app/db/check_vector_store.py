# 测试向量库（Milvus）连通性，可删除。

from __future__ import annotations

from app.common.logger import get_logger
from app.config.settings import settings
from app.db.milvus_store import get_milvus_client

logger = get_logger(__name__)


def main() -> None:
    logger.info("MILVUS_URI=%s", settings.MILVUS_URI)

    client = get_milvus_client()
    collections = client.list_collections()
    print("VECTOR_STORE_OK")
    print("milvus_collections=", collections)


if __name__ == "__main__":
    main()
