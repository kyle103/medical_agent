# 测试向量库（Chroma file persist）连通性，可删除。

import uuid

from app.common.logger import get_logger
from app.config.settings import settings
from app.db.vector_store import init_chroma_collection

logger = get_logger(__name__)


def main() -> None:
    logger.info("VECTOR_STORE_TYPE=%s", settings.VECTOR_STORE_TYPE)
    logger.info("CHROMA_PERSIST_DIRECTORY=%s", settings.CHROMA_PERSIST_DIRECTORY)

    col = init_chroma_collection(collection_name="health_archive_test")

    _id = str(uuid.uuid4())
    col.add(
        ids=[_id],
        documents=["测试文档：用药记录 对乙酰氨基酚"],
        metadatas=[{"user_id": "u_test"}],
    )

    res = col.query(
        query_texts=["对乙酰氨基酚"],
        n_results=1,
        where={"user_id": "u_test"},
    )

    got = (res.get("ids") or [[]])[0]

    if not got or got[0] != _id:
        raise RuntimeError("向量库查询未命中，配置或持久化可能异常")

    print("VECTOR_STORE_OK")
    print("collection=health_archive_test")
    print("hit_id=", got[0])


if __name__ == "__main__":
    main()
