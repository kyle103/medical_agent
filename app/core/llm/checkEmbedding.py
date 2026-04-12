# 测试 Embedding 连通性，可删除。

import asyncio

from app.common.logger import get_logger
from app.config.settings import settings
from app.core.llm.embedding_service import EmbeddingService

logger = get_logger(__name__)


async def main() -> None:
    logger.info("EMBEDDING_TYPE=%s", settings.EMBEDDING_TYPE)
    logger.info("EMBEDDING_API_BASE=%s", settings.EMBEDDING_API_BASE)
    logger.info("EMBEDDING_MODEL_NAME=%s", settings.EMBEDDING_MODEL_NAME)

    texts = [
        "测试文本：对乙酰氨基酚",
        "测试文本：血糖 7.0 mmol/L",
    ]

    svc = EmbeddingService()
    vecs = await svc.embed_documents(texts)

    if not vecs:
        raise RuntimeError("Embedding 返回空结果")

    # 仅打印维度与部分数值，避免输出过长
    dim = len(vecs[0])
    head = vecs[0][:5]

    print("EMBEDDING_OK")
    print("count=", len(vecs))
    print("dim=", dim)
    print("vec0_head=", head)


if __name__ == "__main__":
    asyncio.run(main())
