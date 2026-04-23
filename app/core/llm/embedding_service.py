from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.common.exceptions import ServiceUnavailableException
from app.common.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self):
        self.embedding_type = settings.EMBEDDING_TYPE
        self.model = None
        self.client = None

        if self.embedding_type == "local":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)
            except Exception as e:
                logger.error("加载本地Embedding模型失败: %s", str(e))
                raise ServiceUnavailableException("Embedding模型不可用") from e
        else:
            self.client = AsyncOpenAI(
                api_key=settings.EMBEDDING_API_KEY, base_url=settings.EMBEDDING_API_BASE
            )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if self.embedding_type == "local":
            assert self.model is not None
            vecs = self.model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vecs]

        assert self.client is not None
        try:
            max_batch = int(getattr(settings, "EMBEDDING_API_MAX_BATCH", 10))
            vectors: list[list[float]] = []
            for i in range(0, len(texts), max_batch):
                batch = texts[i : i + max_batch]
                resp = await self.client.embeddings.create(
                    model=settings.EMBEDDING_MODEL_NAME,
                    input=batch,
                )
                vectors.extend([d.embedding for d in resp.data])
            return vectors
        except Exception as e:
            logger.error("Embedding API调用失败: %s", str(e))
            raise ServiceUnavailableException("Embedding服务暂不可用") from e
