from __future__ import annotations

from typing import Any

from app.common.langfuse_helper import elapsed_ms, time_block, track_rag_retrieval
from app.common.logger import get_logger
from app.config.settings import settings
from app.core.llm.embedding_service import EmbeddingService
from app.db.milvus_store import (
    build_metadata_like,
    parse_metadata,
    query_by_filter,
    query_like,
    vector_search,
)


DEFAULT_COLLECTION = "kb_general"

logger = get_logger(__name__)


class PublicKnowledgeService:
    """公共医学知识库检索服务（独立于长期记忆）。"""

    def __init__(self, *, collection_name: str | None = None):
        self.collection_name = collection_name or getattr(
            settings, "MILVUS_PUBLIC_KB_COLLECTION", DEFAULT_COLLECTION
        )
        self.embedder = EmbeddingService()

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").split())

    def _apply_window_expand(self, *, items: list[dict[str, Any]], expand_window: int) -> None:
        if expand_window <= 0:
            return

        grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for it in items:
            source_name = str(it.get("source_name") or "")
            record_line = int(it.get("record_line") or 0)
            grouped.setdefault((source_name, record_line), []).append(it)

        for (source_name, record_line), hits in grouped.items():
            if not source_name or record_line <= 0:
                continue

            filters = [
                build_metadata_like("source_name", source_name),
                build_metadata_like("record_line", int(record_line)),
            ]
            expr = " AND ".join(filters)
            rows = query_by_filter(
                collection_name=self.collection_name,
                filter_expr=expr,
                limit=512,
                output_fields=["document", "metadata"],
            )

            index_to_text: dict[int, str] = {}
            max_index = 0
            for row in rows:
                md = parse_metadata(row.get("metadata"))
                c_index = int(md.get("chunk_index") or 0)
                index_to_text[c_index] = str(row.get("document") or "")
                max_index = max(max_index, c_index)

            for it in hits:
                c_index = int(it.get("chunk_index") or 0)
                start = max(0, c_index - expand_window)
                end = min(max_index, c_index + expand_window)
                parts = [index_to_text.get(i, "") for i in range(start, end + 1)]
                context = "\n".join([p for p in parts if p])
                if context:
                    it["context"] = context
                    it["text"] = context

    @classmethod
    def refresh_cache(cls, *, collection_name: str | None = None) -> int:
        name = collection_name or getattr(settings, "MILVUS_PUBLIC_KB_COLLECTION", DEFAULT_COLLECTION)
        logger.info("public_kb cache refresh noop for Milvus collection=%s", name)
        return 0

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        expand_window: int | None = None,
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        start = time_block()
        try:
            if top_k is None:
                top_k = int(getattr(settings, "PUBLIC_KB_TOP_K", 5))
            if expand_window is None:
                expand_window = int(getattr(settings, "PUBLIC_KB_EXPAND_WINDOW", 1))
            bm25_top_k = int(getattr(settings, "PUBLIC_KB_BM25_TOP_K", 0))
            rrf_k = int(getattr(settings, "PUBLIC_KB_RRF_K", 60))

            logger.info(
                "public_kb retrieve start collection=%s top_k=%s bm25_top_k=%s rrf_k=%s",
                self.collection_name,
                top_k,
                bm25_top_k,
                rrf_k,
            )

            vectors = await self.embedder.embed_documents([q])
            dense_hits = vector_search(
                collection_name=self.collection_name,
                query_vectors=vectors,
                limit=max(1, int(top_k)),
                output_fields=["document", "metadata"],
            )

            items: list[dict[str, Any]] = []
            dense_rank: dict[str, int] = {}

            for i, hit in enumerate(dense_hits):
                entity = hit.get("entity") or {}
                md = parse_metadata(entity.get("metadata"))
                doc_id = str(hit.get("id") or "")
                dense_rank[doc_id] = i + 1
                items.append(
                    {
                        "id": doc_id,
                        "text": str(entity.get("document") or ""),
                        "score": float(hit.get("distance")) if hit.get("distance") is not None else None,
                        "source_name": str(md.get("source_name") or ""),
                        "source_type": str(md.get("source_type") or ""),
                        "record_line": int(md.get("record_line") or 0),
                        "chunk_index": int(md.get("chunk_index") or 0),
                    }
                )

            if bm25_top_k <= 0:
                self._apply_window_expand(items=items, expand_window=expand_window)
                deduped: list[dict[str, Any]] = []
                seen: set[str] = set()
                for it in items:
                    key = self._normalize_text(str(it.get("text") or ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(it)
                logger.info("public_kb retrieve done dense_only count=%s", len(deduped))
                final = deduped[: max(1, int(top_k))]
                track_rag_retrieval(
                    source="public_kb",
                    count=len(final),
                    latency_ms=elapsed_ms(start),
                    success=True,
                    extra={"mode": "dense_only"},
                )
                return final

            like_rows = query_like(
                collection_name=self.collection_name,
                field="document",
                text=q,
                limit=bm25_top_k,
                output_fields=["document", "metadata"],
            )

            bm25_rank: dict[str, int] = {}
            bm25_items: list[dict[str, Any]] = []
            for rank, row in enumerate(like_rows, 1):
                doc_id = str(row.get("id") or "")
                bm25_rank[doc_id] = rank
                md = parse_metadata(row.get("metadata"))
                bm25_items.append(
                    {
                        "id": doc_id,
                        "text": str(row.get("document") or ""),
                        "bm25_score": None,
                        "source_name": str(md.get("source_name") or ""),
                        "source_type": str(md.get("source_type") or ""),
                        "record_line": int(md.get("record_line") or 0),
                        "chunk_index": int(md.get("chunk_index") or 0),
                    }
                )

            merged: dict[str, dict[str, Any]] = {it["id"]: it for it in items}
            for it in bm25_items:
                if it["id"] not in merged:
                    merged[it["id"]] = it

            def _rrf(rank: int) -> float:
                return 1.0 / (rrf_k + rank)

            out: list[dict[str, Any]] = []
            for doc_id, it in merged.items():
                d_rank = dense_rank.get(doc_id)
                b_rank = bm25_rank.get(doc_id)
                score = 0.0
                if d_rank:
                    score += _rrf(d_rank)
                if b_rank:
                    score += _rrf(b_rank)
                it["rrf_score"] = score
                it["dense_rank"] = d_rank
                it["bm25_rank"] = b_rank
                out.append(it)

            out.sort(key=lambda x: float(x.get("rrf_score") or 0.0), reverse=True)
            self._apply_window_expand(items=out, expand_window=expand_window)
            deduped: list[dict[str, Any]] = []
            seen: set[str] = set()
            for it in out:
                key = self._normalize_text(str(it.get("text") or ""))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(it)
            final = deduped[: max(1, int(top_k))]
            logger.info(
                "public_kb retrieve done hybrid count=%s dense_hits=%s bm25_hits=%s",
                len(final),
                len(items),
                len(bm25_items),
            )
            track_rag_retrieval(
                source="public_kb",
                count=len(final),
                latency_ms=elapsed_ms(start),
                success=True,
                extra={"mode": "hybrid", "dense_hits": len(items), "bm25_hits": len(bm25_items)},
            )
            return final
        except Exception as exc:
            track_rag_retrieval(
                source="public_kb",
                count=0,
                latency_ms=elapsed_ms(start),
                success=False,
                error=str(exc),
            )
            raise
