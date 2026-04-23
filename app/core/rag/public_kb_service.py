from __future__ import annotations

from typing import Any
import logging
import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from app.core.llm.embedding_service import EmbeddingService
from app.config.settings import settings
from app.db.vector_store import init_chroma_collection


DEFAULT_COLLECTION = "kb_general"

from app.common.logger import get_logger

logger = get_logger(__name__)


class _Bm25Cache:
    def __init__(
        self,
        *,
        ids: list[str],
        docs: list[str],
        metas: list[dict[str, Any]],
        tokens: list[list[str]] | None = None,
    ):
        self.ids = ids
        self.docs = docs
        self.metas = metas
        self.tokens = tokens or [list(jieba.cut(d)) for d in docs]
        self.bm25 = BM25Okapi(self.tokens)


class PublicKnowledgeService:
    """公共医学知识库检索服务（独立于长期记忆）。"""

    _bm25_cache: dict[str, _Bm25Cache] = {}

    def __init__(self, *, collection_name: str | None = None):
        self.collection_name = collection_name or getattr(settings, "PUBLIC_KB_COLLECTION", DEFAULT_COLLECTION)
        self.embedder = EmbeddingService()

    def _collection(self):
        return init_chroma_collection(collection_name=self.collection_name)

    def _cache_dir(self) -> Path:
        base = getattr(settings, "PUBLIC_KB_BM25_CACHE_DIR", "data/bm25_cache")
        path = Path(base)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cache_path(self) -> Path:
        safe = self.collection_name.replace("/", "_").replace("\\", "_")
        return self._cache_dir() / f"bm25_{safe}.pkl"

    def _load_cache_from_disk(self) -> _Bm25Cache | None:
        path = self._cache_path()
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
            cache = _Bm25Cache(
                ids=[str(i) for i in payload.get("ids") or []],
                docs=[str(d) for d in payload.get("docs") or []],
                metas=payload.get("metas") or [],
                tokens=payload.get("tokens"),
            )
            logger.info(
                "public_kb bm25 cache loaded collection=%s docs=%s path=%s",
                self.collection_name,
                len(cache.docs),
                str(path),
            )
            return cache
        except Exception as e:
            logger.warning("public_kb bm25 cache load failed: %s", str(e))
            return None

    def _save_cache_to_disk(self, cache: _Bm25Cache) -> None:
        path = self._cache_path()
        payload = {
            "ids": cache.ids,
            "docs": cache.docs,
            "metas": cache.metas,
            "tokens": cache.tokens,
        }
        try:
            with path.open("wb") as f:
                pickle.dump(payload, f)
            logger.info(
                "public_kb bm25 cache saved collection=%s docs=%s path=%s",
                self.collection_name,
                len(cache.docs),
                str(path),
            )
        except Exception as e:
            logger.warning("public_kb bm25 cache save failed: %s", str(e))

    def _get_bm25_cache(self) -> _Bm25Cache:
        cache = self._bm25_cache.get(self.collection_name)
        if cache:
            return cache

        disk_cache = self._load_cache_from_disk()
        if disk_cache:
            self._bm25_cache[self.collection_name] = disk_cache
            return disk_cache

        col = self._collection()
        res = col.get()
        docs = res.get("documents") or []
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []

        if not isinstance(docs, list) or not isinstance(ids, list):
            docs = []
            ids = []
        if not isinstance(metas, list):
            metas = []

        cache = _Bm25Cache(ids=[str(i) for i in ids], docs=[str(d) for d in docs], metas=metas)
        self._bm25_cache[self.collection_name] = cache
        self._save_cache_to_disk(cache)
        logger.info("public_kb bm25 cache built collection=%s docs=%s", self.collection_name, len(cache.docs))
        return cache

    def _normalize_text(self, text: str) -> str:
        return " ".join((text or "").split())

    def _apply_window_expand(self, *, col, items: list[dict[str, Any]], expand_window: int) -> None:
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

            res = col.get(where={"source_name": source_name, "record_line": record_line})
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []

            if not isinstance(docs, list) or not isinstance(metas, list):
                continue

            index_to_text: dict[int, str] = {}
            max_index = 0
            for i in range(min(len(docs), len(metas))):
                md = metas[i] if isinstance(metas[i], dict) else {}
                c_index = int(md.get("chunk_index") or 0)
                index_to_text[c_index] = str(docs[i] or "")
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
        name = collection_name or getattr(settings, "PUBLIC_KB_COLLECTION", DEFAULT_COLLECTION)
        cls._bm25_cache.pop(name, None)
        cache = cls(collection_name=name)._get_bm25_cache()
        return len(cache.docs)

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

        col = self._collection()
        vectors = await self.embedder.embed_documents([q])
        res = col.query(query_embeddings=vectors, n_results=max(1, int(top_k)))

        ids = res.get("ids")
        docs = res.get("documents")
        metas = res.get("metadatas")
        dists = res.get("distances")

        first_ids = ids[0] if isinstance(ids, list) and ids else []
        first_docs = docs[0] if isinstance(docs, list) and docs else []
        first_metas = metas[0] if isinstance(metas, list) and metas else []
        first_dists = dists[0] if isinstance(dists, list) and dists else []

        if not isinstance(first_ids, list) or not isinstance(first_docs, list):
            return []

        items: list[dict[str, Any]] = []
        dense_rank: dict[str, int] = {}

        for i in range(min(len(first_ids), len(first_docs))):
            md = first_metas[i] if i < len(first_metas) and isinstance(first_metas[i], dict) else {}
            dist = first_dists[i] if i < len(first_dists) else None
            doc_id = str(first_ids[i])
            dense_rank[doc_id] = i + 1

            items.append(
                {
                    "id": doc_id,
                    "text": str(first_docs[i] or ""),
                    "score": float(dist) if dist is not None else None,
                    "source_name": str(md.get("source_name") or ""),
                    "source_type": str(md.get("source_type") or ""),
                    "record_line": int(md.get("record_line") or 0),
                    "chunk_index": int(md.get("chunk_index") or 0),
                }
            )

        if bm25_top_k <= 0:
            self._apply_window_expand(col=col, items=items, expand_window=expand_window)
            deduped: list[dict[str, Any]] = []
            seen: set[str] = set()
            for it in items:
                key = self._normalize_text(str(it.get("text") or ""))
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(it)
            logger.info("public_kb retrieve done dense_only count=%s", len(deduped))
            return deduped[: max(1, int(top_k))]

        cache = self._get_bm25_cache()
        if not cache.docs:
            logger.info("public_kb retrieve done empty_bm25 count=%s", len(items))
            return items

        query_tokens = list(jieba.cut(q))
        scores = cache.bm25.get_scores(query_tokens)
        bm25_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:bm25_top_k]

        bm25_rank: dict[str, int] = {}
        bm25_items: list[dict[str, Any]] = []
        for rank, idx in enumerate(bm25_idx, 1):
            doc_id = cache.ids[idx]
            bm25_rank[doc_id] = rank
            md = cache.metas[idx] if idx < len(cache.metas) and isinstance(cache.metas[idx], dict) else {}
            bm25_items.append(
                {
                    "id": doc_id,
                    "text": cache.docs[idx],
                    "bm25_score": float(scores[idx]) if idx < len(scores) else 0.0,
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
        self._apply_window_expand(col=col, items=out, expand_window=expand_window)
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
        return final
