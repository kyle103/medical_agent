"""PublicKnowledgeService.retrieve() 混合检索路径的 mock 集成测试。

本地 Chroma 可用性已用真实子集验证；本测试 mock 掉存储层与 embedding，
验证 retrieve() 的编排逻辑（关键词→打分→加权 RRF 融合）正确执行。
"""

import asyncio
from unittest import mock

from app.config.settings import settings
from app.core.rag.public_kb_service import PublicKnowledgeService


def _hit(doc_id: str, text: str, source: str) -> dict:
    return {
        "id": doc_id,
        "distance": 0.5,
        "entity": {
            "document": text,
            "metadata": (
                f'{{"source_name": "{source}", "source_type": "t", '
                f'"record_line": 0, "chunk_index": 0}}'
            ),
        },
    }


def _kw_record(doc_id: str, text: str, source: str) -> dict:
    return {
        "id": doc_id,
        "document": text,
        "metadata": (
            f'{{"source_name": "{source}", "source_type": "t", '
            f'"record_line": 0, "chunk_index": 0}}'
        ),
    }


def test_hybrid_retrieve_keyword_path_or_mode():
    async def _run():
        svc = PublicKnowledgeService(collection_name="kb_general")

        dense_hits = [
            _hit("d1", "布洛芬与阿司匹林相互作用需注意", "s1"),
            _hit("d2", "血压血糖日常管理", "s2"),
        ]
        kw_records = [
            _kw_record("d1", "布洛芬与阿司匹林相互作用需注意", "s1"),
            _kw_record("k3", "布洛芬退烧作用说明", "s3"),
        ]

        with mock.patch.object(svc.embedder, "embed_documents", return_value=[[0.1] * 8]), \
             mock.patch("app.core.rag.public_kb_service.vector_search", return_value=dense_hits), \
             mock.patch("app.core.rag.public_kb_service.keyword_search", return_value=kw_records) as ks:
            out = await svc.retrieve(query="布洛芬和阿司匹林一起吃有相互作用吗", top_k=5, expand_window=0)

        # 默认 or 模式：keyword_search 只调用一次，mode=or
        ks.assert_called_once()
        assert ks.call_args.kwargs.get("mode") == "or"

        ids = {it["id"] for it in out}
        assert "d1" in ids, "dense 命中应保留"
        assert "d2" in ids, "仅 dense 命中应保留"
        assert "k3" in ids, "仅关键词命中应被召回"

        for it in out:
            assert "rrf_score" in it

        kw_item = next(it for it in out if it["id"] == "k3")
        assert kw_item["kw_score"] > 0, "关键词命中应有打分"
        assert kw_item["bm25_score"] is not None

        return out

    asyncio.run(_run())


def test_hybrid_retrieve_keyword_path_and_mode():
    async def _run():
        svc = PublicKnowledgeService(collection_name="kb_general")
        dense_hits = [_hit("d1", "布洛芬与阿司匹林相互作用需注意", "s1")]
        kw_records = [
            _kw_record("d1", "布洛芬与阿司匹林相互作用需注意", "s1"),
            _kw_record("k3", "布洛芬退烧作用说明", "s3"),
        ]

        with mock.patch.object(svc.embedder, "embed_documents", return_value=[[0.1] * 8]), \
             mock.patch("app.core.rag.public_kb_service.vector_search", return_value=dense_hits), \
             mock.patch("app.core.rag.public_kb_service.keyword_search", return_value=kw_records) as ks, \
             mock.patch.object(settings, "PUBLIC_KB_KEYWORD_MODE", "and"):
            out = await svc.retrieve(query="布洛芬和阿司匹林一起吃有相互作用吗", top_k=5, expand_window=0)

        # and 模式：mode 透传给 keyword_search
        assert ks.call_args.kwargs.get("mode") == "and"
        ids = {it["id"] for it in out}
        assert "d1" in ids
        assert "k3" in ids
        return out

    asyncio.run(_run())
