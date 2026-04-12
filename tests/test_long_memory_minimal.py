from __future__ import annotations

import uuid

import pytest

from app.core.memory.long_memory_service import LongMemoryService


@pytest.mark.asyncio
async def test_long_memory_add_and_recall_by_user_id():
    """极简长期记忆：写入 -> 过滤 user_id -> 召回。

    说明：
    - 该测试依赖 VECTOR_STORE_TYPE=chroma_file。
    - 默认使用本地持久化目录（settings.CHROMA_PERSIST_DIRECTORY 为空/占位会回退到 ./data/chroma）。
    """

    svc = LongMemoryService()
    if not svc.is_enabled():
        pytest.skip("vector store not enabled")

    user_id = "u_test_" + uuid.uuid4().hex
    session_id = "s_test"

    items = svc.extract_candidates(user_input="我对青霉素过敏")
    assert items

    svc.add_items(user_id=user_id, session_id=session_id, items=items)

    hits = svc.recall(user_id=user_id, query="我有什么过敏史", top_k=3)
    assert hits
    assert any("过敏" in (h.text or "") for h in hits)

    # 隔离性：别的 user_id 不应命中
    hits2 = svc.recall(user_id=user_id + "_other", query="我有什么过敏史", top_k=3)
    assert hits2 == []
