from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import os
import time
from typing import Any, Dict, List, Tuple

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from app.core.llm.embedding_service import EmbeddingService
from app.db.milvus_store import build_metadata_like, parse_metadata, query_by_filter, vector_search


def apply_window_expand(collection_name: str, items: List[Dict[str, Any]], expand_window: int) -> None:
    """应用窗口扩展，获取目标句子的前后窗口句子"""
    if expand_window <= 0:
        return

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
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
            collection_name=collection_name,
            filter_expr=expr,
            limit=512,
            output_fields=["document", "metadata"],
        )

        index_to_text: Dict[int, str] = {}
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


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test vector retrieval speed")
    parser.add_argument("--collection", type=str, default="kb_general")
    parser.add_argument("--query", type=str, default="肚子疼")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--expand-window", type=int, default=2, help="Window size for context expansion")
    args = parser.parse_args()

    logger.info(f"Starting vector retrieval test with collection={args.collection}, query={args.query}, top_k={args.top_k}, expand_window={args.expand_window}")
    
    try:
        # 初始化嵌入服务
        embedder = EmbeddingService()
        logger.info("EmbeddingService initialized successfully")
        
        # Milvus 使用即连即查，初始化时间仅用于日志
        start_time = time.time()
        init_time = time.time() - start_time
        logger.info(f"Milvus client ready in {init_time:.2f} seconds")
        
        # 生成查询向量
        start_time = time.time()
        vectors = await embedder.embed_documents([args.query])
        embed_time = time.time() - start_time
        logger.info(f"Query embedded in {embed_time:.2f} seconds")
        
        # 执行向量检索
        start_time = time.time()
        res = vector_search(
            collection_name=args.collection,
            query_vectors=vectors,
            limit=args.top_k,
            output_fields=["document", "metadata"],
        )
        query_time = time.time() - start_time
        logger.info(f"Vector query completed in {query_time:.2f} seconds")
        
        # 处理结果
        items = []
        logger.info(f"Processing {len(res)} results")
        try:
            for i, hit in enumerate(res):
                entity = hit.get("entity") or {}
                meta = parse_metadata(entity.get("metadata"))
                text = entity.get("document", "")
                doc_id = hit.get("id")
                logger.info(f"Result {i+1}: ID={doc_id}, Text={text[:50]}..., Meta={meta}")
                item = {
                    "id": doc_id,
                    "text": text,
                    "source_name": meta.get("source_name", ""),
                    "record_line": meta.get("record_line", 0),
                    "chunk_index": meta.get("chunk_index", 0)
                }
                items.append(item)
            logger.info(f"Created {len(items)} items")
        except Exception as e:
            logger.error(f"Error processing results: {e}", exc_info=True)
        
        # 应用窗口扩展
        start_time = time.time()
        try:
            apply_window_expand(args.collection, items, args.expand_window)
        except Exception as e:
            logger.error(f"Error during window expansion: {e}", exc_info=True)
        expand_time = time.time() - start_time
        logger.info(f"Window expansion completed in {expand_time:.2f} seconds")
        
        logger.info(f"Retrieve completed, got {len(res)} results")
        print(f"collection={args.collection} query={args.query} hits={len(res)}")
        print(f"Initialization time: {init_time:.2f} seconds")
        print(f"Embedding time: {embed_time:.2f} seconds")
        print(f"Query time: {query_time:.2f} seconds")
        print(f"Window expansion time: {expand_time:.2f} seconds")
        
        # 确保结果被打印
        print(f"\nDetailed results:")
        print(f"Number of items: {len(items)}")
        try:
            for i, item in enumerate(items):
                print(f"\nResult {i+1}:")
                print(f"ID: {item['id']}")
                print(f"Source: {item.get('source_name', 'N/A')}")
                print(f"Record Line: {item.get('record_line', 'N/A')}")
                print(f"Chunk Index: {item.get('chunk_index', 'N/A')}")
                print(f"Expanded Text: {item['text']}")
                print(f"Text length: {len(item['text'])} characters")
        except Exception as e:
            logger.error(f"Error printing results: {e}", exc_info=True)
        
        # 强制刷新输出
        import sys
        sys.stdout.flush()
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())