from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import os
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from app.db.milvus_store import parse_metadata, query_like


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test BM25 retrieval speed")
    parser.add_argument("--collection", type=str, default="kb_general")
    parser.add_argument("--query", type=str, default="胃胀痛")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    logger.info(f"Starting BM25 retrieval test with collection={args.collection}, query={args.query}, top_k={args.top_k}")
    
    try:
        # 执行 LIKE 文本检索
        start_time = time.time()
        rows = query_like(
            collection_name=args.collection,
            field="document",
            text=args.query,
            limit=args.top_k,
            output_fields=["document", "metadata"],
        )
        retrieval_time = time.time() - start_time
        logger.info(f"LIKE retrieval completed in {retrieval_time:.2f} seconds")

        results = []
        for row in rows:
            doc_id = row.get("id")
            text = row.get("document", "")
            meta = parse_metadata(row.get("metadata"))
            results.append((doc_id, text, meta))
        
        logger.info(f"Retrieve completed, got {len(results)} results")
        print(f"collection={args.collection} query={args.query} hits={len(results)}")
        print(f"LIKE retrieval time: {retrieval_time:.2f} seconds")
        
        for i, (doc_id, text, meta) in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"ID: {doc_id}")
            print(f"Meta: {meta}")
            print(f"Text: {text[:120]}...")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())