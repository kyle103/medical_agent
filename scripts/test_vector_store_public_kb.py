from __future__ import annotations

import argparse
import asyncio

from app.core.llm.embedding_service import EmbeddingService
from app.db.vector_store import init_chroma_collection


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test public KB vector store query")
    parser.add_argument("--collection", type=str, default="kb_general")
    parser.add_argument("--query", type=str, default="胃胀痛")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    col = init_chroma_collection(collection_name=args.collection)
    embedder = EmbeddingService()
    vectors = await embedder.embed_documents([args.query])
    res = col.query(query_embeddings=vectors, n_results=max(1, int(args.top_k)))

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]

    print(f"collection={args.collection} query={args.query} hits={len(ids)}")
    for i in range(min(len(ids), len(docs))):
        doc_id = str(ids[i])
        text = str(docs[i])
        print(f"- {doc_id}: {text[:120]}")


if __name__ == "__main__":
    asyncio.run(main())
