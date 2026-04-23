from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from app.core.llm.embedding_service import EmbeddingService
from app.core.utils.text_splitter import TextSplitter
from app.db.vector_store import init_chroma_collection


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT / "data" / "Source_data"
DEFAULT_COLLECTION = "kb_general"

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("skip bad json line: %s:%s", path.name, line_no)
                continue
            if isinstance(obj, dict):
                yield line_no, obj


def _normalize_questions(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                for q in item:
                    if isinstance(q, str) and q.strip():
                        out.append(q.strip())
            elif isinstance(item, str) and item.strip():
                out.append(item.strip())
    elif isinstance(value, str) and value.strip():
        out = [value.strip()]

    seen: set[str] = set()
    deduped: list[str] = []
    for q in out:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def _normalize_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [a.strip() for a in value if isinstance(a, str) and a.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _build_document(questions: list[str], answers: list[str]) -> str:
    q_text = " / ".join(questions) if questions else ""
    a_text = "\n".join(answers) if answers else ""
    if q_text and a_text:
        return f"问题: {q_text}\n回答: {a_text}"
    return q_text or a_text


def _make_id(base: str, chunk_index: int) -> str:
    seed = f"{base}-{chunk_index}".encode("utf-8")
    return hashlib.sha1(seed).hexdigest()


async def _upsert_batches(
    *,
    collection,
    embedder: EmbeddingService,
    ids: list[str],
    docs: list[str],
    metas: list[dict[str, Any]],
    batch_size: int,
) -> int:
    total = 0
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = docs[i : i + batch_size]
        batch_metas = metas[i : i + batch_size]
        vectors = await embedder.embed_documents(batch_docs)

        if hasattr(collection, "upsert"):
            collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=vectors,
            )
        else:
            try:
                collection.delete(ids=batch_ids)
            except Exception:
                pass
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=vectors,
            )
        total += len(batch_ids)
    return total


async def ingest(
    *,
    source_dir: Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> None:
    col = init_chroma_collection(collection_name=collection_name)
    embedder = EmbeddingService()

    files = sorted(source_dir.glob("*.jsonl"))
    if not files:
        raise RuntimeError(f"no jsonl files found in {source_dir}")

    total_docs = 0

    for path in files:
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []

        for line_no, obj in _iter_jsonl(path):
            questions = _normalize_questions(obj.get("questions"))
            answers = _normalize_answers(obj.get("answers"))
            doc = _build_document(questions, answers)
            if not doc:
                continue

            chunks = TextSplitter.split_text(
                doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            if not chunks:
                continue

            base_id = f"{path.name}:{line_no}"
            for idx, chunk in enumerate(chunks):
                ids.append(_make_id(base_id, idx))
                docs.append(chunk)
                metas.append(
                    {
                        "kb_type": "public_medical",
                        "source_type": "jsonl",
                        "source_name": path.name,
                        "split": path.stem,
                        "record_line": line_no,
                        "question_count": len(questions),
                        "answer_count": len(answers),
                        "chunk_index": idx,
                    }
                )

        if ids:
            inserted = await _upsert_batches(
                collection=col,
                embedder=embedder,
                ids=ids,
                docs=docs,
                metas=metas,
                batch_size=batch_size,
            )
            total_docs += inserted
            print(f"seeded {inserted} docs -> {collection_name} ({path.name})")

    print(f"DONE: total docs inserted = {total_docs}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest public medical KB into Chroma")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Folder with jsonl files",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Chroma collection name",
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    asyncio.run(
        ingest(
            source_dir=args.source_dir,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
