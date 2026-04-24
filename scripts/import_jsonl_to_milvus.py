from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from app.config.settings import settings


def _milvus_uri() -> str:
    return (getattr(settings, "MILVUS_URI", "") or os.getenv("MILVUS_URI", "")).strip()


def _milvus_token() -> str:
    return (getattr(settings, "MILVUS_TOKEN", "") or os.getenv("MILVUS_TOKEN", "")).strip()


def _load_first_embedding(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            emb = obj.get("embedding") or []
            if isinstance(emb, list) and emb:
                return emb
    return []


def _ensure_collection(name: str, dim: int) -> Collection:
    if utility.has_collection(name):
        return Collection(name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description=f"imported from chroma: {name}")
    col = Collection(name, schema)
    col.create_index(
        field_name="embedding",
        index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
    )
    return col


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import JSONL into Milvus (Aliyun managed)")
    parser.add_argument("--input", type=Path, required=True, help="JSONL file exported from Chroma")
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    uri = _milvus_uri()
    if not uri:
        raise RuntimeError("MILVUS_URI is required")

    token = _milvus_token()
    connections.connect(uri=uri, token=token)

    first_emb = _load_first_embedding(args.input)
    if not first_emb:
        raise RuntimeError("No embeddings found in input file")

    col = _ensure_collection(args.collection, dim=len(first_emb))

    ids: list[str] = []
    docs: list[str] = []
    metas: list[str] = []
    embs: list[list[float]] = []

    for obj in _iter_jsonl(args.input):
        emb = obj.get("embedding") or []
        if not isinstance(emb, list) or not emb:
            continue
        ids.append(str(obj.get("id") or ""))
        docs.append(str(obj.get("document") or ""))
        metas.append(json.dumps(obj.get("metadata") or {}, ensure_ascii=False))
        embs.append(emb)

        if len(ids) >= args.batch_size:
            col.insert([ids, docs, metas, embs])
            ids, docs, metas, embs = [], [], [], []

    if ids:
        col.insert([ids, docs, metas, embs])

    col.flush()
    print(f"imported {args.input} -> {args.collection}")


if __name__ == "__main__":
    main()
