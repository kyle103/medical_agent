"""Seed small test knowledge bases into Chroma collections.

Creates/updates collections:
- kb_general (public knowledge base, merged with kb_public_med)
- kb_drug
- kb_lab

Each line of *.jsonl should be:
{
  "id": "...",
  "document": "...",
  "metadata": {...}
}

This is for local testing only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.db.vector_store import init_chroma_collection


ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "knowledge_base"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _upsert(*, collection_name: str, items: list[dict[str, Any]]) -> int:
    col = init_chroma_collection(collection_name=collection_name)

    ids = [it["id"] for it in items]
    docs = [it["document"] for it in items]
    metas = [it.get("metadata") or {} for it in items]

    # Chroma python API: upsert exists in newer versions; fall back to add.
    if hasattr(col, "upsert"):
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    # naive fallback: try delete then add
    try:
        col.delete(ids=ids)
    except Exception:
        pass
    col.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def main() -> None:
    mapping = {
        "kb_general": KB_DIR / "kb_seed_general.jsonl",
        "kb_drug": KB_DIR / "kb_seed_drug.jsonl",
        "kb_lab": KB_DIR / "kb_seed_lab.jsonl",
    }

    for cname, fpath in mapping.items():
        items = _load_jsonl(fpath)
        n = _upsert(collection_name=cname, items=items)
        print(f"seeded {n} docs -> {cname} ({fpath.name})")


if __name__ == "__main__":
    main()
