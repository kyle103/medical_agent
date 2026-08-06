"""把 Chroma 导出 JSONL（每行含 id/document/metadata/embedding）导入本地 Chroma。

支持子集化采样：548k chunk 全量导入会让 Chroma 常驻 ~3-4GB 内存，
这台机器（16GB / 当前空闲 1.9GB）扛不住。默认按 id 哈希均匀采样子集，
向量内存降到几十~两百 MB，检索更聚焦。

用法：
    # 知识库子集（源文件 embedding 为 1024 维，直接复用）
    python -m scripts.import_jsonl_to_chroma \
        --input data/export/chroma_jsonl/kb_general.jsonl \
        --collection kb_general --size 30000 --reset

    # 长期记忆（源文件可能为旧模型 384 维，需 --reembed 用当前服务重嵌为 1024 维）
    python -m scripts.import_jsonl_to_chroma \
        --input data/export/chroma_jsonl/user_long_memory.jsonl \
        --collection user_long_memory --size 999999 --reset --reembed

说明：
- embedding 默认复用文件中的；--reembed 时忽略文件 embedding，用 app 的
  EmbeddingService（当前 text-embedding-v4, 1024 维）重新生成，保证与应用查询维度一致。
- metadata 扁平化为标量存入 Chroma，另存 json_meta（完整 JSON 字符串），
  与现有 parse_metadata() 兼容。
- 导入完成后可删除 13GB 源 jsonl 释放磁盘。
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

# 关闭遥测，必须在 import chromadb 之前设置
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "True")

import chromadb


def _keep(doc_id: str, size: int, total: int) -> bool:
    """确定性哈希均匀采样：id 映射到 [0, 2^32)，小于阈值则保留。

    与文件顺序无关，单遍流式即可，同一 id 结果稳定。
    """
    if size >= total:
        return True
    h = int(hashlib.md5(str(doc_id).encode("utf-8")).hexdigest()[:8], 16)
    return h < (size / total) * (1 << 32)


def _flatten_meta(meta: dict) -> dict:
    flat: dict = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
    flat["json_meta"] = json.dumps(meta or {}, ensure_ascii=False)
    return flat


async def _flush(
    col,
    ids: list[str],
    docs: list[str],
    metas: list[dict],
    embs: list[list[float]],
    reembed: bool,
    embedder,
) -> int:
    if reembed:
        embs = await embedder.embed_documents(docs)
    col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    return len(embs[0]) if embs else 0


async def main() -> None:
    parser = argparse.ArgumentParser(description="Import Chroma-export JSONL into local Chroma (subset-able)")
    parser.add_argument("--input", type=Path, required=True, help="Chroma 导出 JSONL（含 embedding）")
    parser.add_argument("--collection", default="kb_general")
    parser.add_argument("--size", type=int, default=30000, help="目标子集大小")
    parser.add_argument("--total", type=int, default=548532, help="源总行数（kb_general.jsonl 行数）")
    parser.add_argument("--max-lines", type=int, default=0, help="仅读取前 N 行（快速验证用，0=读全部）")
    parser.add_argument("--persist", default="./data/chroma", help="Chroma 持久化目录")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--reset", action="store_true", help="导入前删除同名集合（正式重建用，避免与验证数据混存）")
    parser.add_argument("--reembed", action="store_true", help="忽略文件 embedding，用 app 的 EmbeddingService 重新生成（维度与当前服务一致）")
    args = parser.parse_args()

    os.makedirs(args.persist, exist_ok=True)
    client = chromadb.PersistentClient(path=args.persist)
    if args.reset:
        try:
            client.delete_collection(name=args.collection)
            print(f"reset: deleted existing collection {args.collection}", flush=True)
        except Exception:
            pass
    col = client.get_or_create_collection(name=args.collection)

    if args.reembed:
        from app.core.llm.embedding_service import EmbeddingService
        embedder = EmbeddingService()
    else:
        embedder = None

    eff_total = args.total
    if args.max_lines and args.max_lines < args.total:
        eff_total = args.max_lines

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    embs: list[list[float]] = []
    kept = 0
    scanned = 0
    dim = 0

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scanned += 1
            if args.max_lines and scanned > args.max_lines:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not args.reembed:
                emb = obj.get("embedding") or []
                if not isinstance(emb, list) or not emb:
                    continue
            doc_id = str(obj.get("id") or "")
            if not _keep(doc_id, args.size, eff_total):
                continue
            ids.append(doc_id)
            docs.append(str(obj.get("document") or ""))
            metas.append(_flatten_meta(obj.get("metadata") or {}))
            if not args.reembed:
                embs.append(emb)
            kept += 1
            if len(ids) >= args.batch_size:
                dim = max(dim, await _flush(col, ids, docs, metas, embs, args.reembed, embedder))
                ids, docs, metas, embs = [], [], [], []
                print(f"  upserted {kept} ...", flush=True)

    if ids:
        dim = max(dim, await _flush(col, ids, docs, metas, embs, args.reembed, embedder))

    n = col.count()
    vec_ram = (n * dim * 4) / 1024**3 if dim else 0.0
    print(f"done: scanned={scanned} kept={kept} collection_count={n} dim={dim}")
    print(f"estimate vector RAM: {vec_ram:.2f} GB (float32, HNSW 索引另加开销)")


if __name__ == "__main__":
    asyncio.run(main())
