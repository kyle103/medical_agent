"""本地 Chroma 存储层（替代云端 Milvus）。

提供与 `milvus_store` 语义相近的函数名（vector_search / query_by_filter /
keyword_search / parse_metadata / build_metadata_like / insert_long_memory ...），
生产调用方只需改 import 并把 Milvus 过滤字符串换成 Chroma where/where_document dict。

关键差异：
- Milvus `filter_expr='user_id == "x"'`  →  Chroma `where={'user_id': 'x'}`
- Milvus `document LIKE '%kw%'`          →  Chroma `where_document={'$contains': kw}`
- Milvus `build_metadata_like(...) 字符串` →  Chroma dict（调用方合并）
- metadata 以扁平标量 + json_meta(完整JSON串) 存储，parse_metadata 兼容旧逻辑
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from app.common.exceptions import ServiceUnavailableException
from app.common.logger import get_logger
from app.config.settings import settings

# 遥测关闭必须在 import chromadb 之前
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "True")

logger = get_logger(__name__)

_client: Any = None
_client_lock = threading.Lock()
_available: bool | None = None


def _persist_dir() -> str:
    d = str(getattr(settings, "CHROMA_PERSIST_DIRECTORY", "") or "").strip()
    if not d or d.startswith("{{"):
        d = "./data/chroma"
    return d


def get_chroma_client() -> Any:
    global _client, _available
    if _client is not None:
        return _client
    if _available is False:
        raise ServiceUnavailableException("Chroma 不可用（已标记为不可达）")
    with _client_lock:
        if _client is not None:
            return _client
        try:
            import chromadb
            persist = _persist_dir()
            os.makedirs(persist, exist_ok=True)
            _client = chromadb.PersistentClient(path=persist)
            _available = True
        except Exception as exc:
            _available = False
            raise ServiceUnavailableException(f"Chroma 初始化失败: {exc}") from exc
    return _client


def is_milvus_configured() -> bool:
    """兼容旧调用名：本地 Chroma 视为已配置。"""
    try:
        get_chroma_client()
        return True
    except ServiceUnavailableException:
        return False


def _get_collection(collection_name: str):
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)


def parse_metadata(raw: Any) -> dict[str, Any]:
    """兼容旧逻辑：dict（含 json_meta）或 JSON 字符串均可解析。"""
    if isinstance(raw, dict):
        if "json_meta" in raw:
            try:
                obj = json.loads(raw["json_meta"])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return raw
    if isinstance(raw, str) and raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


def _meta_json_str(meta: dict) -> str:
    return meta.get("json_meta") or json.dumps(meta, ensure_ascii=False)


def vector_search(
    *,
    collection_name: str,
    query_vectors: list[list[float]],
    limit: int,
    output_fields: list[str] | None = None,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """向量检索，返回 [{id, distance, entity:{document, metadata, ...}}]，兼容旧 shape。"""
    col = _get_collection(collection_name)
    try:
        res = col.query(
            query_embeddings=query_vectors,
            n_results=max(1, int(limit)),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise ServiceUnavailableException(f"Chroma 向量检索失败: {exc}") from exc

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: list[dict[str, Any]] = []
    for i in range(len(ids)):
        meta = metas[i] or {} if i < len(metas) else {}
        if not isinstance(meta, dict):
            meta = {}
        entity: dict[str, Any] = {"document": docs[i] if i < len(docs) else ""}
        entity["metadata"] = _meta_json_str(meta)
        for f in (output_fields or []):
            if f in meta and f not in ("metadata",):
                entity[f] = meta[f]
        hits.append({
            "id": ids[i],
            "distance": dists[i] if i < len(dists) else None,
            "entity": entity,
        })
    return hits


def _rows_to_records(res: dict) -> list[dict[str, Any]]:
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    out: list[dict[str, Any]] = []
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        if not isinstance(meta, dict):
            meta = {}
        out.append({
            "id": ids[i],
            "document": docs[i] if i < len(docs) else "",
            "metadata": _meta_json_str(meta),
        })
    return out


def query_by_filter(
    *,
    collection_name: str,
    filter_expr: dict,
    limit: int,
    output_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """按元数据过滤查询（Chroma where dict），返回 [{id, document, metadata}]。"""
    col = _get_collection(collection_name)
    where = filter_expr if isinstance(filter_expr, dict) else None
    if where is not None and len(where) > 1:
        # Chroma where 顶层只允许单个操作符，多字段需包 $and
        where = {"$and": [{k: v} for k, v in where.items()]}
    try:
        res = col.get(where=where, limit=max(1, int(limit)), include=["documents", "metadatas"])
    except Exception as exc:
        raise ServiceUnavailableException(f"Chroma 查询失败: {exc}") from exc
    return _rows_to_records(res)


def keyword_search(
    *,
    collection_name: str,
    keywords: list[str],
    limit: int,
    mode: str = "or",
) -> list[dict[str, Any]]:
    """伪 BM25 关键词检索：Chroma where_document $contains 子串匹配。

    - mode="or"（默认）：前 6 个关键词 $or 组合，宽召回
    - mode="and"：前 3 个关键词逐词过滤取交集，严（避免依赖 $and 支持）
    返回 [{id, document, metadata}]。
    """
    if not keywords:
        return []
    col = _get_collection(collection_name)
    try:
        if mode == "and":
            hits: list[dict[str, Any]] | None = None
            for kw in keywords[:3]:
                res = col.get(where_document={"$contains": kw}, limit=max(1, int(limit)), include=["documents", "metadatas"])
                cur = {r["id"]: r for r in _rows_to_records(res)}
                hits = cur if hits is None else {k: v for k, v in hits.items() if k in cur}
                if not hits:
                    break
            return list((hits or {}).values())
        conds = [{"$contains": k} for k in keywords[:6]]
        where_doc = {"$or": conds} if len(conds) > 1 else conds[0]
        res = col.get(where_document=where_doc, limit=max(1, int(limit)), include=["documents", "metadatas"])
        return _rows_to_records(res)
    except Exception as exc:
        raise ServiceUnavailableException(f"Chroma 关键词检索失败: {exc}") from exc


def build_metadata_like(field: str, value: Any) -> dict:
    """返回 {field: value}，调用方合并为 Chroma where。"""
    return {field: value}


def ensure_long_memory_collection(*, collection_name: str, dim: int) -> None:
    """维度由 Chroma 首次写入自动推断，此处仅确保集合存在。"""
    _get_collection(collection_name)


def insert_long_memory(
    *,
    collection_name: str,
    ids: list[str],
    user_ids: list[str],
    documents: list[str],
    metadatas: list[str],
    embeddings: list[list[float]],
) -> None:
    col = _get_collection(collection_name)
    chroma_metas: list[dict] = []
    for i, meta_str in enumerate(metadatas):
        chroma_metas.append({
            "user_id": user_ids[i] if i < len(user_ids) else "",
            "json_meta": meta_str,
        })
    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=chroma_metas)
