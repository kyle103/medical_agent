from __future__ import annotations

import json
import os
import threading
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections, utility

from app.common.exceptions import ServiceUnavailableException
from app.common.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)

_client: MilvusClient | None = None
_client_lock = threading.Lock()


def _milvus_uri() -> str:
    uri = (getattr(settings, "MILVUS_URI", "") or os.getenv("MILVUS_URI", "")).strip()
    if not uri or uri.startswith("{{"):
        return ""
    return uri


def _milvus_token() -> str:
    token = (getattr(settings, "MILVUS_TOKEN", "") or os.getenv("MILVUS_TOKEN", "")).strip()
    if not token or token.startswith("{{"):
        return ""
    return token


def _ensure_connection() -> None:
    uri = _milvus_uri()
    if not uri:
        raise ServiceUnavailableException("MILVUS_URI 未配置")
    token = _milvus_token()
    try:
        connections.connect(uri=uri, token=token)
    except Exception as exc:
        raise ServiceUnavailableException(f"Milvus 连接失败: {exc}") from exc


def get_milvus_client() -> MilvusClient:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        uri = _milvus_uri()
        if not uri:
            raise ServiceUnavailableException("MILVUS_URI 未配置")
        token = _milvus_token()
        try:
            _client = MilvusClient(uri=uri, token=token)
        except Exception as exc:
            raise ServiceUnavailableException(f"MilvusClient 初始化失败: {exc}") from exc
    return _client


def parse_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return {}
    return {}


def _escape_like(value: str) -> str:
    return value.replace("'", "''")


def build_metadata_like(field: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False)
    escaped = _escape_like(payload)
    return f"metadata LIKE '%\"{field}\": {escaped}%'"


def normalize_search_hits(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    hits = raw
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        hits = raw[0]
    out: list[dict[str, Any]] = []
    for hit in hits:
        if isinstance(hit, dict):
            entity = hit.get("entity") or {}
            out.append(
                {
                    "id": hit.get("id") or entity.get("id"),
                    "distance": hit.get("distance"),
                    "entity": entity,
                }
            )
        else:
            entity = getattr(hit, "entity", {})
            out.append(
                {
                    "id": getattr(hit, "id", None),
                    "distance": getattr(hit, "distance", None),
                    "entity": entity,
                }
            )
    return out


def vector_search(
    *,
    collection_name: str,
    query_vectors: list[list[float]],
    limit: int,
    output_fields: list[str] | None = None,
    vector_field: str = "embedding",
    filter_expr: str | None = None,
) -> list[dict[str, Any]]:
    client = get_milvus_client()
    output_fields = output_fields or []
    try:
        raw = client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=max(1, int(limit)),
            output_fields=output_fields,
            vector_field=vector_field,
            filter=filter_expr,
        )
    except Exception as exc:
        raise ServiceUnavailableException(f"Milvus 向量检索失败: {exc}") from exc
    return normalize_search_hits(raw)


def query_like(
    *,
    collection_name: str,
    field: str,
    text: str,
    limit: int,
    output_fields: list[str] | None = None,
    extra_filter: str | None = None,
) -> list[dict[str, Any]]:
    client = get_milvus_client()
    output_fields = output_fields or []
    escaped = _escape_like(text)
    expr = f"{field} LIKE '%{escaped}%'"
    if extra_filter:
        expr = f"({expr}) AND ({extra_filter})"
    try:
        return client.query(
            collection_name=collection_name,
            filter=expr,
            limit=max(1, int(limit)),
            output_fields=output_fields,
        )
    except Exception as exc:
        raise ServiceUnavailableException(f"Milvus 文本检索失败: {exc}") from exc


def query_by_filter(
    *,
    collection_name: str,
    filter_expr: str,
    limit: int,
    output_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    client = get_milvus_client()
    output_fields = output_fields or []
    try:
        return client.query(
            collection_name=collection_name,
            filter=filter_expr,
            limit=max(1, int(limit)),
            output_fields=output_fields,
        )
    except Exception as exc:
        raise ServiceUnavailableException(f"Milvus 查询失败: {exc}") from exc


def ensure_long_memory_collection(*, collection_name: str, dim: int) -> None:
    _ensure_connection()
    if utility.has_collection(collection_name):
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True, auto_id=False),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description=f"long memory collection: {collection_name}")
    Collection(name=collection_name, schema=schema, shards_num=1)

    try:
        col = Collection(collection_name)
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        col.load()
    except Exception as exc:
        logger.warning("Milvus long memory index init failed: %s", str(exc))


def insert_long_memory(
    *,
    collection_name: str,
    ids: list[str],
    user_ids: list[str],
    documents: list[str],
    metadatas: list[str],
    embeddings: list[list[float]],
) -> None:
    _ensure_connection()
    col = Collection(collection_name)
    col.insert([ids, user_ids, documents, metadatas, embeddings])
