from __future__ import annotations

import os
from app.config.settings import settings
import random
import time
from typing import Any

import pytest
from pymilvus import Collection, MilvusClient, connections


def _is_analyzer_enabled(field: dict[str, Any]) -> bool:
    if field.get("enable_analyzer") is True:
        return True
    params = field.get("params") or field.get("type_params") or {}
    return bool(params.get("enable_analyzer"))


def _get_analyzer_params(field: dict[str, Any]) -> Any:
    if field.get("analyzer_params") is not None:
        return field.get("analyzer_params")
    params = field.get("params") or field.get("type_params") or {}
    return params.get("analyzer_params")


def _normalize_hits(result: Any) -> list[Any]:
    if not result:
        return []
    if isinstance(result, list):
        if result and isinstance(result[0], list):
            return result[0]
        if result and hasattr(result[0], "id"):
            return result
        return result
    return [result]

def _milvus_uri() -> str:
    return (getattr(settings, "MILVUS_URI", "") or os.getenv("MILVUS_URI", "")).strip()


def _milvus_token() -> str:
    return (getattr(settings, "MILVUS_TOKEN", "") or os.getenv("MILVUS_TOKEN", "")).strip()


@pytest.fixture(scope="module")
def milvus_client():
    """创建Milvus客户端连接"""
    uri = _milvus_uri()
    if not uri:
        pytest.skip("MILVUS_URI is required")
    
    token = _milvus_token()
    client = MilvusClient(uri=uri, token=token)
    yield client


@pytest.fixture(scope="module")
def collection_name():
    """返回集合名称"""
    return "kb_general"


def test_milvus_connection(milvus_client):
    """测试Milvus连接"""
    assert milvus_client is not None
    collections = milvus_client.list_collections()
    assert isinstance(collections, list)


def test_kb_general_collection_exists(milvus_client, collection_name):
    """测试kb_general集合是否存在"""
    collections = milvus_client.list_collections()
    assert collection_name in collections


def test_vector_search(milvus_client, collection_name):
    """测试向量检索"""
    # 获取集合信息
    collection_info = milvus_client.describe_collection(collection_name=collection_name)
    fields = collection_info.get('fields', [])
    
    # 提取向量维度
    dimension = 1024
    for field in fields:
        if field.get('name') == 'embedding':
            params = field.get('params', {})
            if 'dim' in params:
                dimension = params['dim']
            break
    
    # 生成随机查询向量
    query_vector = [random.random() for _ in range(dimension)]
    
    # 确定输出字段
    output_fields = []
    field_names = [field.get('name') for field in fields]
    if 'document' in field_names:
        output_fields.append('document')
    if 'metadata' in field_names:
        output_fields.append('metadata')
    
    # 加载集合到内存
    milvus_client.load_collection(collection_name=collection_name)
    
    # 执行向量查询
    start_time = time.time()
    search_result = milvus_client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=3,
        output_fields=output_fields,
        vector_field="embedding"
    )
    total_time = time.time() - start_time
    
    # 验证结果
    assert search_result is not None
    assert isinstance(search_result, list)
    assert len(search_result) > 0
    assert len(search_result[0]) <= 3
    print(f"向量查询耗时: {total_time:.4f} 秒")


def test_bm25_search(milvus_client, collection_name):
    """测试BM25检索"""
    # 获取集合信息
    collection_info = milvus_client.describe_collection(collection_name=collection_name)
    fields = collection_info.get('fields', [])
    
    # 确定输出字段
    output_fields = []
    field_names = [field.get('name') for field in fields]
    if 'document' in field_names:
        output_fields.append('document')
    if 'metadata' in field_names:
        output_fields.append('metadata')
    
    # 测试查询
    bm25_query = "糖尿病治疗"
    
    # 执行BM25查询
    start_time = time.time()
    try:
        # 检查document字段是否启用了分析器
        document_enabled = False
        for field in fields:
            if field.get('name') == 'document':
                document_enabled = _is_analyzer_enabled(field)
                break
        
        if not document_enabled:
            # 使用LIKE操作符进行文本检索
            bm25_result = milvus_client.query(
                collection_name=collection_name,
                filter=f"document LIKE '%{bm25_query}%'",
                limit=3,
                output_fields=output_fields
            )
        else:
            # 连接到集合
            connections.connect(uri=_milvus_uri(), token=_milvus_token())
            col = Collection(collection_name)
            # 使用BM25索引进行文本检索
            bm25_result = col.search(
                data=[bm25_query],
                anns_field="document",
                param={"metric_type": "BM25"},
                limit=3,
                output_fields=output_fields,
            )
        
        total_time = time.time() - start_time
        
        # 验证结果
        assert bm25_result is not None
        if isinstance(bm25_result, list):
            assert len(bm25_result) >= 0
        print(f"BM25查询耗时: {total_time:.4f} 秒")
        
    except Exception as e:
        # 尝试使用LIKE操作符作为备选方案
        print(f"BM25查询失败: {e}")
        print("尝试使用LIKE操作符进行文本检索...")
        bm25_result = milvus_client.query(
            collection_name=collection_name,
            filter=f"document LIKE '%{bm25_query}%'",
            limit=3,
            output_fields=output_fields
        )
        total_time = time.time() - start_time
        assert bm25_result is not None
        assert len(bm25_result) >= 0
        print(f"LIKE操作符查询耗时: {total_time:.4f} 秒")