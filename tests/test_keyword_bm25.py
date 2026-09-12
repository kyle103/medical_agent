"""伪 BM25（关键词 LIKE 模拟）纯函数单元测试。

不依赖 Milvus：仅测 LIKE 表达式构建、位置加权打分与 jieba 关键词抽取。
"""

from app.core.rag.keyword_extractor import KeywordExtractor
from app.core.rag.public_kb_service import build_keyword_like_expr, score_keyword_hits


def test_build_keyword_like_expr_or_mode():
    expr = build_keyword_like_expr(["布洛芬", "相互作用", "阿司匹林"], mode="or")
    assert expr == (
        "(document LIKE '%布洛芬%' OR document LIKE '%相互作用%' OR document LIKE '%阿司匹林%')"
    )


def test_build_keyword_like_expr_and_uses_core_top():
    expr = build_keyword_like_expr(["布洛芬", "相互作用", "阿司匹林", "禁忌"], mode="and", core_top=3)
    assert "OR" not in expr
    assert "布洛芬" in expr
    assert "禁忌" not in expr  # 超出 core_top 的词不进 AND


def test_build_keyword_like_expr_single_quote_escaped():
    expr = build_keyword_like_expr(["a'b"], mode="or")
    assert "a''b" in expr


def test_build_keyword_like_expr_empty():
    assert build_keyword_like_expr([], mode="or") is None


def test_score_keyword_hits_position_weighted():
    # 首词命中权重高于次词命中
    first = [{"text": "布洛芬"}]
    second = [{"text": "相互作用"}]
    kws = ["布洛芬", "相互作用"]
    score_keyword_hits(first, kws)
    score_keyword_hits(second, kws)
    assert first[0]["kw_score"] > second[0]["kw_score"]
    assert first[0]["kw_score"] > 0


def test_score_keyword_hits_multi_occurrence_capped():
    # 命中计数封顶 3：多次命中不线性放大
    once = [{"text": "布洛芬"}]
    many = [{"text": "布洛芬 布洛芬 布洛芬 布洛芬 布洛芬"}]
    score_keyword_hits(once, ["布洛芬"])
    score_keyword_hits(many, ["布洛芬"])
    assert many[0]["kw_score"] > once[0]["kw_score"]
    # 5 次命中 vs 1 次：权重 1/(0+1)=1，min(5,3)=3 vs min(1,3)=1
    assert many[0]["kw_score"] == 3.0 * once[0]["kw_score"]


def test_score_keyword_hits_missing_keyword_zero():
    items = [{"text": "感冒"}]
    score_keyword_hits(items, ["布洛芬", "阿司匹林"])
    assert items[0]["kw_score"] == 0.0


def test_keyword_extractor_jieba_no_stopwords():
    extractor = KeywordExtractor(use_llm=False)
    kws = extractor.extract_jieba("布洛芬和阿司匹林一起吃有相互作用吗", top_k=8)
    assert isinstance(kws, list)
    assert kws  # 非空
    for kw in kws:
        assert len(kw) > 1  # 过滤单字/停用词


def test_keyword_extractor_extract_short_query_jieba_only():
    # 默认 use_llm=False：任何长度都走 jieba，且 async extract 返回列表
    import asyncio

    extractor = KeywordExtractor(use_llm=False)
    kws = asyncio.run(extractor.extract("高血压吃什么药", top_k=8))
    assert isinstance(kws, list)
    assert all(len(k) > 1 for k in kws)
