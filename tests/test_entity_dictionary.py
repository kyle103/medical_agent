"""EntityDictionary 单元测试（纯内存，无需 DB/LLM）。

覆盖：前向最大匹配找回粘连药名、别名归一、重叠最长匹配、剂型后缀剥离、
否定/假设标注、泛称过滤、既有规则抽取契约不回归。
"""

from app.core.rag.entity_dictionary import EntityDictionary
from app.core.tools.drug_entity_extractor import DrugEntityExtractor


def _payload(name: str) -> dict:
    return {
        "drug_name": name,
        "drug_alias": "",
        "interaction_drugs": "[]",
        "interaction_desc": "{}",
    }


def _seed_dict() -> EntityDictionary:
    d = EntityDictionary()
    d.add_entry(kind="drug", canonical_name="对乙酰氨基酚",
                terms=["对乙酰氨基酚", "扑热息痛"], payload=_payload("对乙酰氨基酚"))
    d.add_entry(kind="drug", canonical_name="复方氨酚烷胺片",
                terms=["复方氨酚烷胺片", "感康"], payload=_payload("复方氨酚烷胺片"))
    d.add_entry(kind="drug", canonical_name="布洛芬",
                terms=["布洛芬"], payload=_payload("布洛芬"))
    d.add_entry(kind="drug", canonical_name="阿司匹林",
                terms=["阿司匹林"], payload=_payload("阿司匹林"))
    d.mark_ready()
    return d


def test_forward_max_match_recovers_undelimited_dosage_form():
    """无分隔符、带剂型粘连时找回标准名：旧规则会把整句当一个候选而 LIKE 失败。"""
    d = _seed_dict()
    text = "我同时吃了对乙酰氨基酚缓释片一片和扑热息痛"
    # 命中 对乙酰氨基酚（原文粘连）+ 别名 扑热息痛 → 归一为同一个标准名
    assert d.resolve(text) == ["对乙酰氨基酚"]


def test_alias_normalized_to_canonical():
    d = _seed_dict()
    assert d.resolve("感康") == ["复方氨酚烷胺片"]
    assert d.resolve("今天喝了点扑热息痛") == ["对乙酰氨基酚"]


def test_multiple_drugs_order_preserved_deduped():
    d = _seed_dict()
    assert d.resolve("布洛芬和阿司匹林一起吃可以吗") == ["布洛芬", "阿司匹林"]
    assert d.resolve("扑热息痛、布洛芬、对乙酰氨基酚") == ["对乙酰氨基酚", "布洛芬"]


def test_longest_match_wins_for_overlap():
    """重叠词条只保留最长命中，避免把一个药名拆成多个子串。"""
    d = _seed_dict()
    d.add_entry(kind="drug", canonical_name="XXX复方", terms=["阿司匹"], payload=_payload("XXX复方"))
    hits = d.scan_hits("阿司匹林")
    canonicals = [h["canonical_name"] for h in hits]
    assert canonicals == ["阿司匹林"]  # 阿司匹(3) 被最长 阿司匹林(4) 覆盖


def test_canonicalize_candidate_strips_dosage_suffix():
    d = _seed_dict()
    hit = d.canonicalize_candidate("对乙酰氨基酚缓释片", kinds=("drug",))
    assert hit is not None and hit["drug_name"] == "对乙酰氨基酚"

    assert d.canonicalize_candidate("布洛芬", kinds=("drug",))["drug_name"] == "布洛芬"
    # 别名同样可解析到标准名
    assert d.canonicalize_candidate("感康胶囊", kinds=("drug",))["drug_name"] == "复方氨酚烷胺片"


def test_canonicalize_does_not_false_merge():
    d = _seed_dict()
    # 部分词 / 非药词 → 留给 DB 兜底（返回 None）
    assert d.canonicalize_candidate("匹林", kinds=("drug",)) is None
    assert d.canonicalize_candidate("感冒药", kinds=("drug",)) is None
    assert d.canonicalize_candidate("头孢克肟", kinds=("drug",)) is None


def test_negation_flag_and_drop():
    d = _seed_dict()
    hits = d.scan_hits("我没吃阿司匹林")
    assert hits and hits[0]["canonical_name"] == "阿司匹林"
    assert hits[0]["negated"] is True

    assert d.resolve("我没吃阿司匹林", drop_negated=True) == []
    assert d.resolve("我没吃阿司匹林", drop_negated=False) == ["阿司匹林"]


def test_hypothetical_query_kept_by_default():
    """冲突类问句本质是假设性提问，resolve 默认保留 hypothetical 命中。"""
    d = _seed_dict()
    names = d.resolve("如果同时吃阿司匹林和布洛芬会有冲突吗")
    assert names == ["阿司匹林", "布洛芬"]
    # 标注仍在，供后续消费方按语义取舍
    hits = {h["canonical_name"]: h for h in d.scan_hits("如果吃阿司匹林会怎样")}
    assert hits["阿司匹林"]["hypothetical"] is True


def test_generic_drug_terms_filtered_by_rule_extractor():
    assert DrugEntityExtractor.extract_drug_candidates("布洛芬和感冒药") == ["布洛芬"]


def test_rule_extractor_basic_contract_preserved():
    assert DrugEntityExtractor.extract_drug_candidates("阿司匹林、布洛芬") == ["阿司匹林", "布洛芬"]
