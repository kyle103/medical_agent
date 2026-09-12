"""医疗实体内存词典 + 前向最大匹配。

把 drug_knowledge_base / lab_item_reference_base 加载进内存，
直接在【原文】上做前向最大匹配（无需先按分隔符切词），得到 {规范名/别名/商品名} → 标准名 的解析。

与旧链路（候选 → 逐条 SQL LIKE）的关系：
1. 匹配方向反转：词典词条直接扫原文，能找回被剂量/剂型粘连、无分隔符时的药名，
   例如 "对乙酰氨基酚缓释片一片" → 命中 对乙酰氨基酚（旧规则会把整句当一个候选，LIKE 必然失败）。
2. 共指消解：别名/商品名命中统一归一到 drug_name（扑热息痛 → 对乙酰氨基酚）。
3. 否定/假设标注：命中带 negated/hypothetical 标记，resolve(..., drop_negated=True) 可过滤。

注意：本词典是【基础设施层】，不替代数据库。match_drugs 仍以 DB 为准——
词典只在“能确定把候选解析到某个已知标准名”时短路一次查询；否则走原有 DB 兜底，
保证知识库热更新/软删除后语义不被快照误导。

本模块主体（EntityDictionary）不依赖 DB/日志，可直接单测。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

from sqlalchemy import select

from app.common.logger import get_logger
from app.db.database import get_sessionmaker
from app.db.models import DrugKnowledgeBase, LabItemReferenceBase

logger = get_logger(__name__)

_WS_RE = re.compile(r"[\s　 ​]+")


def normalize_text(text: str) -> str:
    """归一化：NFKC 折叠全角/半角、小写、去掉所有空白（中英混排按字匹配）。"""
    if not text:
        return ""
    return _WS_RE.sub("", unicodedata.normalize("NFKC", text).lower())


# 命中前出现即视为“否定摄入/已停用”
NEGATED_CUES = (
    "没有吃过",
    "没吃过",
    "没有吃",
    "没有服用",
    "没服用",
    "没吃",
    "不吃",
    "别吃",
    "不要吃",
    "不能吃",
    "不可服用",
    "未服用",
    "未吃",
    "停用",
    "停药",
    "停了",
    "停服",
    "戒断",
    "撤药",
    "忌用",
    "禁止",
    "禁用",
    "过敏不能吃",
    "不是吃",
)

# 命中前出现即视为“假设/未发生的摄入”（冲突类问句天然是假设，消费方按需取舍）
HYPOTHETICAL_CUES = (
    "如果要吃",
    "如果想吃",
    "如果服用",
    "如果吃",
    "如果用",
    "要是吃",
    "假如吃",
    "假设吃",
    "想吃",
    "打算吃",
    "准备吃",
    "考虑吃",
    "计划吃",
    "要不要吃",
    "是否要吃",
    "需不需要吃",
    "该不该吃",
    "可不可以吃",
)

# canonicalize_candidate：候选 = 已知标准名 + 剂型/剂型后缀 时，允许剥离的后缀
DOSAGE_SUFFIXES = (
    "缓释片",
    "肠溶片",
    "缓释胶囊",
    "咀嚼片",
    "泡腾片",
    "分散片",
    "普通片",
    "肠溶胶囊",
    "软胶囊",
    "硬胶囊",
    "缓释颗粒",
    "控释片",
    "胶囊",
    "颗粒",
    "片",
    "散",
    "丸",
    "液",
    "口服液",
    "注射液",
    "滴丸",
    "贴",
    "膏",
)


def _cue_applies(win: str, cues: Sequence[str]) -> bool:
    """cue 是否落在命中词紧前方（允许 ≤2 个夹字）。"""
    for cue in cues:
        idx = win.rfind(cue)
        if idx != -1 and (len(win) - (idx + len(cue))) <= 2:
            return True
    return False


class EntityDictionary:
    """内存实体词典：term(规范名/别名/英文名) → {kind, canonical_name, payload}。"""

    def __init__(self) -> None:
        self._term_meta: dict[str, dict] = {}
        self._canon_payloads: dict[tuple[str, str], dict] = {}
        self._max_len = 0
        self._ready = False

    # ---- 构建 ------------------------------------------------------------
    def is_ready(self) -> bool:
        return self._ready

    def clear(self) -> None:
        self._term_meta.clear()
        self._canon_payloads.clear()
        self._max_len = 0
        self._ready = False

    def add_entry(
        self,
        *,
        kind: str,
        canonical_name: str,
        terms: Iterable[str],
        payload: dict,
    ) -> None:
        canonical_key = (kind, canonical_name)
        self._canon_payloads.setdefault(canonical_key, dict(payload))
        for term in terms:
            nt = normalize_text(term)
            if not nt:
                continue
            # 已存在的规范名保留首见 payload；同词不同 canonical 以先到为准
            self._term_meta.setdefault(nt, {"kind": kind, "canonical_name": canonical_name})
            if len(nt) > self._max_len:
                self._max_len = len(nt)

    def mark_ready(self) -> None:
        self._ready = True

    @property
    def term_count(self) -> int:
        return len(self._term_meta)

    @property
    def canonical_count(self) -> int:
        return len(self._canon_payloads)

    # ---- 命中扫描 --------------------------------------------------------
    def scan_hits(self, text: str, kinds: Sequence[str] | None = None) -> list[dict]:
        """前向最大匹配原文，返回有序命中（按规范名去重由 resolve 负责）。"""
        if not text or not self._term_meta:
            return []
        nt = normalize_text(text)
        hits: list[dict] = []
        i, n = 0, len(nt)
        kind_set = set(kinds) if kinds else None
        while i < n:
            upper = min(self._max_len, n - i)
            meta = None
            matched_len = 0
            for length in range(upper, 0, -1):
                m = self._term_meta.get(nt[i : i + length])
                if m is not None:
                    meta = m
                    matched_len = length
                    break
            if meta is None:
                i += 1
                continue
            if kind_set is not None and meta["kind"] not in kind_set:
                # 命中的是其他领域词条：让位，允许同位继续匹配目标领域
                i += 1
                continue
            win = nt[max(0, i - 8) : i]
            hits.append(
                {
                    "term": nt[i : i + matched_len],
                    "canonical_name": meta["canonical_name"],
                    "kind": meta["kind"],
                    "start": i,
                    "end": i + matched_len,
                    "negated": _cue_applies(win, NEGATED_CUES),
                    "hypothetical": _cue_applies(win, HYPOTHETICAL_CUES),
                }
            )
            i += matched_len
        return hits

    def resolve(
        self,
        text: str,
        kinds: Sequence[str] | None = None,
        *,
        drop_negated: bool = False,
    ) -> list[str]:
        """标准名列表（保序、去重）。drop_negated=True 时过滤“否定摄入”命中。

        说明：hypothetical 命中默认保留——冲突/知识类问句本质是假设性提问，
        直接丢弃会误伤“阿司匹林和布洛芬能一起吃吗”这类合法查询。
        """
        names: list[str] = []
        seen: set[str] = set()
        for hit in self.scan_hits(text, kinds=kinds):
            if drop_negated and hit["negated"]:
                continue
            canon = hit["canonical_name"]
            if canon not in seen:
                seen.add(canon)
                names.append(canon)
        return names

    # ---- 候选解析 --------------------------------------------------------
    def canonical_name_for(self, name: str, kinds: Sequence[str] | None = None) -> str | None:
        """把“候选名”解析到已知标准名（含标准名+剂型后缀形式）。

        命中返回标准名；否则返回 None（交给 DB 兜底）。
        只处理【前缀 + 可剥离剂型后缀】，不做任意子串匹配，避免过度归并。
        """
        if not self._term_meta:
            return None
        nt = normalize_text(name)
        if not nt:
            return None
        for length in range(min(self._max_len, len(nt)), 0, -1):
            meta = self._term_meta.get(nt[:length])
            if meta is None:
                continue
            if kinds is not None and meta["kind"] not in kinds:
                continue
            residue = nt[length:]
            if residue == "" or residue in DOSAGE_SUFFIXES:
                return meta["canonical_name"]
        return None

    def _payload_for(self, canonical: str, kinds: Sequence[str] | None) -> dict | None:
        for (kind, canon), payload in self._canon_payloads.items():
            if canon == canonical and (kinds is None or kind in kinds):
                return dict(payload)
        return None

    def canonicalize_candidate(self, name: str, kinds: Sequence[str] | None = None) -> dict | None:
        """canonical_name_for 的 payload 版本（命中返回标准名快照）。"""
        canonical = self.canonical_name_for(name, kinds=kinds)
        if canonical is None:
            return None
        return self._payload_for(canonical, kinds=kinds)

    def describe(self) -> str:
        return f"EntityDictionary(ready={self._ready}, terms={self.term_count}, canon={self.canonical_count})"


# ---------------------------------------------------------------------------
# 模块级单例 + 异步 DB 加载
# ---------------------------------------------------------------------------
_dictionary = EntityDictionary()


def get_dictionary() -> EntityDictionary:
    return _dictionary


def _split_aliases(alias_raw: str | None) -> list[str]:
    if not alias_raw:
        return []
    out: list[str] = []
    for part in re.split(r"[,，、;；/|]+", alias_raw):
        part = (part or "").strip()
        if part and part not in out:
            out.append(part)
    return out


async def reload_entity_dictionary(*, log: bool = True) -> EntityDictionary:
    """强制（重新）从 DB 加载词典。startup 预热与知识库导入后刷新都走这里。"""
    global _dictionary
    try:
        async_session = get_sessionmaker()
        async with async_session() as session:
            drug_rows = (
                (
                    await session.execute(
                        select(DrugKnowledgeBase).where(DrugKnowledgeBase.is_deleted == 0)
                    )
                )
                .scalars()
                .all()
            )
            lab_rows = (
                (
                    await session.execute(
                        select(LabItemReferenceBase).where(LabItemReferenceBase.is_deleted == 0)
                    )
                )
                .scalars()
                .all()
            )
    except Exception as e:  # noqa: BLE001 - DB 未初始化/表缺失时静默降级，不阻塞启动
        if log:
            logger.warning("entity_dictionary reload skipped (db not ready): %s", e)
        _dictionary = EntityDictionary()
        return _dictionary

    builder = EntityDictionary()
    for row in drug_rows:
        terms = [row.drug_name]
        terms.extend(_split_aliases(row.drug_alias))
        seen: set[str] = set()
        dedup_terms: list[str] = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                dedup_terms.append(t)
        builder.add_entry(
            kind="drug",
            canonical_name=row.drug_name,
            terms=dedup_terms,
            payload={
                "drug_name": row.drug_name,
                "drug_alias": row.drug_alias,
                "interaction_drugs": row.interaction_drugs,
                "interaction_desc": row.interaction_desc,
            },
        )
    for row in lab_rows:
        terms = [row.item_name]
        if row.item_en_name:
            terms.append(row.item_en_name)
        builder.add_entry(
            kind="lab",
            canonical_name=row.item_name,
            terms=terms,
            payload={
                "item_name": row.item_name,
                "item_en_name": row.item_en_name,
                "reference_range": row.reference_range,
                "unit": row.unit,
                "high_meaning": row.high_meaning,
                "low_meaning": row.low_meaning,
            },
        )
    builder.mark_ready()
    _dictionary = builder
    if log:
        logger.info("entity_dictionary loaded: %s", builder.describe())
    return _dictionary


async def ensure_entity_dictionary_loaded() -> EntityDictionary:
    """首次使用时惰性加载；已加载则直接返回。"""
    if _dictionary.is_ready():
        return _dictionary
    return await reload_entity_dictionary()
