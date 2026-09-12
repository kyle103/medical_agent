from __future__ import annotations

import json

from sqlalchemy import or_, select

from app.common.exceptions import NotFoundException
from app.core.rag.entity_dictionary import (
    ensure_entity_dictionary_loaded,
)
from app.db.database import get_sessionmaker
from app.db.models import DrugKnowledgeBase


class DrugKnowledgeService:
    async def _ensure_dict(self):
        """返回已加载词典；DB 未就绪时返回空实例（退化为纯 DB 链路）。"""
        try:
            return await ensure_entity_dictionary_loaded()
        except Exception:  # noqa: BLE001 - 词典只是加速层，失败不阻塞
            from app.core.rag.entity_dictionary import EntityDictionary

            return EntityDictionary()

    async def resolve_text(self, text: str, *, drop_negated: bool = False) -> list[str]:
        """对原文做词典前向最大匹配，返回标准药名（保序去重）。

        用于找回规则切词漏掉的药名（无分隔符 / 带剂型粘连），并做别名→标准名归一。
        """
        if not text or not text.strip():
            return []
        dictionary = await self._ensure_dict()
        return dictionary.resolve(text, kinds=("drug",), drop_negated=drop_negated)

    async def canonicalize_names(self, names: list[str]) -> list[str]:
        """把已有候选名归一到标准名（保序去重）。

        归一顺序：
        1. 前缀 + 可剥离剂型后缀 → 标准名（对乙酰氨基酚缓释片 → 对乙酰氨基酚）
        2. 词典扫描候选本身 → 应对切词粘连（"昨晚对乙酰氨基酚缓释片一片" → 对乙酰氨基酚）
        3. 均失败则保留原值，交给 DB 的 LIKE 兜底（新药 / 未收录药名不丢）
        """
        if not names:
            return []
        dictionary = await self._ensure_dict()
        out: list[str] = []
        seen: set[str] = set()

        def _add(value: str) -> None:
            if value and value not in seen:
                seen.add(value)
                out.append(value)

        for name in names:
            canon = dictionary.canonical_name_for(name, kinds=("drug",))
            if canon:
                _add(canon)
                continue
            inner = dictionary.resolve(name, kinds=("drug",))
            if inner:
                for c in inner:
                    _add(c)
                continue
            _add(name)
        return out

    async def match_drugs(self, drug_names: list[str]) -> list[dict]:
        if not drug_names:
            return []

        dictionary = await self._ensure_dict()
        async_session = get_sessionmaker()
        results: list[dict] = []

        async with async_session() as session:
            for name in drug_names:
                # 词典先把候选归一到标准名（别名/商品名/带剂型 → drug_name），
                # 再用 DB 精确等值查询，保证知识库热更新/软删除不被内存快照误导。
                canonical = dictionary.canonical_name_for(name, kinds=("drug",))
                if canonical is not None:
                    q = select(DrugKnowledgeBase).where(
                        DrugKnowledgeBase.is_deleted == 0,
                        DrugKnowledgeBase.drug_name == canonical,
                    )
                else:
                    q = select(DrugKnowledgeBase).where(
                        DrugKnowledgeBase.is_deleted == 0,
                        or_(
                            DrugKnowledgeBase.drug_name == name,
                            DrugKnowledgeBase.drug_alias.like(f"%{name}%"),
                        ),
                    )
                res = await session.execute(q)
                row = res.scalars().first()
                if not row:
                    results.append({"query": name, "match": None})
                else:
                    results.append(
                        {
                            "query": name,
                            "match": {
                                "drug_name": row.drug_name,
                                "drug_alias": row.drug_alias,
                                "interaction_drugs": row.interaction_drugs,
                                "interaction_desc": row.interaction_desc,
                            },
                        }
                    )

        return results

    @staticmethod
    def parse_interactions(row_match: dict) -> tuple[list[str], dict]:
        drugs_raw = row_match.get("interaction_drugs") or "[]"
        desc_raw = row_match.get("interaction_desc") or "{}"
        try:
            drugs = json.loads(drugs_raw)
            if not isinstance(drugs, list):
                drugs = []
        except Exception:
            drugs = []

        try:
            desc = json.loads(desc_raw)
            if not isinstance(desc, dict):
                desc = {}
        except Exception:
            desc = {}

        return drugs, desc
