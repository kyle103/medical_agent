from __future__ import annotations

import re
from typing import Any

from app.core.rag.drug_knowledge_service import DrugKnowledgeService
from app.core.rag.lab_reference_service import LabReferenceService
from app.core.tools.drug_entity_extractor import DrugEntityExtractor


class MedicalKnowledgeService:
    """医疗专业知识检索聚合服务。"""

    LAB_ITEM_KEYWORDS = [
        "血糖",
        "血压",
        "血脂",
        "胆固醇",
        "肝功能",
        "肾功能",
        "白细胞",
        "红细胞",
        "血小板",
        "尿酸",
        "转氨酶",
    ]

    async def retrieve(self, user_input: str, intent: str = "general") -> dict[str, Any]:
        text = (user_input or "").strip()
        if not text:
            return {}

        out: dict[str, Any] = {}
        drug_candidates = DrugEntityExtractor.extract_drug_candidates(text, max_items=6)
        if drug_candidates:
            matched_drugs = await DrugKnowledgeService().match_drugs(drug_candidates)
            out["drug_knowledge"] = [m for m in matched_drugs if isinstance(m, dict)]

        lab_candidates = self._extract_lab_candidates(text)
        if lab_candidates:
            matched_items = await LabReferenceService().match_items(lab_candidates[:6])
            out["lab_reference"] = [m for m in matched_items if isinstance(m, dict)]

        if intent == "drug" and "drug_knowledge" not in out:
            out["drug_knowledge"] = []
        if intent == "lab" and "lab_reference" not in out:
            out["lab_reference"] = []
        return out

    def _extract_lab_candidates(self, text: str) -> list[str]:
        found: list[str] = []
        for kw in self.LAB_ITEM_KEYWORDS:
            if kw in text:
                found.append(kw)
        for hit in re.findall(r"(?:化验|检验|指标)[:：]?\s*([^\s，。；;]{1,20})", text):
            found.append(hit.strip())

        dedup: list[str] = []
        seen = set()
        for item in found:
            if item and item not in seen:
                seen.add(item)
                dedup.append(item)
        return dedup

