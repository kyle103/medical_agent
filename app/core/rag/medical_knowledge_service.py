from __future__ import annotations

import re
import time
from typing import Any

from app.common.logger import get_logger, log_rag_retrieval
from app.core.rag.drug_knowledge_service import DrugKnowledgeService
from app.core.rag.lab_reference_service import LabReferenceService
from app.core.tools.drug_entity_extractor import DrugEntityExtractor

logger = get_logger(__name__)


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

        start_time = time.perf_counter()
        out: dict[str, Any] = {}

        try:
            drug_candidates = DrugEntityExtractor.extract_drug_candidates(text, max_items=6)
            if drug_candidates:
                drug_start = time.perf_counter()
                matched_drugs = await DrugKnowledgeService().match_drugs(drug_candidates)
                drug_ms = int((time.perf_counter() - drug_start) * 1000)
                out["drug_knowledge"] = [m for m in matched_drugs if isinstance(m, dict)]
                log_rag_retrieval(
                    source="drug_knowledge",
                    query=text,
                    method="sql_match",
                    top_k=len(drug_candidates),
                    result_count=len(out["drug_knowledge"]),
                    latency_ms=drug_ms,
                    success=True,
                )

            lab_candidates = self._extract_lab_candidates(text)
            if lab_candidates:
                lab_start = time.perf_counter()
                matched_items = await LabReferenceService().match_items(lab_candidates[:6])
                lab_ms = int((time.perf_counter() - lab_start) * 1000)
                out["lab_reference"] = [m for m in matched_items if isinstance(m, dict)]
                log_rag_retrieval(
                    source="lab_reference",
                    query=text,
                    method="sql_match",
                    top_k=len(lab_candidates[:6]),
                    result_count=len(out["lab_reference"]),
                    latency_ms=lab_ms,
                    success=True,
                )

            if intent == "drug" and "drug_knowledge" not in out:
                out["drug_knowledge"] = []
            if intent == "lab" and "lab_reference" not in out:
                out["lab_reference"] = []

            total_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(
                "MedicalKnowledgeService.retrieve intent=%s drug_candidates=%d lab_candidates=%d total_latency=%dms",
                intent,
                len(drug_candidates) if drug_candidates else 0,
                len(lab_candidates),
                total_ms,
            )
            return out
        except Exception as e:
            total_ms = int((time.perf_counter() - start_time) * 1000)
            log_rag_retrieval(
                source="medical_knowledge",
                query=text,
                method="aggregate",
                result_count=0,
                latency_ms=total_ms,
                success=False,
                error=str(e)[:100],
            )
            raise

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
