from __future__ import annotations

from typing import Any

from app.core.tools.archive_query_tool import ArchiveQueryTool


class MedicationRecallSkill:
    """回忆用户用药信息：优先档案，回退会话历史。"""

    COMMON_DRUGS = [
        "布洛芬",
        "阿司匹林",
        "青霉素",
        "头孢",
        "降压药",
        "降糖药",
        "抗生素",
        "止痛药",
        "感冒药",
        "消炎药",
        "消食片",
    ]

    def __init__(self):
        self.archive_tool = ArchiveQueryTool()

    async def recall_recent_drugs(self, *, user_id: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            tool_result = await self.archive_tool.query(
                user_id=user_id,
                query_type="drug_records",
                query_conditions={},
            )
            drug_records = tool_result.get("items", [])
            if drug_records:
                return {"source": "archive", "records": drug_records}
        except Exception:
            pass

        mentioned = self._extract_from_history(history)
        if mentioned:
            return {"source": "history", "records": [{"drug_name": d} for d in mentioned]}
        return {"source": "none", "records": []}

    def _extract_from_history(self, history: list[dict[str, Any]]) -> list[str]:
        mentioned: list[str] = []
        for message in history:
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            for drug in self.COMMON_DRUGS:
                if drug in content and drug not in mentioned:
                    mentioned.append(drug)
        return mentioned

