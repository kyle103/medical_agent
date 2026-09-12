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
        # 1) SQL 结构化档案优先（权威记录）
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

        # 2) 向量长期记忆兜底（对话中提过但没建档的用药/过敏等）
        try:
            from app.core.memory.long_memory_service import LongMemoryService
            svc = LongMemoryService()
            if svc.is_enabled():
                items = await svc.recall(user_id=user_id, query="用药 药物 吃了 服用", top_k=5)
                mentions = [
                    {"drug_name": self._extract_drug(it.text), "note": it.text}
                    for it in items
                    if "药" in it.text or any(k in it.text for k in self.COMMON_DRUGS)
                ]
                mentions = [m for m in mentions if m["drug_name"]]
                if mentions:
                    return {"source": "long_memory", "records": mentions}
        except Exception:
            pass

        # 3) 会话历史兜底
        mentioned = self._extract_from_history(history)
        if mentioned:
            return {"source": "history", "records": [{"drug_name": d} for d in mentioned]}
        return {"source": "none", "records": []}

    @staticmethod
    def _extract_drug(text: str) -> str:
        """从记忆文本粗提取药名（无 LLM，启发式）。

        先匹配已知药名表（精确），再用用药动词正则兜底。
        注意避开"用户"里的"用"造成的误匹配——只认 吃/服用 等强动词。
        """
        import re
        t = text or ""
        for drug in MedicationRecallSkill.COMMON_DRUGS:
            if drug in t:
                return drug
        m = re.search(r"(?:吃了|服用了|服用过|使用了|使用过|服用了|服用)([^，。！？\s：:，]{1,12})", t)
        if m:
            name = m.group(1).strip().strip("了")
            if 2 <= len(name) <= 24:
                return name
        return ""

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

