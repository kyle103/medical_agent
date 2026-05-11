from __future__ import annotations

import json
from typing import Any

from app.common.logger import get_logger
from app.config.settings import settings
from app.core.llm.llm_service import LLMService

logger = get_logger(__name__)

CAPABILITY_REGISTRY = [
    {
        "name": "drug_interaction",
        "type": "tool",
        "description": "查询两种或多种药物之间的相互作用、配伍禁忌、能否同服。输入：药品名称列表。输出：相互作用结果。",
        "when_to_use": "用户询问两种及以上药物能否一起吃、是否有冲突、相互作用、配伍禁忌等。",
    },
    {
        "name": "drug_record_agent",
        "type": "agent",
        "description": "管理用药记录：添加、查询、删除用户的用药信息。输入：药品名称、剂量、频率等。输出：操作确认或记录列表。",
        "when_to_use": "用户明确想记录/添加/删除自己的用药信息（如'我吃了XX药'、'帮我记录用药'），或查询自己的用药记录列表。注意：如果用户只是问'可以吃什么药'或'推荐什么药'，应路由到main_qa_agent。",
    },
    {
        "name": "main_qa_agent",
        "type": "agent",
        "description": "通用医疗问答、档案查询与药物推荐。输入：用户问题。输出：基于档案或知识的回答，包括疾病用药推荐等科普信息。",
        "when_to_use": "用户查询自己的健康档案、就诊记录、历史用药，提出通用健康科普问题，或询问某种疾病可以吃什么药、推荐用药等。",
    },
    {
        "name": "lab_report",
        "type": "tool",
        "description": "解读化验单指标。输入：检验指标名称和数值。输出：基于参考范围的指标解读。",
        "when_to_use": "用户要求解读化验单、血常规、尿常规等检验指标。",
    },
]


def _llm_enabled() -> bool:
    def _ok(v: str) -> bool:
        v = (v or "").strip()
        return bool(v) and not (v.startswith("{{") and v.endswith("}}"))
    return _ok(settings.LLM_API_BASE) and _ok(settings.LLM_API_KEY) and _ok(settings.LLM_MODEL_NAME)


class LLMDecisionService:
    """LLM 优先决策服务：将工具/Agent 描述暴露给 LLM，由 LLM 统一决策。

    所有决策方法遵循同一模式：
    1. LLM 优先（带超时保护）
    2. 正则/关键词兜底（LLM 失败或不可用时）
    """

    def __init__(self):
        self.llm = LLMService()

    async def classify_intent_and_route(self, text: str) -> dict | None:
        """LLM 优先：意图分类 + 路由决策（一步完成）。

        返回格式：
        {
            "intent": "archive|drug|lab|general",
            "intent_type": "archive|drug_conflict|drug_record|lab_report|general",
            "target_type": "agent|tool",
            "target_name": "main_qa_agent|drug_record_agent|drug_interaction|lab_report",
            "confidence": 0.0~1.0,
            "reason": "..."
        }
        """
        if not _llm_enabled():
            return None

        capabilities_desc = "\n".join(
            f"- {c['name']} (类型: {c['type']}): {c['description']}\n  适用场景: {c['when_to_use']}"
            for c in CAPABILITY_REGISTRY
        )

        system_prompt = (
            "你是一个医疗问答系统的意图分类与路由决策器。\n"
            "以下是系统可用的工具和Agent：\n\n"
            f"{capabilities_desc}\n\n"
            "请根据用户输入，判断其意图并选择最合适的工具/Agent。\n"
            "你必须只输出合法JSON，不要输出Markdown标记，不要有任何其他解释内容。\n"
            "JSON字段：\n"
            '- intent: "archive"(档案查询) | "drug"(药物相关) | "lab"(化验解读) | "general"(通用问答)\n'
            '- intent_type: "archive" | "drug_conflict" | "drug_record" | "lab_report" | "general"\n'
            '- target_type: "agent" | "tool"\n'
            '- target_name: 从上面的工具/Agent列表中选择\n'
            "- confidence: 0.0~1.0\n"
            "- reason: 简短说明决策理由"
        )

        user_prompt = f"用户输入：{text}\n\n请输出决策JSON："

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=8.0,
                max_tokens=300,
            )
            if not raw:
                return None
            match = __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL)
            json_str = match.group(0) if match else raw
            data = json.loads(json_str)

            valid_intents = {"archive", "drug", "lab", "general"}
            valid_targets = {c["name"] for c in CAPABILITY_REGISTRY}
            valid_types = {"agent", "tool"}

            intent = data.get("intent", "")
            target_name = data.get("target_name", "")
            target_type = data.get("target_type", "")
            confidence = float(data.get("confidence", 0.0) or 0.0)

            if intent not in valid_intents:
                return None
            if target_name not in valid_targets:
                return None
            if target_type not in valid_types:
                return None

            expected_type = next((c["type"] for c in CAPABILITY_REGISTRY if c["name"] == target_name), None)
            if expected_type and target_type != expected_type:
                target_type = expected_type

            return {
                "intent": intent,
                "intent_type": data.get("intent_type", intent),
                "target_type": target_type,
                "target_name": target_name,
                "confidence": max(min(confidence, 1.0), 0.0),
                "reason": data.get("reason", ""),
            }
        except Exception as e:
            logger.warning("LLMDecisionService.classify_intent_and_route failed: %s", e)
            return None

    async def batch_route_queries(self, queries: list[str]) -> list[dict | None]:
        """LLM 批量路由：一次调用完成所有子查询的路由决策。

        返回与 queries 等长的列表，每个元素格式同 classify_intent_and_route。
        """
        if not _llm_enabled() or not queries:
            return [None] * len(queries)
        if len(queries) == 1:
            result = await self.classify_intent_and_route(queries[0])
            return [result]

        capabilities_desc = "\n".join(
            f"- {c['name']} (类型: {c['type']}): {c['description']}\n  适用场景: {c['when_to_use']}"
            for c in CAPABILITY_REGISTRY
        )

        queries_desc = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

        system_prompt = (
            "你是一个医疗问答系统的批量意图分类与路由决策器。\n"
            "以下是系统可用的工具和Agent：\n\n"
            f"{capabilities_desc}\n\n"
            "请对每个子查询分别判断意图并选择最合适的工具/Agent。\n"
            "你必须只输出合法JSON数组，不要输出Markdown标记，不要有任何其他解释内容。\n"
            "数组长度必须与输入子查询数量一致。\n"
            "每个元素的JSON字段：\n"
            '- intent: "archive"(档案查询) | "drug"(药物相关) | "lab"(化验解读) | "general"(通用问答)\n'
            '- intent_type: "archive" | "drug_conflict" | "drug_record" | "drug_query" | "lab_report" | "general"\n'
            '- target_type: "agent" | "tool"\n'
            '- target_name: 从上面的工具/Agent列表中选择\n'
            "- confidence: 0.0~1.0\n"
            "- reason: 简短说明决策理由"
        )

        user_prompt = f"子查询列表：\n{queries_desc}\n\n请输出决策JSON数组："

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=10.0,
                max_tokens=800,
            )
            if not raw:
                return [None] * len(queries)
            import re
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            json_str = match.group(0) if match else raw
            data = json.loads(json_str)
            if not isinstance(data, list):
                return [None] * len(queries)

            valid_intents = {"archive", "drug", "lab", "general"}
            valid_targets = {c["name"] for c in CAPABILITY_REGISTRY}
            valid_types = {"agent", "tool"}

            results: list[dict | None] = []
            for item in data:
                if not isinstance(item, dict):
                    results.append(None)
                    continue
                intent = item.get("intent", "")
                target_name = item.get("target_name", "")
                target_type = item.get("target_type", "")
                confidence = float(item.get("confidence", 0.0) or 0.0)
                if intent not in valid_intents or target_name not in valid_targets or target_type not in valid_types:
                    results.append(None)
                    continue
                expected_type = next((c["type"] for c in CAPABILITY_REGISTRY if c["name"] == target_name), None)
                if expected_type and target_type != expected_type:
                    target_type = expected_type
                results.append({
                    "intent": intent,
                    "intent_type": item.get("intent_type", intent),
                    "target_type": target_type,
                    "target_name": target_name,
                    "confidence": max(min(confidence, 1.0), 0.0),
                    "reason": item.get("reason", ""),
                })

            while len(results) < len(queries):
                results.append(None)
            return results[:len(queries)]
        except Exception as e:
            logger.warning("LLMDecisionService.batch_route_queries failed: %s", e)
            return [None] * len(queries)

    async def split_queries(self, text: str) -> list[str] | None:
        """LLM 优先：将多意图输入拆分为独立子查询。"""
        if not _llm_enabled():
            return None

        system_prompt = (
            "你是一个查询拆分助手。用户可能在一条消息中包含多个独立的意图/问题。\n"
            "请将用户输入拆分为独立的子查询，每个子查询包含一个完整意图。\n"
            "你必须只输出合法JSON数组，不要输出Markdown标记，不要有任何其他解释内容。\n"
            '示例：["子查询1", "子查询2", "子查询3"]\n'
            "如果只有一个意图，返回包含单个元素的数组。"
        )

        user_prompt = f"用户输入：{text}\n\n请输出拆分结果："

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=6.0,
                max_tokens=300,
            )
            if not raw:
                return None
            match = __import__("re").search(r"\[.*\]", raw, __import__("re").DOTALL)
            json_str = match.group(0) if match else raw
            data = json.loads(json_str)
            if not isinstance(data, list):
                return None
            queries = [str(q).strip() for q in data if str(q).strip()]
            return queries if queries else None
        except Exception as e:
            logger.warning("LLMDecisionService.split_queries failed: %s", e)
            return None

    async def extract_entities(self, text: str, intent: str) -> dict | None:
        """LLM 优先：从用户输入中提取结构化实体。"""
        if not _llm_enabled():
            return None

        if intent == "drug":
            return await self._extract_drug_entities(text)
        elif intent == "lab":
            return await self._extract_lab_entities(text)
        return None

    async def _extract_drug_entities(self, text: str) -> dict | None:
        system_prompt = (
            "你是医疗信息抽取助手。请从用户输入中提取药品相关信息。\n"
            "你必须只输出合法JSON，不要输出Markdown标记，不要有任何其他解释内容。\n"
            "JSON字段：\n"
            '- drug_name_list: 药品名称数组，例如 ["阿司匹林", "布洛芬"]\n'
            '- dosage: 剂量，例如 "100mg"，未提及则为空字符串\n'
            '- frequency: 频率，例如 "每天一次"，未提及则为空字符串\n'
            '- start_date_text: 开始日期，例如 "今天"，未提及则为空字符串\n'
            '- purpose: 用药目的，未提及则为空字符串'
        )

        user_prompt = f"用户输入：{text}\n\n请输出提取结果："

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=6.0,
                max_tokens=300,
            )
            if not raw:
                return None
            match = __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL)
            json_str = match.group(0) if match else raw
            data = json.loads(json_str)
            drug_names = data.get("drug_name_list", [])
            if isinstance(drug_names, list) and drug_names:
                return {
                    "drug_name_list": [str(n).strip() for n in drug_names if str(n).strip()],
                    "dosage": str(data.get("dosage", "") or ""),
                    "frequency": str(data.get("frequency", "") or ""),
                    "start_date_text": str(data.get("start_date_text", "") or ""),
                    "purpose": str(data.get("purpose", "") or ""),
                }
            return None
        except Exception as e:
            logger.warning("LLMDecisionService._extract_drug_entities failed: %s", e)
            return None

    async def _extract_lab_entities(self, text: str) -> dict | None:
        system_prompt = (
            "你是医疗信息抽取助手。请从用户输入中提取检验指标相关信息。\n"
            "你必须只输出合法JSON，不要输出Markdown标记，不要有任何其他解释内容。\n"
            "JSON字段：\n"
            '- lab_items: 检验指标数组，每个元素包含 item_name(指标名)、test_value(数值)、unit(单位，可选)\n'
            '示例：{"lab_items": [{"item_name": "血糖", "test_value": "6.5", "unit": "mmol/L"}]}'
        )

        user_prompt = f"用户输入：{text}\n\n请输出提取结果："

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=6.0,
                max_tokens=300,
            )
            if not raw:
                return None
            match = __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL)
            json_str = match.group(0) if match else raw
            data = json.loads(json_str)
            lab_items = data.get("lab_items", [])
            if isinstance(lab_items, list) and lab_items:
                return {"lab_items": lab_items, "raw": text}
            return None
        except Exception as e:
            logger.warning("LLMDecisionService._extract_lab_entities failed: %s", e)
            return None

    async def classify_operation_type(self, text: str, history: list[dict] | None = None) -> str | None:
        """LLM 优先：判断用药记录操作类型。返回 'add'|'query'|'delete'|'general'。"""
        if not _llm_enabled():
            return None

        ctx_msgs = (history or [])[-6:]
        system_prompt = (
            "你是一个专门负责判断用户在用药记录方面意图的助手。\n"
            "你需要根据用户的最新输入以及上下文，判断用户是要：\n"
            "1. 'add'：记录、添加、补充自己吃了什么药（即使只是补充某个时间、频率、剂量等细节，也是 'add'）。\n"
            "2. 'query'：查询、查看自己的用药历史记录。\n"
            "3. 'delete'：删除自己的用药记录。\n"
            "4. 'general'：其他情况。\n"
            "请只输出一个字符串：'add', 'query', 'delete' 或者是 'general'，不要有多余字符。"
        )
        user_prompt = f"上下文记录：{ctx_msgs}\n\n当前用户输入：{text}\n请输出判断结果："

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=5.0,
                max_tokens=20,
            )
            if not raw:
                return None
            op = raw.strip().lower()
            for valid in ("add", "query", "delete", "general"):
                if valid in op:
                    return valid
            return None
        except Exception as e:
            logger.warning("LLMDecisionService.classify_operation_type failed: %s", e)
            return None

    async def extract_drug_info(self, text: str, history: list[dict] | None = None) -> dict | None:
        """LLM 优先：提取用药记录详细信息。"""
        if not _llm_enabled():
            return None

        ctx_msgs = (history or [])[-6:]
        system_prompt = (
            "你是一个医疗信息抽取助手。请从用户的最新回复和上下文中，提取用药记录信息。\n"
            "以JSON格式返回，包含以下字段：\n"
            "1. drug_name 药品名称，如果是补充信息且未提及药名，请从上下文中找到药名并填入。如果仍然找不到，填空字符串。\n"
            "2. dosage 剂量，如'100mg'，'1片'。\n"
            "3. frequency 频率，如'每天一次'，'早晚各一次'。\n"
            "4. start_date_text 用药时间，如'昨天晚上八点'、'今天中午'，不要用现在的系统时间。\n"
            "5. purpose 用药目的，如'降压'、'退烧'，未提及则填空字符串。\n"
            "如果字段没有提及并没有在上下文中，请填空字符串。\n"
            "必须且只输出合法的 JSON，不要输出 Markdown 标记，也不要有任何其他解释内容。"
        )
        user_prompt = f"对话上下文：\n{ctx_msgs}\n\n用户最新输入：{text}"

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=6.0,
                max_tokens=300,
            )
            if not raw:
                return None
            match = __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL)
            json_str = match.group(0) if match else raw
            data = json.loads(json_str)
            drug_name = str(data.get("drug_name", "") or "").strip()
            if drug_name:
                return {
                    "drug_name": drug_name,
                    "dosage": str(data.get("dosage", "") or ""),
                    "frequency": str(data.get("frequency", "") or ""),
                    "start_date_text": str(data.get("start_date_text", "") or ""),
                    "purpose": str(data.get("purpose", "") or ""),
                }
            return None
        except Exception as e:
            logger.warning("LLMDecisionService.extract_drug_info failed: %s", e)
            return None

    async def extract_drug_name_from_event(self, text: str) -> str | None:
        """LLM 优先：从用药事件文本中提取药品名称。"""
        if not _llm_enabled():
            return None

        system_prompt = (
            "请从以下文本中提取药品名称。只输出药品名称，不要输出其他内容。\n"
            "如果文本中包含多个药品，只输出第一个。如果无法识别，输出空字符串。"
        )
        user_prompt = f"文本：{text}"

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=4.0,
                max_tokens=30,
            )
            if raw:
                name = raw.strip()
                if len(name) >= 2 and len(name) <= 24:
                    return name
            return None
        except Exception as e:
            logger.warning("LLMDecisionService.extract_drug_name_from_event failed: %s", e)
            return None
