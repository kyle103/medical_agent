from __future__ import annotations

import json
from typing import Any

from app.common.logger import get_logger
from app.config.settings import settings
from app.core.llm.llm_service import LLMService

logger = get_logger(__name__)


def _decision_ctx_block(ctx: str) -> str:
    """把上一轮决策上下文拼成供路由/规划 LLM 消解指代的前置块。

    空串时返回空，不影响不涉及跨轮指代的常规调用。
    """
    ctx = (ctx or "").strip()
    if not ctx:
        return ""
    return (
        "对话上文（仅供消解“它/那药/刚才/上面提到的药”等指代与省略；"
        "决策对象是下面的“用户输入”，不要把上文当作本次需要处理的新问题）：\n"
        f"{ctx}\n\n"
    )


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

    async def classify_intent_and_route(self, text: str, ctx: str = "") -> dict | None:
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

        user_prompt = _decision_ctx_block(ctx) + f"用户输入：{text}\n\n请输出决策JSON："

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

    async def classify_route_and_extract(self, text: str, ctx: str = "") -> dict | None:
        """一次 LLM 调用同时完成：意图分类 + 路由决策 + 实体提取。

        返回格式：
        {
            "intent": "archive|drug|lab|general",
            "intent_type": "archive|drug_conflict|drug_record|lab_report|general",
            "target_type": "agent|tool",
            "target_name": "...",
            "confidence": 0.0~1.0,
            "reason": "...",
            "entities": {
                "drug_name_list": [...],
                "dosage": "...",
                "frequency": "...",
                "start_date_text": "...",
                "purpose": "...",
                "lab_items": [...]
            }
        }
        """
        if not _llm_enabled():
            return None

        capabilities_desc = "\n".join(
            f"- {c['name']} (类型: {c['type']}): {c['description']}\n  适用场景: {c['when_to_use']}"
            for c in CAPABILITY_REGISTRY
        )

        system_prompt = (
            "你是一个医疗问答系统的意图分类、路由决策与实体提取器。\n"
            "以下是系统可用的工具和Agent：\n\n"
            f"{capabilities_desc}\n\n"
            "请根据用户输入，同时完成以下任务：\n"
            "1. 判断意图并选择最合适的工具/Agent\n"
            "2. 提取相关实体信息\n\n"
            "你必须只输出合法JSON，不要输出Markdown标记，不要有任何其他解释内容。\n"
            "JSON字段：\n"
            '- intent: "archive"(档案查询) | "drug"(药物相关) | "lab"(化验解读) | "general"(通用问答)\n'
            '- intent_type: "archive" | "drug_conflict" | "drug_record" | "lab_report" | "general"\n'
            '- target_type: "agent" | "tool"\n'
            '- target_name: 从上面的工具/Agent列表中选择\n'
            "- confidence: 0.0~1.0\n"
            "- reason: 简短说明决策理由\n"
            '- is_multi_intent: true/false —— 该输入是否包含【需要不同工具/Agent 分别处理】的多个独立意图。'
            "只有当多个意图会路由到不同 target 时才为 true（如『我的血糖正常吗？布洛芬能和它一起吃吗』= lab + drug）。"
            "同一 target 能一并回答的多个小问题（如『头痛吃什么药，还要注意什么』都归 main_qa_agent）为 false。\n"
            "- entities: 提取的实体信息，格式如下：\n"
            '  - 如果意图为 drug：{"drug_name_list": ["药品1", "药品2"], "dosage": "剂量", "frequency": "频率", "start_date_text": "开始时间", "purpose": "目的"}。若当前输入用"它/这些药/上面的药/之前那种"等指代上文的药物，可在 drug_name_list 中一并补全所指药名，以便后续做冲突查询等。\n'
            '  - 如果意图为 lab：{"lab_items": [{"item_name": "指标名", "test_value": "数值", "unit": "单位"}]}\n'
            '  - 其他意图：entities 为空对象 {}'
        )

        user_prompt = _decision_ctx_block(ctx) + f"用户输入：{text}\n\n请输出决策与实体JSON："

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=10.0,
                max_tokens=500,
            )
            if not raw:
                return None
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
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

            result = {
                "intent": intent,
                "intent_type": data.get("intent_type", intent),
                "target_type": target_type,
                "target_name": target_name,
                "confidence": max(min(confidence, 1.0), 0.0),
                "reason": data.get("reason", ""),
                "entities": data.get("entities", {}),
                "is_multi_intent": bool(data.get("is_multi_intent", False)),
            }

            entities = result["entities"]
            if intent == "drug" and isinstance(entities, dict):
                drug_names = entities.get("drug_name_list", [])
                if isinstance(drug_names, list):
                    entities["drug_name_list"] = [str(n).strip() for n in drug_names if str(n).strip()]
            elif intent == "lab" and isinstance(entities, dict):
                lab_items = entities.get("lab_items", [])
                if isinstance(lab_items, list):
                    entities["lab_items"] = lab_items

            return result
        except Exception as e:
            logger.warning("LLMDecisionService.classify_route_and_extract failed: %s", e)
            return None

    async def batch_route_queries(self, queries: list[str], ctx: str = "") -> list[dict | None]:
        """LLM 批量路由：一次调用完成所有子查询的路由决策。

        返回与 queries 等长的列表，每个元素格式同 classify_intent_and_route。
        """
        if not _llm_enabled() or not queries:
            return [None] * len(queries)
        if len(queries) == 1:
            result = await self.classify_intent_and_route(queries[0], ctx=ctx)
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

        user_prompt = _decision_ctx_block(ctx) + f"子查询列表：\n{queries_desc}\n\n请输出决策JSON数组："

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

    async def batch_route_with_deps(self, queries: list[str], ctx: str = "") -> tuple[list[dict | None], list[list[str]]]:
        """一次 LLM 调用同时完成：批量路由 + 依赖关系检测。

        将原来的 batch_route_queries + _detect_dependencies_llm 合并，
        减少一次 LLM 往返。

        返回:
          (routes, deps) — routes 格式同 batch_route_queries，
          deps 为每个查询的依赖步骤编号列表。
        """
        default_routes = [None] * len(queries)
        default_deps: list[list[str]] = [[] for _ in queries]
        if not _llm_enabled() or not queries:
            return default_routes, default_deps
        if len(queries) == 1:
            result = await self.classify_intent_and_route(queries[0], ctx=ctx)
            return [result], [[]]

        capabilities_desc = "\n".join(
            f"- {c['name']} (类型: {c['type']}): {c['description']}\n  适用场景: {c['when_to_use']}"
            for c in CAPABILITY_REGISTRY
        )

        queries_desc = "\n".join(f"s{i+1}: {q}" for i, q in enumerate(queries))

        system_prompt = (
            "你是一个医疗问答系统的批量路由与依赖分析器。\n"
            "以下是系统可用的工具和Agent：\n\n"
            f"{capabilities_desc}\n\n"
            "请对每个子查询同时完成两项任务：\n"
            "1. 路由决策：判断意图并选择最合适的工具/Agent\n"
            "2. 依赖分析：判断该查询是否依赖前面查询的结果才能回答\n\n"
            "依赖的常见情形：(a) 代词/序号指代前置内容 (b) 需要前置步骤给出的药名/诊断\n"
            "(c) 对前置推荐结果的追问（效果、用量、对比）\n"
            "注意：因果叙事的连贯句子（如'因为过敏所以住院'）不应标记为依赖，它们是一个事件的完整叙述。\n\n"
            "你必须只输出合法JSON对象，不要输出Markdown标记。\n"
            "JSON格式：\n"
            "{\n"
            '  "routes": {\n'
            '    "s1": {\n'
            '      "intent": "archive|drug|lab|general",\n'
            '      "intent_type": "archive|drug_conflict|drug_record|drug_query|lab_report|general",\n'
            '      "target_type": "agent|tool",\n'
            '      "target_name": "main_qa_agent|drug_record_agent|drug_interaction|lab_report",\n'
            '      "confidence": 0.0~1.0,\n'
            '      "reason": "..."\n'
            "    }\n"
            "  },\n"
            '  "deps": {\n'
            '    "s2": ["s1"],\n'
            '    "s3": ["s1"]\n'
            "  }\n"
            "}\n"
            "s1 不可能有依赖，不出现在 deps 中。无依赖的步骤省略。"
        )

        user_prompt = _decision_ctx_block(ctx) + f"子查询列表：\n{queries_desc}\n\n请输出路由与依赖JSON："

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=12.0,
                max_tokens=900,
            )
            if not raw:
                return default_routes, default_deps

            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            json_str = match.group(0) if match else raw
            data = json.loads(json_str)

            # 解析 routes
            routes_data = data.get("routes", {}) if isinstance(data, dict) else {}
            valid_intents = {"archive", "drug", "lab", "general"}
            valid_targets = {c["name"] for c in CAPABILITY_REGISTRY}
            valid_types = {"agent", "tool"}

            routes: list[dict | None] = [None] * len(queries)
            for i in range(len(queries)):
                key = f"s{i+1}"
                item = routes_data.get(key)
                if not isinstance(item, dict):
                    continue
                intent = item.get("intent", "")
                target_name = item.get("target_name", "")
                target_type = item.get("target_type", "")
                confidence = float(item.get("confidence", 0.0) or 0.0)
                if intent not in valid_intents or target_name not in valid_targets or target_type not in valid_types:
                    continue
                expected_type = next((c["type"] for c in CAPABILITY_REGISTRY if c["name"] == target_name), None)
                if expected_type and target_type != expected_type:
                    target_type = expected_type
                routes[i] = {
                    "intent": intent,
                    "intent_type": item.get("intent_type", intent),
                    "target_type": target_type,
                    "target_name": target_name,
                    "confidence": max(min(confidence, 1.0), 0.0),
                    "reason": item.get("reason", ""),
                }

            # 解析 dependencies
            deps_data = data.get("deps", {}) if isinstance(data, dict) else {}
            all_ids = {f"s{i+1}" for i in range(len(queries))}
            deps: list[list[str]] = [[] for _ in queries]
            for key, dep_list in deps_data.items():
                if not isinstance(dep_list, list):
                    continue
                idx = int(key[1:]) - 1 if key.startswith("s") and key[1:].isdigit() else -1
                if 0 <= idx < len(queries):
                    valid = [d for d in dep_list if isinstance(d, str) and d in all_ids]
                    deps[idx] = valid

            return routes, deps
        except Exception as e:
            logger.warning("LLMDecisionService.batch_route_with_deps failed: %s", e)
            return default_routes, default_deps

    async def split_route_deps(
        self, text: str, ctx: str = ""
    ) -> tuple[list[str] | None, list[dict | None] | None, list[list[str]] | None]:
        """一次 LLM 调用同时完成：多意图拆分 + 逐个路由 + 依赖判定。

        把原来的 split_queries + batch_route_with_deps 两次往返合并为一次，
        并且拆分与路由在同一上下文中完成，边界与路由结果天然自洽。

        返回 (sub_queries, routes, deps)；任一步失败返回 None，由调用方回落旧两步链路。
        """
        if not _llm_enabled() or not (text or "").strip():
            return None, None, None

        capabilities_desc = "\n".join(
            f"- {c['name']} (类型: {c['type']}): {c['description']}\n  适用场景: {c['when_to_use']}"
            for c in CAPABILITY_REGISTRY
        )

        system_prompt = (
            "你是一个医疗问答系统的规划器，需要一次性完成三件事：拆分、路由、依赖分析。\n"
            "以下是系统可用的工具和Agent：\n\n"
            f"{capabilities_desc}\n\n"
            "【任务1 拆分】把用户输入拆成独立子查询。只有需要【不同工具/Agent】分别处理的意图才拆；"
            "因果/叙事连贯的句子（『因为A所以B』）、补充同一事件的从句不要拆。不确定时倾向不拆。\n"
            "【任务2 路由】为每个子查询选择最合适的工具/Agent。\n"
            "【任务3 依赖】判断子查询是否需要前置子查询的结果才能回答：\n"
            "  (a) 代词/序号指代前置内容 (b) 需要前置步骤给出的药名/诊断 (c) 对前置推荐结果的追问。\n"
            "  注意：只写库不产出可读结论的步骤（如用药记录写入）不应被依赖。\n\n"
            "你必须只输出合法JSON对象，不要输出Markdown标记，不要有任何其他解释内容。\n"
            "JSON格式：\n"
            "{\n"
            '  "sub_queries": ["子查询1", "子查询2"],\n'
            '  "routes": {\n'
            '    "s1": {"intent": "archive|drug|lab|general", '
            '"intent_type": "archive|drug_conflict|drug_record|drug_query|lab_report|general", '
            '"target_type": "agent|tool", "target_name": "...", "confidence": 0.0~1.0, "reason": "..."}\n'
            "  },\n"
            '  "deps": {"s2": ["s1"]}\n'
            "}\n"
            "s1 不可能有依赖，不出现在 deps 中。无依赖的步骤省略。"
        )

        user_prompt = _decision_ctx_block(ctx) + f"用户输入：{text}\n\n请输出拆分+路由+依赖JSON："

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=12.0,
                max_tokens=900,
            )
            if not raw:
                return None, None, None

            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group(0) if match else raw)
            if not isinstance(data, dict):
                return None, None, None

            # --- 解析 sub_queries ---
            raw_queries = data.get("sub_queries")
            if not isinstance(raw_queries, list):
                return None, None, None
            sub_queries = [str(q).strip() for q in raw_queries if str(q).strip()]
            if not sub_queries:
                return None, None, None

            n = len(sub_queries)
            # --- 解析 routes ---
            routes_data = data.get("routes", {})
            valid_intents = {"archive", "drug", "lab", "general"}
            valid_targets = {c["name"] for c in CAPABILITY_REGISTRY}
            valid_types = {"agent", "tool"}

            routes: list[dict | None] = [None] * n
            if isinstance(routes_data, dict):
                for i in range(n):
                    item = routes_data.get(f"s{i+1}")
                    if not isinstance(item, dict):
                        continue
                    intent = item.get("intent", "")
                    target_name = item.get("target_name", "")
                    target_type = item.get("target_type", "")
                    if intent not in valid_intents or target_name not in valid_targets or target_type not in valid_types:
                        continue
                    expected_type = next((c["type"] for c in CAPABILITY_REGISTRY if c["name"] == target_name), None)
                    if expected_type and target_type != expected_type:
                        target_type = expected_type
                    routes[i] = {
                        "intent": intent,
                        "intent_type": item.get("intent_type", intent),
                        "target_type": target_type,
                        "target_name": target_name,
                        "confidence": max(min(float(item.get("confidence", 0.0) or 0.0), 1.0), 0.0),
                        "reason": item.get("reason", ""),
                    }

            # --- 解析 deps ---
            deps: list[list[str]] = [[] for _ in range(n)]
            deps_data = data.get("deps", {})
            all_ids = {f"s{i+1}" for i in range(n)}
            if isinstance(deps_data, dict):
                for key, dep_list in deps_data.items():
                    if not isinstance(dep_list, list):
                        continue
                    idx = int(key[1:]) - 1 if key.startswith("s") and key[1:].isdigit() else -1
                    if 0 <= idx < n:
                        deps[idx] = [d for d in dep_list if isinstance(d, str) and d in all_ids and d != f"s{idx+1}"]

            return sub_queries, routes, deps
        except Exception as e:
            logger.warning("LLMDecisionService.split_route_deps failed: %s", e)
            return None, None, None

    async def replan_failed_steps(
        self,
        original_input: str,
        failed_steps: list[dict],
        completed_steps: list[dict],
        ctx: str = "",
    ) -> list[dict] | None:
        """基于失败原因重新规划：让 LLM 决定「重试 / 改写 / 换目标 / 放弃」。

        避免把失败步骤原样重放——确定性失败（药名匹配不到、参数缺失）重跑必然再失败。

        failed_steps 元素: {"step_id", "query", "target_name", "error_msg"}
        completed_steps 元素: {"step_id", "query", "target_name", "summary"}

        返回动作列表: [{"step_id", "action": "retry|rewrite|reroute|drop", "query", "target_name", "reason"}]
        失败返回 None，由调用方回落为原样重放。
        """
        if not _llm_enabled() or not failed_steps:
            return None

        capabilities_desc = "\n".join(
            f"- {c['name']} (类型: {c['type']}): {c['description']}\n  适用场景: {c['when_to_use']}"
            for c in CAPABILITY_REGISTRY
        )

        failed_desc = "\n".join(
            f"- {s['step_id']}: 目标={s.get('target_name','')} 问题={s.get('query','')}\n  失败原因={s.get('error_msg','未知')}"
            for s in failed_steps
        )
        done_desc = "\n".join(
            f"- {s['step_id']}: 目标={s.get('target_name','')} 问题={s.get('query','')}\n  结果摘要={s.get('summary','')}"
            for s in completed_steps
        ) or "（无已完成步骤）"

        system_prompt = (
            "你是一个医疗问答系统的重规划器。部分执行步骤失败了，请针对【失败的步骤】给出修正方案。\n"
            "以下是系统可用的工具和Agent：\n\n"
            f"{capabilities_desc}\n\n"
            "对每个失败步骤，从四种动作里选一个：\n"
            '1. "retry"   —— 失败是偶发的（超时/网络），原样重试即可\n'
            '2. "rewrite" —— 失败是因为问题表述不清/缺少必要信息（如没给出药名、指标数值），'
            "请改写 query 使其可被处理（可结合已完成步骤的结果补全信息）\n"
            '3. "reroute" —— 失败是因为选错了工具/Agent，请改选更合适的 target_name\n'
            '4. "drop"    —— 该步骤无法完成且无关紧要，放弃它（不要为了让流程走通而伪造信息）\n\n'
            "重要约束：\n"
            "- 只处理下面列出的失败步骤，不要新增、不要改动已完成的步骤\n"
            "- 医疗场景下宁可 drop 也不要猜测：拿不到药品名就 drop，不要编造药名\n"
            "- rewrite 时必须保留原始语义，不能改变用户的问题\n\n"
            "你必须只输出合法JSON，不要输出Markdown标记。\n"
            'JSON格式：{"actions": [{"step_id": "s2", "action": "rewrite", '
            '"query": "改写后的问题", "target_name": "...", "reason": "..."}]}'
        )

        user_prompt = (
            _decision_ctx_block(ctx)
            + f"用户原始问题：{original_input}\n\n"
            f"已完成的步骤：\n{done_desc}\n\n"
            f"失败的步骤：\n{failed_desc}\n\n请输出修正动作JSON："
        )

        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=8.0,
                max_tokens=500,
            )
            if not raw:
                return None
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group(0) if match else raw)
            actions = data.get("actions") if isinstance(data, dict) else None
            if not isinstance(actions, list):
                return None

            valid_targets = {c["name"] for c in CAPABILITY_REGISTRY}
            valid_actions = {"retry", "rewrite", "reroute", "drop"}
            allowed_ids = {s["step_id"] for s in failed_steps}

            out: list[dict] = []
            for a in actions:
                if not isinstance(a, dict):
                    continue
                step_id = a.get("step_id", "")
                action = a.get("action", "")
                if step_id not in allowed_ids or action not in valid_actions:
                    continue
                target_name = a.get("target_name", "")
                if action == "reroute" and target_name not in valid_targets:
                    action = "drop"  # 目标非法 → 退化为放弃，避免路由到不存在的执行体
                out.append({
                    "step_id": step_id,
                    "action": action,
                    "query": str(a.get("query", "")).strip(),
                    "target_name": target_name if target_name in valid_targets else "",
                    "reason": str(a.get("reason", "")).strip(),
                })
            return out or None
        except Exception as e:
            logger.warning("LLMDecisionService.replan_failed_steps failed: %s", e)
            return None

    async def split_queries(self, text: str) -> list[str] | None:
        """LLM 优先：将多意图输入拆分为独立子查询。"""
        if not _llm_enabled():
            return None

        system_prompt = (
            "你是一个查询拆分助手。用户可能在一条消息中包含多个独立的意图/问题。\n"
            "请将用户输入拆分为独立的子查询，每个子查询包含一个完整意图。\n"
            "重要拆分原则：\n"
            "1. 只有真正独立的问题才拆分——比如不同主题的多个问题。\n"
            "2. 不要拆分因果/叙事连贯的句子：'因为A所以B'/'A导致B'/'A，结果B' 是整个事件的叙述，不应拆分。\n"
            "3. 逗号/逗号连接的从句如果是在补充说明同一事件，不要拆开。\n"
            "4. 当用户陈述一个既有原因又有结果的个人经历时（如'我对X过敏，吃了X住院了'），保持为一个查询。\n"
            "5. 不确定时，倾向于不拆分（保持原样）。\n"
            "你必须只输出合法JSON数组，不要输出Markdown标记，不要有任何其他解释内容。\n"
            '示例：["子查询1", "子查询2"]\n'
            "如果只有一个意图或无法确定是否应拆分，返回包含单个原始输入的数组。"
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
            "1. 'add'：记录、添加自己新的一次用药（如'我吃了布洛芬'、'记录一下今天吃了阿司匹林'）。\n"
            "2. 'update'：修改/补充已有的用药记录（如'布洛芬的剂量改成100mg'、'补一下布洛芬的时间'、'把昨天的记录改成一天两次'）。\n"
            "3. 'query'：查询、查看自己的用药历史记录。\n"
            "4. 'delete'：删除自己的用药记录。\n"
            "5. 'general'：其他情况。\n"
            "请只输出一个字符串：'add', 'update', 'query', 'delete' 或 'general'，不要有多余字符。"
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
            for valid in ("add", "update", "query", "delete", "general"):
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
