from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.common.logger import get_logger
from app.config.settings import settings
# CAPABILITY_REGISTRY 只做 re-export，保持 planner_agent 等既有导入路径不变
from app.core.agent.capabilities import CAPABILITY_REGISTRY  # noqa: F401
from app.core.agent.capabilities import expected_target_type, valid_target_names
from app.core.agent.schemas import (
    BatchRouteDecision,
    BatchRouteWithDeps,
    DrugEntities,
    DrugRecordInfo,
    LabEntities,
    QuerySplit,
    ReplanResult,
    RouteAndEntities,
    RouteDecision,
    SplitRouteDeps,
)
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


# CAPABILITY_REGISTRY 已迁至 app/core/agent/capabilities.py，并在文件头 re-export，
# 以保证 planner_agent 等既有调用方的导入路径不变。


def _capabilities_block() -> str:
    """把注册表渲染成提示词里的能力清单（多个决策方法共用）。"""
    return "\n".join(
        f"- {c['name']} (类型: {c['type']}): {c['description']}\n  适用场景: {c['when_to_use']}"
        for c in CAPABILITY_REGISTRY
    )


def _brief_validation_error(e: ValidationError, limit: int = 4) -> str:
    """把 ValidationError 压成"字段路径: 原因"的一行，用于日志。"""
    parts = []
    for x in e.errors()[:limit]:
        loc = ".".join(str(p) for p in (x.get("loc") or ())) or "(根)"
        parts.append(f"{loc}: {x.get('msg', '')}")
    return "; ".join(parts)


def _step_id_to_index(step_id: str, n: int) -> int:
    """把 `"s1"` / `"s2"` 解析成 0-based 下标；格式非法或越界返回 -1。

    改造前这些 id 是字典的**键**——键不会重复，也天然落在循环范围内。
    现在它们是数组元素的**字段**，"重复"和"越界"成了新的可能输入，
    必须显式收敛，否则 `routes[idx] = ...` 会静默写到错误的位置。
    """
    s = (step_id or "").strip()
    if not s.startswith("s") or not s[1:].isdigit():
        return -1
    idx = int(s[1:]) - 1
    return idx if 0 <= idx < n else -1


def _strict_route(item: Any) -> tuple[dict | None, str]:
    """把"结构已校验、枚举故意放宽"的路由元素升级为严格决策。

    返回 `(决策 dict 或 None, 失败原因)`。

    为什么要再走一遍 `RouteDecision.model_validate`：元素类型（`BatchRouteItem` /
    `StepRouteItem`）只约束结构——`intent` / `target_type` 是裸 `str`，
    `target_name` 也没查注册表。这一步补上枚举与注册表校验。

    **只挑 `RouteDecision` 认识的字段再送进去**：`StepRouteItem` 比它多一个
    `step_id`，而 `RouteDecision` 是 `extra="forbid"`——整份 `model_dump()` 直接塞
    会因为这个多出来的字段被判非法，导致**所有**带 step_id 的元素升格失败。
    （这个坑是 `test_batch_route_with_deps_parses_arrays` 抓出来的。）

    失败**只影响该元素**（返回 None），这是批量场景下的"逐项容错"：
    不能因为一条非法退化成整批失败。
    """
    payload = {k: v for k, v in item.model_dump().items() if k in RouteDecision.model_fields}
    try:
        strict = RouteDecision.model_validate(payload)
    except ValidationError as e:
        return None, _brief_validation_error(e)
    return _route_decision_to_dict(strict), ""


def _route_decision_to_dict(d: RouteDecision) -> dict:
    """把 schema 对象转成既有契约的 dict。

    保留了改造前 `classify_intent_and_route` 的两处后处理，避免语义漂移：
    - `target_type` 以注册表为准纠正（模型写错类型时不该丢弃整条决策）
    - `confidence` 收敛到 [0, 1]（schema 只校验类型，不拒绝越界值——
      把越界值判为非法会让一次本来可用的决策变成丢弃）

    另有一处**有意收紧**：`intent_type` 为空串时回退到 `intent`。
    改造前是 `data.get("intent_type", intent)`，"字段存在但为空串"时会返回空串；
    而下游多处按 `.get("intent_type", "general")` 取值，空串会被当成有效值一路传下去。
    """
    target_type = d.target_type
    expected = expected_target_type(d.target_name)
    if expected and target_type != expected:
        target_type = expected
    return {
        "intent": d.intent,
        "intent_type": d.intent_type or d.intent,
        "target_type": target_type,
        "target_name": d.target_name,
        "confidence": max(min(d.confidence, 1.0), 0.0),
        "reason": d.reason,
    }


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

        system_prompt = (
            "你是一个医疗问答系统的意图分类与路由决策器。\n"
            "以下是系统可用的工具和Agent：\n\n"
            f"{_capabilities_block()}\n\n"
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
            # schema 负责结构与枚举，校验失败会带字段路径回灌重试一次，仍失败才返回 None。
            # 这替代了原先"正则捞 JSON → 手写 valid_xxx 集合 → 静默 return None"的三段式：
            # 失败不再是黑盒，且拿到了一次自愈机会。
            decision = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=RouteDecision,
                timeout_s=8.0,
                max_tokens=300,
            )
            if decision is None:
                return None
            return _route_decision_to_dict(decision)
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

        system_prompt = (
            "你是一个医疗问答系统的意图分类、路由决策与实体提取器。\n"
            "以下是系统可用的工具和Agent：\n\n"
            f"{_capabilities_block()}\n\n"
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
            "- 实体字段（**平铺在同一个 JSON 里**，不要再嵌套一层 entities 对象）：\n"
            "  - 意图为 drug 时填：drug_name_list（药品名数组）、dosage（剂量）、frequency（频率）、"
            "start_date_text（开始时间）、purpose（目的）。"
            '若当前输入用"它/这些药/上面的药/之前那种"等指代上文的药物，'
            "可在 drug_name_list 中一并补全所指药名，以便后续做冲突查询等。\n"
            "  - 意图为 lab 时填：lab_items（数组，元素含 item_name / test_value / unit）。\n"
            "  - 其他意图：这些字段留空（数组给 []，字符串给 \"\"）。"
        )

        user_prompt = _decision_ctx_block(ctx) + f"用户输入：{text}\n\n请输出决策与实体JSON："

        try:
            # max_tokens 由 500 提到 700：schema 要求实体字段全部出现（strict 模式把它们都列进 required），
            # 输出比原来"缺字段也行"的形态更长，留出余量避免截断后再触发重试。
            decision = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=RouteAndEntities,
                timeout_s=10.0,
                max_tokens=700,
            )
            if decision is None:
                return None

            result = _route_decision_to_dict(decision)
            result["is_multi_intent"] = bool(decision.is_multi_intent)
            # 把拍平的实体字段重新组装成既有契约的 entities 字典（下游按 intent 取用）。
            if decision.intent == "drug":
                result["entities"] = {
                    "drug_name_list": [
                        str(n).strip() for n in decision.drug_name_list if str(n).strip()
                    ],
                    "dosage": decision.dosage,
                    "frequency": decision.frequency,
                    "start_date_text": decision.start_date_text,
                    "purpose": decision.purpose,
                }
            elif decision.intent == "lab":
                result["entities"] = {"lab_items": [i.model_dump() for i in decision.lab_items]}
            else:
                # 文档约定：非 drug/lab 意图 entities 为空对象。
                # 改造前这里是"模型给什么就透传什么"，与文档不符；现按文档收敛。
                result["entities"] = {}
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

        queries_desc = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))

        system_prompt = (
            "你是一个医疗问答系统的批量意图分类与路由决策器。\n"
            "以下是系统可用的工具和Agent：\n\n"
            f"{_capabilities_block()}\n\n"
            "请对每个子查询分别判断意图并选择最合适的工具/Agent。\n"
            "你必须只输出一个合法JSON对象，不要输出Markdown标记，不要有任何其他解释内容。\n"
            '输出形如 {"decisions": [...]}，decisions 数组长度必须与输入子查询数量一致、顺序一一对应。\n'
            "每个元素的JSON字段：\n"
            '- intent: "archive"(档案查询) | "drug"(药物相关) | "lab"(化验解读) | "general"(通用问答)\n'
            '- intent_type: "archive" | "drug_conflict" | "drug_record" | "drug_query" | "lab_report" | "general"\n'
            '- target_type: "agent" | "tool"\n'
            '- target_name: 从上面的工具/Agent列表中选择\n'
            "- confidence: 0.0~1.0\n"
            "- reason: 简短说明决策理由"
        )

        user_prompt = _decision_ctx_block(ctx) + f"子查询列表：\n{queries_desc}\n\n请输出决策JSON："

        try:
            batch = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=BatchRouteDecision,
                timeout_s=10.0,
                max_tokens=800,
            )
            results: list[dict | None] = [None] * len(queries)
            if batch is None:
                return results

            # 逐项严格校验：schema 层只保证结构，枚举与注册表在这里逐条把关。
            # 这样单条不合法只影响该条（与改造前的逐项容错一致），但这次会留下日志。
            for idx, item in enumerate(batch.decisions[: len(queries)]):
                decision, why = _strict_route(item)
                if decision is None:
                    logger.info(
                        "batch_route_queries 第 %d 条不合法，置 None（其余照常返回）: %s",
                        idx + 1,
                        why,
                    )
                    continue
                results[idx] = decision
            return results
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
            '  "routes": [\n'
            '    {"step_id": "s1", "intent": "archive|drug|lab|general", '
            '"intent_type": "archive|drug_conflict|drug_record|drug_query|lab_report|general", '
            '"target_type": "agent|tool", '
            '"target_name": "main_qa_agent|drug_record_agent|drug_interaction|lab_report", '
            '"confidence": 0.0, "reason": "..."}\n'
            "  ],\n"
            '  "deps": [\n'
            '    {"step_id": "s2", "depends_on": ["s1"]}\n'
            "  ]\n"
            "}\n"
            'routes 必须为每个子查询给出恰好一条，step_id 依次为 "s1" ~ "sN"，与输入编号一一对应。\n'
            "s1 不可能有依赖，不出现在 deps 中。无依赖的步骤省略。"
        )

        user_prompt = _decision_ctx_block(ctx) + f"子查询列表：\n{queries_desc}\n\n请输出路由与依赖JSON："

        try:
            # 输出契约由 BatchRouteWithDeps 固定。注意这里把 routes/deps 从
            # "以步骤 id 为键的字典"改成了"带显式 step_id 的数组"——
            # strict json_schema 表达不了动态键对象（schemas.py 顶部约束 1）。
            # max_tokens 900→1100：schema 要求每个元素字段齐全，输出比原来更长。
            batch = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=BatchRouteWithDeps,
                timeout_s=12.0,
                max_tokens=1100,
            )
            if batch is None:
                return default_routes, default_deps

            n = len(queries)

            # 逐条严格校验：schema 只保证结构（且故意放宽枚举），
            # 注册表与枚举在这里逐条把关，单条不合法只影响该条。
            routes: list[dict | None] = [None] * n
            for item in batch.routes:
                idx = _step_id_to_index(item.step_id, n)
                if idx < 0:
                    logger.info(
                        "batch_route_with_deps 忽略无法识别的 step_id=%r（应为 s1~s%d）",
                        item.step_id,
                        n,
                    )
                    continue
                decision, why = _strict_route(item)
                if decision is None:
                    logger.info(
                        "batch_route_with_deps %s 不合法，置 None（其余照常返回）: %s",
                        item.step_id,
                        why,
                    )
                    continue
                routes[idx] = decision

            # 依赖解析：只保留指向合法步骤 id 的项（与改造前一致）
            all_ids = {f"s{i+1}" for i in range(n)}
            deps: list[list[str]] = [[] for _ in range(n)]
            for d in batch.deps:
                idx = _step_id_to_index(d.step_id, n)
                if idx < 0:
                    continue
                # 注意：这里**不**排除自依赖，而 split_route_deps 会排除。
                # 这是改造前就存在的不一致，两者各自保持原样，不要顺手统一。
                deps[idx] = [x for x in d.depends_on if x in all_ids]

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
            '  "routes": [\n'
            '    {"step_id": "s1", "intent": "archive|drug|lab|general", '
            '"intent_type": "archive|drug_conflict|drug_record|drug_query|lab_report|general", '
            '"target_type": "agent|tool", "target_name": "...", "confidence": 0.0, "reason": "..."}\n'
            "  ],\n"
            '  "deps": [\n'
            '    {"step_id": "s2", "depends_on": ["s1"]}\n'
            "  ]\n"
            "}\n"
            'routes 必须为每个子查询给出恰好一条，step_id 依次为 "s1" ~ "sN"。\n'
            "s1 不可能有依赖，不出现在 deps 中。无依赖的步骤省略。"
        )

        user_prompt = _decision_ctx_block(ctx) + f"用户输入：{text}\n\n请输出拆分+路由+依赖JSON："

        try:
            # 与 batch_route_with_deps 同因：动态键字典改成带显式 step_id 的数组。
            # max_tokens 900→1200：schema 要求每个元素字段齐全，输出比原来长。
            plan = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=SplitRouteDeps,
                timeout_s=12.0,
                max_tokens=1200,
            )
            if plan is None:
                return None, None, None

            sub_queries = [q.strip() for q in plan.sub_queries if q.strip()]
            if not sub_queries:
                return None, None, None

            n = len(sub_queries)

            # --- 路由：逐条严格校验，单条不合法只影响该条 ---
            routes: list[dict | None] = [None] * n
            for item in plan.routes:
                idx = _step_id_to_index(item.step_id, n)
                if idx < 0:
                    logger.info(
                        "split_route_deps 忽略无法识别的 step_id=%r（应为 s1~s%d）", item.step_id, n
                    )
                    continue
                decision, why = _strict_route(item)
                if decision is None:
                    logger.info(
                        "split_route_deps %s 不合法，置 None（其余照常返回）: %s", item.step_id, why
                    )
                    continue
                routes[idx] = decision

            # --- 依赖：只保留指向合法步骤的项 ---
            all_ids = {f"s{i+1}" for i in range(n)}
            deps: list[list[str]] = [[] for _ in range(n)]
            for d in plan.deps:
                idx = _step_id_to_index(d.step_id, n)
                if idx < 0:
                    continue
                # 这里排除自依赖（batch_route_with_deps 不排除——改造前就是如此，别统一）
                deps[idx] = [x for x in d.depends_on if x in all_ids and x != f"s{idx+1}"]

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
            # ReplanAction.action 故意是裸 str 而非 `Literal[...]`：非法动作要**逐条过滤**，
            # 用 Literal 会让一条非法导致整批校验失败（与 BatchRouteItem 同一套理由）。
            plan = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=ReplanResult,
                timeout_s=8.0,
                max_tokens=600,
            )
            if plan is None:
                return None

            valid_targets = valid_target_names()
            valid_actions = {"retry", "rewrite", "reroute", "drop"}
            allowed_ids = {s["step_id"] for s in failed_steps}

            out: list[dict] = []
            for a in plan.actions:
                step_id = a.step_id
                action = a.action
                if step_id not in allowed_ids or action not in valid_actions:
                    # 逐条容错：非法的单条直接丢弃，不影响其余（与改造前一致）
                    continue
                if action == "reroute" and a.target_name not in valid_targets:
                    action = "drop"  # 目标非法 → 退化为放弃，避免路由到不存在的执行体
                out.append({
                    "step_id": step_id,
                    "action": action,
                    "query": a.query.strip(),
                    "target_name": a.target_name if a.target_name in valid_targets else "",
                    "reason": a.reason.strip(),
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
            "你必须只输出一个合法JSON对象，不要输出Markdown标记，不要有任何其他解释内容。\n"
            '示例：{"queries": ["子查询1", "子查询2"]}\n'
            "如果只有一个意图或无法确定是否应拆分，返回只含单个原始输入的对象。"
        )

        user_prompt = f"用户输入：{text}\n\n请输出拆分结果："

        try:
            # 改造前是**数组根** `["q1","q2"]`；strict json_schema 要求根类型是 object，
            # 数组根会被拒 → 包一层 queries（上面的提示词已同步改）。
            split = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=QuerySplit,
                timeout_s=6.0,
                max_tokens=400,
            )
            if split is None:
                return None
            queries = [q.strip() for q in split.queries if q.strip()]
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
            ent = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=DrugEntities,
                timeout_s=6.0,
                max_tokens=400,
            )
            if ent is None:
                return None
            names = [n.strip() for n in ent.drug_name_list if n.strip()]
            if not names:
                # 改造前语义：「一个药名都拿不到」= 抽取失败，返回 None 让调用方走兜底
                return None
            return {
                "drug_name_list": names,
                "dosage": ent.dosage,
                "frequency": ent.frequency,
                "start_date_text": ent.start_date_text,
                "purpose": ent.purpose,
            }
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
            ent = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=LabEntities,
                timeout_s=6.0,
                max_tokens=500,
            )
            if ent is None:
                return None
            if not ent.lab_items:
                return None
            # `raw` 字段是既有契约的一部分（调用方会读），必须保留。
            # 元素统一 model_dump 成 item_name/test_value/unit 三个字段：
            # 改造前是"模型给什么就透传什么"，缺 unit 或夹带杂字段都会漏到下游。
            return {"lab_items": [i.model_dump() for i in ent.lab_items], "raw": text}
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
            info = await self.llm.chat_completion_json(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=DrugRecordInfo,
                timeout_s=6.0,
                max_tokens=400,
            )
            if info is None:
                return None
            drug_name = info.drug_name.strip()
            if not drug_name:
                # 拿不到药名就没有可记录的内容 → None（改造前语义）
                return None
            return {
                "drug_name": drug_name,
                "dosage": info.dosage,
                "frequency": info.frequency,
                "start_date_text": info.start_date_text,
                "purpose": info.purpose,
            }
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
