from __future__ import annotations

import json
from typing import Any

from app.common.logger import get_logger
from app.config.settings import settings
from app.core.agent.intent_classifier import IntentClassifier
from app.core.agent.llm_decision_service import LLMDecisionService
from app.core.agent.state import ExecutionPlan, PlanStep
from app.core.llm.llm_service import LLMService
from app.core.skills.drug_record_state_machine import DrugRecordPhase, DrugRecordStateMachine

logger = get_logger(__name__)

MAX_REPLAN = 2

AGENT_TARGETS = {"drug_record_agent", "main_qa_agent"}
TOOL_TARGETS = {"drug_interaction", "lab_report"}

# ---- dependency detection: rule patterns ----

_CONTINUATION_PRONOUNS = [
    "它", "这", "那个", "上面", "刚才", "之前", "继续", "然后", "还要", "还用", "还需要",
    "其中", "哪些", "这些", "那些", "哪种", "第一个", "第二个", "第一种", "第二种",
]

_CONDITIONAL_REF_PATTERNS = [
    "如果是", "如果这样", "这样的话", "那这样的话", "如果是这样",
    "那就是说", "那么", "那样的话", "如果是真的",
]

_EVALUATION_PATTERNS = [
    "怎么样", "有用吗", "有效吗", "管用吗", "安全吗", "副作用",
    "哪个更好", "哪个更", "哪种更", "区别", "对比",
]

_FOLLOWUP_PATTERNS = [
    "剂量", "怎么吃", "怎么服用", "用量", "吃多少", "多久",
    "需要注意什么", "有什么注意", "禁忌", "不能和", "不能跟",
]

_IMPLICIT_RECOMMENDATION = [
    "可以吃什么药", "吃什么药", "用什么药", "有什么药", "该吃什么", "要吃什么", "吃什么好",
    "推荐", "建议用什么", "能用什么",
]

_IMPLICIT_CAUSE = [
    "原因", "症状", "怎么办", "腹泻", "发烧", "咳嗽", "头痛", "疼痛",
    "感冒", "炎症", "感染", "什么病", "得了", "患有", "诊断",
]

_DRUG_CONFLICT_KEYWORDS = [
    "一起吃", "同服", "相互作用", "配伍", "冲突", "禁忌", "能不能一起", "可以一起",
    "同时服用", "一起用", "混合", "并用",
]

_RESULT_REF_KEYWORDS = [
    "其中哪些药", "这些药", "那些药", "上面的药", "推荐的药", "上面提到的药",
    "以上", "上述", "前面", "提到的",
]

_DEPENDENCY_HINT = ";".join(
    _CONTINUATION_PRONOUNS + _EVALUATION_PATTERNS + _FOLLOWUP_PATTERNS
    + _RESULT_REF_KEYWORDS + _DRUG_CONFLICT_KEYWORDS
)


def _quick_intent_classify(text: str) -> str:
    """快速规则意图分类：用于 LLM 超时后的兜底路由。

    仅依赖关键词匹配，不需要 LLM 调用。
    返回 intent 值：drug / lab / archive / general

    重要：如果查询混合了医学症状描述和用药问题，返回 general
    （由 main_qa_agent 统一处理），避免单一工具丢失其他信息。
    """
    t = (text or "").strip()

    _DRUG_SIGNALS = [
        "冲突", "相互作用", "一起吃", "同服", "配伍", "禁忌",
        "能不能一起", "可以一起", "同时服用",
        "吃什么药", "用什么药", "止痛药", "退烧药", "消炎药",
        "降压药", "降糖药", "能吃什么", "该吃什么", "要吃什么",
        "吃了", "服用", "用药记录", "添加", "剂量", "mg", "毫克",
        "用药", "处方", "忌口", "副作用", "药",
    ]
    _MEDICAL_SYMPTOM_SIGNALS = [
        "病", "症", "疼", "痛", "头痛", "发烧", "发热", "咳嗽", "感冒",
        "发炎", "感染", "恶心", "呕吐", "头晕", "乏力", "胸闷", "心慌",
        "气短", "水肿", "出血", "皮疹", "瘙痒", "红肿", "溃疡",
        "是不是", "可能是", "请问", "会不会是", "需不需要",
        "高血压", "糖尿病", "哮喘", "过敏", "腹泻", "便秘",
    ]
    _LAB_SIGNALS = [
        "检查", "化验", "检验", "指标", "CT", "MRI", "X光", "B超",
        "报告", "化验单", "体检", "血常规", "尿常规",
    ]
    _ARCHIVE_SIGNALS = ["档案", "病历", "历史记录", "就诊记录"]

    has_drug = any(k in t for k in _DRUG_SIGNALS)
    has_medical = any(k in t for k in _MEDICAL_SYMPTOM_SIGNALS)
    has_lab = any(k in t for k in _LAB_SIGNALS)
    has_archive = any(k in t for k in _ARCHIVE_SIGNALS)

    # 混合查询：既有症状描述又有用药问题 → general（避免单一工具丢失信息）
    if has_drug and has_medical:
        return "general"

    # 纯药物查询或冲突检查
    if has_drug:
        return "drug"

    # 化验/检查
    if has_lab:
        return "lab"

    # 档案查询
    if has_archive:
        return "archive"

    # 有医学症状但无用药/检查 → general
    if has_medical:
        return "general"

    return "general"


def _route_by_intent_and_text(state: dict) -> dict:
    intent = (state.get("intent") or "").strip().lower()
    text = state.get("user_input", "").strip()
    entities = state.get("extract_entities") or {}
    drug_names = entities.get("drug_name_list") if isinstance(entities, dict) else []

    if intent == "lab":
        return {"target_type": "tool", "target_name": "lab_report", "intent_type": "lab_report", "confidence": float(state.get("intent_confidence") or 0.9), "reason": "route by intent=lab"}
    if intent == "archive":
        return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "archive", "confidence": float(state.get("intent_confidence") or 0.9), "reason": "route by intent=archive"}
    if intent == "general":
        return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "general", "confidence": float(state.get("intent_confidence") or 0.8), "reason": "route by intent=general"}

    if intent == "drug":
        conflict_keywords = ["相互作用", "一起吃", "同服", "配伍", "冲突", "禁忌", "能不能一起", "可以一起"]
        record_keywords = ["记录", "添加用药", "我吃了", "我服用", "我用了", "用药记录", "剂量", "频次", "每天", "每次", "mg", "毫克"]
        delete_keywords = ["删除", "移除", "清空"]
        query_drug_keywords = ["可以吃什么药", "吃什么药", "能用什么药", "有什么药", "该吃什么", "要吃什么", "吃什么好"]
        allergy_keywords = ["过敏"]

        is_conflict = any(k in text for k in conflict_keywords) or ("药" in text and "一起" in text)
        is_record = any(k in text for k in record_keywords) and not any(k in text for k in query_drug_keywords)
        is_delete = any(k in text for k in delete_keywords)
        is_query_drug = any(k in text for k in query_drug_keywords)
        is_allergy = any(k in text for k in allergy_keywords)

        # 过敏声明（如"我对XX过敏"）不是用药记录，路由到通用问答
        if is_allergy and is_record:
            return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "allergy_record", "confidence": 0.9, "reason": "route by allergy keywords (not drug record)"}

        if is_conflict and not is_record and not is_delete:
            return {"target_type": "tool", "target_name": "drug_interaction", "intent_type": "drug_conflict", "confidence": float(state.get("intent_confidence") or 0.85), "reason": "route by drug conflict keywords"}
        if is_query_drug:
            return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "drug_query", "confidence": float(state.get("intent_confidence") or 0.85), "reason": "route by drug query keywords"}
        if (is_record or is_delete) and not is_conflict:
            return {"target_type": "agent", "target_name": "drug_record_agent", "intent_type": "drug_record", "confidence": float(state.get("intent_confidence") or 0.85), "reason": "route by drug record keywords"}
        if isinstance(drug_names, list) and len(drug_names) >= 2:
            return {"target_type": "tool", "target_name": "drug_interaction", "intent_type": "drug_conflict", "confidence": float(state.get("intent_confidence") or 0.75), "reason": "route by multi-drug entities"}
        return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "drug_query", "confidence": float(state.get("intent_confidence") or 0.7), "reason": "route by drug default to qa"}

    if any(k in text for k in ["化验", "检验", "血常规", "尿常规", "指标"]):
        return {"target_type": "tool", "target_name": "lab_report", "intent_type": "lab_report", "confidence": 0.75, "reason": "route by text: lab"}
    if any(k in text for k in ["相互作用", "一起吃", "同服", "冲突", "禁忌"]):
        return {"target_type": "tool", "target_name": "drug_interaction", "intent_type": "drug_conflict", "confidence": 0.75, "reason": "route by text: drug conflict"}
    if any(k in text for k in ["用药记录", "记录", "添加", "吃了", "服用", "mg", "毫克"]):
        # 过敏声明优先——路由到通用问答而非用药记录
        if any(k in text for k in ["过敏"]):
            return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "allergy_record", "confidence": 0.78, "reason": "route by text: allergy (not drug record)"}
        return {"target_type": "agent", "target_name": "drug_record_agent", "intent_type": "drug_record", "confidence": 0.7, "reason": "route by text: drug record"}
    if any(k in text for k in ["档案", "病历", "历史记录", "就诊"]):
        return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "archive", "confidence": 0.7, "reason": "route by text: archive"}
    return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "general", "confidence": 0.6, "reason": "route by text: default general"}


def _detect_dependencies_rule(query: str, previous_queries: list[str]) -> list[str]:
    """规则优先的依赖检测（扩展版）。

    按四个维度检测当前 query 是否依赖前置查询的结果：
    1. 指代引用 — 代词/序号指向前面提到的内容
    2. 评估追问 — 询问前文推荐的效果/安全性/对比
    3. 细节追问 — 询问前文药物的剂量/用法/注意事项
    4. 隐式因果 — 药物推荐/冲突查询依赖症状分析/药物列表
    """
    if not previous_queries:
        return []

    deps: set[int] = set()
    prev_count = len(previous_queries)

    # 1) 代词/序号引用 → 依赖所有前置步骤（保守策略）
    if any(p in query for p in _CONTINUATION_PRONOUNS):
        deps.update(range(prev_count))

    # 2) "结果引用" → 依赖包含药/症状关键词的前置步骤
    if any(k in query for k in _RESULT_REF_KEYWORDS):
        for i, prev_q in enumerate(previous_queries):
            if any(kw in prev_q for kw in ["药", "原因", "症状", "治疗", "怎么办", "腹泻", "发烧", "咳嗽", "头痛", "疼痛", "感冒", "炎症", "感染"]):
                deps.add(i)

    # 3) "评估追问"（XX怎么样？有用吗？哪个更好？）→ 依赖包含药/治疗关键词的前置步骤
    _eval_hit = any(k in query for k in _EVALUATION_PATTERNS)
    _detail_hit = any(k in query for k in _FOLLOWUP_PATTERNS)
    if _eval_hit or _detail_hit:
        for i, prev_q in enumerate(previous_queries):
            if any(kw in prev_q for kw in ["药", "推荐", "治疗", "布洛芬", "阿司匹林", "阿莫西林", "头孢"]):
                deps.add(i)

    # 4) 药物冲突查询 → 依赖提供药名的前置步骤
    if any(k in query for k in _DRUG_CONFLICT_KEYWORDS):
        for i, prev_q in enumerate(previous_queries):
            if any(kw in prev_q for kw in ["药", "吃什么", "用什么", "推荐"]):
                deps.add(i)

    # 5) 隐式依赖：药物推荐 → 依赖症状分析
    if any(k in query for k in _IMPLICIT_RECOMMENDATION):
        for i, prev_q in enumerate(previous_queries):
            if any(kw in prev_q for kw in _IMPLICIT_CAUSE):
                deps.add(i)

    # 6) "为什么" / "怎么会" 追问 → 依赖前面有实质内容的步骤
    if any(k in query for k in ["为什么", "怎么会", "原因是", "是什么原因"]):
        for i, prev_q in enumerate(previous_queries):
            if len(prev_q) >= 6:
                deps.add(i)

    # 7) 条件引用（"如果是"、"如果这样"）→ 依赖最近的结论性前置步骤
    if any(query.startswith(k) or k in query for k in _CONDITIONAL_REF_PATTERNS):
        for i in range(prev_count - 1, -1, -1):
            prev_q = previous_queries[i]
            if any(kw in prev_q for kw in ["可能", "是不是", "是否", "请问", "吗", "吧"]):
                deps.add(i)
                break

    # 8) "还" / "也" 位于句首 → 补充描述，依赖前一步
    _stripped = query.strip()
    if _stripped.startswith("还") or _stripped.startswith("也"):
        if prev_count > 0:
            deps.add(prev_count - 1)

    return [f"s{i + 1}" for i in sorted(deps)]




async def _detect_dependencies_batch(queries: list[str]) -> list[list[str]]:
    """批量检测依赖关系：规则结果与 LLM 结果取并集。

    规则提供高精度快速覆盖，LLM 捕捉规则遗漏的语义依赖。
    最终每个查询的依赖 = 规则 ∪ LLM。
    """
    # 规则层
    rule_results: list[list[str]] = []
    for i, q in enumerate(queries):
        rule_deps = _detect_dependencies_rule(q, queries[:i])
        rule_results.append(rule_deps)

    if len(queries) <= 1:
        return rule_results

    # LLM 补充层 — 始终运行，作为规则的补充
    llm_results: list[list[str]] = [[] for _ in queries]
    if _llm_enabled():
        llm_results = await _detect_dependencies_llm(queries)

    # 合并：规则 ∪ LLM
    merged: list[list[str]] = []
    for i in range(len(queries)):
        combined = list(set(rule_results[i]) | set(llm_results[i]))
        merged.append(sorted(combined, key=lambda x: int(x[1:])))
    return merged


async def _detect_dependencies_llm(queries: list[str]) -> list[list[str]]:
    """单次 LLM 调用检测所有查询间的依赖关系。"""
    llm = LLMService()
    queries_desc = "\n".join(f"s{i+1}: {q}" for i, q in enumerate(queries))
    prompt = (
        f"分析以下查询之间的依赖关系。如果当前查询需要之前某个查询的结果才能回答，标记为依赖。\n"
        f"依赖的常见情形：(1) 使用了代词或序号指代前置内容 (2) 需要前置步骤给出的药品名/诊断结果 "
        f"(3) 对前置推荐结果做进一步追问（效果、副作用、用量、对比等）\n"
        f"所有查询：\n{queries_desc}\n\n"
        f"输出JSON对象，键为步骤编号(s2,s3,...)，值为依赖的步骤编号数组。s1 不可能有依赖。\n"
        f"无依赖的步骤省略，或给空数组。\n"
        f"示例：{{\"s2\": [\"s1\"], \"s3\": [\"s1\", \"s2\"]}}\n"
        f"只输出JSON，不要其他内容。"
    )
    try:
        raw = await llm.chat_completion(
            prompt=prompt,
            system_prompt="你是依赖分析助手，只输出JSON对象。",
            stream=False,
            timeout_s=6.0,
            max_tokens=200,
        )
        import re
        match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
        json_str = match.group(0) if match else raw.strip()
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return [[] for _ in queries]
        results: list[list[str]] = [[] for _ in queries]
        all_ids = {f"s{i+1}" for i in range(len(queries))}
        for key, deps in data.items():
            if not isinstance(deps, list):
                continue
            idx = int(key[1:]) - 1 if key.startswith("s") and key[1:].isdigit() else -1
            if 0 <= idx < len(queries):
                valid = [d for d in deps if isinstance(d, str) and d in all_ids]
                results[idx] = valid
        return results
    except Exception:
        logger.debug("LLM dependency detection failed, using rule-only results")
        return [[] for _ in queries]


def _llm_enabled() -> bool:
    def _ok(v: str) -> bool:
        v = (v or '').strip()
        return bool(v) and not (v.startswith('{{') and v.endswith('}}'))
    return _ok(settings.LLM_API_BASE) and _ok(settings.LLM_API_KEY) and _ok(settings.LLM_MODEL_NAME)


def _split_user_queries_rule(text: str) -> list[str]:
    """规则拆分：仅在明确的多意图转换标记处拆分。

    保守策略——宁可少拆也不误拆：
    - 只在显式标记处拆分（"另外"、"此外"、"顺便问" 等）
    - 不按句号/问号等标点拆分，避免将背景陈述和补充描述切成碎片
    - 如果拆分后只有 1 段，返回原文本
    """
    import re
    raw = (text or "").strip()
    if not raw:
        return []

    # 显式多意图标记：这些词明确表示"我要问另一个问题了"
    _EXPLICIT_MARKERS = [
        "另外", "此外", "顺便问", "还想问", "还想知道", "再问", "再请教",
        "第一个问题", "第二个问题", "第三个问题",
        "一是", "二是", "三是", "第一", "第二", "第三",
        "问题一", "问题二", "问题三",
    ]

    # 用这些标记拆分
    marker_pattern = "|".join(re.escape(m) for m in _EXPLICIT_MARKERS)
    parts = re.split(rf"(?={marker_pattern})", raw)

    out: list[str] = []
    for part in parts:
        part = part.strip(" ，,;；。！？!?\n\r")
        if part:
            out.append(part)

    # 去重
    seen: set[str] = set()
    dedup: list[str] = []
    for q in out:
        if q in seen:
            continue
        seen.add(q)
        dedup.append(q)

    return dedup if dedup else [raw]


_MULTI_INTENT_MARKERS = [
    "另外", "还有", "并且", "同时", "顺便", "此外",
    "第一个问题", "第二个问题", "一是", "二是", "第一", "第二",
]


def _is_likely_single_intent(text: str) -> bool:
    """快速规则判断一段文本是否很可能是单意图（不需要 LLM 拆分）。"""
    t = (text or "").strip()
    if not t:
        return True
    # 已含明确分隔标记
    if any(m in t for m in _MULTI_INTENT_MARKERS):
        return False
    # 多个问号
    if t.count("?") + t.count("？") >= 2:
        return False
    # 短文本通常不需要拆分
    if len(t) <= 50:
        return True
    # 中等长度、没有多意图标记 → 倾向不拆分
    if len(t) <= 120 and t.count("，") <= 2:
        return True
    return False


async def _split_user_queries(text: str) -> list[str]:
    """拆分用户输入为独立子查询。规则预检 → LLM 增强。

    如果规则预检判定为单意图，跳过 LLM 调用直接返回规则结果。
    """
    rule_result = _split_user_queries_rule(text)
    if len(rule_result) <= 1 and _is_likely_single_intent(text):
        logger.info("_split_user_queries: rule pre-check single-intent, skip LLM")
        return rule_result if rule_result else [text]

    if _llm_enabled():
        llm_decision = LLMDecisionService()
        llm_result = await llm_decision.split_queries(text)
        if llm_result and len(llm_result) > 0:
            logger.info("_split_user_queries: LLM split into %d queries", len(llm_result))
            return llm_result
    logger.info("_split_user_queries: rule split into %d queries", len(rule_result))
    return rule_result if rule_result else [text]


async def _predict_intent_for_query(query: str) -> dict | None:
    try:
        clf = IntentClassifier()
        sub_intent = await clf.predict(text=query, stream=False)
        return {"intent": sub_intent.intent, "confidence": sub_intent.confidence, "reason": sub_intent.reason}
    except Exception:
        return None


async def _route_single_query(query: str, state: dict) -> dict:
    if _llm_enabled():
        llm_decision = LLMDecisionService()
        llm_route = await llm_decision.classify_intent_and_route(query)
        if llm_route and llm_route.get("confidence", 0) >= 0.5:
            logger.info("_route_single_query: LLM route query=%s -> %s", query[:20], llm_route.get("target_name"))
            return llm_route

    pred = await _predict_intent_for_query(query)
    intent_val = pred.get("intent", "general") if pred else "general"
    conf_val = pred.get("confidence", 0.5) if pred else 0.5

    sub_state = dict(state)
    sub_state["user_input"] = query
    sub_state["intent"] = intent_val
    sub_state["intent_confidence"] = conf_val
    rule_route = _route_by_intent_and_text(sub_state)
    logger.info("_route_single_query: rule route query=%s -> %s", query[:20], rule_route.get("target_name"))
    return rule_route


class PlannerAgent:
    """Plan-and-Execute 协调者：负责计划生成、执行评估与动态重规划。

    设计原则：
    - LLM 优先（暴露工具/Agent 描述，由 LLM 统一决策）
    - 正则/关键词兜底（LLM 不可用或失败时）
    - 支持重规划循环（最多 MAX_REPLAN 次）
    """

    def __init__(self):
        self.llm = LLMService()

    async def generate_plan(self, state: dict) -> dict:
        if state.get("error_msg"):
            return state

        user_input = state.get("user_input", "")

        sm_data = (state.get("private_scratchpads") or {}).get("drug_record_sm")
        if sm_data:
            sm = DrugRecordStateMachine.from_dict(sm_data)
            if sm.is_active():
                state["execution_plan"] = ExecutionPlan(
                    steps=[PlanStep(
                        step_id="s1",
                        query=user_input,
                        target_type="agent",
                        target_name="drug_record_agent",
                        intent_type="drug_record_sm",
                        depends_on=[],
                        execution_strategy="serial",
                    )],
                    strategy="single",
                    conflict_resolution_policy="none",
                )
                state["plan_phase"] = "planning"
                logger.info("PlannerAgent plan: active state machine -> drug_record_agent")
                return state

        sub_queries = await _split_user_queries(user_input)

        if len(sub_queries) <= 1:
            existing_route = state.get("intent_analysis")
            if existing_route and isinstance(existing_route, dict) and existing_route.get("target_name"):
                route_result = existing_route
                logger.info("generate_plan: reuse intent_analysis route=%s", route_result.get("target_name"))
            else:
                route_result = await _route_single_query(user_input, state)
            state["intent_analysis"] = route_result
            state["target_agent"] = route_result["target_name"]
            state["intent_type"] = route_result.get("intent_type", state.get("intent", "general"))

            plan = ExecutionPlan(
                steps=[PlanStep(
                    step_id="s1",
                    query=user_input,
                    target_type=route_result.get("target_type", "agent"),
                    target_name=route_result["target_name"],
                    intent_type=route_result.get("intent_type", "general"),
                    depends_on=[],
                    execution_strategy="serial",
                )],
                strategy="single",
                conflict_resolution_policy="evidence_priority",
            )
        else:
            llm_decision = LLMDecisionService()
            batch_routes, llm_deps = await llm_decision.batch_route_with_deps(sub_queries)

            # 规则层依赖检测 + LLM 结果合并（并集），减少一次 LLM 往返
            rule_deps: list[list[str]] = []
            for i, q in enumerate(sub_queries):
                rule_deps.append(_detect_dependencies_rule(q, sub_queries[:i]))
            dep_results: list[list[str]] = []
            for i in range(len(sub_queries)):
                combined = list(set(rule_deps[i]) | set(llm_deps[i]))
                dep_results.append(sorted(combined, key=lambda x: int(x[1:])))

            steps = []
            for i, q in enumerate(sub_queries):
                route_result = batch_routes[i]
                if not route_result or not isinstance(route_result, dict):
                    route_result = _route_by_intent_and_text({
                        "user_input": q,
                        "intent": _quick_intent_classify(q),
                        "intent_confidence": 0.5,
                        "extract_entities": {},
                    })

                deps = dep_results[i]

                steps.append(PlanStep(
                    step_id=f"s{i + 1}",
                    query=q,
                    target_type=route_result.get("target_type", "agent"),
                    target_name=route_result["target_name"],
                    intent_type=route_result.get("intent_type", "general"),
                    depends_on=deps,
                    execution_strategy="parallel" if not deps else "serial",
                ))

            plan = ExecutionPlan(
                steps=steps,
                strategy="topological",
                conflict_resolution_policy="evidence_priority",
            )

        state["execution_plan"] = plan
        state["plan_phase"] = "planning"
        logger.info("PlannerAgent plan=%s", json.dumps({k: v for k, v in plan.items()}, ensure_ascii=False, default=str))
        return state

    async def evaluate_for_replan(self, state: dict) -> dict:
        results = state.get("plan_step_results", {})
        plan = state.get("execution_plan", {})
        steps = plan.get("steps", [])
        replan_count = state.get("replan_count", 0)

        if replan_count >= MAX_REPLAN:
            logger.info("PlannerAgent evaluate: replan_count=%s >= MAX_REPLAN, skip", replan_count)
            state["needs_replan"] = False
            return state

        failed_steps = []
        for step_id, result in results.items():
            if isinstance(result, dict) and result.get("error_msg"):
                failed_steps.append(step_id)

        if not failed_steps:
            cross_conflict = self._detect_cross_step_conflict(results, steps)
            if cross_conflict:
                state["needs_replan"] = True
                state["replan_reason"] = f"跨步骤药物冲突: {cross_conflict}"
                state["replan_count"] = replan_count + 1
                logger.info("PlannerAgent evaluate: needs_replan=True reason=%s", cross_conflict)
                return state

            state["needs_replan"] = False
            return state

        if len(failed_steps) == len(steps):
            state["needs_replan"] = False
            logger.info("PlannerAgent evaluate: all steps failed, no replan")
            return state

        retry_steps = []
        for step in steps:
            if step["step_id"] in failed_steps:
                retry_steps.append(PlanStep(
                    step_id=f"{step['step_id']}_retry",
                    query=step["query"],
                    target_type=step.get("target_type", "agent"),
                    target_name=step.get("target_name", ""),
                    intent_type=step.get("intent_type", "general"),
                    depends_on=[],
                    execution_strategy="serial",
                ))

        if retry_steps:
            existing_steps = [s for s in steps if s["step_id"] not in failed_steps]
            revised_steps = existing_steps + retry_steps
            state["execution_plan"] = ExecutionPlan(
                steps=revised_steps,
                strategy="topological",
                conflict_resolution_policy="evidence_priority",
            )
            state["needs_replan"] = True
            state["replan_reason"] = f"重试失败步骤: {failed_steps}"
            state["replan_count"] = replan_count + 1
            logger.info("PlannerAgent evaluate: needs_replan=True retry_steps=%s", [s["step_id"] for s in retry_steps])
            return state

        state["needs_replan"] = False
        return state

    def _detect_cross_step_conflict(self, results: dict, steps: list[PlanStep]) -> str | None:
        drug_names_from_steps: list[str] = []
        for step in steps:
            result = results.get(step["step_id"], {})
            if not isinstance(result, dict):
                continue
            if step.get("intent_type") == "drug_conflict":
                interactions = (result.get("tool_result") or {}).get("interaction_result", [])
                if interactions:
                    return f"步骤{step['step_id']}已检测到药物冲突"
            if step.get("intent_type") == "drug_record":
                entities = result.get("extract_entities") or {}
                if isinstance(entities, dict):
                    names = entities.get("drug_name_list", [])
                    drug_names_from_steps.extend(names)

        if len(drug_names_from_steps) >= 2:
            return f"多步骤涉及药物{drug_names_from_steps}，需补充冲突检查"
        return None
