from __future__ import annotations

import json
from typing import Any

from app.common.logger import get_logger
from app.config.settings import settings
from app.core.agent.intent_classifier import IntentClassifier
from app.core.agent.llm_decision_service import CAPABILITY_REGISTRY, LLMDecisionService
from app.core.agent.state import ExecutionPlan, PlanStep
from app.core.llm.llm_service import LLMService

logger = get_logger(__name__)

MAX_REPLAN = 2

# ---- 跨轮冲突检测：档案用药的分层判据与容量上限 ----
_ARCHIVE_RECENT_DAYS = 7          # 近期记录：纳入检查
_ARCHIVE_LONGTERM_MIN_COUNT = 2   # 重复记录 ≥2 次视为长期在用，与时间无关
_MAX_CONFLICT_DRUGS = 6           # 单次补检最多带入的药味数
_MIN_ARCHIVE_SLOTS = 2            # 档案药至少占的坑位，避免被本轮药全挤掉

AGENT_TARGETS = {"drug_record_agent", "main_qa_agent"}
TOOL_TARGETS = {"drug_interaction", "lab_report"}


def _describe_archive_drug(entry: dict) -> str:
    """把档案用药渲染成带来源与时间的描述：阿司匹林（2026-09-05 记录，共 3 次）。"""
    seg = entry.get("name", "")
    detail = []
    if entry.get("record_date"):
        detail.append(f"{entry['record_date']} 记录")
    if int(entry.get("record_count") or 1) > 1:
        detail.append(f"共 {entry['record_count']} 次")
    if detail:
        seg += "（" + "，".join(detail) + "）"
    return seg


def _build_conflict_query(
    names: list[str], current: list[str], archive: list[dict], truncated_current: int = 0
) -> str:
    """构造补检步骤的 query：**明确区分药物来源**，不让档案药与本轮药混为一谈。"""
    parts: list[str] = []
    if current:
        parts.append("本轮对话中提到的药物：" + "、".join(current))
    if archive:
        used = [e for e in archive if e.get("name") in names]
        if used:
            parts.append("用户用药档案中记录的药物：" + "、".join(_describe_archive_drug(e) for e in used))
    if not parts:
        parts.append("涉及药物：" + "、".join(names))
    tail = "请检查【本轮提到的药物】与【档案中记录的药物】之间的相互作用或配伍禁忌。"
    if truncated_current > 0:
        tail += f"（本轮药物较多，仅取前 {len(current) - truncated_current} 种）"
    if len(names) >= _MAX_CONFLICT_DRUGS:
        tail += f"（药物较多，本次仅核查其中 {len(names)} 种）"
    return "。".join(parts) + "。" + tail


def _target_type_of(target_name: str) -> str | None:
    """按能力注册表裁定 target 是 tool 还是 agent；未知返回 None（由调用方沿用原值）。"""
    return next((c["type"] for c in CAPABILITY_REGISTRY if c["name"] == target_name), None)

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


# ---- target-aware 依赖修正：按「前置步骤实际会产出什么」裁决依赖边 ----

# 只写库、不产出可被后续步骤引用的可读结论
_NON_PRODUCING_TARGETS = {"drug_record_agent"}
# 会产出药品名/推荐结论，可被「这些药/怎么样/怎么吃」类追问引用
_PRODUCING_DRUG_TARGETS = {"drug_interaction", "main_qa_agent"}
# 引用/追问前步结论的强信号（不含泛指代词——代词已由规则层按"依赖全部前置"处理）
_RESULT_REF_ALL = list(dict.fromkeys(_RESULT_REF_KEYWORDS + _EVALUATION_PATTERNS + _FOLLOWUP_PATTERNS))
# 明确在问「记录/档案有没有存下」，此时写库步骤才可被依赖
_RECORD_REF_KEYWORDS = ["记录", "记下", "存了", "存起来", "档案里", "有没有记"]


def _refine_deps_by_target(
    query: str,
    previous_queries: list[str],
    previous_routes: list[dict | None] | None,
    deps: list[str],
) -> list[str]:
    """用前置步骤的路由结果修正依赖边（target-aware）。

    两层作用：
    1. **否决**：前置步骤只写库不产出可读结论（如 drug_record_agent）时，
       除非本句明确在问记录本身，否则移除该边——LLM 常把写库步骤当成普通前置连上。
    2. **补强**：前置步骤会产出药名/推荐且本句出现引用/评估/细节追问信号时，确保连边。
    """
    if not previous_routes:
        return deps

    def _target_of(idx: int) -> str | None:
        if 0 <= idx < len(previous_routes):
            route = previous_routes[idx]
            if isinstance(route, dict):
                return route.get("target_name")
        return None

    asks_about_record = any(k in query for k in _RECORD_REF_KEYWORDS)
    kept: set[str] = set()
    for d in deps:
        if not (d.startswith("s") and d[1:].isdigit()):
            continue
        idx = int(d[1:]) - 1
        target = _target_of(idx)
        if target in _NON_PRODUCING_TARGETS and not asks_about_record:
            continue  # 否决：写库步骤不产出可引用结论
        kept.add(d)

    # 补强：前步产出药名/推荐且本句在引用或追问其结论
    if any(k in query for k in _RESULT_REF_ALL):
        for i in range(len(previous_queries)):
            if _target_of(i) in _PRODUCING_DRUG_TARGETS:
                kept.add(f"s{i + 1}")

    return sorted(kept, key=lambda x: int(x[1:]))


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


# 只有「需要不同 target 处理」的域组合才算真·多意图；
# 症状词与药物词同现并不算——它们都归 main_qa_agent，拆了反而丢上下文。
_LAB_SIGNALS = ["化验", "检验", "指标", "血常规", "尿常规", "报告", "体检", "血糖", "血压", "血脂"]
_ARCHIVE_SIGNALS = ["档案", "病历", "历史记录", "就诊记录", "用药记录"]
_DRUG_SIGNALS = ["一起吃", "同服", "相互作用", "冲突", "禁忌", "配伍", "吃了", "服用", "剂量", "mg", "毫克"]


def _has_multi_target_signals(text: str) -> bool:
    """跨域信号检测：出现两个需要不同 target 的域 → 很可能是多意图。

    比单纯"加严长度/标点阈值"更精准：不会因为「头痛吃什么药」这种单意图句
    （症状词+药物词都归 main_qa_agent）而误判成多意图，白白多付一次 LLM 往返。
    """
    has_lab = any(k in text for k in _LAB_SIGNALS)
    has_archive = any(k in text for k in _ARCHIVE_SIGNALS)
    has_drug = any(k in text for k in _DRUG_SIGNALS)
    return (has_lab and has_drug) or (has_lab and has_archive) or (has_archive and has_drug)


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
    # 跨域信号：需要不同工具分别处理的意图同现
    if _has_multi_target_signals(t):
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
        ctx = (state.get("decision_context") or "") if isinstance(state, dict) else ""
        llm_route = await llm_decision.classify_intent_and_route(query, ctx=ctx)
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
        ctx = state.get("decision_context") or ""

        # ---- 1. 是否需要拆分：规则预检 ∪ intent 节点的零成本多意图标记 ----
        # 规则漏判（把多意图判成单意图）原本不可逆——跳过拆分后没有任何环节再检查；
        # 这里用 intent 那次必调 LLM 顺带产出的 is_multi_intent 兜底，不增加任何往返。
        rule_split = _split_user_queries_rule(user_input)
        rule_says_multi = len(rule_split) > 1 or not _is_likely_single_intent(user_input)
        llm_says_multi = bool(state.get("is_multi_intent"))

        if not (rule_says_multi or llm_says_multi):
            logger.info("generate_plan: single-intent path, skip split LLM (rule_multi=%s)", rule_says_multi)
            return await self._build_single_step_plan(state, user_input)

        # ---- 2. 多意图：一次调用同时完成 拆分 + 路由 + 依赖 ----
        llm_decision = LLMDecisionService()
        sub_queries, routes, deps = await llm_decision.split_route_deps(user_input, ctx=ctx)

        # ---- 3. 一次调用失败 → 回落旧的「拆分 + 批量路由」两步链路 ----
        if not sub_queries:
            logger.info("generate_plan: split_route_deps unavailable, fallback to two-step")
            sub_queries = await _split_user_queries(user_input)
            if len(sub_queries) <= 1:
                return await self._build_single_step_plan(state, user_input)
            routes, deps = await llm_decision.batch_route_with_deps(sub_queries, ctx=ctx)

        if len(sub_queries) <= 1:
            return await self._build_single_step_plan(state, user_input)

        # ---- 4. 依赖 = 规则 ∪ LLM，再按前置步骤的 target 修正（可加边也可减边）----
        dep_results: list[list[str]] = []
        for i, q in enumerate(sub_queries):
            rule_deps = _detect_dependencies_rule(q, sub_queries[:i])
            llm_deps_i = (deps[i] if deps and i < len(deps) else []) or []
            combined = sorted(set(rule_deps) | set(llm_deps_i), key=lambda x: int(x[1:]))
            dep_results.append(_refine_deps_by_target(q, sub_queries[:i], routes, combined))

        steps = []
        for i, q in enumerate(sub_queries):
            route_result = routes[i] if routes and i < len(routes) else None
            if not route_result or not isinstance(route_result, dict):
                # 首个子查询优先复用 intent 节点在完整原句上算出的路由，语义最接近
                if i == 0 and isinstance(state.get("intent_analysis"), dict) and state["intent_analysis"].get("target_name"):
                    route_result = state["intent_analysis"]
                else:
                    route_result = _route_by_intent_and_text({
                        "user_input": q,
                        "intent": _quick_intent_classify(q),
                        "intent_confidence": 0.5,
                        "extract_entities": {},
                    })

            step_deps = dep_results[i]
            steps.append(PlanStep(
                step_id=f"s{i + 1}",
                query=q,
                target_type=route_result.get("target_type", "agent"),
                target_name=route_result.get("target_name", "main_qa_agent"),
                intent_type=route_result.get("intent_type", "general"),
                depends_on=step_deps,
                execution_strategy="parallel" if not step_deps else "serial",
            ))

        plan = ExecutionPlan(
            steps=steps,
            strategy="topological",
            conflict_resolution_policy="evidence_priority",
        )

        state["execution_plan"] = plan
        state["plan_phase"] = "planning"
        # 主链路仍取首个步骤的路由，供下游 target_agent 等字段使用
        state.setdefault("target_agent", steps[0]["target_name"])
        logger.info("PlannerAgent multi-step plan steps=%s deps=%s", len(steps), [s["depends_on"] for s in steps])
        return state

    async def _build_single_step_plan(self, state: dict, user_input: str) -> dict:
        """单意图路径：复用 intent 节点的路由结果，0 次额外 LLM 往返。"""
        existing_route = state.get("intent_analysis")
        if isinstance(existing_route, dict) and existing_route.get("target_name"):
            route_result = existing_route
        else:
            route_result = await _route_single_query(user_input, state)

        state["intent_analysis"] = route_result
        state["target_agent"] = route_result["target_name"]
        state["intent_type"] = route_result.get("intent_type", state.get("intent", "general"))

        state["execution_plan"] = ExecutionPlan(
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
        state["plan_phase"] = "planning"
        return state

    async def evaluate_for_replan(self, state: dict) -> dict:
        """只做【判定 + 上下文收集】，不再自行拼接重试计划。

        生成修正计划的职责交给 build_revised_plan：失败原因多样，需要判断是
        原样重试、改写问题还是换执行体，直接重放对确定性失败（药名匹配不到、
        参数缺失）必然再次失败。
        """
        results = state.get("plan_step_results", {})
        plan = state.get("execution_plan", {})
        steps = plan.get("steps", [])
        replan_count = state.get("replan_count", 0)

        if replan_count >= MAX_REPLAN:
            state["needs_replan"] = False
            return state

        step_ids = [s["step_id"] for s in steps]
        failed_ids = [
            sid for sid in step_ids
            if isinstance(results.get(sid), dict) and results[sid].get("error_msg")
        ]

        # 1) 有失败步骤 → 收集失败原因与已完成结果，交给 LLM 重规划
        if failed_ids:
            if len(failed_ids) >= len(steps):
                # 全失败：重试拿不到任何新信息
                state["needs_replan"] = False
                return state
            state["replan_context"] = await self._build_replan_context(steps, results, failed_ids)
            state["needs_replan"] = True
            state["replan_reason"] = f"步骤执行失败: {failed_ids}"
            state["replan_count"] = replan_count + 1
            return state

        # 2) 无失败 → 检查跨步骤冲突（含跨轮：本轮药物 vs 档案中在服药物）
        conflict = await self._detect_cross_step_conflict(results, steps, state)
        if conflict:
            state["cross_step_conflict"] = conflict
            if conflict.get("needs_check"):
                state["needs_replan"] = True
                state["replan_reason"] = conflict.get("reason", "")
                state["replan_count"] = replan_count + 1
                return state
            # 某一步已经查出冲突：不需要再补一步，结论留给下游呈现
            state["needs_replan"] = False
            return state

        state["needs_replan"] = False
        return state

    async def _build_replan_context(
        self, steps: list[PlanStep], results: dict, failed_ids: list[str]
    ) -> dict:
        """组装给重规划 LLM 的上下文：失败步骤的错误 + 已完成步骤的结果摘要。"""
        failed_desc: list[dict] = []
        completed_desc: list[dict] = []

        for step in steps:
            sid = step["step_id"]
            result = results.get(sid) or {}
            if not isinstance(result, dict):
                continue
            if sid in failed_ids:
                failed_desc.append({
                    "step_id": sid,
                    "query": step.get("query", ""),
                    "target_name": step.get("target_name", ""),
                    "error_msg": str(result.get("error_msg", "未知错误"))[:300],
                })
                continue
            summary = result.get("final_response") or ""
            if not summary and isinstance(result.get("tool_result"), dict):
                summary = json.dumps(result["tool_result"], ensure_ascii=False)
            completed_desc.append({
                "step_id": sid,
                "query": step.get("query", ""),
                "target_name": step.get("target_name", ""),
                "summary": str(summary)[:400],
            })

        return {"failed_steps": failed_desc, "completed_steps": completed_desc}

    async def build_revised_plan(self, state: dict) -> dict:
        """重规划分支：按「能确定处理就确定处理」的原则分两条路。

        - 跨步骤冲突 → **确定性补一个 drug_interaction 步骤**（不调 LLM，
          因为该做什么完全确定，交给 LLM 只会增加延迟与不确定性）
        - 步骤失败 → **交给 LLM 判断**（retry/rewrite/reroute/drop），
          失败原因多样，需要判断换策略；LLM 不可用时回落为原样重放
        """
        plan = state.get("execution_plan") or {}
        steps = list(plan.get("steps", []))
        attempts = int(state.get("replan_count", 1) or 1)

        # --- 路径 A：跨步骤冲突 → 确定性补步 ---
        conflict = state.get("cross_step_conflict") or {}
        if conflict.get("needs_check"):
            current = [n for n in (conflict.get("current_drugs") or []) if n]
            archive = [e for e in (conflict.get("archive_drugs") or []) if isinstance(e, dict) and e.get("name")]
            # 本轮药物优先，同时给档案药预留坑位——本轮药很多时若直接截断，
            # 档案药会被静默丢掉，跨轮检测等于失效
            archive_names_all = [e["name"] for e in archive]
            slots_for_archive = 0
            if archive_names_all:
                slots_for_archive = min(
                    max(_MIN_ARCHIVE_SLOTS, _MAX_CONFLICT_DRUGS - len(current)),
                    _MAX_CONFLICT_DRUGS,
                )
            current_cap = _MAX_CONFLICT_DRUGS - slots_for_archive
            names = current[:current_cap] + archive_names_all[:slots_for_archive]
            truncated_current = len(current) - min(len(current), current_cap)
            if len(names) >= 2:
                steps.append(PlanStep(
                    step_id=f"s_conflict_check_{attempts}",
                    query=_build_conflict_query(names, current, archive, truncated_current),
                    target_type="tool",
                    target_name="drug_interaction",
                    intent_type="drug_conflict",
                    depends_on=[],
                    execution_strategy="serial",
                ))
                state["execution_plan"] = ExecutionPlan(
                    steps=steps, strategy="topological", conflict_resolution_policy="evidence_priority",
                )
                state["plan_phase"] = "planning"
                logger.info("build_revised_plan: append conflict-check step drugs=%s", names)
                return state

        # --- 路径 B：步骤失败 → LLM 修正 ---
        replan_ctx = state.get("replan_context") or {}
        failed = replan_ctx.get("failed_steps") or []
        if not failed:
            return state

        by_id = {s["step_id"]: s for s in steps if isinstance(s, dict)}
        actions: list[dict] | None = None
        if _llm_enabled():
            try:
                actions = await LLMDecisionService().replan_failed_steps(
                    original_input=state.get("original_user_input") or state.get("user_input", ""),
                    failed_steps=failed,
                    completed_steps=replan_ctx.get("completed_steps") or [],
                    ctx=state.get("decision_context") or "",
                )
            except Exception as e:
                logger.warning("build_revised_plan: replan LLM failed: %s", e)
                actions = None

        new_steps: list[PlanStep] = []
        if actions:
            for a in actions:
                orig = by_id.get(a.get("step_id", ""))
                if not orig:
                    continue
                action = a.get("action", "retry")
                if action == "drop":
                    logger.info("build_revised_plan: drop step=%s reason=%s", a.get("step_id"), a.get("reason"))
                    continue
                # rewrite 用 LLM 改写后的问句；reroute 换目标；retry 保持原样
                query = a.get("query") or orig.get("query", "")
                if action != "rewrite":
                    query = orig.get("query", "")
                target_name = a.get("target_name") or orig.get("target_name", "")
                new_steps.append(PlanStep(
                    step_id=f"{a['step_id']}_r{attempts}",
                    query=query,
                    target_type=_target_type_of(target_name) or orig.get("target_type", "agent"),
                    target_name=target_name,
                    intent_type=orig.get("intent_type", "general"),
                    depends_on=[],   # 重试步骤改串行，避免再被上游结果污染
                    execution_strategy="serial",
                ))
        else:
            # 回落：LLM 不可用/输出非法时保持原样重放，保证仍有一次重试机会
            for f in failed:
                orig = by_id.get(f.get("step_id", ""))
                if not orig:
                    continue
                new_steps.append(PlanStep(
                    step_id=f"{f['step_id']}_r{attempts}",
                    query=orig.get("query", ""),
                    target_type=orig.get("target_type", "agent"),
                    target_name=orig.get("target_name", ""),
                    intent_type=orig.get("intent_type", "general"),
                    depends_on=[],
                    execution_strategy="serial",
                ))

        if new_steps:
            state["execution_plan"] = ExecutionPlan(
                steps=steps + new_steps,
                strategy="topological",
                conflict_resolution_policy="evidence_priority",
            )
            state["plan_phase"] = "planning"
            logger.info("build_revised_plan: llm_actions=%s new_steps=%s", bool(actions), len(new_steps))
        return state

    async def _detect_cross_step_conflict(
        self, results: dict, steps: list[PlanStep], state: dict | None = None
    ) -> dict | None:
        """跨步骤 / 跨轮药物冲突检测。

        - **跨步骤（同轮）**：药名分散在多个步骤、且没有任何一步查过相互作用
        - **跨轮（跨会话）**：本轮提到的药 + 用户档案中在服的药物。
          现实里用户极少一句话说完所有在吃的药，跨轮累积才是常态，
          所以这条比同轮检测更有价值。

        返回:
          {"needs_check": True, ...}  —— 需要补一个 drug_interaction 步骤
          {"needs_check": False, ...} —— 某步已查出冲突，结论交给下游呈现
          None —— 无冲突迹象
        """
        drug_names: list[str] = []
        already_checked_detail: list[dict] = []
        archive_drugs: list[str] = []

        for step in steps:
            result = results.get(step["step_id"], {})
            if not isinstance(result, dict):
                continue

            if step.get("intent_type") == "drug_conflict":
                interactions = (result.get("tool_result") or {}).get("interaction_result", [])
                if interactions:
                    already_checked_detail.extend(interactions)

            # 结构化来源优先（已经是标准名）
            entities = result.get("extract_entities") or {}
            if isinstance(entities, dict):
                names = entities.get("drug_name_list", [])
                if isinstance(names, list):
                    drug_names.extend(str(n).strip() for n in names if str(n).strip())

            tool_result = result.get("tool_result") or {}
            if isinstance(tool_result, dict):
                for d in tool_result.get("drug_list", []) or []:
                    if isinstance(d, dict) and d.get("match_status") == "匹配成功":
                        dn = str(d.get("drug_name", "")).strip()
                        if dn:
                            drug_names.append(dn)

        # 结构化没拿到足够的药名时，用实体词典扫回答文本补召回
        if len(drug_names) < 2:
            drug_names.extend(await self._scan_drug_names_from_responses(results, steps))

        # 跨轮：纳入档案中在服的药物（仅在本轮确实涉及用药/冲突意图时才查，避免无谓开销与噪音）
        archive_entries: list[dict] = []
        if state and self._should_include_archive(steps):
            archive_entries = await self._load_archive_drugs(state)

        current_names = list(dict.fromkeys(drug_names))
        # 档案药只保留本轮未提及的，避免自比对；分层判据见 _load_archive_drugs
        ongoing = [e for e in archive_entries if e.get("ongoing") and e["name"] not in current_names]
        stale = [e for e in archive_entries if not e.get("ongoing") and e["name"] not in current_names]
        archive_names = [e["name"] for e in ongoing]
        all_names = current_names + archive_names

        if already_checked_detail:
            return {
                "needs_check": False,
                "current_drugs": current_names,
                "archive_drugs": ongoing,
                "stale_drugs": stale,
                "drug_names": all_names,
                "detail": already_checked_detail,
                "reason": "已在计划内步骤中检测到药物相互作用",
            }

        if len(all_names) >= 2:
            scope = "本轮药物 + 档案记录药物" if archive_names else "本轮多步骤"
            return {
                "needs_check": True,
                "current_drugs": current_names,
                "archive_drugs": ongoing,
                "stale_drugs": stale,
                "drug_names": all_names,
                "reason": f"{scope}涉及药物{all_names}，但没有任何步骤做过相互作用检查",
            }
        return None

    @staticmethod
    def _should_include_archive(steps: list[PlanStep]) -> bool:
        """只有本轮确实在【记录用药】或【查冲突】时才纳入档案药物。

        否则用户随便问一句"感冒吃什么药"都会被拉去和历史用药做全组合比对，
        既浪费一次查表又容易产生无关告警。
        """
        trigger_types = {"drug_record", "drug_conflict"}
        for s in steps:
            if isinstance(s, dict) and (
                s.get("intent_type") in trigger_types or s.get("target_name") == "drug_record_agent"
            ):
                return True
        return False

    @staticmethod
    async def _load_archive_drugs(state: dict) -> list[dict]:
        """读取用户档案中的用药记录（已归一），并按分层判据标注是否在服。

        不用单一时间窗——药物相互作用的风险窗口不是固定的几天：
        长期在服药（降压/降糖/抗凝）与新药的冲突恰恰是跨周跨月才出现。
        判据：
        - 硬排除：end_date 已过（在 SQL 层完成）
        - ongoing：最近 _ARCHIVE_RECENT_DAYS 天内记录 **或** 该药被记录过 ≥2 次
          （重复记录 = 长期在用，这个判据与时间无关，最可靠）
        """
        user_id = state.get("user_id")
        if not user_id:
            return []
        try:
            from app.db.crud.archive_crud import ArchiveCRUD
            entries = await ArchiveCRUD().list_drug_entries(user_id=user_id)
        except Exception as e:
            logger.debug("cross-turn archive drugs load skipped: %s", e)
            return []
        if not entries:
            return []

        names = [e["name"] for e in entries]
        try:
            from app.core.rag.drug_knowledge_service import DrugKnowledgeService
            canon = await DrugKnowledgeService().canonicalize_names(names)
        except Exception as e:
            logger.debug("cross-turn archive drugs normalize skipped: %s", e)
            canon = names

        out: list[dict] = []
        seen: set[str] = set()
        for entry, cname in zip(entries, canon):
            name = cname or entry["name"]
            if not name or name in seen:
                continue
            seen.add(name)
            days = entry.get("days_ago")
            count = int(entry.get("record_count") or 1)
            ongoing = (days is not None and days <= _ARCHIVE_RECENT_DAYS) or count >= _ARCHIVE_LONGTERM_MIN_COUNT
            out.append({**entry, "name": name, "ongoing": ongoing})
        return out

    @staticmethod
    async def _scan_drug_names_from_responses(results: dict, steps: list[PlanStep]) -> list[str]:
        """用实体词典从步骤回答文本中扫药名（应对推荐类步骤没回写结构化实体的情况）。"""
        texts = []
        for step in steps:
            result = results.get(step["step_id"], {})
            if isinstance(result, dict):
                resp = result.get("final_response")
                if resp:
                    texts.append(str(resp)[:1200])
        if not texts:
            return []
        try:
            from app.core.rag.drug_knowledge_service import DrugKnowledgeService
            svc = DrugKnowledgeService()
            found: list[str] = []
            for t in texts:
                found.extend(await svc.resolve_text(t))
            return found
        except Exception as e:
            logger.debug("cross-step drug scan skipped: %s", e)
            return []
