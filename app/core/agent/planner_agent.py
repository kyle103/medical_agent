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

        is_conflict = any(k in text for k in conflict_keywords) or ("药" in text and "一起" in text)
        is_record = any(k in text for k in record_keywords) and not any(k in text for k in query_drug_keywords)
        is_delete = any(k in text for k in delete_keywords)
        is_query_drug = any(k in text for k in query_drug_keywords)

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
        return {"target_type": "agent", "target_name": "drug_record_agent", "intent_type": "drug_record", "confidence": 0.7, "reason": "route by text: drug record"}
    if any(k in text for k in ["档案", "病历", "历史记录", "就诊"]):
        return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "archive", "confidence": 0.7, "reason": "route by text: archive"}
    return {"target_type": "agent", "target_name": "main_qa_agent", "intent_type": "general", "confidence": 0.6, "reason": "route by text: default general"}


def _detect_dependencies_rule(query: str, previous_queries: list[str]) -> list[str]:
    deps: list[str] = []
    pronouns = ["它", "这", "那个", "上面", "刚才", "之前", "继续", "然后", "还有", "其中"]
    if any(p in query for p in pronouns):
        for i, _ in enumerate(previous_queries):
            deps.append(f"s{i + 1}")
    result_ref_keywords = ["其中哪些药", "这些药", "那些药", "上面的药", "推荐的药", "上面提到的药"]
    if any(k in query for k in result_ref_keywords) and previous_queries:
        for i, prev_q in enumerate(previous_queries):
            if any(kw in prev_q for kw in ["药", "原因", "症状", "治疗", "怎么办", "腹泻", "发烧", "咳嗽", "头痛", "疼痛", "感冒", "炎症", "感染"]):
                deps.append(f"s{i + 1}")
    conflict_keywords = ["一起吃", "同服", "相互作用", "配伍", "冲突", "禁忌"]
    if any(k in query for k in conflict_keywords) and previous_queries:
        for i, prev_q in enumerate(previous_queries):
            if any(kw in prev_q for kw in ["药", "吃什么", "用什么", "推荐"]):
                if f"s{i + 1}" not in deps:
                    deps.append(f"s{i + 1}")
    implicit_dep_keywords = ["可以吃什么药", "吃什么药", "用什么药", "有什么药", "该吃什么", "要吃什么", "吃什么好", "怎么办", "怎么治", "如何治疗", "怎么处理", "怎么缓解", "如何缓解"]
    if any(k in query for k in implicit_dep_keywords) and previous_queries:
        for i, prev_q in enumerate(previous_queries):
            if any(kw in prev_q for kw in ["原因", "症状", "怎么办", "腹泻", "发烧", "咳嗽", "头痛", "疼痛", "感冒", "炎症", "感染", "什么病", "得了", "患有", "诊断"]):
                if f"s{i + 1}" not in deps:
                    deps.append(f"s{i + 1}")
    return list(set(deps))


async def _detect_dependencies_semantic(query: str, all_queries: list[str], query_index: int) -> list[str]:
    rule_deps = _detect_dependencies_rule(query, all_queries[:query_index])
    if rule_deps:
        return rule_deps
    if len(all_queries) <= 1:
        return []
    if not _llm_enabled():
        return []

    llm = LLMService()
    prev = all_queries[:query_index]
    if not prev:
        return []

    prompt = (
        f"分析以下查询之间的依赖关系。\n"
        f"当前查询：{query}\n"
        f"之前的查询：{json.dumps(prev, ensure_ascii=False)}\n"
        f"如果当前查询需要之前某个查询的结果才能回答（例如包含代词引用、需要前文药名等），返回依赖的步骤编号列表。\n"
        f"步骤编号格式为 s1, s2, ...（对应第1、2...个查询）。\n"
        f"如果没有依赖，返回空数组 []。\n"
        f"只输出 JSON 数组，不要其他内容。"
    )
    try:
        raw = await llm.chat_completion(
            prompt=prompt,
            system_prompt="你是依赖分析助手，只输出JSON数组。",
            stream=False,
            timeout_s=5.0,
            max_tokens=60,
        )
        deps = json.loads(raw.strip())
        if isinstance(deps, list):
            valid = [d for d in deps if isinstance(d, str) and d.startswith("s")]
            return valid
    except Exception:
        logger.debug("semantic dependency detection fallback to rule")
    return []


async def _detect_dependencies_batch(queries: list[str]) -> list[list[str]]:
    """批量检测多个查询之间的依赖关系。规则优先，仅对规则未覆盖的查询做一次LLM调用。"""
    results: list[list[str]] = []
    need_llm_indices: list[int] = []
    for i, q in enumerate(queries):
        rule_deps = _detect_dependencies_rule(q, queries[:i])
        results.append(rule_deps)
        if not rule_deps and i > 0:
            need_llm_indices.append(i)

    if not need_llm_indices or not _llm_enabled() or len(queries) <= 1:
        return results

    llm = LLMService()
    queries_desc = "\n".join(f"s{i+1}: {q}" for i, q in enumerate(queries))
    need_desc = ", ".join(f"s{i+1}" for i in need_llm_indices)
    prompt = (
        f"分析以下查询之间的依赖关系。\n"
        f"所有查询：\n{queries_desc}\n\n"
        f"请判断 {need_desc} 是否依赖前面的查询结果（如代词引用、需要前文药名等）。\n"
        f"输出JSON对象，键为步骤编号(s1,s2,...)，值为依赖的步骤编号数组。无依赖则为空数组。\n"
        f"示例：{{\"s2\": [\"s1\"], \"s3\": [\"s2\"]}}\n"
        f"只输出JSON，不要其他内容。"
    )
    try:
        raw = await llm.chat_completion(
            prompt=prompt,
            system_prompt="你是依赖分析助手，只输出JSON对象。",
            stream=False,
            timeout_s=3.0,
            max_tokens=120,
        )
        data = json.loads(raw.strip())
        if isinstance(data, dict):
            for idx in need_llm_indices:
                key = f"s{idx + 1}"
                deps = data.get(key, [])
                if isinstance(deps, list):
                    valid = [d for d in deps if isinstance(d, str) and d.startswith("s")]
                    if valid:
                        results[idx] = valid
    except Exception:
        logger.debug("batch semantic dependency detection fallback to rule")

    return results


def _llm_enabled() -> bool:
    def _ok(v: str) -> bool:
        v = (v or '').strip()
        return bool(v) and not (v.startswith('{{') and v.endswith('}}'))
    return _ok(settings.LLM_API_BASE) and _ok(settings.LLM_API_KEY) and _ok(settings.LLM_MODEL_NAME)


def _split_user_queries_rule(text: str) -> list[str]:
    import re
    raw = (text or "").strip()
    if not raw:
        return []
    parts = re.split(r"[。！？!?；;]+", raw)
    parts = [p.strip(" ，,") for p in parts if p and p.strip(" ，,")]
    out: list[str] = []
    for p in parts:
        sub = re.split(r"(?=帮我|请帮我|另外|还有|并且|同时|我是否|我有|顺便)", p)
        for s in sub:
            s = s.strip(" ，,")
            if s:
                out.append(s)
    dedup: list[str] = []
    seen = set()
    for q in out:
        if q in seen:
            continue
        seen.add(q)
        dedup.append(q)
    return dedup


async def _split_user_queries(text: str) -> list[str]:
    if _llm_enabled():
        llm_decision = LLMDecisionService()
        llm_result = await llm_decision.split_queries(text)
        if llm_result and len(llm_result) > 0:
            logger.info("_split_user_queries: LLM split into %d queries", len(llm_result))
            return llm_result
    rule_result = _split_user_queries_rule(text)
    logger.info("_split_user_queries: rule split into %d queries", len(rule_result))
    return rule_result


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
                    strategy="serial",
                    conflict_resolution_policy="none",
                )
                state["plan_phase"] = "planning"
                logger.info("PlannerAgent plan: active state machine -> drug_record_agent")
                return state

        sub_queries = await _split_user_queries(user_input)

        if len(sub_queries) <= 1:
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
                strategy="serial",
                conflict_resolution_policy="evidence_priority",
            )
        else:
            llm_decision = LLMDecisionService()
            batch_routes = await llm_decision.batch_route_queries(sub_queries)

            dep_results = await _detect_dependencies_batch(sub_queries)

            steps = []
            for i, q in enumerate(sub_queries):
                route_result = batch_routes[i]
                if not route_result or not isinstance(route_result, dict):
                    route_result = _route_by_intent_and_text({"user_input": q, "intent": "general", "intent_confidence": 0.5, "extract_entities": {}})

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

            has_deps = any(s.get("depends_on") for s in steps)
            plan = ExecutionPlan(
                steps=steps,
                strategy="hybrid" if has_deps else "parallel",
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
                strategy="serial",
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
