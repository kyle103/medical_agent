from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import date, datetime

from sqlalchemy import and_, select

from app.common.logger import get_logger
from app.core.agent.intent_classifier import IntentClassifier
from app.core.agent.state import ExecutionPlan, PlanStep
from app.core.llm.llm_service import LLMService
from app.core.memory.long_memory_service import LongMemoryService
from app.core.memory.memory_service import MemoryService
from app.core.rag.medical_knowledge_service import MedicalKnowledgeService
from app.core.rag.public_kb_service import PublicKnowledgeService
from app.core.session.agent_state_store import AgentStateStore
from app.core.skills.drug_record_state_machine import DrugRecordPhase, DrugRecordStateMachine
from app.core.skills.input_classifier import InputClassifier
from app.core.skills.medication_confirmation_skill import MedicationConfirmationSkill
from app.core.tools.archive_query_tool import ArchiveQueryTool
from app.core.tools.drug_entity_extractor import DrugEntityExtractor
from app.core.tools.drug_record_tool import DrugRecordTool

from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord

logger = get_logger(__name__)

INTENTS = {
    "archive": "档案查询",
    "drug": "药物相关（冲突查询/用药记录添加）",
    "lab": "化验单解读",
    "general": "通用问答",
}


def _need_contextual_memory(user_input: str) -> bool:
    t = (user_input or "").strip()
    if not t:
        return False

    contains_drug_statement = (
        "吃了" in t or "服用" in t or "用了" in t
        or "需要添加用药记录" in t or "添加用药" in t
    )
    exclude_combinations = ["吃药", "服药"]
    is_excluded = any(combo in t for combo in exclude_combinations) and "需要添加用药记录" not in t and "添加用药" not in t

    if contains_drug_statement and not is_excluded:
        if not any(query in t for query in ["吃什么药", "什么药", "哪些药", "哪种药"]):
            return False

    if "记得" in t and "药" in t:
        return True

    if len(t) <= 12:
        simple_statements = ["我有", "我是", "我在", "我要", "我想"]
        if not any(statement in t for statement in simple_statements):
            return True

    pronouns = ["那个", "它", "这", "这样", "上面", "刚才", "之前", "继续", "然后", "还要", "还用", "还需要"]
    if any(p in t for p in pronouns):
        return True

    recall = ["总结", "回顾", "复盘", "你还记得", "你记得", "还记得", "之前", "刚才", "上次", "昨天", "今天", "最近", "回顾", "总结"]
    if any(k in t for k in recall):
        return True

    drug_queries = ["吃过什么药", "吃了什么药", "服用过什么", "用过什么药", "今天吃了什么药", "昨天吃了什么药"]
    if any(q in t for q in drug_queries):
        return True

    return False


def _short_window_history(history: list[dict], max_turns: int = 4) -> str:
    if not history:
        return ""
    window = history[-max(1, max_turns * 2):]
    return _format_history(window, max_chars=900)


def _format_history(history: list[dict], max_chars: int = 1400) -> str:
    lines: list[str] = []
    for h in history:
        role = h.get("role")
        content = (h.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"用户：{content}")
        elif role == "assistant":
            lines.append(f"助手：{content}")
        else:
            lines.append(f"{role}：{content}")

    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _split_user_queries(text: str) -> list[str]:
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


def _route_by_intent_and_text(state: dict) -> dict:
    intent = (state.get("intent") or "").strip().lower()
    text = state.get("user_input", "").strip()
    entities = state.get("extract_entities") or {}
    drug_names = entities.get("drug_name_list") if isinstance(entities, dict) else []

    if intent == "lab":
        return {"target_agent": "lab_report_agent", "intent_type": "lab_report", "confidence": float(state.get("intent_confidence") or 0.9), "reason": "route by intent=lab"}
    if intent == "archive":
        return {"target_agent": "main_qa_agent", "intent_type": "archive", "confidence": float(state.get("intent_confidence") or 0.9), "reason": "route by intent=archive"}
    if intent == "general":
        return {"target_agent": "main_qa_agent", "intent_type": "general", "confidence": float(state.get("intent_confidence") or 0.8), "reason": "route by intent=general"}

    if intent == "drug":
        conflict_keywords = ["相互作用", "一起吃", "同服", "配伍", "冲突", "禁忌", "能不能一起", "可以一起"]
        record_keywords = ["记录", "添加", "我吃了", "我服用", "我用了", "用药记录", "剂量", "频次", "每天", "每次", "mg", "毫克"]
        delete_keywords = ["删除", "移除", "清空"]

        is_conflict = any(k in text for k in conflict_keywords) or ("药" in text and "一起" in text)
        is_record = any(k in text for k in record_keywords)
        is_delete = any(k in text for k in delete_keywords)

        if is_conflict and not is_record and not is_delete:
            return {"target_agent": "drug_conflict_agent", "intent_type": "drug_conflict", "confidence": float(state.get("intent_confidence") or 0.85), "reason": "route by drug conflict keywords"}
        if (is_record or is_delete) and not is_conflict:
            return {"target_agent": "drug_record_agent", "intent_type": "drug_record", "confidence": float(state.get("intent_confidence") or 0.85), "reason": "route by drug record keywords"}
        if isinstance(drug_names, list) and len(drug_names) >= 2:
            return {"target_agent": "drug_conflict_agent", "intent_type": "drug_conflict", "confidence": float(state.get("intent_confidence") or 0.75), "reason": "route by multi-drug entities"}
        return {"target_agent": "drug_record_agent", "intent_type": "drug_record", "confidence": float(state.get("intent_confidence") or 0.7), "reason": "route by drug default"}

    if any(k in text for k in ["化验", "检验", "血常规", "尿常规", "指标"]):
        return {"target_agent": "lab_report_agent", "intent_type": "lab_report", "confidence": 0.75, "reason": "route by text: lab"}
    if any(k in text for k in ["相互作用", "一起吃", "同服", "冲突", "禁忌"]):
        return {"target_agent": "drug_conflict_agent", "intent_type": "drug_conflict", "confidence": 0.75, "reason": "route by text: drug conflict"}
    if any(k in text for k in ["用药记录", "记录", "添加", "吃了", "服用", "mg", "毫克"]):
        return {"target_agent": "drug_record_agent", "intent_type": "drug_record", "confidence": 0.7, "reason": "route by text: drug record"}
    if any(k in text for k in ["档案", "病历", "历史记录", "就诊"]):
        return {"target_agent": "main_qa_agent", "intent_type": "archive", "confidence": 0.7, "reason": "route by text: archive"}
    return {"target_agent": "main_qa_agent", "intent_type": "general", "confidence": 0.6, "reason": "route by text: default general"}


async def _predict_intent_for_query(query: str) -> dict | None:
    try:
        clf = IntentClassifier()
        sub_intent = await clf.predict(text=query, stream=False)
        return {"intent": sub_intent.intent, "confidence": sub_intent.confidence, "reason": sub_intent.reason}
    except Exception:
        return None


def _detect_dependencies(query: str, previous_queries: list[str]) -> list[str]:
    deps: list[str] = []
    pronouns = ["它", "这", "那个", "上面", "刚才", "之前", "继续", "然后", "还有"]
    if any(p in query for p in pronouns):
        for i, _ in enumerate(previous_queries):
            deps.append(f"s{i + 1}")
    return deps


async def _extract_drug_info_from_text(text: str) -> dict:
    drug_info: dict = {"dosage": "未指定", "frequency": "未指定", "start_date_text": "未指定", "purpose": "未指定"}

    dosage_patterns = [
        r"(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)",
        r"(一次|每次)\s*(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)",
        r"(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)\s*(一次|每次)",
    ]
    frequency_patterns = [
        r"(一天|每日)\s*(\d+)\s*次",
        r"(\d+)\s*次\s*(一天|每日)",
        r"(早晚|早中晚|早中晚各一次|早晚各一次|早中晚各一次)",
        r"(需要时|必要时|疼痛时|不适时)",
    ]
    start_date_patterns = [
        r"(今天|昨天|前天|\d+月\d+日|\d+年\d+月\d+日|\d{4}-\d{2}-\d{2})",
        r"(从|自)\s*(今天|昨天|前天|\d+月\d+日|\d+年\d+月\d+日|\d{4}-\d{2}-\d{2})",
        r"(开始|起)\s*(今天|昨天|前天|\d+月\d+日|\d+年\d+月\d+日|\d{4}-\d{2}-\d{2})",
    ]
    purpose_patterns = [
        r"(用于|治疗|缓解|针对)\s*([^，。！？]{1,20})",
        r"(因为|由于)\s*([^，。！？]{1,20})\s*(而|所以)",
        r"(头痛|发烧|感冒|疼痛|炎症|高血压|糖尿病|冠心病|哮喘)",
    ]

    for pattern in dosage_patterns:
        match = re.search(pattern, text)
        if match:
            drug_info["dosage"] = match.group(0)
            break
    for pattern in frequency_patterns:
        match = re.search(pattern, text)
        if match:
            drug_info["frequency"] = match.group(0)
            break
    for pattern in start_date_patterns:
        match = re.search(pattern, text)
        if match:
            drug_info["start_date_text"] = match.group(1) if match.groups() else match.group(0)
            break
    for pattern in purpose_patterns:
        match = re.search(pattern, text)
        if match:
            drug_info["purpose"] = match.group(2) if len(match.groups()) > 1 else match.group(1)
            break

    return drug_info


async def _commit_drug_record(user_id: str, sm: DrugRecordStateMachine) -> dict:
    tool = DrugRecordTool()
    info = sm.collected_info
    start_date = None
    date_text = info.get("start_date_text", "")
    if date_text and date_text != "未指定":
        start_date = tool._parse_date_text(date_text)

    result = await tool.add_record(
        user_id=user_id,
        drug_name=sm.drug_name,
        dosage=info.get("dosage", "未指定"),
        frequency=info.get("frequency", "未指定"),
        time_text=date_text if date_text != "未指定" else "",
        start_date=start_date,
    )
    return result


async def _process_drug_record_state_machine(state: dict) -> dict:
    user_input = (state.get("user_input") or "").strip()
    user_id = state.get("user_id")

    sm_data = (state.get("private_scratchpads") or {}).get("drug_record_sm")

    if sm_data:
        sm = DrugRecordStateMachine.from_dict(sm_data)

        if sm.is_expired():
            sm.transition(DrugRecordPhase.EXPIRED)
            state.setdefault("private_scratchpads", {})["drug_record_sm"] = None
            state["final_response"] = "用药记录收集已超时取消。如需重新记录，请随时告诉我。"
            return state

        input_type = InputClassifier.classify(user_input, sm.current_field)

        if input_type == "negative":
            sm.transition(DrugRecordPhase.CANCELLED)
            state.setdefault("private_scratchpads", {})["drug_record_sm"] = None
            state["final_response"] = "好的，已取消用药记录收集。"
            return state

        if sm.phase == DrugRecordPhase.COLLECTING:
            if input_type == "irrelevant":
                result = sm.handle_irrelevant_input(user_input)
                if result["phase"] == "cancelled":
                    state.setdefault("private_scratchpads", {})["drug_record_sm"] = None
                    state["final_response"] = result["message"]
                    return state
                state["final_response"] = result["message"]
                state.setdefault("private_scratchpads", {})["drug_record_sm"] = sm.to_dict()
                return state

            result = sm.collect_answer(user_input)
            if result["phase"] == "confirming":
                state["final_response"] = result["summary"]
            else:
                state["final_response"] = result["question"]
            state.setdefault("private_scratchpads", {})["drug_record_sm"] = sm.to_dict()
            return state

        if sm.phase == DrugRecordPhase.CONFIRMING:
            if input_type == "affirmative":
                commit_result = await _commit_drug_record(user_id, sm)
                sm.transition(DrugRecordPhase.COMMITTED)
                state.setdefault("private_scratchpads", {})["drug_record_sm"] = None
                if commit_result.get("created"):
                    state["final_response"] = f"已为您记录用药信息：{sm.drug_name}。如需修改，请随时告诉我。"
                else:
                    state["final_response"] = commit_result.get("message", "用药记录已存在。")
                return state
            elif input_type == "negative":
                sm.transition(DrugRecordPhase.CANCELLED)
                state.setdefault("private_scratchpads", {})["drug_record_sm"] = None
                state["final_response"] = "好的，已取消用药记录。"
                return state
            else:
                sm.transition(DrugRecordPhase.COLLECTING)
                result = sm.collect_answer(user_input)
                state["final_response"] = result.get("question", result.get("summary", ""))
                state.setdefault("private_scratchpads", {})["drug_record_sm"] = sm.to_dict()
                return state

    candidate_events = state.get("candidate_drug_events")
    if candidate_events and user_id:
        drug_event = candidate_events[0]
        extracted = await _extract_drug_info_from_text(drug_event.get("full_text", ""))

        sm = DrugRecordStateMachine(drug_name=drug_event["drug_name"], initial_info=extracted)
        sm.transition(DrugRecordPhase.COLLECTING)

        missing = sm.get_missing_fields()
        if not missing:
            sm.transition(DrugRecordPhase.CONFIRMING)
            state["final_response"] = sm.confirmation_summary
        else:
            question = sm.next_question()
            state["final_response"] = question

        state.setdefault("private_scratchpads", {})["drug_record_sm"] = sm.to_dict()
        state.pop("candidate_drug_events", None)

    return state


async def input_check(state: dict) -> dict:
    return state


async def memory_load(state: dict) -> dict:
    if state.get("error_msg"):
        return state

    user_id = state.get("user_id")
    session_id = state.get("session_id")

    if not user_id or not session_id:
        state["history"] = []
        state["history_text"] = ""
        state["memory_summary"] = ""
        state["long_memory_items"] = []
        state["long_memory_text"] = ""
        state["shared_facts"] = {}
        state["private_scratchpads"] = {}
        state["proposed_updates"] = []
        state["skill_ctx"] = {}
        state["retrieved_knowledge"] = {}
        return state

    mem = MemoryService()
    history = await mem.get_user_memory(user_id=user_id, session_id=session_id, limit=12)
    state["history"] = history
    state["history_text"] = _format_history(history, max_chars=1400)
    state["memory_summary"] = await mem.get_memory_summary(user_id=user_id, session_id=session_id)
    state.setdefault("shared_facts", {})
    state.setdefault("private_scratchpads", {})
    state.setdefault("proposed_updates", [])
    state.setdefault("skill_ctx", {})
    state.setdefault("retrieved_knowledge", {})

    try:
        rt_state = await AgentStateStore().get_state(user_id=user_id, session_id=session_id)
        state["session_runtime_state"] = rt_state
        pending = rt_state.get("pending_confirmation") if isinstance(rt_state, dict) else None
        if isinstance(pending, dict):
            state["pending_confirmation"] = pending

        saved_sm = (rt_state.get("private_scratchpads") or {}).get("drug_record_sm") if isinstance(rt_state, dict) else None
        if saved_sm:
            state.setdefault("private_scratchpads", {})["drug_record_sm"] = saved_sm
    except Exception:
        state["session_runtime_state"] = {}

    state["long_memory_items"] = []
    state["long_memory_text"] = ""
    try:
        start_time = time.time()
        svc = LongMemoryService()
        query = state.get("user_input", "")
        if svc.is_enabled():
            items = await svc.recall(user_id=user_id, query=query, top_k=6)

            drug_keywords = ["药", "药物", "服用", "吃了", "吃过", "布洛芬", "阿司匹林", "抗生素", "降压药", "降糖药"]
            drug_related_items = []
            other_items = []

            seen = set()
            for it in items:
                if it.memory_id in seen:
                    continue
                seen.add(it.memory_id)
                text_val = it.text
                is_drug_related = any(keyword in text_val for keyword in drug_keywords)
                if is_drug_related:
                    drug_related_items.append(it)
                else:
                    other_items.append(it)

            filtered_items = drug_related_items + other_items
            filtered_items = filtered_items[:5]

            state["long_memory_items"] = [
                {"memory_id": it.memory_id, "text": it.text, "memory_type": it.memory_type, "source": it.source, "session_id": it.session_id, "created_at": it.created_at}
                for it in filtered_items
            ]
            if filtered_items:
                state["long_memory_text"] = "\n".join([f"- {it.text}" for it in filtered_items])

            retrieval_time_ms = int((time.time() - start_time) * 1000)
            logger.info("long_memory recall done count=%s cost_ms=%s", len(filtered_items), retrieval_time_ms)
    except Exception:
        logger.exception("long_memory recall failed")

    if not state.get("long_memory_items"):
        logger.debug("long_memory recall empty")

    return state


async def intent_recognition(state: dict) -> dict:
    from app.core.agent.intent_classifier import IntentClassifier

    text = state.get("user_input", "").strip().lower()

    clf = IntentClassifier()
    res = await clf.predict(text=text, stream=bool(state.get("stream", False)))

    state["intent"] = res.intent
    state["intent_confidence"] = res.confidence
    state["intent_reason"] = res.reason

    return state


async def entity_extraction(state: dict) -> dict:
    intent = state.get("intent")
    text = state.get("user_input", "")

    entities: dict = {}
    if intent == "drug":
        entities["drug_name_list"] = DrugEntityExtractor.extract_drug_candidates(text, max_items=10)
    elif intent == "lab":
        entities["raw"] = text
    else:
        entities["query"] = text

    state["extract_entities"] = entities
    return state


async def knowledge_retrieve(state: dict) -> dict:
    if state.get("error_msg"):
        return state
    start_time = time.time()
    try:
        svc = MedicalKnowledgeService()
        state["retrieved_knowledge"] = await svc.retrieve(
            user_input=state.get("user_input", ""),
            intent=state.get("intent", "general"),
        )
    except Exception:
        state["retrieved_knowledge"] = {}

    logger.info("knowledge_retrieve core intent=%s items=%s elapsed=%.3fs", state.get("intent"), len(state.get("retrieved_knowledge") or {}), time.time() - start_time)

    if state.get("intent") == "general":
        try:
            public_kb = PublicKnowledgeService()
            state["retrieved_knowledge"]["public_kb"] = await public_kb.retrieve(query=state.get("user_input", ""))
        except Exception:
            state.setdefault("retrieved_knowledge", {})["public_kb"] = []
        logger.info("knowledge_retrieve public_kb count=%s", len((state.get("retrieved_knowledge") or {}).get("public_kb") or []))
    return state


async def plan_node(state: dict) -> dict:
    if state.get("error_msg"):
        return state

    user_input = state.get("user_input", "")

    sm_data = (state.get("private_scratchpads") or {}).get("drug_record_sm")
    if sm_data:
        sm = DrugRecordStateMachine.from_dict(sm_data)
        if sm.is_active():
            state["execution_plan"] = ExecutionPlan(
                steps=[PlanStep(step_id="s1", query=user_input, target_agent="drug_record_agent", intent_type="drug_record_sm", depends_on=[], execution_strategy="serial")],
                strategy="serial",
                conflict_resolution_policy="none",
            )
            state["plan_phase"] = "planning"
            return state

    sub_queries = _split_user_queries(user_input)

    if len(sub_queries) <= 1:
        route_result = _route_by_intent_and_text(state)
        state["intent_analysis"] = route_result
        state["target_agent"] = route_result["target_agent"]
        state["intent_type"] = route_result.get("intent_type", state.get("intent", "general"))

        plan = ExecutionPlan(
            steps=[PlanStep(
                step_id="s1",
                query=user_input,
                target_agent=route_result["target_agent"],
                intent_type=route_result.get("intent_type", "general"),
                depends_on=[],
                execution_strategy="serial",
            )],
            strategy="serial",
            conflict_resolution_policy="evidence_priority",
        )
    else:
        steps = []
        for i, q in enumerate(sub_queries):
            pred = await _predict_intent_for_query(q)
            intent_val = pred.get("intent", "general") if pred else "general"
            conf_val = pred.get("confidence", 0.5) if pred else 0.5

            sub_state = dict(state)
            sub_state["user_input"] = q
            sub_state["intent"] = intent_val
            sub_state["intent_confidence"] = conf_val
            route_result = _route_by_intent_and_text(sub_state)

            deps = _detect_dependencies(q, sub_queries[:i])

            steps.append(PlanStep(
                step_id=f"s{i + 1}",
                query=q,
                target_agent=route_result["target_agent"],
                intent_type=route_result.get("intent_type", intent_val),
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

    logger.info("plan_node plan=%s", json.dumps({k: v for k, v in plan.items()}, ensure_ascii=False, default=str))
    return state


def _group_steps_by_dependency(steps: list[PlanStep]) -> list[list[PlanStep]]:
    topo: list[list[PlanStep]] = []
    remaining = list(steps)
    completed_ids: set[str] = set()
    while remaining:
        ready = [s for s in remaining if all(d in completed_ids for d in s.get("depends_on", []))]
        if not ready:
            ready = [remaining[0]]
        topo.append(ready)
        for s in ready:
            completed_ids.add(s["step_id"])
            remaining.remove(s)
    return topo


def _build_sub_state(state: dict, step: PlanStep) -> dict:
    sub_state = dict(state)
    sub_state["user_input"] = step["query"]
    sub_state["target_agent"] = step["target_agent"]
    sub_state["intent_type"] = step.get("intent_type", "general")
    for key in ("final_response", "error_msg", "intent_analysis", "extract_entities", "tool_result", "llm_output"):
        sub_state.pop(key, None)
    return sub_state


async def _execute_single_step(sub_state: dict, step: PlanStep) -> dict:
    from app.core.agent.agent_router import AgentRouter

    target_agent = step["target_agent"]

    if target_agent == "drug_record_agent" and step.get("intent_type") == "drug_record_sm":
        result = await _process_drug_record_state_machine(sub_state)
        result["intent_type"] = "drug_record_sm"
        return result

    router = AgentRouter()
    try:
        result_state = await router.route_and_execute(sub_state)
        result_state["intent_type"] = step.get("intent_type", "general")
        return result_state
    except Exception as e:
        logger.error("execute_single_step failed step=%s error=%s", step["step_id"], e)
        sub_state["error_msg"] = f"Agent执行失败: {str(e)}"
        sub_state["final_response"] = f"处理'{step['query']}'时出现错误，请稍后重试。"
        sub_state["intent_type"] = step.get("intent_type", "general")
        return sub_state


async def execute_node(state: dict) -> dict:
    if state.get("error_msg"):
        return state

    plan = state.get("execution_plan", {})
    steps = plan.get("steps", [])
    strategy = plan.get("strategy", "serial")
    results: dict[str, dict] = {}

    if not steps:
        state["plan_step_results"] = results
        state["plan_phase"] = "executing"
        return state

    if strategy == "serial" or len(steps) <= 1:
        for step in steps:
            sub_state = _build_sub_state(state, step)
            result = await _execute_single_step(sub_state, step)
            results[step["step_id"]] = result
            state.update({k: v for k, v in result.items() if k in ("final_response", "error_msg", "tool_result", "llm_output", "extract_entities")})
    else:
        groups = _group_steps_by_dependency(steps)
        for group in groups:
            tasks = [_execute_single_step(_build_sub_state(state, step), step) for step in group]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            for step, result in zip(group, group_results):
                if isinstance(result, Exception):
                    results[step["step_id"]] = {"error_msg": str(result), "final_response": f"处理'{step['query']}'时出现错误。", "intent_type": step.get("intent_type", "general")}
                else:
                    results[step["step_id"]] = result

    state["plan_step_results"] = results
    state["plan_phase"] = "executing"

    logger.info("execute_node done steps=%s results_keys=%s", [s["step_id"] for s in steps], list(results.keys()))
    return state


async def reconcile_node(state: dict) -> dict:
    if state.get("error_msg") and not state.get("plan_step_results"):
        return state

    results = state.get("plan_step_results", {})
    plan = state.get("execution_plan", {})

    if not results:
        return state

    if len(results) == 1:
        single = next(iter(results.values()))
        if single.get("final_response"):
            state["final_response"] = single["final_response"]
        if single.get("error_msg") and not state.get("final_response"):
            state["final_response"] = single["error_msg"]
        state["plan_phase"] = "reconciling"
        return state

    sections: list[str] = []
    drug_conflict_interactions: list[dict] = []

    for step_id, result in results.items():
        step = next((s for s in plan.get("steps", []) if s["step_id"] == step_id), None)
        query = step.get("query", "") if step else ""
        content = result.get("final_response", "") or result.get("error_msg", "")
        intent_type = result.get("intent_type", "general")

        if content:
            sections.append(f"**{query}**\n{content}")

        if intent_type == "drug_conflict":
            interactions = (result.get("tool_result") or {}).get("interaction_result", [])
            if interactions:
                drug_conflict_interactions.extend(interactions)

    if drug_conflict_interactions:
        lines = ["⚠️ 跨任务药物冲突提醒："]
        for it in drug_conflict_interactions:
            lines.append(f"- {it.get('drug_a', '')} + {it.get('drug_b', '')}：{it.get('interaction_desc', '')}")
        sections.append("\n".join(lines))

    if sections:
        state["final_response"] = "我分条为你处理如下：\n\n" + "\n\n".join([f"{i + 1}. {s}" for i, s in enumerate(sections)])
        state["intent"] = "multi"
        state["intent_type"] = "multi"

    state["plan_phase"] = "reconciling"
    return state


async def response_plan(state: dict) -> dict:
    if state.get("error_msg") and state.get("final_response"):
        return state

    intent = state.get("intent") or "general"
    user_input = state.get("user_input", "")
    tool_name = state.get("tool_name") or ""

    mode = "llm_chat"
    if intent in ("archive", "drug", "lab") or tool_name in ("archive", "drug_interaction", "lab_report"):
        mode = "llm_format"

    long_mem = (state.get("long_memory_text") or "").strip()
    knowledge = state.get("retrieved_knowledge") or {}
    need_mem = _need_contextual_memory(user_input) or bool(long_mem)
    if knowledge:
        need_mem = True
    if not (state.get("history") or state.get("memory_summary") or long_mem or knowledge):
        need_mem = False

    state["response_mode"] = mode
    state["inject_memory"] = bool(need_mem)
    return state


async def llm_generate(state: dict) -> dict:
    if state.get("final_response"):
        state["llm_output"] = state["final_response"]
        return state

    if state.get("needs_confirmation") and state.get("confirmation_message"):
        state["llm_output"] = state["confirmation_message"]
        return state

    content = (state.get("tool_result") or {}).get("final_desc") or ""
    if not content:
        content = "当前未获取到有效工具结果。"

    mode = state.get("response_mode") or "llm_chat"
    inject_memory = bool(state.get("inject_memory"))

    mem_summary = (state.get("memory_summary") or "").strip()
    if not mem_summary and inject_memory:
        mem_summary = _short_window_history(state.get("history") or [], max_turns=4)

    long_mem = (state.get("long_memory_text") or "").strip()
    retrieved_knowledge = state.get("retrieved_knowledge") or {}
    if retrieved_knowledge:
        logger.info("llm_generate inject_knowledge keys=%s", list(retrieved_knowledge.keys()))

    candidate_drug_events = state.get("candidate_drug_events")
    if candidate_drug_events:
        skill = MedicationConfirmationSkill()
        state["llm_output"] = skill.build_confirmation_message(candidate_drug_events)
        state["skill_ctx"] = state.get("skill_ctx") or {}
        state["skill_ctx"]["medication_confirmation"] = {"candidate_events": candidate_drug_events}
        return state

    if mode == "llm_format":
        system_prompt = (
            "你是医疗问答助手，任务是把\"工具/数据库查询结果\"用清晰、自然、结构化的中文表达出来。\n"
            "要求：\n"
            "1) 只能基于提供的工具结果输出，不得编造任何未给出的事实。\n"
            "2) 输出尽量简洁，分点呈现，必要时补充就医建议边界。\n"
            "3) 禁止给出诊断结论或处方/调整用药建议。\n"
        )
        user_prompt = (f"用户问题：{state.get('user_input', '')}\n\n" f"工具结果：\n{content}\n")
        if long_mem:
            user_prompt = f"长期记忆（用户历史偏好/事实，供参考，可能与本轮有关）：\n{long_mem}\n\n" + user_prompt
        if inject_memory and mem_summary:
            user_prompt = f"会话记忆（可能与本轮有关）：\n{mem_summary}\n\n" + user_prompt
        if retrieved_knowledge:
            user_prompt = f"医疗知识库检索结果（供参考）：\n{json.dumps(retrieved_knowledge, ensure_ascii=False)}\n\n" + user_prompt
    else:
        system_prompt = (
            "你是医疗问答助手，需要用自然的对话方式回答用户。\n"
            "原则：\n"
            "1) 如提供了会话记忆/长期记忆或工具结果，你必须优先使用它们来保持上下文一致。\n"
            "2) 不得编造不存在的个人信息/检查结果/用药记录。\n"
            "3) 禁止诊断与处方/调整用药建议；可以给出通用科普与就医指引。\n"
        )
        parts = []
        if long_mem:
            parts.append("长期记忆（用户历史偏好/事实）：\n" + long_mem)
        if inject_memory and mem_summary:
            parts.append("会话记忆：\n" + mem_summary)
        if content:
            parts.append("工具/检索结果：\n" + content)
        if retrieved_knowledge:
            parts.append("医疗知识库检索结果（供参考）：\n" + json.dumps(retrieved_knowledge, ensure_ascii=False))
        parts.append("用户输入：\n" + (state.get("user_input", "") or ""))
        user_prompt = "\n\n".join(parts)

    try:
        llm = LLMService()
        raw = await llm.chat_completion(prompt=user_prompt, system_prompt=system_prompt, stream=False, timeout_s=12.0, max_tokens=900)
        state["llm_output"] = (raw or "").strip() or content
    except Exception as e:
        logger.error(f"LLM生成失败: {e}")
        state["llm_output"] = content

    return state


async def output_check_and_disclaimer(state: dict) -> dict:
    state["final_response"] = state.get("llm_output", "") or state.get("final_response", "")

    proposed = state.get("proposed_updates") or []
    proposed.append({"scope": "shared", "key": "latest_response", "value": state.get("final_response", ""), "source": "out"})
    state["proposed_updates"] = proposed

    return state


async def commit_gate(state: dict) -> dict:
    if state.get("error_msg"):
        return state

    shared = dict(state.get("shared_facts") or {})
    allow_keys = {"intent", "target_agent", "extract_entities", "retrieved_knowledge", "latest_response"}

    updates_by_key: dict[str, dict] = {}
    for item in (state.get("proposed_updates") or []):
        if not isinstance(item, dict):
            continue
        if item.get("scope") != "shared":
            continue
        key = item.get("key")
        if key not in allow_keys:
            logger.warning("commit_gate rejected key=%s from source=%s", key, item.get("source"))
            continue
        if key in updates_by_key:
            existing_priority = updates_by_key[key].get("priority", 0)
            new_priority = item.get("priority", 0)
            if new_priority > existing_priority:
                updates_by_key[key] = item
        else:
            updates_by_key[key] = item

    for key, item in updates_by_key.items():
        shared[key] = item.get("value")

    if state.get("intent"):
        shared["intent"] = state.get("intent")
    if state.get("target_agent"):
        shared["target_agent"] = state.get("target_agent")
    if state.get("extract_entities"):
        shared["extract_entities"] = state.get("extract_entities")
    if state.get("retrieved_knowledge"):
        shared["retrieved_knowledge"] = state.get("retrieved_knowledge")

    state["shared_facts"] = shared
    state["proposed_updates"] = []
    return state


async def memory_update(state: dict) -> dict:
    if state.get("error_msg"):
        return state

    mem = MemoryService()
    await mem.update_user_memory(state["user_id"], state["session_id"], "user", state["user_input"])
    if "final_response" in state:
        await mem.update_user_memory(state["user_id"], state["session_id"], "assistant", state["final_response"])

    try:
        start_time = time.time()
        svc = LongMemoryService()
        if svc.is_enabled():
            items = await svc.extract_candidates(user_input=state.get("user_input", ""))
            if items:
                await svc.add_items(user_id=state["user_id"], session_id=state["session_id"], items=items)
                write_time_ms = int((time.time() - start_time) * 1000)
                logger.info("long_memory write done count=%s cost_ms=%s", len(items), write_time_ms)

                drug_events = [item for item in items if item.memory_type == "drug_event"]
                if drug_events:
                    drug_info_list = []
                    for event in drug_events:
                        original_text = event.text.replace("用户", "我")
                        drug_match = re.search(r"(?:吃了|服用了|用了|吃|服用|使用|用)([^，。！？\s]{1,30})", original_text)
                        if drug_match:
                            drug_name = drug_match.group(1).strip()
                            drug_info_list.append({"drug_name": drug_name, "full_text": original_text, "confidence": event.confidence})

                    if drug_info_list:
                        state["candidate_drug_events"] = drug_info_list
                        skill_ctx = state.get("skill_ctx") or {}
                        skill_ctx["medication_confirmation"] = {"candidate_events": drug_info_list}
                        state["skill_ctx"] = skill_ctx
    except Exception:
        logger.exception("long_memory write failed")

    try:
        user_id = state.get("user_id")
        session_id = state.get("session_id")
        runtime_state = state.get("session_runtime_state")
        if not isinstance(runtime_state, dict):
            runtime_state = {}
        pending = state.get("pending_confirmation")
        if isinstance(pending, dict) and pending:
            runtime_state["pending_confirmation"] = pending
        else:
            runtime_state.pop("pending_confirmation", None)

        scratchpads = state.get("private_scratchpads") or {}
        if scratchpads:
            runtime_state["private_scratchpads"] = scratchpads

        await AgentStateStore().upsert_state(user_id=user_id, session_id=session_id, state=runtime_state)
    except Exception:
        pass

    return state


async def error_finalize(state: dict) -> dict:
    if state.get("error_msg"):
        state["final_response"] = state["error_msg"]
    return state
