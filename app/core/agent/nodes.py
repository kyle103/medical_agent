from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import date, datetime

from sqlalchemy import and_, select

from app.common.logger import get_logger, log_node_execution, log_step_execution
from app.core.agent.intent_classifier import IntentClassifier, IntentResult
from app.core.agent.planner_agent import PlannerAgent
from app.core.agent.state import ExecutionPlan, PlanStep
from app.core.agent.tool_executor import ToolExecutor
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

_planner = PlannerAgent()
_tool_executor = ToolExecutor()


def _llm_enabled_for_nodes() -> bool:
    from app.config.settings import settings
    def _ok(v: str) -> bool:
        v = (v or '').strip()
        return bool(v) and not (v.startswith('{{') and v.endswith('}}'))
    return _ok(settings.LLM_API_BASE) and _ok(settings.LLM_API_KEY) and _ok(settings.LLM_MODEL_NAME)

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


_MEDICAL_QUERY_KEYWORDS = [
    # 疾病与症状
    "病", "症", "疼", "痛", "发烧", "发热", "咳嗽", "感冒", "发炎", "感染", "癌", "肿瘤",
    "高血压", "糖尿病", "哮喘", "过敏", "腹泻", "便秘", "失眠", "抑郁", "焦虑",
    "恶心", "呕吐", "头晕", "乏力", "胸闷", "心慌", "气短", "水肿", "出血",
    "皮疹", "瘙痒", "红肿", "溃疡", "结节", "囊肿", "息肉", "结石",
    # 药物与治疗
    "药", "用药", "服用", "剂量", "治疗", "手术", "检查", "化验", "检验",
    "打针", "输液", "吃药", "开药", "处方", "忌口",
    # 身体部位
    "心脏", "肝", "肾", "肺", "胃", "肠", "脑", "血管", "血液", "骨骼", "关节", "皮肤",
    "眼睛", "耳朵", "鼻子", "喉咙", "牙齿", "颈椎", "腰椎", "膝盖",
    # 医学概念
    "副作用", "禁忌", "相互作用", "疫苗", "预防", "康复", "护理", "营养",
    "指标", "血糖", "血压", "血脂", "尿酸", "转氨酶",
    "CT", "MRI", "X光", "B超", "核磁", "体检", "报告", "化验单",
    # 就医相关
    "挂号", "就诊", "就医", "看病", "科室", "医生", "医院", "急诊", "住院",
    # 健康疑问
    "什么原因", "怎么办", "怎么回事", "要注意什么", "会不会", "需不需要",
    "要不要", "能不能", "可以吗",
]


def _is_medical_query(text: str, intent: str, entities: dict | None = None) -> bool:
    """判断查询是否需要医疗知识检索。

    优先复用上游意图识别 + 实体提取的结果（零额外成本），
    仅当上游信息不足以判断时才回退到关键词匹配。

    drug/lab/archive 意图始终走 RAG；general 意图根据实体和关键词综合判断。
    """
    if intent in ("drug", "lab", "archive"):
        return True
    if intent != "general":
        return True
    t = (text or "").strip()
    if not t:
        return False
    # 上游已提取到医疗实体 → 走 RAG
    if entities:
        if entities.get("drug_name_list"):
            return True
        if entities.get("lab_items"):
            return True
    # 关键词兜底
    if any(kw in t for kw in _MEDICAL_QUERY_KEYWORDS):
        return True
    # 短文本 → 可能是闲聊
    if len(t) <= 30:
        return False
    if len(t) <= 80 and "?" not in t and "？" not in t:
        return False
    return True


async def _extract_drug_info_from_text(text: str) -> dict:
    from app.core.agent.llm_decision_service import LLMDecisionService

    default_info: dict = {"dosage": "未指定", "frequency": "未指定", "start_date_text": "未指定", "purpose": "未指定"}

    llm_decision = LLMDecisionService()
    llm_result = await llm_decision.extract_drug_info(text)
    if llm_result and llm_result.get("drug_name"):
        drug_info = dict(default_info)
        if llm_result.get("dosage"):
            drug_info["dosage"] = llm_result["dosage"]
        if llm_result.get("frequency"):
            drug_info["frequency"] = llm_result["frequency"]
        if llm_result.get("start_date_text"):
            drug_info["start_date_text"] = llm_result["start_date_text"]
        if llm_result.get("purpose"):
            drug_info["purpose"] = llm_result["purpose"]
        return drug_info

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
            default_info["dosage"] = match.group(0)
            break
    for pattern in frequency_patterns:
        match = re.search(pattern, text)
        if match:
            default_info["frequency"] = match.group(0)
            break
    for pattern in start_date_patterns:
        match = re.search(pattern, text)
        if match:
            default_info["start_date_text"] = match.group(1) if match.groups() else match.group(0)
            break
    for pattern in purpose_patterns:
        match = re.search(pattern, text)
        if match:
            default_info["purpose"] = match.group(2) if len(match.groups()) > 1 else match.group(1)
            break

    return default_info


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
    _t0 = time.perf_counter()
    from app.core.compliance.compliance_service import ComplianceService

    user_input = state.get("user_input", "")
    ok, msg = ComplianceService().input_compliance_check(user_input)
    if not ok:
        state["error_msg"] = msg
        logger.warning("input_check compliance blocked: %s", msg)

    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="input_check", latency_ms=latency_ms, blocked=bool(state.get("error_msg")))
    return state


async def memory_load(state: dict) -> dict:
    _t0 = time.perf_counter()
    if state.get("error_msg"):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="memory_load", latency_ms=latency_ms, skipped=True)
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

        if isinstance(rt_state, dict) and not rt_state.get("long_memory_flushed"):
            prev_history = await mem.get_user_memory(user_id=user_id, session_id=session_id, limit=50)
            if prev_history and len(prev_history) >= 2:
                asyncio.create_task(_async_flush_session_long_memory(user_id=user_id, session_id=session_id, history=prev_history))
                logger.info("long_memory session_end flush triggered: session=%s history_count=%s", session_id, len(prev_history))
                try:
                    rt_state["long_memory_flushed"] = True
                    await AgentStateStore().upsert_state(user_id=user_id, session_id=session_id, state=rt_state)
                except Exception:
                    pass
    except Exception:
        state["session_runtime_state"] = {}

    if len(history) > 20:
        try:
            svc = LongMemoryService()
            if svc.is_enabled():
                asyncio.create_task(_async_compress_and_write_long_memory(user_id=user_id, session_id=session_id, history=history))
                logger.info("long_memory compress_pre_write triggered: session=%s history_count=%s", session_id, len(history))
        except Exception:
            pass

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

    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="memory_load", latency_ms=latency_ms, long_mem_count=len(state.get("long_memory_items") or []))
    return state


async def intent_recognition(state: dict) -> dict:
    _t0 = time.perf_counter()
    from app.core.agent.intent_classifier import IntentClassifier
    from app.core.agent.llm_decision_service import LLMDecisionService

    text = state.get("user_input", "").strip().lower()
    user_input_raw = state.get("user_input", "")

    if _detect_memory_save_intent(user_input_raw):
        state["force_long_memory_write"] = True
        state["long_memory_write_source"] = "explicit"
        logger.info("memory_save intent detected: user_input=%s", user_input_raw[:50])

    if _llm_enabled_for_nodes():
        llm_decision = LLMDecisionService()
        combined = await llm_decision.classify_route_and_extract(user_input_raw)

        if combined and combined.get("target_name") and combined.get("confidence", 0) >= 0.5:
            state["intent"] = combined.get("intent", "general")
            state["intent_confidence"] = combined.get("confidence", 0.8)
            state["intent_reason"] = combined.get("reason", "llm_route")
            state["intent_analysis"] = combined
            state["target_agent"] = combined.get("target_name", "")
            state["intent_type"] = combined.get("intent_type", "general")

            entities: dict = {}
            llm_entities = combined.get("entities", {})
            intent = state["intent"]

            if intent == "drug" and isinstance(llm_entities, dict):
                if llm_entities.get("drug_name_list"):
                    entities["drug_name_list"] = llm_entities["drug_name_list"]
                    if llm_entities.get("dosage"):
                        entities["dosage"] = llm_entities["dosage"]
                    if llm_entities.get("frequency"):
                        entities["frequency"] = llm_entities["frequency"]
                    if llm_entities.get("start_date_text"):
                        entities["start_date_text"] = llm_entities["start_date_text"]
                if not entities.get("drug_name_list"):
                    entities["drug_name_list"] = DrugEntityExtractor.extract_drug_candidates(text, max_items=10)
            elif intent == "lab" and isinstance(llm_entities, dict):
                if llm_entities.get("lab_items"):
                    entities.update(llm_entities)
                else:
                    entities["raw"] = text
            else:
                entities["query"] = text

            state["extract_entities"] = entities
            latency_ms = int((time.perf_counter() - _t0) * 1000)
            log_node_execution(node_name="intent_recognition", latency_ms=latency_ms, intent=state.get("intent"), confidence=state.get("intent_confidence"), entity_keys=list(entities.keys()), merged_llm=True)
            return state

    clf = IntentClassifier()
    try:
        route_result = await clf.predict(text=text, stream=False)
    except Exception:
        route_result = IntentResult(intent="general", confidence=0.5, reason="fallback")

    state["intent"] = route_result.intent
    state["intent_confidence"] = route_result.confidence
    state["intent_reason"] = route_result.reason

    entities: dict = {}
    intent = state.get("intent", "general")
    if intent == "drug":
        entities["drug_name_list"] = DrugEntityExtractor.extract_drug_candidates(text, max_items=10)
    elif intent == "lab":
        entities["raw"] = text
    else:
        entities["query"] = text

    state["extract_entities"] = entities
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="intent_recognition", latency_ms=latency_ms, intent=state.get("intent"), confidence=state.get("intent_confidence"), entity_keys=list(entities.keys()), fallback=True)
    return state


async def entity_extraction(state: dict) -> dict:
    _t0 = time.perf_counter()

    if state.get("extract_entities"):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="entity_extraction", latency_ms=latency_ms, skipped=True, reason="pre_extracted")
        return state

    from app.core.agent.llm_decision_service import LLMDecisionService

    intent = state.get("intent")
    text = state.get("user_input", "")

    entities: dict = {}
    if intent == "drug":
        llm_decision = LLMDecisionService()
        try:
            llm_entities = await llm_decision.extract_entities(text, "drug")
        except Exception:
            llm_entities = None
        if llm_entities and llm_entities.get("drug_name_list"):
            entities["drug_name_list"] = llm_entities["drug_name_list"]
            if llm_entities.get("dosage"):
                entities["dosage"] = llm_entities["dosage"]
            if llm_entities.get("frequency"):
                entities["frequency"] = llm_entities["frequency"]
            if llm_entities.get("start_date_text"):
                entities["start_date_text"] = llm_entities["start_date_text"]
        else:
            entities["drug_name_list"] = DrugEntityExtractor.extract_drug_candidates(text, max_items=10)
            entities["_fallback"] = "regex"
    elif intent == "lab":
        llm_decision = LLMDecisionService()
        try:
            llm_entities = await llm_decision.extract_entities(text, "lab")
        except Exception:
            llm_entities = None
        if llm_entities and llm_entities.get("lab_items"):
            entities.update(llm_entities)
        else:
            entities["raw"] = text
            entities["_fallback"] = "raw"
    else:
        entities["query"] = text

    state["extract_entities"] = entities
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    fallback = entities.pop("_fallback", None)
    log_node_execution(node_name="entity_extraction", latency_ms=latency_ms, intent=intent, entity_keys=list(entities.keys()), fallback=fallback)
    return state


async def knowledge_retrieve(state: dict) -> dict:
    _t0 = time.perf_counter()
    if state.get("error_msg"):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="knowledge_retrieve", latency_ms=latency_ms, skipped=True)
        return state

    user_input = state.get("user_input", "")
    intent = state.get("intent", "general")

    from app.config.settings import settings
    if settings.ENABLE_SELECTIVE_RAG and not _is_medical_query(user_input, intent, state.get("extract_entities")):
        state["retrieved_knowledge"] = {}
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="knowledge_retrieve", latency_ms=latency_ms, skipped=True, reason="non_medical")
        return state

    async def _retrieve_drug_knowledge():
        try:
            from app.db.chroma_store import is_milvus_configured
            if is_milvus_configured():
                svc = MedicalKnowledgeService()
                return await svc.retrieve(user_input=user_input, intent=intent)
            return {}
        except Exception:
            return {}

    async def _retrieve_public_kb():
        try:
            from app.db.chroma_store import is_milvus_configured
            if is_milvus_configured():
                public_kb = PublicKnowledgeService()
                return await public_kb.retrieve(query=user_input)
            return []
        except Exception:
            return []

    need_public = intent == "general"
    if need_public:
        drug_result, public_result = await asyncio.gather(
            _retrieve_drug_knowledge(),
            _retrieve_public_kb(),
        )
        state["retrieved_knowledge"] = drug_result
        state["retrieved_knowledge"]["public_kb"] = public_result
    else:
        state["retrieved_knowledge"] = await _retrieve_drug_knowledge()

    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="knowledge_retrieve", latency_ms=latency_ms, intent=intent, knowledge_keys=list((state.get("retrieved_knowledge") or {}).keys()))
    return state


async def plan_node(state: dict) -> dict:
    _t0 = time.perf_counter()
    if state.get("error_msg"):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="plan_node", latency_ms=latency_ms, skipped=True)
        return state

    if state.get("needs_replan"):
        state["needs_replan"] = False
        state["plan_phase"] = "planning"
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="plan_node", latency_ms=latency_ms, replan=True, replan_reason=state.get("replan_reason", ""))
        logger.info("plan_node: replan using revised plan, plan_phase=%s", state.get("plan_phase"))
        return state

    state = await _planner.generate_plan(state)

    route_result = state.get("intent_analysis") or {}
    if route_result:
        state["target_agent"] = route_result.get("target_name", "")

    plan = state.get("execution_plan", {})
    steps = plan.get("steps", [])
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="plan_node", latency_ms=latency_ms, step_count=len(steps), strategy=plan.get("strategy", ""))
    return state


def _group_steps_by_dependency(steps: list[PlanStep]) -> list[list[PlanStep]]:
    """按依赖关系拓扑分层：同层步骤可并行，层间必须串行。

    如果某个步骤声明的依赖指向不存在的步骤 ID（孤依赖），
    将其降级为无依赖步骤并告警，而不是静默打乱执行顺序。
    """
    all_ids = {s["step_id"] for s in steps}
    topo: list[list[PlanStep]] = []
    remaining = list(steps)
    completed_ids: set[str] = set()

    for s in remaining:
        orphans = [d for d in s.get("depends_on", []) if d not in all_ids]
        if orphans:
            logger.warning(
                "_group_steps_by_dependency: step=%s has orphan deps=%s — treating as no deps",
                s["step_id"], orphans,
            )
            s["depends_on"] = [d for d in s.get("depends_on", []) if d not in orphans]

    while remaining:
        ready = [s for s in remaining if all(d in completed_ids for d in s.get("depends_on", []))]
        if not ready:
            remaining_ids = [s["step_id"] for s in remaining]
            logger.error(
                "_group_steps_by_dependency: circular or unresolvable deps among %s — falling back to serial",
                remaining_ids,
            )
            for s in remaining:
                topo.append([s])
                completed_ids.add(s["step_id"])
            break
        topo.append(ready)
        for s in ready:
            completed_ids.add(s["step_id"])
            remaining.remove(s)
    return topo


def _build_sub_state(state: dict, step: PlanStep, step_results: dict | None = None) -> dict:
    sub_state = dict(state)
    sub_state["user_input"] = step["query"]
    sub_state["target_agent"] = step.get("target_name", "")
    sub_state["intent_type"] = step.get("intent_type", "general")

    for key in ("final_response", "error_msg", "intent_analysis", "extract_entities", "tool_result", "llm_output"):
        sub_state.pop(key, None)

    original_input = state.get("original_user_input") or state.get("user_input", "")
    step_query = step.get("query", "")
    has_deps = step.get("depends_on") and step_results

    if not has_deps and original_input and step_query and original_input != step_query:
        sub_state["user_input"] = f"[背景信息：用户原始问题是「{original_input}」]\n当前需要回答的部分：{step_query}"

    if step_results and step.get("depends_on"):
        structured_ctx = _build_structured_context(step, step_results, original_input, step_query)
        sub_state["user_input"] = structured_ctx["prompt"]
        sub_state["step_context"] = structured_ctx["context"]
        sub_state["extract_entities"] = structured_ctx["merged_entities"]

    return sub_state


def _DEFAULT_TRUNCATE_CHARS() -> int:
    return 800


def _build_structured_context(
    step: PlanStep,
    step_results: dict,
    original_input: str,
    step_query: str,
) -> dict:
    """从依赖步骤结果中提取结构化上下文，供下游步骤使用。

    返回:
      prompt: 拼接后的 user_input 文本
      context: 结构化上下文 dict，含 summaries / entities / lab_items / key_findings
      merged_entities: 合并后的实体 dict（供 extract_entities 使用）
    """
    dep_summaries: list[str] = []
    merged_drug_names: list[str] = []
    merged_lab_items: list[dict] = []
    key_findings: list[str] = []
    max_chars = _DEFAULT_TRUNCATE_CHARS()

    for dep_id in step["depends_on"]:
        dep_result = step_results.get(dep_id)
        if not isinstance(dep_result, dict):
            continue

        dep_response = dep_result.get("final_response", "")
        if dep_response:
            truncated = dep_response[:max_chars] + ("...(内容过长已截断)" if len(dep_response) > max_chars else "")
            dep_summaries.append(f"[步骤{dep_id}的结果]: {truncated}")

            findings = _extract_key_findings(dep_response)
            if findings:
                key_findings.extend(findings)
        elif dep_result.get("tool_result"):
            tool_res = dep_result["tool_result"]
            if isinstance(tool_res, dict):
                tool_text = json.dumps(tool_res, ensure_ascii=False)[:max_chars]
                dep_summaries.append(f"[步骤{dep_id}的工具结果]: {tool_text}")

        # 合并实体：药品名称
        dep_entities = dep_result.get("extract_entities") or {}
        if isinstance(dep_entities, dict):
            dep_drug_names = dep_entities.get("drug_name_list", [])
            if isinstance(dep_drug_names, list):
                merged_drug_names.extend(dep_drug_names)

        # 合并实体：化验指标
        dep_tool_result = dep_result.get("tool_result") or {}
        if isinstance(dep_tool_result, dict):
            tool_drug_list = dep_tool_result.get("drug_list", [])
            for d in tool_drug_list:
                dn = d.get("drug_name", "") if isinstance(d, dict) else ""
                if dn and d.get("match_status") == "匹配成功":
                    merged_drug_names.append(dn)

            tool_lab_items = dep_tool_result.get("item_list", [])
            if isinstance(tool_lab_items, list) and tool_lab_items:
                for item in tool_lab_items:
                    if isinstance(item, dict):
                        merged_lab_items.append({
                            "item_name": item.get("item_name", ""),
                            "test_value": item.get("test_value", ""),
                            "reference_range": item.get("reference_range", ""),
                            "abnormal_flag": item.get("abnormal_flag", ""),
                        })

    # 如果从结构化结果中没拿到药名，从 final_response 文本中正则兜底提取
    if not merged_drug_names:
        for dep_id in step["depends_on"]:
            dep_result = step_results.get(dep_id)
            if not isinstance(dep_result, dict):
                continue
            dep_response = dep_result.get("final_response", "")
            if dep_response:
                dep_names = DrugEntityExtractor.extract_drug_candidates(dep_response, max_items=10)
                merged_drug_names.extend(dep_names)

    # 构建 prompt
    context_parts: list[str] = []
    if original_input and step_query and original_input != step_query and step_query not in original_input:
        context_parts.append(f"[用户原始问题]: {original_input}")
    context_parts.append(f"[当前需要回答的问题]: {step_query}")
    context_parts.extend(dep_summaries)
    if dep_summaries:
        context_parts.append("请基于以上前置步骤的结果来回答当前问题。")
    prompt = "\n".join(context_parts)

    # 构建 merged_entities
    merged_entities: dict = {}
    if merged_drug_names:
        merged_entities["drug_name_list"] = list(set(merged_drug_names))
    if merged_lab_items:
        merged_entities["lab_items"] = merged_lab_items

    # 结构化上下文
    structured_ctx: dict = {
        "dep_summaries": dep_summaries,
        "key_findings": key_findings,
        "drug_names": list(set(merged_drug_names)),
        "lab_items": merged_lab_items,
    }

    return {
        "prompt": prompt,
        "context": structured_ctx,
        "merged_entities": merged_entities,
    }


def _extract_key_findings(text: str) -> list[str]:
    """从步骤回答文本中提取关键结论（规则兜底）。"""
    if not text:
        return []
    findings: list[str] = []
    patterns = [
        r"(?:总之|综上所述|因此|所以|核心结论[：:]?)\s*(.{10,120}?)(?:[。；]|$)",
        r"(?:常用\S*?包括|推荐\S*?包括|主要有)\s*(.{10,120}?)(?:[。；]|$)",
        r"(?:注意|需注意|注意事项)[：:]\s*(.{10,120}?)(?:[。；]|$)",
    ]
    import re as _re
    for pat in patterns:
        for m in _re.finditer(pat, text):
            finding = m.group(1).strip()
            if len(finding) >= 6 and finding not in findings:
                findings.append(finding)
    return findings[:5]


async def _execute_single_step(sub_state: dict, step: PlanStep) -> dict:
    target_type = step.get("target_type", "agent")
    target_name = step.get("target_name", "")

    if target_type == "tool":
        try:
            tool_result = await _tool_executor.execute(target_name, sub_state)
            sub_state["tool_result"] = tool_result.get("tool_result", tool_result)
            if tool_result.get("extract_entities"):
                sub_state["extract_entities"] = tool_result["extract_entities"]
            if tool_result.get("intent_type"):
                sub_state["intent_type"] = tool_result["intent_type"]
            if tool_result.get("error_msg"):
                sub_state["error_msg"] = tool_result["error_msg"]
            return sub_state
        except Exception as e:
            logger.error("_execute_single_step tool=%s failed: %s", target_name, e)
            sub_state["error_msg"] = str(e)
            return sub_state

    if target_name == "drug_record_agent" and step.get("intent_type") == "drug_record_sm":
        result = await _process_drug_record_state_machine(sub_state)
        result["intent_type"] = "drug_record_sm"
        return result

    from app.core.agent.agent_router import AgentRouter
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
    _t0 = time.perf_counter()
    if state.get("error_msg"):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="execute_node", latency_ms=latency_ms, skipped=True)
        return state

    plan = state.get("execution_plan", {})
    steps = plan.get("steps", [])
    results: dict[str, dict] = state.get("plan_step_results") or {}

    if "original_user_input" not in state:
        state["original_user_input"] = state.get("user_input", "")

    if not steps:
        state["plan_step_results"] = results
        state["plan_phase"] = "executing"
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="execute_node", latency_ms=latency_ms, step_count=0)
        return state

    if len(steps) <= 1:
        for step in steps:
            if step["step_id"] in results:
                continue
            step_t0 = time.perf_counter()
            sub_state = _build_sub_state(state, step, step_results=results)
            result = await _execute_single_step(sub_state, step)
            results[step["step_id"]] = result
            state.update({k: v for k, v in result.items() if k in ("final_response", "error_msg", "tool_result", "llm_output", "extract_entities")})
            state["plan_step_results"] = results

            step_latency_ms = int((time.perf_counter() - step_t0) * 1000)
            log_step_execution(
                step_id=step["step_id"],
                target=step.get("target_name", ""),
                query=step.get("query", ""),
                latency_ms=step_latency_ms,
                has_error=bool(result.get("error_msg")),
                depends_on=step.get("depends_on"),
            )

            state = await _planner.evaluate_for_replan(state)
            if state.get("needs_replan"):
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                log_node_execution(
                    node_name="execute_node",
                    latency_ms=latency_ms,
                    needs_replan=True,
                    replan_reason=state.get("replan_reason", ""),
                    completed_step=step["step_id"],
                )
                return state
    else:
        groups = _group_steps_by_dependency(steps)
        for group in groups:
            tasks = []
            group_steps = []
            for step in group:
                if step["step_id"] in results:
                    continue
                sub_state = _build_sub_state(state, step, step_results=results)
                tasks.append(_execute_single_step(sub_state, step))
                group_steps.append(step)
            if not tasks:
                continue
            group_t0 = time.perf_counter()
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            group_latency_ms = int((time.perf_counter() - group_t0) * 1000)
            for step, result in zip(group_steps, group_results):
                if isinstance(result, Exception):
                    results[step["step_id"]] = {"error_msg": str(result), "final_response": f"处理'{step['query']}'时出现错误。", "intent_type": step.get("intent_type", "general")}
                else:
                    results[step["step_id"]] = result

            state["plan_step_results"] = results
            for step in group_steps:
                step_result = results.get(step["step_id"], {})
                log_step_execution(
                    step_id=step["step_id"],
                    target=step.get("target_name", ""),
                    query=step.get("query", ""),
                    latency_ms=group_latency_ms,
                    has_error=bool(step_result.get("error_msg")),
                    depends_on=step.get("depends_on"),
                )

            state = await _planner.evaluate_for_replan(state)
            if state.get("needs_replan"):
                latency_ms = int((time.perf_counter() - _t0) * 1000)
                log_node_execution(
                    node_name="execute_node",
                    latency_ms=latency_ms,
                    needs_replan=True,
                    replan_reason=state.get("replan_reason", ""),
                )
                logger.info(
                    "execute_node: needs_replan after group reason=%s",
                    state.get("replan_reason"),
                )
                return state

    state["plan_step_results"] = results
    state["plan_phase"] = "executing"

    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(
        node_name="execute_node",
        latency_ms=latency_ms,
        step_count=len(steps),
        results_count=len(results),
    )
    return state


async def reconcile_node(state: dict) -> dict:
    _t0 = time.perf_counter()
    if state.get("error_msg") and not state.get("plan_step_results"):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="reconcile_node", latency_ms=latency_ms, skipped=True)
        return state

    results = state.get("plan_step_results", {})
    plan = state.get("execution_plan", {})

    if not results:
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="reconcile_node", latency_ms=latency_ms, skipped=True)
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

        if not content and result.get("tool_result"):
            tool_result = result["tool_result"]
            if isinstance(tool_result, dict):
                content = tool_result.get("final_desc", "")
                if not content and tool_result.get("interaction_result"):
                    interactions = tool_result["interaction_result"]
                    if isinstance(interactions, list) and len(interactions) > 0:
                        lines = []
                        for it in interactions:
                            drug_a = it.get("drug_a", "")
                            drug_b = it.get("drug_b", "")
                            desc = it.get("interaction_desc", "")
                            lines.append(f"{drug_a} + {drug_b}：{desc}")
                        content = "\n".join(lines)
                    else:
                        content = tool_result.get("message", "已匹配到药品，但未查询到两两相互作用记录；如需更精确信息，请查阅说明书或咨询执业药师。")

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
        state["reconciled_sections"] = sections
        state["intent"] = "multi"
        state["intent_type"] = "multi"

    state["plan_phase"] = "reconciling"
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="reconcile_node", latency_ms=latency_ms, result_count=len(results), has_sections=bool(sections))
    return state


async def response_plan(state: dict) -> dict:
    _t0 = time.perf_counter()
    if state.get("error_msg") and state.get("final_response"):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="response_plan", latency_ms=latency_ms, skipped=True)
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
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="response_plan", latency_ms=latency_ms, mode=mode, inject_memory=bool(need_mem))
    return state


async def llm_generate(state: dict) -> dict:
    _t0 = time.perf_counter()

    reconciled_sections = state.get("reconciled_sections")
    if reconciled_sections and len(reconciled_sections) > 1:
        system_prompt = (
            "你是医疗问答助手，需要将多个子问题的回答整合为一个清晰、自然的回复。\n"
            "原则：\n"
            "1) 必须保留每个子问题的回答要点，不得遗漏或添加原文未提及的医学事实。\n"
            "2) 每个子问题用二级标题（##）分隔，标题即为子问题本身。\n"
            "3) 每个子问题的回答要简洁精炼，去除重复和冗余内容。\n"
            "4) 使用**加粗**标记关键信息（如药名、症状、注意事项）。\n"
            "5) 使用项目符号或编号列表组织多条信息，每条之间空一行。\n"
            "6) 语言自然亲切，像一位耐心的家庭医生在和你聊天。\n"
            "7) 如果某个子问题无法回答（如工具查询失败），用简短一句话说明，不要输出原始错误信息或凭空补充。\n"
            "8) 整体回复结尾用一句温馨提示收束。\n"
        )
        sections_text = ""
        for i, section in enumerate(reconciled_sections):
            sections_text += f"\n\n--- 子问题 {i + 1} ---\n{section}"
        user_prompt = f"用户原始问题：{state.get('user_input', '')}\n\n以下是各子问题的回答：{sections_text}"

        try:
            llm = LLMService()
            raw = await llm.chat_completion(prompt=user_prompt, system_prompt=system_prompt, timeout_s=15.0, max_tokens=1200)
            state["llm_output"] = (raw or "").strip()
            state["final_response"] = state["llm_output"]
        except Exception as e:
            logger.error("llm_generate multi-intent failed: %s", e)
            combined = "\n\n".join([f"## {s}" for s in reconciled_sections])
            state["llm_output"] = combined
            state["final_response"] = combined

        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="llm_generate", latency_ms=latency_ms, mode="multi_intent", section_count=len(reconciled_sections))
        return state

    if state.get("final_response"):
        state["llm_output"] = state["final_response"]
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="llm_generate", latency_ms=latency_ms, shortcut="final_response")
        return state

    if state.get("needs_confirmation") and state.get("confirmation_message"):
        state["llm_output"] = state["confirmation_message"]
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="llm_generate", latency_ms=latency_ms, shortcut="confirmation")
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
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="llm_generate", latency_ms=latency_ms, shortcut="drug_confirmation")
        return state

    if mode == "llm_format":
        system_prompt = (
            "你是医疗问答助手，任务是把\"工具/数据库查询结果\"用清晰、自然、结构化的中文表达出来。\n"
            "核心要求：\n"
            "1) 严格基于提供的工具/检索结果输出，不得添加任何结果中未提及的医学事实、数据或结论。\n"
            "2) 如果结果中某项信息缺失或无法确定，直接说明\"该信息未在查询结果中体现\"，不要自行补充。\n"
            "3) 输出尽量简洁，分点呈现，必要时补充就医建议边界。\n"
            "4) 禁止给出诊断结论或处方/调整用药建议。\n"
            "5) 回复格式要求：\n"
            "   - 使用**加粗**标记关键信息（如药名、指标名）\n"
            "   - 使用编号列表或项目符号组织多条信息\n"
            "   - 每个要点之间用空行分隔，保持视觉清晰\n"
            "   - 语言自然亲切，像医生对患者解释一样，避免生硬的罗列\n"
            "   - 结尾用一段简短的温馨提示收束\n"
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
            "核心原则：\n"
            "1) 如果提供了知识库检索结果、工具查询结果、或会话/长期记忆，你的回答必须基于这些信息。\n"
            "   知识库未收录的内容应明确说\"目前知识库中未查到相关信息\"，不得凭训练数据编造医学事实。\n"
            "2) 不得编造不存在的个人信息/检查结果/用药记录/药物数据。\n"
            "3) 禁止诊断与处方/调整用药建议；可以给出通用科普与就医指引。\n"
            "4) 对于纯闲聊或非医疗问题（如\"你好\"），正常友好回复即可，不需要强行关联医学内容。\n"
            "5) 回复格式要求：\n"
            "   - 语气亲切自然，像一位耐心的家庭医生在和你聊天\n"
            "   - 使用**加粗**标记关键信息（如药名、症状、注意事项）\n"
            "   - 多条信息用编号列表或项目符号组织，每条之间空一行\n"
            "   - 先给简短总结，再展开说明，避免一上来就堆砌大量文字\n"
            "   - 结尾用一句温馨提示收束（如：如有不适请及时就医）\n"
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

    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="llm_generate", latency_ms=latency_ms, mode=mode, output_len=len(state.get("llm_output") or ""))
    return state


async def output_check_and_disclaimer(state: dict) -> dict:
    _t0 = time.perf_counter()
    from app.core.compliance.compliance_service import ComplianceService

    state["final_response"] = state.get("llm_output", "") or state.get("final_response", "")

    compliance = ComplianceService()
    ok, msg = compliance.output_compliance_check(state["final_response"])
    if not ok:
        # 统一输出合规：任何路径（工具/多意图/Agent）产出的 final_response 都过闸
        logger.warning("output_check compliance blocked: %s", msg)
        state["final_response"] = (
            "抱歉，该回答涉及医疗红线内容，无法提供具体建议。"
            "如有健康问题，请及时就医，并在医生指导下用药。"
        )
    state["final_response"] = compliance.add_disclaimer(state["final_response"])

    proposed = state.get("proposed_updates") or []
    proposed.append({"scope": "shared", "key": "latest_response", "value": state.get("final_response", ""), "source": "out"})
    state["proposed_updates"] = proposed

    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="output_check_and_disclaimer", latency_ms=latency_ms, blocked=not ok)
    return state


async def commit_gate(state: dict) -> dict:
    _t0 = time.perf_counter()
    if state.get("error_msg"):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="commit_gate", latency_ms=latency_ms, skipped=True)
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
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="commit_gate", latency_ms=latency_ms)
    return state


async def memory_update(state: dict) -> dict:
    _t0 = time.perf_counter()
    if state.get("error_msg"):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="memory_update", latency_ms=latency_ms, skipped=True)
        return state

    mem = MemoryService()
    await mem.update_user_memory(state["user_id"], state["session_id"], "user", state["user_input"])
    if "final_response" in state:
        await mem.update_user_memory(state["user_id"], state["session_id"], "assistant", state["final_response"])

    if state.get("force_long_memory_write"):
        asyncio.create_task(_async_long_memory_write(state, source=state.get("long_memory_write_source", "explicit")))
        logger.info("long_memory write triggered by force: source=%s", state.get("long_memory_write_source", "explicit"))
    else:
        logger.debug("long_memory write skipped (not forced, will write on session end)")

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

    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="memory_update", latency_ms=latency_ms, forced_write=bool(state.get("force_long_memory_write")))
    return state


async def _async_long_memory_write(state: dict, source: str = "chat"):
    try:
        start_time = time.time()
        svc = LongMemoryService()
        if not svc.is_enabled():
            return
        items = await svc.extract_candidates(user_input=state.get("user_input", ""))
        if not items:
            return
        result = await svc.write_with_conflict_check(
            user_id=state["user_id"], session_id=state["session_id"], items=items, source=source
        )
        write_time_ms = int((time.time() - start_time) * 1000)
        logger.info("long_memory write done source=%s result=%s cost_ms=%s", source, result, write_time_ms)

        drug_events = [item for item in items if item.memory_type == "drug_event"]
        if drug_events:
            drug_info_list = []
            from app.core.agent.llm_decision_service import LLMDecisionService
            llm_decision = LLMDecisionService()
            for event in drug_events:
                original_text = event.text.replace("用户", "我")
                drug_name = await llm_decision.extract_drug_name_from_event(original_text)
                if not drug_name:
                    drug_match = re.search(r"(?:吃了|服用了|用了|吃|服用|使用|用)([^，。！？\s]{1,30})", original_text)
                    if drug_match:
                        drug_name = drug_match.group(1).strip()
                if drug_name:
                    drug_info_list.append({"drug_name": drug_name, "full_text": original_text, "confidence": event.confidence})

            if drug_info_list:
                logger.info("async_long_memory_write: drug_events=%s", [d["drug_name"] for d in drug_info_list])
    except Exception:
        logger.exception("async_long_memory_write failed")


async def _async_flush_session_long_memory(*, user_id: str, session_id: str, history: list[dict]):
    """对话结束后批量写入长期记忆（session_end 策略）。"""
    try:
        svc = LongMemoryService()
        if not svc.is_enabled():
            return
        result = await svc.batch_write_session(user_id=user_id, session_id=session_id, history=history)
        logger.info("long_memory session_end flush done: session=%s result=%s", session_id, result)
    except Exception:
        logger.exception("async_flush_session_long_memory failed")


async def _async_compress_and_write_long_memory(*, user_id: str, session_id: str, history: list[dict]):
    """短期记忆压缩前写入长期记忆（compress_pre_write 策略）。"""
    try:
        svc = LongMemoryService()
        if not svc.is_enabled():
            return
        result = await svc.batch_write_session(user_id=user_id, session_id=session_id, history=history)
        logger.info("long_memory compress_pre_write done: session=%s result=%s", session_id, result)
    except Exception:
        logger.exception("async_compress_and_write_long_memory failed")


_MEMORY_SAVE_PATTERNS = [
    "记住", "帮我记", "记下来", "记录一下", "保存", "别忘了",
    "记住这个", "帮我记住", "记一下", "存一下", "备忘",
]


def _detect_memory_save_intent(user_input: str) -> bool:
    """检测用户是否有显式要求保存记忆的意图。"""
    text = user_input.strip()
    return any(pat in text for pat in _MEMORY_SAVE_PATTERNS)


# ---- fact-checker: medical claim detection patterns ----

_MEDICAL_CLAIM_DRUG_PATTERNS = [
    r"(?:布洛芬|阿司匹林|阿莫西林|头孢\S{0,3}|青霉素|红霉素|氯霉素|四环素|庆大霉素|链霉素)",
    r"(?:硝苯地平|卡托普利|依那普利|氯沙坦|氨氯地平|美托洛尔|比索洛尔|普萘洛尔)",
    r"(?:二甲双胍|格列\S{1,4}|胰岛素|阿卡波糖|罗格列酮|西格列汀)",
    r"(?:奥美拉唑|雷尼替丁|西咪替丁|吗丁啉|多潘立酮|蒙脱石散)",
    r"(?:氯雷他定|西替利嗪|扑尔敏|苯海拉明|特非那定)",
    r"(?:地西泮|艾司唑仑|阿普唑仑|舍曲林|氟西汀|帕罗西汀)",
    r"[一-鿿]{1,3}(?:素|芬|林|唑|坦|普利|地平|洛尔|他汀|贝特)",
]

_MEDICAL_CLAIM_DISEASE_PATTERNS = [
    r"(?:高血压|糖尿病|冠心病|哮喘|COPD|慢阻肺|肝炎|肝硬化|肾炎|肾衰竭)",
    r"(?:脑梗|心梗|中风|偏瘫|心衰|心律失常|房颤|室颤)",
    r"(?:胃癌|肺癌|肝癌|乳腺癌|前列腺癌|结肠癌|白血病|淋巴瘤)",
    r"(?:肺炎|支气管炎|肺结核|肺气肿|肺纤维化|间质性肺炎)",
    r"(?:胃炎|胃溃疡|十二指肠溃疡|溃疡性结肠炎|克罗恩病|肠易激)",
    r"(?:甲亢|甲减|桥本|痛风|骨质疏松|类风湿|红斑狼疮|银屑病)",
    r"(?:抑郁|焦虑|精神分裂|双相|强迫症|恐惧症|惊恐)",
]

_MEDICAL_CLAIM_TREATMENT_PATTERNS = [
    r"(?:治疗|治愈|根治|康复|好转|缓解|改善|控制|预防)",
    r"(?:服用|口服|注射|输液|静脉|外用|涂抹|含服|吞服)",
    r"(?:每天\d次|每日\d次|\d次/天|\d+mg|\d+g|\d+ml|\d+片|\d+粒|\d+支)",
    r"(?:剂量|用量|用法|频次|疗程|停药|换药|加量|减量)",
    r"(?:手术|切除|移植|搭桥|支架|透析|化疗|放疗|靶向|免疫治疗)",
]

_MEDICAL_CLAIM_STATS_PATTERNS = [
    r"\d+\.?\d*\s*%",
    r"\d+/\d+\s*(?:的|人|患者|病例)",
    r"(?:研究表明|研究显示|据统计|数据表明|临床试验|指南推荐)",
    r"(?:发病率|死亡率|治愈率|有效率|生存率|五年生存)",
]


def _response_has_medical_claims(text: str) -> bool:
    """规则检测回答中是否包含医疗相关事实陈述。"""
    if not text:
        return False
    all_patterns = (
        _MEDICAL_CLAIM_DRUG_PATTERNS
        + _MEDICAL_CLAIM_DISEASE_PATTERNS
        + _MEDICAL_CLAIM_TREATMENT_PATTERNS
        + _MEDICAL_CLAIM_STATS_PATTERNS
    )
    for pat in all_patterns:
        if re.search(pat, text):
            return True
    return False


def _has_rag_or_tool_context(state: dict) -> bool:
    """检查是否存在可用于事实校验的实质性知识上下文。

    注意：SQL 药物名称匹配（drug_knowledge）仅表示"该药名在数据库中存在"，
    不提供可用于校验 LLM 输出的医学知识。实质性上下文必须是：
    - Milvus RAG 检索结果（public_kb），或
    - 工具执行结果（drug_interaction / lab_report 等）
    """
    retrieved = state.get("retrieved_knowledge") or {}

    if isinstance(retrieved, dict):
        # public_kb: Milvus 向量/BM25 检索 → 实质性医学知识
        public_kb = retrieved.get("public_kb")
        if isinstance(public_kb, list) and len(public_kb) > 0:
            return True

    # 工具执行结果 → 实质性上下文
    tool_result = state.get("tool_result")
    if isinstance(tool_result, dict):
        if tool_result.get("final_desc") or tool_result.get("interaction_result"):
            return True

    # 多步骤执行结果中的工具输出
    plan_results = state.get("plan_step_results")
    if isinstance(plan_results, dict):
        for result in plan_results.values():
            if isinstance(result, dict):
                tr = result.get("tool_result")
                if isinstance(tr, dict) and (tr.get("final_desc") or tr.get("interaction_result")):
                    return True

    return False


async def fact_check(state: dict) -> dict:
    """事实校验节点：检查 LLM 输出中的医疗声明是否有 RAG/工具上下文支撑。

    插入在 llm_generate 之后、output_check_and_disclaimer 之前。
    纯规则检查（无额外 LLM 调用），延迟 ~0ms。

    逻辑：
    - 非医疗回答（如闲聊）→ 跳过
    - 医疗回答 + 有 RAG/工具上下文 → 信任生成（prompt 已要求 grounding）
    - 医疗回答 + 无 RAG/工具上下文 → 追加核实建议警告
    """
    _t0 = time.perf_counter()
    from app.config.settings import settings

    if not settings.ENABLE_FACT_CHECK:
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="fact_check", latency_ms=latency_ms, skipped=True, reason="disabled")
        return state

    llm_output = state.get("llm_output", "") or state.get("final_response", "")
    if not llm_output or len(llm_output.strip()) < 5:
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="fact_check", latency_ms=latency_ms, skipped=True, reason="empty_output")
        return state

    if not _response_has_medical_claims(llm_output):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="fact_check", latency_ms=latency_ms, skipped=True, reason="no_medical_claims")
        return state

    if _has_rag_or_tool_context(state):
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="fact_check", latency_ms=latency_ms, action="pass", reason="context_exists")
        return state

    logger.warning("fact_check: medical claims detected but no RAG/tool context — appending warning")
    warn = (
        "\n\n---\n"
        "⚠️ 提示：以上部分健康信息未能从当前知识库中充分验证。"
        "AI回答可能存在不准确之处，建议在采纳前咨询执业医师或查阅权威医学资料。"
    )
    state["final_response"] = llm_output + warn
    state["llm_output"] = state["final_response"]
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="fact_check", latency_ms=latency_ms, action="warn_no_context", output_len=len(state["final_response"]))
    return state


async def error_finalize(state: dict) -> dict:
    _t0 = time.perf_counter()
    if state.get("error_msg"):
        state["final_response"] = state["error_msg"]
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="error_finalize", latency_ms=latency_ms, has_error=bool(state.get("error_msg")))
    return state
