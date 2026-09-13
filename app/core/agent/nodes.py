from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import date, datetime

from langgraph.types import StreamWriter
from sqlalchemy import and_, select

from app.common.logger import get_logger, log_node_execution, log_step_execution
from app.core.agent.intent_classifier import IntentClassifier, IntentResult
from app.core.agent.planner_agent import PlannerAgent
from app.core.agent.state import ExecutionPlan, PlanStep
from app.core.agent.stream_events import chunk_payload, intent_payload, iter_sentence_chunks
from app.core.agent.tool_executor import ToolExecutor
from app.core.llm.llm_service import LLMService
from app.core.memory.long_memory_service import LongMemoryService
from app.core.memory.memory_service import MemoryService
from app.core.rag.medical_knowledge_service import MedicalKnowledgeService
from app.core.rag.public_kb_service import PublicKnowledgeService
from app.core.session.agent_state_store import AgentStateStore
from app.core.skills.medication_confirmation_skill import MedicationConfirmationSkill
from app.core.tools.archive_query_tool import ArchiveQueryTool
from app.core.tools.drug_entity_extractor import DrugEntityExtractor
from app.core.tools.drug_record_tool import DrugRecordTool

from app.db.database import get_sessionmaker
from app.db.models import UserDrugRecord

logger = get_logger(__name__)

_planner = PlannerAgent()
_tool_executor = ToolExecutor()

# 长期记忆批量写入：距上次提取新增 ≥ N 条消息才在后台批量提取一次
LONG_MEMORY_BATCH_INTERVAL = 6


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


def _build_decision_context(state: dict, max_chars: int = 600) -> str:
    """为路由/规划构造紧凑的对话上文（结构化决策上下文注入）。

    只拼最近 1-2 轮原文 + 上一轮结构化决策 + 待确认状态 + 会话摘要片断，
    供 intent/plan 的 LLM 消解跨轮指代/省略，避免把整段历史直接倒给分类器。
    无相关内容时返回空串。
    """
    parts: list[str] = []

    history = state.get("history") or []
    if history:
        tail = _short_window_history(history, max_turns=2)
        if tail:
            parts.append("近期对话：\n" + tail)

    last = state.get("last_decision")
    if isinstance(last, dict) and last:
        bits = []
        if last.get("intent"):
            bits.append(f"上一轮意图={last['intent']}")
        if last.get("target_agent"):
            bits.append(f"上一轮目标={last['target_agent']}")
        drugs = last.get("drug_names")
        if isinstance(drugs, list) and drugs:
            bits.append("涉及药物=" + "、".join(str(d) for d in drugs))
        lab = last.get("lab_items")
        if isinstance(lab, list) and lab:
            bits.append("化验指标=" + "、".join(str(i) for i in lab))
        if bits:
            parts.append("上一轮决策：\n" + "；".join(bits))

    pending = state.get("pending_confirmation")
    if isinstance(pending, dict) and pending:
        try:
            pend_str = json.dumps(pending, ensure_ascii=False, default=str)[:400]
        except Exception:
            pend_str = str(pending)[:400]
        parts.append("待用户确认的用药记录：\n" + pend_str)

    summary = (state.get("memory_summary") or "").strip()
    if summary:
        parts.append("会话摘要：\n" + summary[:250])

    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


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




# ─────────────────────────────────────────────────────────────────────────────
# turn_reset：每轮入口重置「语义上是单轮」的字段
#
# 背景（Step 4.5 引入，治 BUG-1 / BUG-3）：
#   - error_msg / final_response / intent* / execution_plan / plan_phase 等都是 tracked 字段
#     （state.py 中无 UntrackedValue 标注），checkpointer 写入快照；
#   - 但它们的**语义**是「本轮产出」，不该跨轮继承；
#   - 否则上轮的 final_response / error_msg 会污染下轮的入口——表现为：
#       BUG-1：合规拦截留下的 error_msg 让后续每轮都在 _need_error 条件边跳到 err
#       BUG-3：上轮的 final_response 触发 build_generation_prompt 短路分支，
#              本轮生成阶段 LLM 调用次数 = 0，直接复读上轮答案
#       同类：force_long_memory_write 等信号字段若跨轮残留，会让 memory_update 误触发
#
# 哪些字段**不**重置：
#   - 请求级输入（user_id / session_id / user_input / stream / enable_archive_link）：
#     这些由 chat_router.py 重新赋值，残留无意义但也无害
#   - 跨轮业务态（pending_confirmation / last_decision / session_runtime_state）：
#     已是 UntrackedValue，且由 AgentStateStore / memory_load 维护
#   - 已是 UntrackedValue 的中间产物（history / retrieved_knowledge / plan_step_results /
#     shared_facts / private_scratchpads / decision_context / 等）：
#     不进快照，不需要"重置"
# ─────────────────────────────────────────────────────────────────────────────
_TURN_LOCAL_FIELDS: tuple[str, ...] = (
    # 拦截/错误（BUG-1 根因）
    "error_msg",
    # 最终产出（BUG-3 根因）
    "final_response",
    # 意图识别（每轮由 intent_node 重算）
    "intent", "intent_type", "intent_confidence", "intent_reason", "intent_analysis",
    "target_agent", "is_multi_intent",
    # 实体抽取（每轮重算）
    "extract_entities",
    # 工具调用
    "tool_name", "tool_result",
    # 生成原文（仅本轮 out/fact_check 消费）
    "llm_output",
    # 确认/合规（每轮由 input_check / out 节点重算）
    "needs_confirmation", "confirmation_message", "compliance_check_result",
    # 计划（每轮由 plan_node 重算）
    "execution_plan",
    "is_multi_section", "plan_phase",
    "replan_count", "replan_reason", "needs_replan",
    "replan_context", "cross_step_conflict",
    # 药物事件（每轮重算）
    "candidate_drug_events",
    # 长期记忆写入信号（每轮由 intent_node 判定；不重置会让上轮的"记住"无限触发）
    "force_long_memory_write", "long_memory_write_source",
)


# 各字段 reset 时的**类型正确的空值**。
#
# ⚠️ 绝不能统一置 `None`：`None` 与「键不存在」对 `state.get(k, 默认值)` **不等价**。
# 写成 None 之后键仍然存在，下游 `.get(k, 默认值)` 拿到的是 None 而不是默认值——
#   - `state.get("replan_count", 0)` → None，紧接着 `None >= MAX_REPLAN` 抛 TypeError
#     （planner_agent.py:700→702，凡有步骤的轮次必过）
#   - `state.get("execution_plan", {})` → None，紧接着 `.get("steps")` 抛 AttributeError
#     （nodes.py:847，目前被 plan_node 先行覆盖掩盖，属于侥幸）
# 所以按 `state.py` 的**声明类型**给空值，保持「已重置」与「本轮尚未产生」在语义上一致。
_TURN_LOCAL_DEFAULTS: dict[str, object] = {
    # str
    "error_msg": "", "final_response": "", "llm_output": "",
    "intent": "", "intent_type": "", "intent_reason": "",
    "target_agent": "", "tool_name": "",
    "confirmation_message": "", "replan_reason": "", "long_memory_write_source": "",
    # Literal["planning", "executing", "reconciling", "responding"]
    "plan_phase": "planning",
    # bool（含已无消费者的 compliance_check_result，仍按声明类型给 False）
    "is_multi_intent": False, "needs_confirmation": False, "compliance_check_result": False,
    "is_multi_section": False, "needs_replan": False, "force_long_memory_write": False,
    # int / float
    "replan_count": 0, "intent_confidence": 0.0,
    # dict
    "intent_analysis": {}, "extract_entities": {}, "tool_result": {},
    "execution_plan": {}, "replan_context": {}, "cross_step_conflict": {},
    # list[dict]
    "candidate_drug_events": [],
}

# 导入期自检：新增 _TURN_LOCAL_FIELDS 却忘了给默认值时，在这里直接失败，
# 而不是静默退化成 None 再到运行期炸在某个 `.get(k, 默认值)` 上。
_MISSING_DEFAULTS = [k for k in _TURN_LOCAL_FIELDS if k not in _TURN_LOCAL_DEFAULTS]
if _MISSING_DEFAULTS:  # pragma: no cover - 导入期不变量
    raise RuntimeError(f"turn_reset 缺少默认值: {_MISSING_DEFAULTS}")


async def turn_reset(state: dict) -> dict:
    _t0 = time.perf_counter()
    # ⚠️ langgraph 的节点合并语义是 `state.update(returned)`——只覆盖、不删除。
    # 所以不能 `state.pop()` 或返回不含这些键的 dict：merge 之后旧键依旧残留。
    # 正确做法是**显式 overwrite 成空/默认值**，让合并后的 state 持有这些「被清掉」的值。
    # 然后下游节点（如 `_need_error` 的 `state.get("error_msg")`）看到的就是空/默认。
    for k in _TURN_LOCAL_FIELDS:
        if k in state:
            state[k] = _TURN_LOCAL_DEFAULTS[k]
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="turn_reset", latency_ms=latency_ms)
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
        state["decision_context"] = ""
        state["last_decision"] = None
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
    state.setdefault("decision_context", "")
    state.setdefault("last_decision", None)

    try:
        rt_state = await AgentStateStore().get_state(user_id=user_id, session_id=session_id)
        state["session_runtime_state"] = rt_state
        pending = rt_state.get("pending_confirmation") if isinstance(rt_state, dict) else None
        if isinstance(pending, dict):
            state["pending_confirmation"] = pending

        last_decision = rt_state.get("last_decision") if isinstance(rt_state, dict) else None
        if isinstance(last_decision, dict) and last_decision:
            state["last_decision"] = last_decision

    except Exception:
        state["session_runtime_state"] = {}
        state["last_decision"] = None

    # 长期记忆批量写入：增量游标去重（cursor），攒够新消息再后台提取一次
    if len(history) >= 2:
        try:
            svc = LongMemoryService()
            if svc.is_enabled():
                latest_chat_id = int(history[-1].get("chat_id") or 0)
                cursor = await svc.get_cursor(user_id=user_id, session_id=session_id)
                if latest_chat_id - cursor >= LONG_MEMORY_BATCH_INTERVAL:
                    asyncio.create_task(_async_flush_session_long_memory(user_id=user_id, session_id=session_id, history=history))
                    logger.info(
                        "long_memory incremental batch triggered: session=%s cursor=%s latest=%s",
                        session_id, cursor, latest_chat_id,
                    )
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

    # 结构化决策上下文注入（方案B）：供分类/路由消解跨轮指代与省略
    if not state.get("decision_context"):
        state["decision_context"] = _build_decision_context(state)
    decision_ctx = state.get("decision_context") or ""

    if _detect_memory_save_intent(user_input_raw):
        state["force_long_memory_write"] = True
        state["long_memory_write_source"] = "explicit"
        logger.info("memory_save intent detected: user_input=%s", user_input_raw[:50])

    if _llm_enabled_for_nodes():
        llm_decision = LLMDecisionService()
        combined = await llm_decision.classify_route_and_extract(user_input_raw, ctx=decision_ctx)

        if combined and combined.get("target_name") and combined.get("confidence", 0) >= 0.5:
            state["intent"] = combined.get("intent", "general")
            state["intent_confidence"] = combined.get("confidence", 0.8)
            state["intent_reason"] = combined.get("reason", "llm_route")
            state["intent_analysis"] = combined
            state["target_agent"] = combined.get("target_name", "")
            state["intent_type"] = combined.get("intent_type", "general")
            # 零额外往返：复用本次调用顺带判定的多意图标记，供 plan 兜底规则漏判
            state["is_multi_intent"] = bool(combined.get("is_multi_intent", False))

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
        # 重规划不再空转：跨步骤冲突 → 确定性补冲突检查步骤；
        # 步骤失败 → 把错误原因交给 LLM 决定 重试/改写/换目标/放弃
        state = await _planner.build_revised_plan(state)
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


def _build_sub_state(state: dict, step: PlanStep, step_results: dict | None = None, *, preserve_entities: bool = False) -> dict:
    sub_state = dict(state)
    sub_state["user_input"] = step["query"]
    sub_state["target_agent"] = step.get("target_name", "")
    sub_state["intent_type"] = step.get("intent_type", "general")

    # 单步（无并行/依赖）路径保留顶层实体：跨轮指代已被 intent 依据决策上下文
    # 解析成药名，需原样带给工具，避免工具只能从当前句正则回退丢失被指代药名。
    # `plan_step_results` 必须剥掉——`dict(state)` 是浅拷贝，`sub_state["plan_step_results"]`
    # 与 `execute_node` 手里的 `results` **是同一个对象**；而 `_execute_single_step` 又把
    # `sub_state` 本身当步骤结果返回，于是 `results[sid]["plan_step_results"] is results`
    # 构成**真循环引用**。后果：checkpointer 的 writes 元数据序列化栈溢出
    # （实测 `ormsgpack.packb` → TypeError: Recursion limit reached，整轮请求失败）。
    # 依赖步骤的结果不从这里读——`_build_structured_context` 走的是 `step_results` 形参。
    drop_keys = [
        "final_response", "error_msg", "intent_analysis", "tool_result", "llm_output",
        "plan_step_results",
    ]
    if not preserve_entities:
        drop_keys.append("extract_entities")
    for key in drop_keys:
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


#: 步骤结果里**下游真正会读**的键——穷举自全部消费点：
#:   reconcile_node（final_response / error_msg / tool_result / intent_type）
#:   _build_structured_context / _detect_cross_step_conflict（final_response / extract_entities / tool_result）
#:   execute_node 的 state.update 回填（final_response / error_msg / tool_result / llm_output / extract_entities）
#:   _has_substantive_context（tool_result）
#: 其余键一律剥掉：`_execute_single_step` 返回的是 sub_state 本身，原样存进
#: `plan_step_results` 会让每个步骤结果里嵌一份近乎完整的 state 副本，
#: 随重规划轮次逐层嵌套（体积 = 步骤数 × 整份状态）。
_STEP_RESULT_KEYS: tuple[str, ...] = (
    "final_response",
    "error_msg",
    "tool_result",
    "llm_output",
    "extract_entities",
    "intent_type",
)


def _project_step_result(result: dict) -> dict:
    """把步骤结果裁剪成只含下游真正读取的键，切断对整份 state 的引用。"""
    if not isinstance(result, dict):
        return {"error_msg": str(result), "final_response": "步骤执行返回了非预期结构。"}
    return {k: result[k] for k in _STEP_RESULT_KEYS if k in result}


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
            return _project_step_result(sub_state)
        except Exception as e:
            logger.error("_execute_single_step tool=%s failed: %s", target_name, e)
            sub_state["error_msg"] = str(e)
            return _project_step_result(sub_state)

    from app.core.agent.agent_router import AgentRouter
    router = AgentRouter()
    try:
        result_state = await router.route_and_execute(sub_state)
        result_state["intent_type"] = step.get("intent_type", "general")
        return _project_step_result(result_state)
    except Exception as e:
        logger.error("execute_single_step failed step=%s error=%s", step["step_id"], e)
        sub_state["error_msg"] = f"Agent执行失败: {str(e)}"
        sub_state["final_response"] = f"处理'{step['query']}'时出现错误，请稍后重试。"
        sub_state["intent_type"] = step.get("intent_type", "general")
        return _project_step_result(sub_state)


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
            sub_state = _build_sub_state(state, step, step_results=results, preserve_entities=True)
            result = await _execute_single_step(sub_state, step)
            results[step["step_id"]] = result
            state.update({k: v for k, v in result.items() if k in ("final_response", "error_msg", "tool_result", "llm_output", "extract_entities")})
            # tool_name 此前从未被赋值，导致生成模式判定里 tool_name 分支恒为死代码
            if step.get("target_type") == "tool":
                state["tool_name"] = step.get("target_name", "")
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
                # 多步场景同样回填 tool_name（取本层最后一个 tool 步骤）
                if step.get("target_type") == "tool":
                    state["tool_name"] = step.get("target_name", "")
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


def _tool_result_content(result: dict) -> str:
    """从工具结果里提取可展示文本。

    tool 步骤不产出 final_response（ToolExecutor 只回 tool_result），
    必须从 final_desc / interaction_result / message 兜底，否则单步工具场景会丢内容。
    """
    tr = result.get("tool_result")
    if not isinstance(tr, dict):
        return ""
    content = tr.get("final_desc") or ""
    if content:
        return content
    interactions = tr.get("interaction_result")
    if isinstance(interactions, list) and interactions:
        return "\n".join(
            f"{it.get('drug_a', '')} + {it.get('drug_b', '')}：{it.get('interaction_desc', '')}"
            for it in interactions if isinstance(it, dict)
        )
    return tr.get("message", "")


def _format_conflict_warning(interactions: list[dict], conflict: dict | None = None) -> str:
    """渲染冲突提醒。

    除了冲突对本身，还要说清**每个药的来源**：本轮对话提到的，还是用药档案里
    哪天记录的。医疗场景下"你凭什么说我在吃这个"必须可回答。
    末尾固定带一句"如已停用请忽略"——项目无法可靠判断在服状态，如实告知不确定性。
    """
    lines = ["⚠️ 跨任务药物冲突提醒："]
    for it in interactions:
        lines.append(f"- {it.get('drug_a', '')} + {it.get('drug_b', '')}：{it.get('interaction_desc', '')}")

    archive = [e for e in ((conflict or {}).get("archive_drugs") or []) if isinstance(e, dict)]
    if archive:
        names = {str(it.get("drug_a", "")) for it in interactions} | {str(it.get("drug_b", "")) for it in interactions}
        hit = [e for e in archive if e.get("name") in names]
        if hit:
            segs = []
            for e in hit:
                seg = e.get("name", "")
                if e.get("record_date"):
                    seg += f"（{e['record_date']} 记录"
                    if int(e.get("record_count") or 1) > 1:
                        seg += f"，共 {e['record_count']} 次"
                    seg += "）"
                segs.append(seg)
            lines.append("其中" + "、".join(segs) + "来自你的用药档案；如已停用请忽略本条提示。")
    return "\n".join(lines)


def _collect_conflict_interactions(results: dict, state: dict) -> dict[tuple, dict]:
    """统一收集跨步骤药物相互作用，按药品对去重。

    两个来源：
    1. 各步骤 tool_result.interaction_result（含 O-04 补检产生的 s_conflict_check_* 步骤）
    2. state["cross_step_conflict"] —— O-04 在重规划阶段落盘的结论（兜底，
       防止补检步骤因重规划次数用尽而未执行时结论丢失）
    """
    conflict_map: dict[tuple, dict] = {}

    def _add(items) -> None:
        for it in items or []:
            if not isinstance(it, dict):
                continue
            key = tuple(sorted([str(it.get("drug_a", "")), str(it.get("drug_b", ""))]))
            if key[0] and key[1]:
                conflict_map[key] = it

    for result in results.values():
        if isinstance(result, dict):
            _add((result.get("tool_result") or {}).get("interaction_result"))

    cross = state.get("cross_step_conflict") or {}
    if isinstance(cross, dict):
        _add(cross.get("detail"))

    return conflict_map


def _build_reconciled_context(results: dict, plan: dict, conflict_map: dict) -> dict:
    """汇总各步骤的【结构化中间结果】，供生成策略判定（_decide_response_mode）与 llm 注入。

    此前 reconcile 只保留文本（final_response / final_desc），结构化字段会丢：
    药名、化验指标明细（含参考范围与异常标记）、工具命中状态、失败步骤清单。
    """
    drug_names: list[str] = []
    lab_items: list[dict] = []
    key_findings: list[str] = []
    tool_targets: list[str] = []
    failed_steps: list[str] = []

    steps = (plan or {}).get("steps", [])
    for step_id, result in results.items():
        if not isinstance(result, dict):
            continue
        step = next((s for s in steps if s.get("step_id") == step_id), None) or {}

        if step.get("target_type") == "tool" and step.get("target_name"):
            tool_targets.append(step["target_name"])
        if result.get("error_msg"):
            failed_steps.append(step_id)

        entities = result.get("extract_entities") or {}
        if isinstance(entities, dict):
            for n in entities.get("drug_name_list") or []:
                n = str(n).strip()
                if n:
                    drug_names.append(n)

        tool_result = result.get("tool_result") or {}
        if isinstance(tool_result, dict):
            for d in tool_result.get("drug_list") or []:
                if isinstance(d, dict) and d.get("match_status") == "匹配成功":
                    n = str(d.get("drug_name", "")).strip()
                    if n:
                        drug_names.append(n)
            for item in tool_result.get("item_list") or []:
                if isinstance(item, dict):
                    lab_items.append({
                        "item_name": item.get("item_name", ""),
                        "test_value": item.get("test_value", ""),
                        "reference_range": item.get("reference_range", ""),
                        "abnormal_flag": item.get("abnormal_flag", ""),
                    })

        final_resp = result.get("final_response") or ""
        if final_resp:
            try:
                key_findings.extend(_extract_key_findings(final_resp))
            except Exception:  # noqa: BLE001 - 摘要抽取失败不影响主流程
                pass

    dedup_items: list[dict] = []
    seen_items: set[tuple] = set()
    for it in lab_items:
        key = (it.get("item_name", ""), it.get("test_value", ""))
        if key in seen_items:
            continue
        seen_items.add(key)
        dedup_items.append(it)

    return {
        "drug_names": list(dict.fromkeys(drug_names)),
        "lab_items": dedup_items,
        "key_findings": list(dict.fromkeys(key_findings))[:10],
        "tool_targets": list(dict.fromkeys(tool_targets)),
        "failed_steps": failed_steps,
        "has_conflict": bool(conflict_map),
        "has_structured_facts": bool(drug_names or dedup_items),
    }


def _format_reconciled_context(ctx: dict | None, max_chars: int = 600) -> str:
    """把结构化上下文压成可注入 prompt 的紧凑文本；无内容返回空串。"""
    if not isinstance(ctx, dict):
        return ""
    parts: list[str] = []
    if ctx.get("drug_names"):
        parts.append("涉及药品（已标准化）：" + "、".join(ctx["drug_names"][:8]))
    if ctx.get("lab_items"):
        lines = []
        for it in ctx["lab_items"][:8]:
            seg = f"{it.get('item_name','')} {it.get('test_value','')}"
            if it.get("reference_range"):
                seg += f"（参考范围 {it['reference_range']}）"
            if it.get("abnormal_flag"):
                seg += f"（{it['abnormal_flag']}）"
            lines.append(seg)
        parts.append("化验指标：" + "；".join(lines))
    if ctx.get("has_conflict"):
        parts.append("注意：本轮已检出药物相互作用")
    if not parts:
        return ""
    text = "\n".join(parts)
    return text[:max_chars]


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

    conflict_map = _collect_conflict_interactions(results, state)

    if len(results) == 1:
        single = next(iter(results.values()))
        content = single.get("final_response") or _tool_result_content(single) or single.get("error_msg") or ""
        if conflict_map:
            warning = _format_conflict_warning(list(conflict_map.values()), state.get("cross_step_conflict"))
            # 工具自身的 final_desc 已按同样格式列出冲突，直接替换为带警示标题的版本，避免重复
            if (single.get("tool_result") or {}).get("interaction_result"):
                content = warning
            else:
                content = (content + "\n\n" if content else "") + warning
        if content:
            state["final_response"] = content
        state["reconciled_context"] = _build_reconciled_context(results, plan, conflict_map)
        state["plan_phase"] = "reconciling"
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="reconcile_node", latency_ms=latency_ms, result_count=1, conflict_count=len(conflict_map))
        return state

    sections: list[str] = []

    for step_id, result in results.items():
        step = next((s for s in plan.get("steps", []) if s["step_id"] == step_id), None)
        query = step.get("query", "") if step else ""
        content = result.get("final_response", "") or result.get("error_msg", "")

        if not content:
            content = _tool_result_content(result)

        if content:
            sections.append(f"**{query}**\n{content}")

    if conflict_map:
        sections.append(_format_conflict_warning(list(conflict_map.values()), state.get("cross_step_conflict")))
    else:
        # 检测到多药但重规划次数用尽、补检步骤未执行 → 显式告知，而不是静默略过
        cross = state.get("cross_step_conflict") or {}
        names = (cross.get("drug_names") or []) if isinstance(cross, dict) else []
        if len(names) >= 2:
            sections.append(
                "⚠️ 本轮涉及多种药物（" + "、".join(names[:6]) + "），但未能完成相互作用核查，"
                "建议核对药品说明书或咨询执业药师。"
            )

    # 档案里有较久远的记录（不满足在服判据）→ 只提示不检查，交用户确认
    stale = [e for e in (state.get("cross_step_conflict") or {}).get("stale_drugs", []) if isinstance(e, dict)]
    if stale:
        segs = []
        for e in stale[:4]:
            seg = e.get("name", "")
            if e.get("record_date"):
                seg += f"（{e['record_date']}）"
            segs.append(seg)
        sections.append(
            "提示：你的用药档案中还有 " + "、".join(segs) + " 的较早记录（未纳入本次相互作用核查）。"
            "如仍在服用，告诉我可以一并核查。"
        )

    if sections:
        state["reconciled_sections"] = sections
        # 不再把 intent / intent_type 改成 "multi"：那会让下游 _decide_response_mode 的
        # 模式判定落到 llm_chat（丢失事实约束 prompt），也会污染记忆里记的意图
        state["is_multi_section"] = True
    state["reconciled_context"] = _build_reconciled_context(results, plan, conflict_map)

    state["plan_phase"] = "reconciling"
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(node_name="reconcile_node", latency_ms=latency_ms, result_count=len(results), has_sections=bool(sections))
    return state


# 这些意图/工具意味着答案是【事实型】的（查表/档案/工具结果），
# 必须用 llm_format 的强事实约束 prompt，不能走宽松的科普口吻
_FACT_INTENTS = {"archive", "drug", "lab", "drug_conflict", "drug_record", "lab_report"}
_FACT_TOOLS = {"drug_interaction", "lab_report", "archive"}


def _decide_response_mode(state: dict) -> str:
    """按【本次实际产出了什么】决定生成模式，而不是只看顶层 intent。

    原因：① 多步场景下顶层 intent 只代表首个步骤，其余步骤可能是事实型查询；
    ② 此前 reconcile 会把 intent 改成 "multi"，而 "multi" 不在事实型集合里，
       导致多步响应全部退回 llm_chat，丢掉「不得添加结果外医学事实」的约束。
    """
    intents: set[str] = set()
    top_intent = state.get("intent")
    if top_intent:
        intents.add(str(top_intent))
    top_intent_type = state.get("intent_type")
    if top_intent_type:
        intents.add(str(top_intent_type))

    for result in (state.get("plan_step_results") or {}).values():
        if isinstance(result, dict):
            it = result.get("intent_type")
            if it:
                intents.add(str(it))

    if intents & _FACT_INTENTS:
        return "llm_format"
    if state.get("tool_name") in _FACT_TOOLS:
        return "llm_format"
    # 有结构化工具结果但意图判定缺失 → 仍按事实型处理，宁严不松
    if state.get("tool_result"):
        return "llm_format"
    # reconcile 汇总出的结构化事实（药名/化验指标）同样说明答案是事实型的
    if (state.get("reconciled_context") or {}).get("has_structured_facts"):
        return "llm_format"
    return "llm_chat"


def _decide_response_strategy(state: dict) -> dict:
    """决定本次生成的约束策略 —— **纯函数，无副作用**。

    原为独立图节点 `response_plan`：每轮无条件执行、并把结果写回 state。
    但它的两个输出只被 `build_generation_prompt` 消费，且短路分支根本用不到，
    因此降级为函数，由 `build_generation_prompt` 在真正需要时按需调用：

    - 两条执行路径（非流式图 / 流式 run_stream）都经过 `build_generation_prompt`，
      策略因此自动带上，不存在"某条路径忘了调用"的漂移风险；
    - 短路分支（final_response / confirmation / drug_confirmation）零开销；
    - 原实现里 `error_msg + final_response` 的跳过守卫不再需要：该场景下
      `build_generation_prompt` 会先返回 `branch="final_response"`，走不到这里。

    返回 {"mode": "llm_chat" | "llm_format", "inject_memory": bool}
    """
    user_input = state.get("user_input", "")
    long_mem = (state.get("long_memory_text") or "").strip()
    knowledge = state.get("retrieved_knowledge") or {}

    need_mem = _need_contextual_memory(user_input) or bool(long_mem)
    if knowledge:
        need_mem = True
    # 历史 / 摘要 / 长期记忆 / 知识全空 → 没有任何可注入的上下文，强制关闭
    if not (state.get("history") or state.get("memory_summary") or long_mem or knowledge):
        need_mem = False

    return {"mode": _decide_response_mode(state), "inject_memory": bool(need_mem)}


def build_generation_prompt(state: dict) -> dict:
    """构造生成阶段所需的 prompt —— **纯函数，不发起 LLM 调用、无副作用**。

    非流式（llm_generate 节点）与流式（workflow.run_stream）共用这一份构造逻辑，
    避免两条路径各自维护一份 prompt 拼装代码而漂移。历史上确实漂移过：流式路径
    缺少 reconciled_context 注入、不写 skill_ctx、也没有 inject_knowledge 日志。

    返回 dict：
      branch        : "multi_intent" | "final_response" | "confirmation" | "drug_confirmation" | "normal"
      system_prompt : 短路分支下为空串
      user_prompt   : 短路分支下为空串
      content       : 工具结果兜底文本（"normal" 分支 LLM 失败时的降级内容）
      mode          : 生成策略，仅 "normal" 分支非空（由 _decide_response_strategy 按需计算），其余为 None
    """
    reconciled_sections = state.get("reconciled_sections") or []
    if len(reconciled_sections) > 1:
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
        return {"branch": "multi_intent", "system_prompt": system_prompt, "user_prompt": user_prompt, "content": "", "mode": None}

    # 以下几个是短路分支：上游已产出可直接返回的文本，无需再走 LLM、也无需计算生成策略
    if state.get("final_response"):
        return {"branch": "final_response", "system_prompt": "", "user_prompt": "", "content": "", "mode": None}
    if state.get("needs_confirmation") and state.get("confirmation_message"):
        return {"branch": "confirmation", "system_prompt": "", "user_prompt": "", "content": "", "mode": None}
    if state.get("candidate_drug_events"):
        return {"branch": "drug_confirmation", "system_prompt": "", "user_prompt": "", "content": "", "mode": None}

    content = (state.get("tool_result") or {}).get("final_desc") or ""
    if not content:
        content = "当前未获取到有效工具结果。"

    # 生成策略按需计算：短路分支在上面都已 return，走不到这里 → 那三分支零开销
    _t_strategy = time.perf_counter()
    strategy = _decide_response_strategy(state)
    mode = strategy["mode"]
    inject_memory = strategy["inject_memory"]
    log_node_execution(
        node_name="response_strategy",
        latency_ms=int((time.perf_counter() - _t_strategy) * 1000),
        mode=mode,
        inject_memory=inject_memory,
        merged_into="build_generation_prompt",
    )

    mem_summary = (state.get("memory_summary") or "").strip()
    if not mem_summary and inject_memory:
        mem_summary = _short_window_history(state.get("history") or [], max_turns=4)

    long_mem = (state.get("long_memory_text") or "").strip()
    retrieved_knowledge = state.get("retrieved_knowledge") or {}
    if retrieved_knowledge:
        logger.info("llm_generate inject_knowledge keys=%s", list(retrieved_knowledge.keys()))

    # 本轮已确认的结构化事实（工具/档案查询产出），优先级高于知识库检索结果
    recon = _format_reconciled_context(state.get("reconciled_context"))

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
        if recon:
            user_prompt = f"本轮已确认的结构化事实（来源：工具/档案查询，优先级高于检索结果）：\n{recon}\n\n" + user_prompt
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
        if recon:
            parts.append("本轮已确认的结构化事实（来源：工具/档案查询，优先级高于检索结果）：\n" + recon)
        parts.append("用户输入：\n" + (state.get("user_input", "") or ""))
        user_prompt = "\n\n".join(parts)

    return {
        "branch": "normal",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "content": content,
        "mode": mode,
    }


#: 生成阶段的 LLM 超时（秒）。改造前非流式用 12s、流式用 15s；合并后统一取 15s
#: （即流式路径的既有值）——保住流式契约不变，非流式那边略微变宽松、更少误超时。
_GEN_TIMEOUT_S = 15.0

#: 短路分支（`final_response` / `confirmation` / `drug_confirmation`）的**模拟流式节拍**。
#:
#: 为什么需要：这几个分支的文本由前置节点**整段**产出，不经过 LLM 逐 token 生成。
#: `iter_sentence_chunks` 把整段切成小片后，若**在同一个事件循环轮次里连续 writer()**，
#: 所有 chunk 会在同一瞬间产生 → `astream` 一次性吐出 → StreamingResponse 在一个 TCP 包
#: 送达 → 前端一次 paint 渲染完。**表现就是"一大坨文字一次性出现"，与真流式肉眼可辨。**
#: （实测：假流式 chunk 总跨度 0ms；真流式 3.5s。）
#:
#: 修复方式：每片之间 `await asyncio.sleep()`，让出事件循环，使 SSE 真正分片送达。
#:
#: 代价与边界：这是**人为等待**，不是让内容变快。总追加时长由 `_FAKE_STREAM_BUDGET_S`
#: 封顶，避免长文本把总时长拖长；把 `_FAKE_STREAM_DELAY_S` 置 0 即整体关闭（退回旧行为）。
_FAKE_STREAM_DELAY_S = 0.02
_FAKE_STREAM_BUDGET_S = 1.2


def _noop_writer(_payload: dict) -> None:
    """writer 兜底：直接调用本节点（不经图执行）时使用。

    图执行时 langgraph 按「参数名 == writer 且注解 == StreamWriter」自动注入真 writer；
    未开启 `stream_mode="custom"` 时它注入的也是等价的 no-op，所以这里不会崩。
    """
    return None


async def _emit_pieces_with_pace(pieces: list[str], writer: StreamWriter) -> int:
    """把整段文本的切片**按节拍**经 writer 推出，返回人为等待的总毫秒数。

    为什么不能直接 for-writer：见 `_FAKE_STREAM_DELAY_S` 的注释——同一事件循环轮次内
    连续 writer() 会让所有 chunk 在同一瞬间产生，SSE 一个包送达，前端一次渲染完，
    用户看到"一大坨"。**必须让出事件循环**，切片才有意义。

    节拍策略：单片间隔不超过 `_FAKE_STREAM_DELAY_S`，且**总等待不超过
    `_FAKE_STREAM_BUDGET_S`**（长文本按片数摊薄），避免文本越长总时长越久。
    末片之后不再等待——那一段是纯白等。
    """
    if not pieces:
        return 0
    delay = 0.0
    if _FAKE_STREAM_DELAY_S > 0 and len(pieces) > 1:
        delay = min(_FAKE_STREAM_DELAY_S, _FAKE_STREAM_BUDGET_S / (len(pieces) - 1))
    waited = 0.0
    for i, piece in enumerate(pieces):
        writer(chunk_payload(piece))
        if delay and i < len(pieces) - 1:
            await asyncio.sleep(delay)
            waited += delay
    return int(waited * 1000)


async def llm_generate(state: dict, writer: StreamWriter = _noop_writer) -> dict:
    """生成阶段节点。

    Step 3 起本节点**同时承担推流职责**：真正的 token 由这里经 writer 推给
    `run_stream` 的 `stream_mode="custom"` 通道。这是能消除 `workflow.py` 手搓
    run_stream 的前提——改造前该节点调的是 `chat_completion`（非流式），节点内不产
    token，所以流式只能绕开图另写一份节点序列。

    约束：**参数名必须是 `writer`、注解必须是 `StreamWriter`**，否则 langgraph 不注入
    （见 `langgraph/utils/runnable.py::KWARGS_CONFIG_KEYS`）；名字错了不会报错，
    只会静默退化成非流式，属于最难查的一类故障。

    state 写回语义与改造前**完全一致**（`llm_output` / `final_response` / `skill_ctx`）。
    """
    _t0 = time.perf_counter()

    plan = build_generation_prompt(state)
    branch = plan["branch"]

    if branch == "multi_intent":
        # 与改造前一致：开吐之前先发一次精简 intent（不带 intent_analysis）
        writer(intent_payload(state, full=False))
        full_response = ""
        try:
            async for tok in LLMService().chat_completion_stream(
                prompt=plan["user_prompt"],
                system_prompt=plan["system_prompt"],
                timeout_s=_GEN_TIMEOUT_S,
                max_tokens=1200,
            ):
                full_response += tok
                writer(chunk_payload(tok))
        except Exception as e:
            logger.error("llm_generate multi-intent failed: %s", e)
            full_response = "\n\n".join([f"## {s}" for s in (state.get("reconciled_sections") or [])])
            writer(chunk_payload(full_response))

        state["llm_output"] = full_response
        state["final_response"] = full_response
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(
            node_name="llm_generate",
            latency_ms=latency_ms,
            mode="multi_intent",
            section_count=len(state.get("reconciled_sections") or []),
        )
        return state

    if branch == "final_response":
        state["llm_output"] = state["final_response"]
        # 上游已产出完整文本，按句切分模拟流式，避免一次性吐出一大段。
        # ⚠️ 每片之间必须 `await` 让出事件循环，否则等于没切——见 _FAKE_STREAM_DELAY_S 注释。
        writer(intent_payload(state, full=False))
        pieces = list(iter_sentence_chunks(state["final_response"]))
        fake_delay_ms = await _emit_pieces_with_pace(pieces, writer)
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(
            node_name="llm_generate",
            latency_ms=max(0, latency_ms - fake_delay_ms),   # 扣掉人为节拍，保持延迟指标可比
            shortcut="final_response",
            chunks=len(pieces),
            fake_delay_ms=fake_delay_ms,
        )
        return state

    if branch == "confirmation":
        # 与改造前一致：该分支不发 intent 事件
        state["llm_output"] = state["confirmation_message"]
        writer(chunk_payload(state["confirmation_message"]))
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="llm_generate", latency_ms=latency_ms, shortcut="confirmation")
        return state

    if branch == "drug_confirmation":
        # 与改造前一致：该分支不发 intent 事件
        state.setdefault("skill_ctx", {})
        state["skill_ctx"]["medication_confirmation"] = {"candidate_events": state["candidate_drug_events"]}
        confirm_msg = MedicationConfirmationSkill().build_confirmation_message(state["candidate_drug_events"])
        state["llm_output"] = confirm_msg
        writer(chunk_payload(confirm_msg))
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        log_node_execution(node_name="llm_generate", latency_ms=latency_ms, shortcut="drug_confirmation")
        return state

    mode = plan["mode"] or "llm_chat"
    writer(intent_payload(state, full=False))
    full_response = ""
    try:
        async for tok in LLMService().chat_completion_stream(
            prompt=plan["user_prompt"],
            system_prompt=plan["system_prompt"],
            timeout_s=_GEN_TIMEOUT_S,
            max_tokens=900,
        ):
            full_response += tok
            writer(chunk_payload(tok))
    except Exception as e:
        logger.error("llm_generate failed: %s", e)
        full_response = plan["content"]
        writer(chunk_payload(full_response))

    # 空输出兜底：改造前**非流式**路径是 `(raw or "").strip() or plan["content"]`，
    # 流式路径没有这层兜底。合并到一条路径后取并集，把兜底补回来——
    # 否则"流建立成功但一个 token 都没吐"（`chat_completion_stream` 不抛异常，
    # 该模型实测偶发 content=''）会让 llm_output 变成空串，且此时一个 chunk 都没发过，
    # 前端气泡停在"正在生成…"，最终落到空回答（output_check 对空输出只跳过校验、不兜底）。
    if not full_response.strip():
        logger.warning("llm_generate 空输出，回退到 plan.content 兜底 (mode=%s)", mode)
        full_response = plan["content"]
        writer(chunk_payload(full_response))

    state["llm_output"] = full_response.strip()
    latency_ms = int((time.perf_counter() - _t0) * 1000)
    log_node_execution(
        node_name="llm_generate",
        latency_ms=latency_ms,
        mode=mode,
        branch=branch,
        output_len=len(state.get("llm_output") or ""),
    )
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
    try:
        await mem.update_user_memory(state["user_id"], state["session_id"], "user", state["user_input"])
        if "final_response" in state:
            await mem.update_user_memory(state["user_id"], state["session_id"], "assistant", state["final_response"])
    except Exception as e:
        # 落库失败不得反噬本轮回答：Step 3 起本节点进入流式 drain，异常上抛会打断一条
        # 已经成功吐完内容的流（改造前流式路径把记忆更新丢在 create_task 里，有 try/except 兜底）。
        # 非流式路径同样受益：DB 抖动不再把一次成功的问答变成 500。
        logger.error("memory_update persist chat record failed: %s", e)

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

        # 持久化"上一轮决策"摘要：供下一轮 intent/plan 消解跨轮指代/省略
        last_decision: dict = {}
        if state.get("intent"):
            last_decision["intent"] = state["intent"]
        if state.get("target_agent"):
            last_decision["target_agent"] = state["target_agent"]
        ents = state.get("extract_entities") or {}
        if isinstance(ents, dict):
            drugs = ents.get("drug_name_list")
            if isinstance(drugs, list):
                cleaned_drugs = [str(d) for d in drugs if str(d).strip()]
                if cleaned_drugs:
                    last_decision["drug_names"] = cleaned_drugs
            lab_items = ents.get("lab_items")
            if isinstance(lab_items, list):
                lab_names = []
                for it in lab_items:
                    if isinstance(it, dict) and it.get("item_name"):
                        lab_names.append(str(it["item_name"]))
                if lab_names:
                    last_decision["lab_items"] = lab_names
        if last_decision:
            runtime_state["last_decision"] = last_decision
        else:
            runtime_state.pop("last_decision", None)

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
        user_id = state["user_id"]
        session_id = state["session_id"]

        # 定位当前消息的 chat_id 作为 source（供游标去重，避免批量路径重复提取）
        source_chat_id = None
        try:
            source_chat_id = await MemoryService().get_latest_user_chat_id(user_id=user_id, session_id=session_id)
        except Exception:
            pass

        items = await svc.extract_candidates(
            user_input=state.get("user_input", ""), source_chat_id=source_chat_id
        )
        if not items:
            return
        result = await svc.write_with_conflict_check(
            user_id=user_id, session_id=session_id, items=items, source=source
        )
        # 更新游标：本消息已提取（drug_event 已由服务分流到 SQL）
        if source_chat_id:
            await svc.update_cursor(user_id=user_id, session_id=session_id, chat_id=source_chat_id)
        write_time_ms = int((time.time() - start_time) * 1000)
        logger.info("long_memory write done source=%s result=%s cost_ms=%s", source, result, write_time_ms)
    except Exception:
        logger.exception("async_long_memory_write failed")


async def _async_flush_session_long_memory(*, user_id: str, session_id: str, history: list[dict]):
    """增量批量写入长期记忆（游标去重，flush/compress/session_end 多时机幂等）。"""
    try:
        svc = LongMemoryService()
        if not svc.is_enabled():
            return
        result = await svc.batch_write_session(user_id=user_id, session_id=session_id, history=history)
        logger.info("long_memory batch done: session=%s result=%s", session_id, result)
    except Exception:
        logger.exception("async_flush_session_long_memory failed")


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
