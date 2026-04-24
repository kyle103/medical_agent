from __future__ import annotations

import json
import os
import time
import logging
from datetime import datetime

from app.core.memory.memory_service import MemoryService
from app.common.logger import get_logger
from app.core.memory.long_memory_service import LongMemoryService
from app.core.rag.medical_knowledge_service import MedicalKnowledgeService
from app.core.rag.public_kb_service import PublicKnowledgeService
from app.core.session.agent_state_store import AgentStateStore
from app.core.llm.llm_service import LLMService
from app.core.skills.medication_confirmation_skill import MedicationConfirmationSkill
from app.core.tools.archive_query_tool import ArchiveQueryTool
from app.core.tools.drug_entity_extractor import DrugEntityExtractor
from app.core.tools.drug_interaction_tool import DrugInteractionTool
from app.core.tools.lab_report_tool import LabReportTool

logger = get_logger(__name__)

# 确保日志目录存在
def ensure_log_directory():
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# 记录向量检索日志
def log_vector_retrieval(user_id, session_id, query, results, retrieval_time, error=None):
    log_dir = ensure_log_directory()
    log_file = os.path.join(log_dir, 'vector_retrieval.log')
    
    log_record = {
        'timestamp': datetime.now().isoformat(),
        'level': 'ERROR' if error else 'INFO',
        'message': 'Vector retrieval completed' if not error else f'Vector retrieval failed: {error}',
        'extra': {
            'user_id': user_id,
            'session_id': session_id,
            'query': query,
            'retrieval_time': round(retrieval_time, 4),
            'result_count': len(results),
            'results': results
        }
    }
    
    # 写入日志文件
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_record) + '\n')

# 记录向量库写入日志
def log_vector_store_write(user_id, session_id, items, write_time, error=None):
    log_dir = ensure_log_directory()
    log_file = os.path.join(log_dir, 'vector_retrieval.log')
    
    log_record = {
        'timestamp': datetime.now().isoformat(),
        'level': 'ERROR' if error else 'INFO',
        'message': 'Vector store write completed' if not error else f'Vector store write failed: {error}',
        'extra': {
            'user_id': user_id,
            'session_id': session_id,
            'write_time': round(write_time, 4),
            'item_count': len(items),
            'items': [
                {
                    'memory_id': item.memory_id,
                    'text': item.text,
                    'memory_type': item.memory_type,
                    'confidence': item.confidence
                }
                for item in items
            ]
        }
    }
    
    # 写入日志文件
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_record) + '\n')


INTENTS = {
    "archive": "档案查询",
    "drug": "药物相关（冲突查询/用药记录添加）",
    "lab": "化验单解读",
    "general": "通用问答"
}


def _need_contextual_memory(user_input: str) -> bool:
    """无需显式提示词的上下文需求判定（轻量规则）。"""
    t = (user_input or "").strip()
    if not t:
        return False

    # 用药陈述类 - 这些是新的独立陈述，不需要上下文
    # 检查是否包含用药陈述关键词，但排除作为其他词汇一部分的情况
    contains_drug_statement = (
        "吃了" in t or "服用" in t or "用了" in t or 
        "需要添加用药记录" in t or "添加用药" in t
    )
    # 排除特定组合词，但保留独立的用药记录请求
    exclude_combinations = ["吃药", "服药"]
    is_excluded = any(combo in t for combo in exclude_combinations) and "需要添加用药记录" not in t and "添加用药" not in t
    
    if contains_drug_statement and not is_excluded:
        # 检查是否是询问之前的用药而非陈述当前用药
        if not any(query in t for query in ["吃什么药", "什么药", "哪些药", "哪种药"]):
            return False

    # 特殊回忆用药场景
    if "记得" in t and "药" in t:
        return True

    # 短追问/承接
    if len(t) <= 12:
        # 排除简单的陈述性语句
        simple_statements = ["我有", "我是", "我在", "我要", "我想"]
        if not any(statement in t for statement in simple_statements):
            return True

    # 指代/省略
    pronouns = ["那个", "它", "这", "这样", "上面", "刚才", "之前", "继续", "然后", "还要", "还用", "还需要"]
    if any(p in t for p in pronouns):
        return True

    # 回忆/核对类
    recall = ["总结", "回顾", "复盘", "你还记得", "你记得", "还记得", "之前", "刚才", "上次", "昨天", "今天", "最近", "回顾", "总结"]
    if any(k in t for k in recall):
        return True

    # 用药追问类
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


async def input_check(state: dict) -> dict:
    # 合规检查已禁用
    return state


async def memory_load(state: dict) -> dict:
    """短期记忆加载：读取 user_id + session_id 最近对话，供后续节点使用。"""
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

    # 读取会话级运行状态（例如：待确认任务）
    try:
        rt_state = await AgentStateStore().get_state(user_id=user_id, session_id=session_id)
        state["session_runtime_state"] = rt_state
        pending = rt_state.get("pending_confirmation") if isinstance(rt_state, dict) else None
        if isinstance(pending, dict):
            state["pending_confirmation"] = pending
    except Exception:
        state["session_runtime_state"] = {}

    # 长期记忆召回（向量库）：不影响主流程，失败则忽略
    state["long_memory_items"] = []
    state["long_memory_text"] = ""
    try:
        start_time = time.time()
        svc = LongMemoryService()
        query = state.get("user_input", "")
        if svc.is_enabled():
            # 增大召回top_k到6，获取更多候选结果
            items = await svc.recall(user_id=user_id, query=query, top_k=6)
            
            # 二次筛选：优先保留包含药物名称的记忆，然后保留其他记忆
            drug_keywords = ["药", "药物", "服用", "吃了", "吃过", "布洛芬", "阿司匹林", "抗生素", "降压药", "降糖药"]
            drug_related_items = []
            other_items = []
            
            seen = set()
            for it in items:
                if it.memory_id in seen:
                    continue
                seen.add(it.memory_id)
                
                # 检查是否包含药物相关关键词（对中文不做lower处理）
                text = it.text
                is_drug_related = any(keyword in text for keyword in drug_keywords)
                
                if is_drug_related:
                    drug_related_items.append(it)
                else:
                    other_items.append(it)
            
            # 合并结果，药物相关记忆优先
            filtered_items = drug_related_items + other_items
            # 最终保留最多5个记忆项
            filtered_items = filtered_items[:5]
            
            state["long_memory_items"] = [
                {
                    "memory_id": it.memory_id,
                    "text": it.text,
                    "memory_type": it.memory_type,
                    "source": it.source,
                    "session_id": it.session_id,
                    "created_at": it.created_at,
                }
                for it in filtered_items
            ]
            
            if filtered_items:
                state["long_memory_text"] = "\n".join([f"- {it.text}" for it in filtered_items])
            
            # 记录向量检索日志
            retrieval_time = time.time() - start_time
            log_vector_retrieval(
                user_id=user_id,
                session_id=session_id,
                query=query,
                results=[
                    {
                        "memory_id": it.memory_id,
                        "text": it.text,
                        "memory_type": it.memory_type,
                        "source": it.source,
                        "session_id": it.session_id,
                        "created_at": str(it.created_at)
                    }
                    for it in filtered_items
                ],
                retrieval_time=retrieval_time
            )
    except Exception as e:
        # 记录错误日志
        log_vector_retrieval(
            user_id=user_id,
            session_id=session_id,
            query=state.get("user_input", ""),
            results=[],
            retrieval_time=0.0,
            error=str(e)
        )
        print(f"长期记忆召回失败: {e}")
        pass

    return state


async def intent_recognition(state: dict) -> dict:
    """意图识别节点（应用级）。

    - 优先使用规则分类，稳定可控
    - 如用户已配置 LLM，则用 LLM 做可选增强（并具备失败回退）
    - 只做意图分类，不做任何医疗结论生成
    """

    from app.core.agent.intent_classifier import IntentClassifier

    text = state.get("user_input", "").strip().lower()
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "")
    
    # 检查用户是否在确认添加用药记录
    if text in ["是", "是的", "确认", "确定", "好", "好的", "y", "yes", "同意", "添加", "保存"]:
        # 如果用户确认，检查是否存在待确认的用药事件
        if "candidate_drug_events" in state:
            # 将候选用药事件移动到待确认列表
            state["pending_drug_events_for_confirmation"] = state.pop("candidate_drug_events")

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
    """医疗专业知识检索（与长期记忆隔离）。"""
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

    logger.info(
        "knowledge_retrieve core intent=%s items=%s elapsed=%.3fs",
        state.get("intent"),
        len(state.get("retrieved_knowledge") or {}),
        time.time() - start_time,
    )

    if state.get("intent") == "general":
        try:
            public_kb = PublicKnowledgeService()
            state["retrieved_knowledge"]["public_kb"] = await public_kb.retrieve(
                query=state.get("user_input", ""),
            )
        except Exception:
            state.setdefault("retrieved_knowledge", {})["public_kb"] = []
        logger.info(
            "knowledge_retrieve public_kb count=%s",
            len((state.get("retrieved_knowledge") or {}).get("public_kb") or []),
        )
    return state


def _route_by_intent_and_text(state: dict) -> dict:
    """主路由规则：始终返回明确 target_agent。"""
    intent = (state.get("intent") or "").strip().lower()
    text = (state.get("user_input") or "").strip()
    entities = state.get("extract_entities") or {}
    drug_names = entities.get("drug_name_list") if isinstance(entities, dict) else []

    # 1) 基于明确意图直接路由
    if intent == "lab":
        return {
            "target_agent": "lab_report_agent",
            "intent_type": "lab_report",
            "confidence": float(state.get("intent_confidence") or 0.9),
            "reason": "route by intent=lab",
        }
    if intent == "archive":
        return {
            "target_agent": "main_qa_agent",
            "intent_type": "archive",
            "confidence": float(state.get("intent_confidence") or 0.9),
            "reason": "route by intent=archive",
        }
    if intent == "general":
        return {
            "target_agent": "main_qa_agent",
            "intent_type": "general",
            "confidence": float(state.get("intent_confidence") or 0.8),
            "reason": "route by intent=general",
        }

    # 2) drug 子意图细分
    if intent == "drug":
        conflict_keywords = ["相互作用", "一起吃", "同服", "配伍", "冲突", "禁忌", "能不能一起", "可以一起"]
        record_keywords = ["记录", "添加", "我吃了", "我服用", "我用了", "用药记录", "剂量", "频次", "每天", "每次", "mg", "毫克"]
        delete_keywords = ["删除", "移除", "清空"]

        is_conflict = any(k in text for k in conflict_keywords) or ("药" in text and "一起" in text)
        is_record = any(k in text for k in record_keywords)
        is_delete = any(k in text for k in delete_keywords)

        if is_conflict and not is_record and not is_delete:
            return {
                "target_agent": "drug_conflict_agent",
                "intent_type": "drug_conflict",
                "confidence": float(state.get("intent_confidence") or 0.85),
                "reason": "route by drug conflict keywords",
            }
        if (is_record or is_delete) and not is_conflict:
            return {
                "target_agent": "drug_record_agent",
                "intent_type": "drug_record",
                "confidence": float(state.get("intent_confidence") or 0.85),
                "reason": "route by drug record keywords",
            }
        if isinstance(drug_names, list) and len(drug_names) >= 2:
            return {
                "target_agent": "drug_conflict_agent",
                "intent_type": "drug_conflict",
                "confidence": float(state.get("intent_confidence") or 0.75),
                "reason": "route by multi-drug entities",
            }
        return {
            "target_agent": "drug_record_agent",
            "intent_type": "drug_record",
            "confidence": float(state.get("intent_confidence") or 0.7),
            "reason": "route by drug default",
        }

    # 3) 当 intent 为空或异常时，按文本快速识别（仍属于主路由，不是兜底）
    if any(k in text for k in ["化验", "检验", "血常规", "尿常规", "指标"]):
        return {"target_agent": "lab_report_agent", "intent_type": "lab_report", "confidence": 0.75, "reason": "route by text: lab"}
    if any(k in text for k in ["相互作用", "一起吃", "同服", "冲突", "禁忌"]):
        return {"target_agent": "drug_conflict_agent", "intent_type": "drug_conflict", "confidence": 0.75, "reason": "route by text: drug conflict"}
    if any(k in text for k in ["用药记录", "记录", "添加", "吃了", "服用", "mg", "毫克"]):
        return {"target_agent": "drug_record_agent", "intent_type": "drug_record", "confidence": 0.7, "reason": "route by text: drug record"}
    if any(k in text for k in ["档案", "病历", "历史记录", "就诊"]):
        return {"target_agent": "main_qa_agent", "intent_type": "archive", "confidence": 0.7, "reason": "route by text: archive"}
    return {"target_agent": "main_qa_agent", "intent_type": "general", "confidence": 0.6, "reason": "route by text: default general"}


async def agent_route(state: dict) -> dict:
    """Agent路由：主路径必须写入 target_agent。"""
    intent_result = _route_by_intent_and_text(state)
    state["intent_analysis"] = intent_result
    state["target_agent"] = intent_result["target_agent"]
    state["intent_type"] = intent_result.get("intent_type", state.get("intent", "general"))

    logger.info(
        "agent_route selected target_agent=%s intent=%s intent_type=%s reason=%s",
        state.get("target_agent"),
        state.get("intent"),
        state.get("intent_type"),
        (state.get("intent_analysis") or {}).get("reason"),
    )
    
    return state


async def _fallback_agent_route(state: dict) -> dict:
    """历史兼容函数：统一走主路由规则。"""
    out = _route_by_intent_and_text(state)
    state["intent_analysis"] = out
    state["target_agent"] = out["target_agent"]
    state["intent_type"] = out.get("intent_type", state.get("intent", "general"))
    return state


async def agent_execute(state: dict) -> dict:
    """Agent执行：调用对应的专业Agent处理请求"""
    from app.core.agent.agent_router import AgentRouter
    
    target_agent = state.get("target_agent")
    
    if not target_agent:
        state["error_msg"] = (
            "路由异常：agent_route 未写入 target_agent "
            f"(intent={state.get('intent')}, intent_type={state.get('intent_type')}, "
            f"has_intent_analysis={bool(state.get('intent_analysis'))})"
        )
        return state
    
    router = AgentRouter()
    
    try:
        result_state = await router.route_and_execute(state)
        state.update(result_state)
            
    except Exception as e:
        state["error_msg"] = f"Agent执行失败: {str(e)}"
    
    return state


async def response_plan(state: dict) -> dict:
    """响应规划：决定是否调用 LLM、是否注入记忆，以及选择 chat/format 提示词。"""
    if state.get("error_msg"):
        return state

    intent = state.get("intent") or "general"
    user_input = state.get("user_input", "")
    tool_name = state.get("tool_name") or ""

    # 默认：更像大模型的对话生成
    mode = "llm_chat"

    # 工具型意图优先走格式化
    if intent in ("archive", "drug", "lab") or tool_name in ("archive", "drug_interaction", "lab_report"):
        mode = "llm_format"

    # 当long_memory_text非空时，无条件注入长期记忆
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
    """真正调用 LLM 生成更自然的输出。"""

    # 检查是否已有final_response或confirmation_message
    if state.get("final_response"):
        state["llm_output"] = state["final_response"]
        return state
    
    # 检查是否需要确认
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
        user_prompt = (f"用户问题：{state.get('user_input','')}\n\n" f"工具结果：\n{content}\n")
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
        
        raw = await llm.chat_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            stream=False,
            timeout_s=12.0,
            max_tokens=900,
        )
        state["llm_output"] = (raw or "").strip() or content
    except Exception as e:
        logger.error(f"LLM生成失败: {e}")
        state["llm_output"] = content

    return state


import re
from datetime import datetime, date
from sqlalchemy import select, and_
from app.db.models import UserDrugRecord
from app.db.database import get_sessionmaker
from app.core.rag.drug_knowledge_service import DrugKnowledgeService


async def _has_same_drug_record(user_id: str, drug_name: str, start_date: date = None) -> bool:
    """检查是否存在相同的用药记录（基于 drug_name + start_date 幂等写入）"""
    async_session = get_sessionmaker()
    async with async_session() as session:
        # 构建查询条件
        conditions = [
            UserDrugRecord.user_id == user_id,
            UserDrugRecord.drug_name == drug_name,
            UserDrugRecord.is_deleted == 0
        ]
        
        # 如果提供了开始日期，则加入日期条件
        if start_date:
            conditions.append(UserDrugRecord.start_date == start_date)
        
        stmt = select(UserDrugRecord).where(and_(*conditions))
        result = await session.execute(stmt)
        existing_record = result.scalars().first()
        
        return existing_record is not None


async def _extract_drug_info_from_text(text: str) -> dict:
    """从文本中提取用药信息（剂量、频次、开始时间、用途）"""
    drug_info = {
        "dosage": "未指定",
        "frequency": "未指定", 
        "start_date": datetime.now(),
        "purpose": "未指定"
    }
    
    # 剂量提取模式
    dosage_patterns = [
        r'(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)',
        r'(一次|每次)\s*(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)',
        r'(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)\s*(一次|每次)'
    ]
    
    # 频次提取模式
    frequency_patterns = [
        r'(一天|每日)\s*(\d+)\s*次',
        r'(\d+)\s*次\s*(一天|每日)',
        r'(早晚|早中晚|早中晚各一次|早晚各一次|早中晚各一次)',
        r'(需要时|必要时|疼痛时|不适时)'
    ]
    
    # 开始时间提取模式
    start_date_patterns = [
        r'(今天|昨天|前天|\d+月\d+日|\d+年\d+月\d+日|\d{4}-\d{2}-\d{2})',
        r'(从|自)\s*(今天|昨天|前天|\d+月\d+日|\d+年\d+月\d+日|\d{4}-\d{2}-\d{2})',
        r'(开始|起)\s*(今天|昨天|前天|\d+月\d+日|\d+年\d+月\d+日|\d{4}-\d{2}-\d{2})'
    ]
    
    # 用途提取模式
    purpose_patterns = [
        r'(用于|治疗|缓解|针对)\s*([^，。！？]{1,20})',
        r'(因为|由于)\s*([^，。！？]{1,20})\s*(而|所以)',
        r'(头痛|发烧|感冒|疼痛|炎症|高血压|糖尿病|冠心病|哮喘)'
    ]
    
    # 提取剂量
    for pattern in dosage_patterns:
        match = re.search(pattern, text)
        if match:
            drug_info["dosage"] = match.group(0)
            break
    
    # 提取频次
    for pattern in frequency_patterns:
        match = re.search(pattern, text)
        if match:
            drug_info["frequency"] = match.group(0)
            break
    
    # 提取开始时间（简化处理，实际应该解析日期）
    for pattern in start_date_patterns:
        match = re.search(pattern, text)
        if match:
            drug_info["start_date_text"] = match.group(1) if match.groups() else match.group(0)
            break
    
    # 提取用途
    for pattern in purpose_patterns:
        match = re.search(pattern, text)
        if match:
            drug_info["purpose"] = match.group(2) if len(match.groups()) > 1 else match.group(1)
            break
    
    return drug_info


async def _identify_drug_name(text: str) -> str:
    """基于药品知识库/NER识别药品名"""
    # 首先尝试从文本中提取可能的药品名称
    drug_patterns = [
        r'(?:吃了|服用了|用了|吃|服用|使用|用)([^，。！？\s]{1,30})',
        r'([^，。！？\s]{1,30})(?:片|粒|胶囊|支|瓶|袋|贴)',
        r'(?:药名|药品|药物)\s*[:：]\s*([^，。！？\s]{1,30})'
    ]
    
    candidate_names = []
    for pattern in drug_patterns:
        matches = re.findall(pattern, text)
        candidate_names.extend(matches)
    
    # 使用药品知识库进行匹配
    if candidate_names:
        svc = DrugKnowledgeService()
        matched_drugs = await svc.match_drugs(candidate_names)
        
        # 优先返回匹配成功的药品名
        for result in matched_drugs:
            if result.get("match"):
                return result["match"]["drug_name"]
        
        # 如果没有匹配到知识库中的药品，返回第一个候选名称
        return candidate_names[0] if candidate_names else "未知药品"
    
    return "未知药品"


async def _collect_missing_drug_info(state: dict, drug_event: dict) -> dict:
    """收集缺失的用药信息（多轮收集）"""
    drug_name = drug_event.get("drug_name", "")
    current_info = drug_event.get("collected_info", {})
    
    # 字段映射：中文字段名（用于多轮收集）到英文字段名（用于数据库存储）
    field_mapping = {
        "剂量": "dosage",
        "频次": "frequency", 
        "开始时间": "start_date_text",
        "用途": "purpose"
    }
    
    # 检查缺失的字段
    missing_fields = []
    for chinese_field, english_field in field_mapping.items():
        # 检查是否已经收集了该字段
        if chinese_field not in current_info:
            # 如果是从文本提取的，检查英文字段
            if english_field not in current_info or current_info.get(english_field) == "未指定":
                missing_fields.append(chinese_field)
        elif current_info.get(chinese_field) == "未指定":
            missing_fields.append(chinese_field)
    
    # 如果所有信息都已收集，返回确认摘要
    if not missing_fields:
        # 构建确认摘要，使用正确的字段映射
        summary = f"请确认以下用药信息：\n"
        summary += f"• 药品名称：{drug_name}\n"
        
        # 使用字段映射获取正确的值
        for chinese_field, english_field in field_mapping.items():
            # 优先使用中文字段名（用户回答的值），如果没有则使用英文字段名（从文本提取的值）
            value = current_info.get(chinese_field, current_info.get(english_field, "未指定"))
            summary += f"• {chinese_field}：{value}\n"
        
        summary += "\n确认无误请回复'确认'，如需修改请直接告诉我需要修改的内容。"
        
        state["drug_confirmation_summary"] = summary
        state["waiting_for_confirmation"] = True
        return state
    
    # 生成追问提示
    question_template = {
        "剂量": f"请问您服用{drug_name}的剂量是多少？（例如：一次1片，或50毫克）",
        "频次": f"请问您服用{drug_name}的频率是怎样的？（例如：一天3次，或早晚各一次）", 
        "开始时间": f"请问您是从什么时候开始服用{drug_name}的？（例如：今天，或2024年1月1日）",
        "用途": f"请问您服用{drug_name}主要是用于治疗什么？（例如：头痛、发烧、高血压等）"
    }
    
    # 按优先级询问（剂量 > 频次 > 开始时间 > 用途）
    priority_order = ["剂量", "频次", "开始时间", "用途"]
    for field in priority_order:
        if field in missing_fields:
            state["current_question"] = question_template[field]
            state["waiting_for_answer"] = field
            state["current_drug_event"] = drug_event
            break
    
    return state


async def _process_drug_confirmation(state: dict) -> dict:
    """
    处理用户对用药记录的确认响应，支持多轮信息收集和确认摘要
    """
    user_input = state.get("user_input", "").strip().lower()
    user_id = state.get("user_id")
    
    # 处理多轮信息收集的响应
    if state.get("waiting_for_answer"):
        current_field = state["waiting_for_answer"]
        current_drug_event = state.get("current_drug_event", {})
        
        # 更新收集到的信息
        if "collected_info" not in current_drug_event:
            current_drug_event["collected_info"] = {}
        
        current_drug_event["collected_info"][current_field] = user_input
        
        # 清除当前等待状态
        state.pop("waiting_for_answer", None)
        state.pop("current_question", None)
        
        # 继续收集下一个缺失字段
        state = await _collect_missing_drug_info(state, current_drug_event)
        
        # 如果已经进入确认阶段，设置确认摘要
        if state.get("waiting_for_confirmation"):
            state["confirmation_messages"] = [state.get("drug_confirmation_summary", "")]
        
        return state
    
    # 处理确认摘要的响应
    if state.get("waiting_for_confirmation"):
        if user_input in ["确认", "是的", "是", "确定", "好", "好的", "y", "yes", "同意"]:
            # 用户确认，保存记录到档案
            current_drug_event = state.get("current_drug_event", {})
            collected_info = current_drug_event.get("collected_info", {})
            
            try:
                async_session = get_sessionmaker()
                async with async_session() as session:
                    # 检查是否已存在相同记录
                    if await _has_same_drug_record(user_id, current_drug_event["drug_name"]):
                        state["confirmation_messages"] = ["用药记录已存在，无需重复添加。"]
                    else:
                        # 创建新的用药记录，使用 remark 字段
                        drug_record = UserDrugRecord(
                            user_id=user_id,
                            drug_name=current_drug_event["drug_name"],
                            dosage=collected_info.get("dosage", "未指定"),
                            frequency=collected_info.get("frequency", "未指定"),
                            start_date=datetime.now(),  # 简化处理，实际应该解析日期
                            end_date=None,
                            remark=f"多轮收集信息: 剂量-{collected_info.get('dosage', '未指定')}, "
                                   f"频次-{collected_info.get('frequency', '未指定')}, "
                                   f"开始时间-{collected_info.get('start_date_text', '未指定')}, "
                                   f"用途-{collected_info.get('purpose', '未指定')}"
                        )
                        
                        # 添加到数据库
                        session.add(drug_record)
                        await session.commit()
                        
                        state["confirmation_messages"] = ["用药记录已成功添加到您的档案中。"]
                
                # 清理状态
                state.pop("waiting_for_confirmation", None)
                state.pop("current_drug_event", None)
                state.pop("drug_confirmation_summary", None)
                
            except Exception as e:
                # 记录错误但不中断流程
                print(f"保存用药记录到档案失败: {e}")
                state["confirmation_messages"] = ["抱歉，保存用药记录到档案时出现错误，请稍后重试。"]
        else:
            # 用户需要修改信息，重新开始收集
            current_drug_event = state.get("current_drug_event", {})
            current_drug_event["collected_info"] = {}
            state = await _collect_missing_drug_info(state, current_drug_event)
            state.pop("waiting_for_confirmation", None)
        
        return state
    
    # 处理初始用药确认（简单模式）
    if user_input in ["是", "是的", "确认", "确定", "好", "好的", "y", "yes", "同意", "添加", "保存"]:
        # 获取候选用药事件
        candidate_drug_events = state.get("candidate_drug_events")
        if not candidate_drug_events:
            skill_ctx = state.get("skill_ctx") or {}
            med_ctx = skill_ctx.get("medication_confirmation") if isinstance(skill_ctx, dict) else {}
            if isinstance(med_ctx, dict):
                candidate_drug_events = med_ctx.get("candidate_events")
        if candidate_drug_events and user_id:
            # 选择第一个用药事件进行多轮收集
            drug_event = candidate_drug_events[0]
            
            # 从文本中提取已有信息
            extracted_info = await _extract_drug_info_from_text(drug_event.get("full_text", ""))
            drug_event["collected_info"] = extracted_info
            
            # 开始多轮信息收集
            state = await _collect_missing_drug_info(state, drug_event)
            
            # 清除候选用药事件，避免重复提示
            state.pop("candidate_drug_events", None)
    
    return state


async def output_check_and_disclaimer(state: dict) -> dict:
    # 合规检查已禁用
    pass

    # 处理用户对用药记录的确认
    state = await _process_drug_confirmation(state)
    
    # 检查是否处于多轮收集状态
    if state.get("current_question"):
        # 在多轮收集过程中，优先显示收集问题
        state["final_response"] = state["current_question"]
    elif state.get("waiting_for_confirmation"):
        # 在确认阶段，显示确认摘要
        state["final_response"] = state.get("drug_confirmation_summary", "")
    else:
        # 正常情况，使用LLM输出
        state["final_response"] = state.get("llm_output", "")
    
    # 如果有确认消息，添加到最终响应中
    confirmation_msgs = state.get("confirmation_messages", [])
    if confirmation_msgs:
        state["final_response"] += "\n\n" + "\n".join(confirmation_msgs)

    proposed = state.get("proposed_updates") or []
    proposed.append({"scope": "shared", "key": "latest_response", "value": state.get("final_response", ""), "source": "out"})
    state["proposed_updates"] = proposed
    
    return state


async def commit_gate(state: dict) -> dict:
    """提交门禁：仅允许白名单字段进入 shared_facts。"""
    if state.get("error_msg"):
        return state

    shared = dict(state.get("shared_facts") or {})
    allow_keys = {"intent", "target_agent", "extract_entities", "retrieved_knowledge", "latest_response"}

    for item in (state.get("proposed_updates") or []):
        if not isinstance(item, dict):
            continue
        if item.get("scope") != "shared":
            continue
        key = item.get("key")
        if key in allow_keys:
            shared[key] = item.get("value")

    # 稳定写入高价值事实（即使没有显式提案）
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
    # 只有在final_response存在时才更新助手记忆
    if "final_response" in state:
        await mem.update_user_memory(state["user_id"], state["session_id"], "assistant", state["final_response"])

    # 写入长期记忆（向量库）：LLM抽取 + add；失败不影响主流程
    try:
        start_time = time.time()
        svc = LongMemoryService()
        if svc.is_enabled():
            items = await svc.extract_candidates(user_input=state.get("user_input", ""))
            if items:
                await svc.add_items(user_id=state["user_id"], session_id=state["session_id"], items=items)
                # 记录向量库写入日志
                write_time = time.time() - start_time
                log_vector_store_write(
                    user_id=state["user_id"],
                    session_id=state["session_id"],
                    items=items,
                    write_time=write_time
                )
                
                # 检查是否有用药事件，如果有则添加到state中，用于后续提示用户确认是否写入档案
                drug_events = [item for item in items if item.memory_type == "drug_event"]
                if drug_events:
                    # 提取药物名称和相关信息，用于构建确认提示
                    drug_info_list = []
                    for event in drug_events:
                        # 从文本中提取药物名称
                        import re
                        # 先尝试从处理过的文本中恢复原始表述，如果包含'用户'则替换成'我'
                        original_text = event.text.replace('用户', '我')
                        drug_match = re.search(r'(?:吃了|服用了|用了|吃|服用|使用|用)([^，。！？\s]{1,30})', original_text)
                        if drug_match:
                            drug_name = drug_match.group(1).strip()
                            drug_info_list.append({
                                "drug_name": drug_name,
                                "full_text": original_text,  # 使用恢复的原始文本
                                "confidence": event.confidence
                            })
                    
                    if drug_info_list:
                        state["candidate_drug_events"] = drug_info_list
                        skill_ctx = state.get("skill_ctx") or {}
                        skill_ctx["medication_confirmation"] = {"candidate_events": drug_info_list}
                        state["skill_ctx"] = skill_ctx
    except Exception as e:
        # 记录错误日志
        write_time = time.time() - start_time
        log_vector_store_write(
            user_id=state.get("user_id", "unknown"),
            session_id=state.get("session_id", "unknown"),
            items=[],
            write_time=write_time,
            error=str(e)
        )
        pass

    # 持久化会话级运行状态（最小化：仅保存待确认信息）
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
        await AgentStateStore().upsert_state(
            user_id=user_id,
            session_id=session_id,
            state=runtime_state,
        )
    except Exception:
        pass

    return state


async def error_finalize(state: dict) -> dict:
    if state.get("error_msg"):
        state["final_response"] = state["error_msg"]
    return state
