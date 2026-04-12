from __future__ import annotations

import json
import os
import time
from datetime import datetime

from app.core.compliance.compliance_service import ComplianceService
from app.core.memory.memory_service import MemoryService
from app.core.memory.long_memory_service import LongMemoryService
from app.core.llm.llm_service import LLMService
from app.core.tools.archive_query_tool import ArchiveQueryTool
from app.core.tools.drug_interaction_tool import DrugInteractionTool
from app.core.tools.lab_report_tool import LabReportTool

# 确保日志目录存在
def ensure_log_directory():
    log_dir = os.path.join(os.getcwd(), 'logs')
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
    "archive": "档案管理",
    "drug": "药物冲突查询",
    "lab": "化验单解读",
    "general": "通用科普",
}


def _need_contextual_memory(user_input: str) -> bool:
    """无需显式提示词的上下文需求判定（轻量规则）。"""
    t = (user_input or "").strip()
    if not t:
        return False

    # 短追问/承接
    if len(t) <= 12:
        return True

    # 指代/省略
    pronouns = ["那个", "它", "这", "这样", "上面", "刚才", "之前", "继续", "然后", "还要", "还用", "还需要"]
    if any(p in t for p in pronouns):
        return True

    # 复盘/总结类
    recall = ["总结", "回顾", "复盘", "你还记得", "你记得", "还记得"]
    if any(k in t for k in recall):
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
    ok, msg = ComplianceService().input_compliance_check(state.get("user_input", ""))
    if not ok:
        state["error_msg"] = msg
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
        return state

    mem = MemoryService()
    history = await mem.get_user_memory(user_id=user_id, session_id=session_id, limit=12)
    state["history"] = history
    state["history_text"] = _format_history(history, max_chars=1400)
    state["memory_summary"] = await mem.get_memory_summary(user_id=user_id, session_id=session_id)

    # 长期记忆召回（向量库）：不影响主流程，失败则忽略
    state["long_memory_items"] = []
    state["long_memory_text"] = ""
    try:
        start_time = time.time()
        svc = LongMemoryService()
        query = state.get("user_input", "")
        if svc.is_enabled():
            items = svc.recall(user_id=user_id, query=query, top_k=3)
            state["long_memory_items"] = [
                {
                    "memory_id": it.memory_id,
                    "text": it.text,
                    "memory_type": it.memory_type,
                    "source": it.source,
                    "session_id": it.session_id,
                    "created_at": it.created_at,
                }
                for it in items
            ]
            if items:
                uniq = []
                seen = set()
                for it in items:
                    if it.memory_id not in seen:
                        seen.add(it.memory_id)
                        uniq.append(it)
                state["long_memory_text"] = "\n".join([f"- {it.text}" for it in uniq])
            
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
                    for it in items
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

    text = state.get("user_input", "")
    stream = bool(state.get("stream", False))

    clf = IntentClassifier()
    res = await clf.predict(text=text, stream=stream)

    state["intent"] = res.intent
    state["intent_confidence"] = res.confidence
    state["intent_reason"] = res.reason
    return state


async def entity_extraction(state: dict) -> dict:
    intent = state.get("intent")
    text = state.get("user_input", "")

    entities: dict = {}
    if intent == "drug":
        parts = [p.strip() for p in text.replace("、", ",").split(",") if p.strip()]
        entities["drug_name_list"] = parts[:10]
    elif intent == "lab":
        entities["raw"] = text
    else:
        entities["query"] = text

    state["extract_entities"] = entities
    return state


async def tool_route(state: dict) -> dict:
    intent = state.get("intent")
    if intent == "drug":
        state["tool_name"] = "drug_interaction"
    elif intent == "lab":
        state["tool_name"] = "lab_report"
    elif intent == "archive":
        state["tool_name"] = "archive"
    else:
        state["tool_name"] = "general"
    return state


async def tool_execute(state: dict) -> dict:
    user_id = state.get("user_id")
    intent = state.get("intent")
    entities = state.get("extract_entities") or {}

    if intent == "drug":
        tool = DrugInteractionTool()
        state["tool_result"] = await tool.check_interactions(
            user_id=user_id,
            drug_name_list=entities.get("drug_name_list") or [],
            sync_to_archive=False,
        )
    elif intent == "lab":
        tool = LabReportTool()
        lab_items = []
        lines = [l.strip() for l in state.get("user_input", "").splitlines() if l.strip()]
        for l in lines:
            if ":" in l:
                n, v = l.split(":", 1)
                lab_items.append({"item_name": n.strip(), "test_value": v.strip(), "unit": None})
        state["tool_result"] = await tool.interpret(
            user_id=user_id,
            lab_item_list=lab_items or [{"item_name": "血糖", "test_value": "0", "unit": "mmol/L"}],
            sync_to_archive=False,
        )
    elif intent == "archive":
        tool = ArchiveQueryTool()
        state["tool_result"] = await tool.query_recent_drugs(user_id=user_id, days=7, limit=20)
    else:
        state["tool_result"] = {
            "final_desc": "我可以提供通用健康科普与就医指引类信息，也可以帮你做药物相互作用科普查询或化验指标通用解读。请描述你的问题或提供药品/指标名称。"
        }

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

    need_mem = _need_contextual_memory(user_input)
    if not (state.get("history") or state.get("memory_summary") or state.get("long_memory_text")):
        need_mem = False

    state["response_mode"] = mode
    state["inject_memory"] = bool(need_mem)
    return state


async def llm_generate(state: dict) -> dict:
    """真正调用 LLM 生成更自然的输出。"""

    content = (state.get("tool_result") or {}).get("final_desc") or ""
    if not content:
        content = "当前未获取到有效工具结果。"

    mode = state.get("response_mode") or "llm_chat"
    inject_memory = bool(state.get("inject_memory"))

    mem_summary = (state.get("memory_summary") or "").strip()
    if not mem_summary and inject_memory:
        mem_summary = _short_window_history(state.get("history") or [], max_turns=4)

    long_mem = (state.get("long_memory_text") or "").strip()

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
    except Exception:
        state["llm_output"] = content

    return state


async def output_check_and_disclaimer(state: dict) -> dict:
    compliance = ComplianceService()

    ok, msg = compliance.output_compliance_check(state.get("llm_output", ""))
    if not ok:
        state["error_msg"] = msg
        return state

    state["final_response"] = compliance.add_disclaimer(state.get("llm_output", ""))
    return state


async def memory_update(state: dict) -> dict:
    if state.get("error_msg"):
        return state

    mem = MemoryService()
    await mem.update_user_memory(state["user_id"], state["session_id"], "user", state["user_input"])
    await mem.update_user_memory(state["user_id"], state["session_id"], "assistant", state["final_response"])

    # 写入长期记忆（向量库）：LLM抽取 + add；失败不影响主流程
    try:
        start_time = time.time()
        svc = LongMemoryService()
        if svc.is_enabled():
            items = await svc.extract_candidates(user_input=state.get("user_input", ""))
            if items:
                svc.add_items(user_id=state["user_id"], session_id=state["session_id"], items=items)
                # 记录向量库写入日志
                write_time = time.time() - start_time
                log_vector_store_write(
                    user_id=state["user_id"],
                    session_id=state["session_id"],
                    items=items,
                    write_time=write_time
                )
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

    return state


async def error_finalize(state: dict) -> dict:
    if state.get("error_msg"):
        state["final_response"] = ComplianceService().add_disclaimer(state["error_msg"])
    return state
