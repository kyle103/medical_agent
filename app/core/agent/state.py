from __future__ import annotations

from typing import Literal
from typing import TypedDict


class AgentState(TypedDict, total=False):
    user_id: str
    session_id: str
    user_input: str

    # 短期记忆（会话上下文）
    history: list[dict]  # [{role, content, create_time}]
    history_text: str
    recall_mode: bool

    # 长期记忆（向量库召回，按 user_id 隔离）
    long_memory_items: list[dict]  # [{memory_id,text,memory_type,source,session_id,created_at}]
    long_memory_text: str

    # 记忆摘要（用于注入LLM，优先摘要；必要时再加短窗口原文）
    memory_summary: str

    # 响应规划：决定是否调用LLM、是否注入记忆、使用何种提示词
    response_mode: Literal["llm_chat", "llm_format", "template_only"]
    inject_memory: bool

    # 可观测性
    intent: str
    intent_confidence: float
    intent_reason: str

    extract_entities: dict
    tool_name: str
    tool_result: dict
    llm_output: str
    compliance_check_result: bool
    final_response: str
    error_msg: str
