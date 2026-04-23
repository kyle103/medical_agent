from __future__ import annotations

from typing import Literal
from typing import TypedDict


class AgentState(TypedDict, total=False):
    user_id: str
    session_id: str
    user_input: str
    stream: bool
    enable_archive_link: bool

    # 短期记忆（会话上下文）
    history: list[dict]  # [{role, content, create_time}]
    history_text: str
    recall_mode: bool

    # 长期记忆（向量库召回，按 user_id 隔离）
    long_memory_items: list[dict]  # [{memory_id,text,memory_type,source,session_id,created_at}]
    long_memory_text: str

    # 记忆摘要（用于注入LLM，优先摘要；必要时再加短窗口原文）
    memory_summary: str

    # 分层状态：共享事实/私有草稿/候选提交
    shared_facts: dict
    private_scratchpads: dict  # {agent_name: {...}}
    proposed_updates: list[dict]  # [{scope,key,value,source}]
    skill_ctx: dict  # 各 skill 私有上下文

    # 响应规划：决定是否调用LLM、是否注入记忆、使用何种提示词
    response_mode: Literal["llm_chat", "llm_format", "template_only"]
    inject_memory: bool

    # 专业知识检索结果（非用户长期记忆）
    retrieved_knowledge: dict

    # 可观测性
    intent: str
    intent_confidence: float
    intent_reason: str
    intent_type: str
    intent_analysis: dict
    target_agent: str

    extract_entities: dict
    tool_name: str
    tool_result: dict
    llm_output: str
    needs_confirmation: bool
    confirmation_message: str
    compliance_check_result: bool
    final_response: str
    error_msg: str

    # 运行过程中的中间状态
    candidate_drug_events: list[dict]
    pending_drug_events_for_confirmation: list[dict]
    waiting_for_answer: str
    waiting_for_confirmation: bool
    current_question: str
    current_drug_event: dict
    drug_confirmation_summary: str
    confirmation_messages: list[str]

    # 会话级运行状态（持久化）
    session_runtime_state: dict
    pending_confirmation: dict
