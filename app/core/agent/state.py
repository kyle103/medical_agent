from __future__ import annotations

from typing import Literal
from typing import TypedDict


class PlanStep(TypedDict, total=False):
    step_id: str
    query: str
    target_type: Literal["agent", "tool"]
    target_name: str
    intent_type: str
    depends_on: list[str]
    execution_strategy: Literal["serial", "parallel"]
    step_context: dict


class ExecutionPlan(TypedDict, total=False):
    steps: list[PlanStep]
    strategy: Literal["single", "topological"]
    conflict_resolution_policy: str


class AgentState(TypedDict, total=False):
    user_id: str
    session_id: str
    user_input: str
    stream: bool
    enable_archive_link: bool

    history: list[dict]
    history_text: str
    recall_mode: bool

    long_memory_items: list[dict]
    long_memory_text: str

    memory_summary: str

    shared_facts: dict
    private_scratchpads: dict
    proposed_updates: list[dict]
    skill_ctx: dict

    decision_context: str
    last_decision: dict

    retrieved_knowledge: dict

    intent: str
    intent_confidence: float
    intent_reason: str
    intent_type: str
    intent_analysis: dict
    target_agent: str
    # 意图节点那次必调的 LLM 调用顺带判定的「是否多意图」，零额外往返地给 plan 兜底
    is_multi_intent: bool

    extract_entities: dict
    tool_name: str
    tool_result: dict
    llm_output: str
    needs_confirmation: bool
    confirmation_message: str
    compliance_check_result: bool
    final_response: str
    error_msg: str

    candidate_drug_events: list[dict]
    pending_drug_events_for_confirmation: list[dict]

    session_runtime_state: dict
    pending_confirmation: dict

    execution_plan: ExecutionPlan
    plan_step_results: dict
    reconciled_sections: list[str]
    # 多段结果标记（不再复用 intent —— 把 intent 改成 "multi" 会污染下游的模式判定与记忆写入）
    is_multi_section: bool
    # 汇总后的结构化上下文：药名/指标/关键结论/是否有冲突等，供生成策略判定与 llm 注入
    reconciled_context: dict
    plan_phase: Literal["planning", "executing", "reconciling", "responding"]

    replan_count: int
    replan_reason: str
    needs_replan: bool
    # 重规划上下文：失败步骤的错误信息 + 已完成步骤的结果摘要，供 LLM 基于错误重新规划
    replan_context: dict
    # 跨步骤冲突：多步分散提及的药物 + 冲突结论，供补检与最终呈现
    cross_step_conflict: dict

    force_long_memory_write: bool
    long_memory_write_source: str
