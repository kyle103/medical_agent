from __future__ import annotations

from typing import Literal
from typing import TypedDict


class PlanStep(TypedDict, total=False):
    step_id: str
    query: str
    target_agent: str
    intent_type: str
    depends_on: list[str]
    execution_strategy: Literal["serial", "parallel"]


class ExecutionPlan(TypedDict, total=False):
    steps: list[PlanStep]
    strategy: Literal["serial", "parallel", "hybrid"]
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

    response_mode: Literal["llm_chat", "llm_format", "template_only"]
    inject_memory: bool

    retrieved_knowledge: dict

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

    candidate_drug_events: list[dict]
    pending_drug_events_for_confirmation: list[dict]

    session_runtime_state: dict
    pending_confirmation: dict

    execution_plan: ExecutionPlan
    plan_step_results: dict
    plan_phase: Literal["planning", "executing", "reconciling", "responding"]
