from __future__ import annotations

from typing import Annotated
from typing import Literal
from typing import TypedDict

from langgraph.channels import UntrackedValue


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
    """图状态。

    ── 关于 `UntrackedValue`（Step 3.5）──

    `langgraph` 的 `UntrackedValue` 语义是「Stores the last value received, never checkpointed」：
    **通道照常存值、节点之间照常传递**，只是 `checkpoint()` 永远抛 `EmptyChannelError`，
    因此**不会进快照**。写入并发性也与默认通道 `LastValue` 完全一致（一步内多写同样报
    `InvalidUpdateError`），所以**标注不改变单次运行内的任何行为**，只改变"落不落盘"。

    标注判据（按 `scripts/probe_state_snapshot_fields.py` 在真实图上的实测结果定，不靠猜）：

      a. 跨轮业务态，且已有独立持久化 owner（`AgentStateStore`）→ 排除
         理由：`memory_load` 每轮从 DB 还原，让 checkpointer 再存一份只会产生不一致。
      b. 每轮由既有持久化层重建或派生的 → 排除
         理由：丢了也能逐字重建，快照里存它纯属重复。
      c. 单次运行内有效、且消费方全在本轮内的中间产物 → 排除
         理由：跨轮恢复时不会被读到；其中 `plan_step_results` 更是**必须排除**——它内部
         嵌着整份 state 副本（`_build_sub_state` 做的是 `dict(state)`），构成循环引用，
         任何 JSON/msgpack 快照序列化器都会失败。
      d. 控制流与小标量 → 保持
      e. 最终产出 → 保持

    ── 已知代价（必须在 Step 4/6 之前记得）──

    `pending_confirmation` / `last_decision` 排除后不再进快照，当前功能不受影响
    （跨轮确认仍由 `AgentStateStore` + `memory_load` 负责），但 **`interrupt()` 的恢复
    依赖快照包含决策字段** → 做 HITL 时要把确认相关字段改回持久化通道，或把 interrupt 的
    payload 设计成自包含。这是已知的前后依赖，不是遗漏。

    `plan_step_results` 排除后，中途续跑会重跑计划步骤（`execute_node` 靠它跳过已完成步骤）。
    根治办法是剥掉子状态里的整份 state 副本、让它重新变成可序列化的小对象，届时可改回保持快照。
    """

    user_id: str
    session_id: str
    user_input: str
    stream: bool
    enable_archive_link: bool

    # (b) memory_load 每轮从 MemoryService 重载 / 由 history 派生
    history: Annotated[list[dict], UntrackedValue]
    history_text: Annotated[str, UntrackedValue]
    recall_mode: bool

    # (b) 每轮从长期记忆层重载 / 由 items 派生
    long_memory_items: Annotated[list[dict], UntrackedValue]
    long_memory_text: Annotated[str, UntrackedValue]

    # (b) 每轮由 get_memory_summary 重建
    memory_summary: Annotated[str, UntrackedValue]

    # (c) 本轮共享事实/私有草稿板/待更新项：单轮有效，且 shared_facts 内含 retrieved_knowledge 副本
    shared_facts: Annotated[dict, UntrackedValue]
    private_scratchpads: Annotated[dict, UntrackedValue]
    proposed_updates: Annotated[list[dict], UntrackedValue]
    # (c) 1435 行写入本轮用药确认上下文（含 candidate_events 副本），单轮有效
    skill_ctx: Annotated[dict, UntrackedValue]

    # (c) 由历史/记忆/长期记忆拼出的决策上下文，单轮用于消解指代
    decision_context: Annotated[str, UntrackedValue]

    # (a) AgentStateStore 为唯一 owner（memory_update 每轮回写、memory_load 每轮还原）
    last_decision: Annotated[dict, UntrackedValue]

    # (c) 本轮 RAG 召回片段
    retrieved_knowledge: Annotated[dict, UntrackedValue]

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
    # (c) 本轮工具返回，仅本轮 reconcile/生成消费
    tool_result: Annotated[dict, UntrackedValue]
    # (c) 生成原文，是 final_response 的前缀子集；仅本轮 out / fact_check 消费，恢复时以 final_response 为准
    llm_output: Annotated[str, UntrackedValue]
    needs_confirmation: bool
    confirmation_message: str
    compliance_check_result: bool
    final_response: str
    error_msg: str

    # (c) 本轮药物事件抽取结果 / 待确认事件（后者为声明但未使用的字段）
    candidate_drug_events: Annotated[list[dict], UntrackedValue]
    pending_drug_events_for_confirmation: Annotated[list[dict], UntrackedValue]

    # (a) memory_load 会把 AgentStateStore 整行返回塞进来；owner 是 AgentStateStore，不让 checkpointer 接管。
    #     注意该字段**永不单独写回**（写回走 memory_update 里的 runtime_state 局部变量），
    #     所以排除出快照不会丢数据。
    session_runtime_state: Annotated[dict, UntrackedValue]
    # (a) 跨轮确认态，owner 是 AgentStateStore
    pending_confirmation: Annotated[dict, UntrackedValue]

    execution_plan: ExecutionPlan
    # (c) 每步结果内部嵌着整份 state 副本 → 循环引用，无法序列化，必须排除
    plan_step_results: Annotated[dict, UntrackedValue]
    # (c) 本轮汇总结论片段
    reconciled_sections: Annotated[list[str], UntrackedValue]
    # 多段结果标记（不再复用 intent —— 把 intent 改成 "multi" 会污染下游的模式判定与记忆写入）
    is_multi_section: bool
    # (c) 汇总后的结构化上下文：药名/指标/关键结论/是否有冲突等，供本轮生成策略判定与 llm 注入
    reconciled_context: Annotated[dict, UntrackedValue]
    plan_phase: Literal["planning", "executing", "reconciling", "responding"]

    replan_count: int
    replan_reason: str
    needs_replan: bool
    # (c) 重规划上下文：失败步骤的错误信息 + 已完成步骤的结果摘要，供本轮 LLM 基于错误重新规划
    replan_context: Annotated[dict, UntrackedValue]
    # (c) 跨步骤冲突：多步分散提及的药物 + 冲突结论，供本轮补检与最终呈现
    cross_step_conflict: Annotated[dict, UntrackedValue]

    force_long_memory_write: bool
    long_memory_write_source: str
