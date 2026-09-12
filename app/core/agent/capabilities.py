"""能力注册表：系统可路由的工具 / Agent 清单。

从 `llm_decision_service.py` 抽出为独立模块，目的只有一个：**打断循环导入**。
`schemas.py` 需要按注册表生成 `target_name` 的合法取值，而 `llm_decision_service.py`
又需要导入 `schemas.py`；若注册表仍留在后者，就会形成
`schemas -> llm_decision_service -> schemas` 的环。

`llm_decision_service` 仍会 re-export 本常量，历史调用方（`planner_agent` 等）无需改动。
"""

from __future__ import annotations

from typing import Any

CAPABILITY_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "drug_interaction",
        "type": "tool",
        "description": "查询两种或多种药物之间的相互作用、配伍禁忌、能否同服。输入：药品名称列表。输出：相互作用结果。",
        "when_to_use": "用户询问两种及以上药物能否一起吃、是否有冲突、相互作用、配伍禁忌等。",
    },
    {
        "name": "drug_record_agent",
        "type": "agent",
        "description": "管理用药记录：添加、查询、删除用户的用药信息。输入：药品名称、剂量、频率等。输出：操作确认或记录列表。",
        "when_to_use": "用户明确想记录/添加/删除自己的用药信息（如'我吃了XX药'、'帮我记录用药'），或查询自己的用药记录列表。注意：如果用户只是问'可以吃什么药'或'推荐什么药'，应路由到main_qa_agent。",
    },
    {
        "name": "main_qa_agent",
        "type": "agent",
        "description": "通用医疗问答、档案查询与药物推荐。输入：用户问题。输出：基于档案或知识的回答，包括疾病用药推荐等科普信息。",
        "when_to_use": "用户查询自己的健康档案、就诊记录、历史用药，提出通用健康科普问题，或询问某种疾病可以吃什么药、推荐用药等。",
    },
    {
        "name": "lab_report",
        "type": "tool",
        "description": "解读化验单指标。输入：检验指标名称和数值。输出：基于参考范围的指标解读。",
        "when_to_use": "用户要求解读化验单、血常规、尿常规等检验指标。",
    },
]


def valid_target_names() -> tuple[str, ...]:
    """注册表里的全部合法 target_name（供 schema 校验与错误提示使用）。"""
    return tuple(c["name"] for c in CAPABILITY_REGISTRY)


def expected_target_type(target_name: str) -> str | None:
    """给定 tool/agent 名字，返回注册表里声明的类型；未注册返回 None。"""
    return next((c["type"] for c in CAPABILITY_REGISTRY if c["name"] == target_name), None)
