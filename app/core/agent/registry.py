from __future__ import annotations

from typing import Dict

from app.core.agent.base_agent import BaseAgent
from app.core.agent.drug_record_agent import DrugRecordAgent
from app.core.agent.main_qa_agent import MainQAAgent
from app.core.agent.planner_agent import TOOL_TARGETS
from app.core.agent.tool_executor import ToolExecutor


def build_agent_registry() -> Dict[str, BaseAgent]:
    agents: list[BaseAgent] = [
        DrugRecordAgent(),
        MainQAAgent(),
    ]
    return {agent.agent_name: agent for agent in agents}


def build_tool_registry() -> Dict[str, ToolExecutor]:
    return {name: ToolExecutor() for name in TOOL_TARGETS}

