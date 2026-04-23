from __future__ import annotations

from typing import Dict

from app.core.agent.base_agent import BaseAgent
from app.core.agent.drug_conflict_agent import DrugConflictAgent
from app.core.agent.drug_record_agent import DrugRecordAgent
from app.core.agent.lab_report_agent import LabReportAgent
from app.core.agent.main_qa_agent import MainQAAgent


def build_agent_registry() -> Dict[str, BaseAgent]:
    """统一注册 Agent，便于调度器基于能力动态发现。"""
    agents: list[BaseAgent] = [
        DrugConflictAgent(),
        DrugRecordAgent(),
        LabReportAgent(),
        MainQAAgent(),
    ]
    return {agent.agent_name: agent for agent in agents}

