"""
Agent路由器
负责将请求路由到对应的专业Agent并执行
"""

from typing import Dict, Any

from app.core.agent.drug_conflict_agent import DrugConflictAgent
from app.core.agent.drug_record_agent import DrugRecordAgent
from app.core.agent.lab_report_agent import LabReportAgent
from app.core.agent.main_qa_agent import MainQAAgent


class AgentRouter:
    """Agent路由器，负责管理各个专业Agent的调用"""
    
    def __init__(self):
        # 初始化各个Agent实例
        self.agents = {
            "drug_conflict_agent": DrugConflictAgent(),
            "drug_record_agent": DrugRecordAgent(),
            "lab_report_agent": LabReportAgent(),
            "main_qa_agent": MainQAAgent(),
        }
    
    async def route_and_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """路由请求到对应的Agent并执行"""
        target_agent = state.get("target_agent")
        
        if not target_agent:
            state["error_msg"] = "无法确定处理请求的Agent"
            return state
        
        if target_agent not in self.agents:
            state["error_msg"] = f"未知的Agent: {target_agent}"
            return state
        
        # 获取对应的Agent实例
        agent = self.agents[target_agent]
        
        # 调用Agent处理请求
        try:
            result_state = await agent.process(state)
            return result_state
        except Exception as e:
            state["error_msg"] = f"Agent执行失败: {str(e)}"
            return state
    
    def get_agent(self, agent_name: str):
        """获取指定的Agent实例"""
        return self.agents.get(agent_name)
    
    def list_agents(self):
        """列出所有可用的Agent"""
        return list(self.agents.keys())