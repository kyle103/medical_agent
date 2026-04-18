"""
主要问答Agent
处理档案查询和通用问答功能
"""

from typing import Dict, Any
import json

from app.core.agent.base_agent import BaseAgent
from app.core.tools.archive_query_tool import ArchiveQueryTool
from app.core.prompts import Prompts


class MainQAAgent(BaseAgent):
    """主要问答Agent - 处理档案查询和通用问答"""
    
    def __init__(self):
        super().__init__("main_qa_agent")
        self.archive_tool = ArchiveQueryTool()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理档案查询和通用问答请求"""
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")
        
        # 判断是否为档案查询
        is_archive_query = self._is_archive_query(user_input)
        
        if is_archive_query:
            return await self._handle_archive_query(state)
        else:
            return await self._handle_general_qa(state)
    
    async def _handle_archive_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理档案查询"""
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")
        
        # 提取查询类型和条件
        query_type, query_conditions = self._parse_archive_query(user_input)
        
        # 调用档案查询工具
        try:
            tool_result = await self.archive_tool.query(
                user_id=user_id,
                query_type=query_type,
                query_conditions=query_conditions,
            )
        except Exception as e:
            state["final_response"] = self._add_disclaimer(
                f"档案查询服务暂时不可用，请稍后重试。错误信息：{str(e)}"
            )
            return state
        
        # 生成档案查询结果
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
用户请求档案查询：
用户输入：{user_input}
查询类型：{query_type}
查询条件：{query_conditions}

查询结果：
{json.dumps(tool_result, ensure_ascii=False, indent=2)}

请基于查询结果生成专业的档案信息展示，要求：
1. 清晰展示用户的档案信息
2. 按时间顺序组织就诊记录、用药记录等
3. 突出重要的健康信息
4. 语言简洁明了，便于用户理解
5. 不涉及疾病诊断或治疗建议
"""
        
        response = await self._call_llm(user_prompt, system_prompt, state)
        
        # 合规校验
        ok, msg = self._check_compliance(response)
        if not ok:
            state["error_msg"] = msg
            state["final_response"] = self._add_disclaimer(
                "抱歉，由于合规要求，无法提供相关档案信息。"
            )
            return state
        
        state["final_response"] = self._add_disclaimer(response)
        
        # 记录Agent调用
        self._log_agent_call(user_id, self.agent_name, user_input, response)
        
        return state
    
    async def _handle_general_qa(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理通用问答"""
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")
        
        # 处理"还记得我之前吃的药吗？"这类问题
        if any(phrase in user_input for phrase in ["还记得我之前吃的药吗", "之前吃过什么药", "我吃了哪些药", "我之前吃的药"]):
            # 优先查询档案中的用药记录
            try:
                tool_result = await self.archive_tool.query(
                    user_id=user_id,
                    query_type="drug_records",
                    query_conditions={},
                )
                
                # 检查是否有用药记录
                drug_records = tool_result.get("items", [])
                if drug_records:
                    # 有档案记录，生成响应
                    system_prompt = self.get_system_prompt()
                    user_prompt = f"""
用户询问之前的用药记录：
用户输入：{user_input}
查询结果：
{tool_result}

请基于查询结果生成专业的用药记录展示，要求：
1. 清晰展示用户的用药记录
2. 按时间顺序组织
3. 语言简洁明了，便于用户理解
4. 不涉及疾病诊断或治疗建议
"""
                    response = await self._call_llm(user_prompt, system_prompt, state)
                else:
                    # 档案为空，检查对话历史
                    history = state.get("history", [])
                    mentioned_drugs = []
                    
                    # 从对话历史中提取提到的药品
                    for message in history:
                        if message.get("role") == "user":
                            content = message.get("content", "")
                            # 简单的药品名称提取
                            common_drugs = ["布洛芬", "阿司匹林", "青霉素", "头孢", "降压药", "降糖药", 
                                           "抗生素", "止痛药", "感冒药", "消炎药", "消食片"]
                            for drug in common_drugs:
                                if drug in content and drug not in mentioned_drugs:
                                    mentioned_drugs.append(drug)
                    
                    if mentioned_drugs:
                        # 有对话中提到的药品
                        response = f"在我们的对话中，您提到过以下药品：{', '.join(mentioned_drugs)}。这些信息尚未写入您的用药档案，如需记录，请告诉我具体的用药信息，我将帮您添加到档案中。"
                    else:
                        # 既没有档案记录，也没有对话中提到的药品
                        response = "根据目前的档案信息，您尚未录入任何用药记录。如果您需要记录用药信息，请告诉我具体的药品名称、用法用量等详情，我将帮您添加到档案中。"
            except Exception as e:
                # 查询失败，检查对话历史
                history = state.get("history", [])
                mentioned_drugs = []
                
                # 从对话历史中提取提到的药品
                for message in history:
                    if message.get("role") == "user":
                        content = message.get("content", "")
                        # 简单的药品名称提取
                        common_drugs = ["布洛芬", "阿司匹林", "青霉素", "头孢", "降压药", "降糖药", 
                                       "抗生素", "止痛药", "感冒药", "消炎药", "消食片"]
                        for drug in common_drugs:
                            if drug in content and drug not in mentioned_drugs:
                                mentioned_drugs.append(drug)
                
                if mentioned_drugs:
                    # 有对话中提到的药品
                    response = f"在我们的对话中，您提到过以下药品：{', '.join(mentioned_drugs)}。这些信息尚未写入您的用药档案，如需记录，请告诉我具体的用药信息，我将帮您添加到档案中。"
                else:
                    # 既没有档案记录，也没有对话中提到的药品
                    response = "根据目前的档案信息，您尚未录入任何用药记录。如果您需要记录用药信息，请告诉我具体的药品名称、用法用量等详情，我将帮您添加到档案中。"
        else:
            # 其他通用问答
            system_prompt = self.get_system_prompt()
            
            # 直接调用LLM进行通用问答
            response = await self._call_llm(user_input, system_prompt, state)
        
        # 合规校验
        ok, msg = self._check_compliance(response)
        if not ok:
            state["error_msg"] = msg
            state["final_response"] = self._add_disclaimer(
                "抱歉，由于合规要求，无法回答该问题。"
            )
            return state
        
        state["final_response"] = self._add_disclaimer(response)
        
        # 记录Agent调用
        self._log_agent_call(user_id, self.agent_name, user_input, response)
        
        return state
    
    def _is_archive_query(self, user_input: str) -> bool:
        """判断是否为档案查询"""
        archive_keywords = [
            "档案", "病历", "就诊记录", "用药记录", "健康档案", 
            "我的信息", "个人信息", "历史记录", "过往病史", "还记得我之前吃的药吗"
        ]
        
        return any(keyword in user_input for keyword in archive_keywords)
    
    def _parse_archive_query(self, user_input: str) -> tuple:
        """解析档案查询类型和条件"""
        query_type = "general"
        query_conditions = {}
        
        # 判断查询类型
        if "就诊" in user_input or "看病" in user_input:
            query_type = "visit_records"
        elif "用药" in user_input or "服药" in user_input:
            query_type = "drug_records"
        elif "化验" in user_input or "检查" in user_input:
            query_type = "lab_records"
        elif "基本信息" in user_input or "个人资料" in user_input:
            query_type = "basic_info"
        
        # 提取时间条件
        import re
        time_patterns = [
            r"(\d+)年", r"(\d+)月", r"(\d+)日",
            r"最近(\d+)天", r"最近(\d+)个月", r"最近(\d+)年"
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, user_input)
            if match:
                query_conditions["time_range"] = user_input
                break
        
        return query_type, query_conditions
    
    def get_system_prompt(self) -> str:
        return Prompts.get_prompt("MAIN_QA_AGENT")