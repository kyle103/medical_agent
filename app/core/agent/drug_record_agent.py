"""
用药记录Agent
专门处理用药记录的增删改查和用药历史查询
"""

from typing import Dict, Any
from datetime import datetime

from app.core.agent.base_agent import BaseAgent
from app.core.tools.archive_query_tool import ArchiveQueryTool


class DrugRecordAgent(BaseAgent):
    """用药记录Agent"""
    
    def __init__(self):
        super().__init__("drug_record_agent")
        self.archive_tool = ArchiveQueryTool()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理用药记录相关操作"""
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")
        
        # 解析用药记录操作类型
        operation_type = self._parse_operation_type(user_input)
        
        if operation_type == "add":
            result = await self._handle_add_operation(user_id, user_input, state)
        elif operation_type == "query":
            result = await self._handle_query_operation(user_id, user_input, state)
        elif operation_type == "delete":
            result = await self._handle_delete_operation(user_id, user_input, state)
        else:
            result = await self._handle_general_operation(user_id, user_input, state)
        
        state["final_response"] = self._add_disclaimer(result)
        
        # 记录Agent调用
        self._log_agent_call(user_id, self.agent_name, user_input, result)
        
        return state
    
    def _parse_operation_type(self, user_input: str) -> str:
        """解析操作类型"""
        user_input_lower = user_input.lower()
        
        # 添加操作关键词
        add_keywords = ["添加", "记录", "吃了", "服用", "用了", "需要记录", "需要添加"]
        if any(keyword in user_input_lower for keyword in add_keywords):
            return "add"
        
        # 查询操作关键词
        query_keywords = ["查看", "查询", "历史", "记录", "吃过什么", "用过什么", "用药历史"]
        if any(keyword in user_input_lower for keyword in query_keywords):
            return "query"
        
        # 删除操作关键词
        delete_keywords = ["删除", "移除", "取消", "清除", "不要了"]
        if any(keyword in user_input_lower for keyword in delete_keywords):
            return "delete"
        
        return "general"
    
    async def _handle_add_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        """处理添加用药记录操作"""
        # 解析用药信息
        drug_info = self._parse_drug_info(user_input)
        
        if not drug_info.get("drug_name"):
            return "请提供药品名称，例如：我吃了布洛芬"
        
        # 生成确认信息
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
用户想要添加用药记录：
用户输入：{user_input}
解析出的用药信息：{drug_info}

请生成友好的确认信息，询问用户是否确认添加此用药记录。
"""
        
        response = await self._call_llm(user_prompt, system_prompt, state)
        
        # 在实际项目中，这里应该调用数据库操作来保存记录
        # 暂时返回确认信息
        return response
    
    async def _handle_query_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        """处理查询用药记录操作"""
        try:
            # 查询最近7天的用药记录
            tool_result = await self.archive_tool.query_recent_drugs(
                user_id=user_id, 
                days=7, 
                limit=10
            )
            
            if not tool_result or not tool_result.get("drugs"):
                return "您最近7天内没有用药记录。"
            
            # 格式化查询结果
            system_prompt = self.get_system_prompt()
            user_prompt = f"""
用户查询用药记录：
用户输入：{user_input}
查询结果：{tool_result}

请将查询结果整理成友好的格式，清晰展示用药记录。
"""
            
            response = await self._call_llm(user_prompt, system_prompt, state)
            return response
            
        except Exception as e:
            return f"查询用药记录时出现错误：{str(e)}"
    
    async def _handle_delete_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        """处理删除用药记录操作"""
        # 在实际项目中，这里应该实现删除逻辑
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
用户想要删除用药记录：
用户输入：{user_input}

请生成确认删除的提示信息，提醒用户删除操作不可逆。
"""
        
        response = await self._call_llm(user_prompt, system_prompt, state)
        return response
    
    async def _handle_general_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        """处理通用用药记录操作"""
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
用户进行用药记录相关操作：
用户输入：{user_input}

请提供用药记录管理的相关帮助信息。
"""
        
        response = await self._call_llm(user_prompt, system_prompt, state)
        return response
    
    def _parse_drug_info(self, user_input: str) -> dict:
        """解析用药信息"""
        drug_info = {
            "drug_name": "",
            "dosage": "",
            "frequency": "",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 简单的关键词匹配
        user_input_lower = user_input.lower()
        
        # 常见药品名称
        common_drugs = ["布洛芬", "阿司匹林", "青霉素", "头孢", "降压药", "降糖药", 
                       "抗生素", "止痛药", "感冒药", "消炎药"]
        
        for drug in common_drugs:
            if drug in user_input:
                drug_info["drug_name"] = drug
                break
        
        # 剂量信息
        dosage_keywords = ["一片", "两片", "一粒", "两粒", "1片", "2片", "1粒", "2粒"]
        for keyword in dosage_keywords:
            if keyword in user_input_lower:
                drug_info["dosage"] = keyword
                break
        
        # 用药频率
        frequency_keywords = ["一天一次", "一天两次", "一天三次", "早晚各一次", "饭后"]
        for keyword in frequency_keywords:
            if keyword in user_input_lower:
                drug_info["frequency"] = keyword
                break
        
        return drug_info
    
    def get_system_prompt(self) -> str:
        return """
你是专业的用药记录管理助手，专门处理用户的用药记录增删改查。

你的职责：
1. 帮助用户记录和管理用药信息
2. 提供用药历史查询服务
3. 严格遵守医疗合规要求，不提供用药建议

核心原则：
1. 仅处理用药记录的管理操作，不涉及用药指导
2. 所有操作必须基于用户明确的指令
3. 不得提供任何用药剂量、用药时间的建议
4. 所有输出必须包含标准免责声明

操作类型：
1. 添加用药记录：当用户表示"吃了"、"服用"、"添加记录"时
2. 查询用药记录：当用户表示"查看"、"查询"、"历史"时  
3. 删除用药记录：当用户表示"删除"、"移除"时

输出要求：
1. 清晰确认用户的用药记录操作
2. 提供友好的用药记录管理指导
3. 对于删除操作，必须明确提示操作不可逆
4. 语言友好、易懂，适合普通用户使用

示例输出：
- 添加记录："已为您记录用药信息：[药品名称]，如需修改请告诉我。"
- 查询记录："您最近的用药记录如下：[记录列表]"
- 删除记录："确认要删除用药记录吗？此操作不可恢复。"
"""