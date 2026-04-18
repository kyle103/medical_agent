"""
用药记录Agent
专门处理用药记录的增删改查和用药历史查询
"""

from typing import Dict, Any
from datetime import datetime

from app.core.agent.base_agent import BaseAgent
from app.core.tools.archive_query_tool import ArchiveQueryTool
from app.core.prompts import Prompts


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
        
        # 查询操作关键词
        query_keywords = ["查看", "查询", "检查", "有哪些", "有什么", "最近", "历史", "用药记录", "档案"]
        if any(keyword in user_input_lower for keyword in query_keywords):
            return "query"
        
        # 添加操作关键词
        add_keywords = ["添加", "录入", "新增", "创建", "我吃了", "我服用了", "我要用", "我需要", "我想", "吃了", "服用", "用了", "需要记录", "需要添加"]
        if any(keyword in user_input_lower for keyword in add_keywords):
            return "add"
        
        # 删除操作关键词
        delete_keywords = ["删除", "移除", "取消", "清除", "不要了"]
        if any(keyword in user_input_lower for keyword in delete_keywords):
            return "delete"
        
        return "general"
    
    async def _handle_add_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        """处理添加用药记录操作"""
        # 解析用药信息
        drug_info = await self._parse_drug_info(user_input)
        
        if not drug_info["drug_name"]:
            return "请提供药品名称，例如：我吃了布洛芬"
        
        # 写入数据库
        try:
            from sqlalchemy import select, and_
            from app.db.models import UserDrugRecord
            from app.db.database import get_sessionmaker
            from datetime import datetime
            
            async_session = get_sessionmaker()
            async with async_session() as session:
                # 检查是否已存在相同记录
                stmt = select(UserDrugRecord).where(
                    and_(
                        UserDrugRecord.user_id == user_id,
                        UserDrugRecord.drug_name == drug_info["drug_name"],
                        UserDrugRecord.is_deleted == 0
                    )
                )
                result = await session.execute(stmt)
                existing_record = result.scalars().first()
                
                if existing_record:
                    return "用药记录已存在，无需重复添加。"
                
                # 创建新的用药记录
                drug_record = UserDrugRecord(
                    user_id=user_id,
                    drug_name=drug_info["drug_name"],
                    dosage=drug_info["dosage"] or "未指定",
                    frequency=drug_info["frequency"] or "未指定",
                    start_date=datetime.now(),
                    end_date=None,
                    remark=f"添加时间: {drug_info['time']}"
                )
                
                # 添加到数据库
                session.add(drug_record)
                await session.commit()
                
                return f"已为您记录用药信息：{drug_info['drug_name']}，{drug_info['frequency']}，{drug_info['dosage']}。如需修改或添加其他用药信息，请随时告诉我。\n重要提醒：本记录仅用于您个人用药信息的管理，具体用药请遵医嘱，如有疑问建议咨询专业医师或药师。"
        except Exception as e:
            return f"保存用药记录时出现错误：{str(e)}"
    
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
    
    async def _parse_drug_info(self, user_input: str) -> dict:
        """解析用药信息"""
        drug_info = {
            "drug_name": "",
            "dosage": "",
            "frequency": "",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 简单的关键词匹配
        user_input_lower = user_input.lower()
        
        # 使用DrugKnowledgeService匹配药品名称
        try:
            from app.core.rag.drug_knowledge_service import DrugKnowledgeService
            
            # 提取可能的药品名称
            import re
            drug_patterns = [
                r'(?:吃了|服用了|用了|吃|服用|使用|用)([^，。！？\s]{1,30})',
                r'([^，。！？\s]{1,30})(?:片|粒|胶囊|支|瓶|袋|贴)',
                r'(?:药名|药品|药物)\s*[:：]\s*([^，。！？\s]{1,30})'
            ]
            
            candidate_names = []
            for pattern in drug_patterns:
                matches = re.findall(pattern, user_input)
                candidate_names.extend(matches)
            
            # 使用药品知识库进行匹配
            if candidate_names:
                svc = DrugKnowledgeService()
                matched_drugs = await svc.match_drugs(candidate_names)
                
                # 优先返回匹配成功的药品名
                for result in matched_drugs:
                    if result.get("match"):
                        drug_info["drug_name"] = result["match"]["drug_name"]
                        break
                
                # 如果没有匹配到，使用第一个候选名称
                if not drug_info["drug_name"] and candidate_names:
                    drug_info["drug_name"] = candidate_names[0]
        except Exception as e:
            # 如果DrugKnowledgeService失败，使用常见药品列表
            common_drugs = ["布洛芬", "阿司匹林", "青霉素", "头孢", "降压药", "降糖药", 
                           "抗生素", "止痛药", "感冒药", "消炎药", "消食片"]
            
            for drug in common_drugs:
                if drug in user_input:
                    drug_info["drug_name"] = drug
                    break
        
        # 剂量信息
        dosage_patterns = [
            r'(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)',
            r'(一次|每次)\s*(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)',
            r'(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋|贴)\s*(一次|每次)'
        ]
        
        for pattern in dosage_patterns:
            match = re.search(pattern, user_input)
            if match:
                drug_info["dosage"] = match.group(0)
                break
        
        # 用药频率
        frequency_patterns = [
            r'(一天|每日)\s*(\d+)\s*次',
            r'(\d+)\s*次\s*(一天|每日)',
            r'(早晚|早中晚|早中晚各一次|早晚各一次|早中晚各一次)',
            r'(需要时|必要时|疼痛时|不适时)'
        ]
        
        for pattern in frequency_patterns:
            match = re.search(pattern, user_input)
            if match:
                drug_info["frequency"] = match.group(0)
                break
        
        return drug_info
    
    def get_system_prompt(self) -> str:
        return Prompts.get_prompt("DRUG_RECORD_AGENT")