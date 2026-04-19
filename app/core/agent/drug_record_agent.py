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
        operation_type = await self._parse_operation_type(user_input, state)
        
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
    
    async def _parse_operation_type(self, user_input: str, state: dict) -> str:
        system_prompt = "你是一个专门负责判断用户在用药记录方面意图的助手。你需要根据用户的最新输入以及上下文，判断用户是要：1. 'add'：记录、添加、补充自己吃了什么药（即使只是补充某个时间、频率、剂量等细节，也是 'add'）。2. 'query'：查询、产看自己的用药历史记录。3. 'delete'：删除自己的用药记录。4. 'general'：其他情况。请只输出一个字符串：'add', 'query', 'delete' 或者是 'general'，不要有多余字符。"
        memory_messages = state.get('history', [])
        ctx_msgs = memory_messages[-6:] if len(memory_messages) > 6 else memory_messages
        user_prompt = f"上下文记录：{ctx_msgs}\n\n当前用户输入：{user_input}\n请输出判断结果："
        try:
            response = await self._call_llm(user_prompt, system_prompt, state)
            op = response.strip().lower()
            if 'add' in op:
                return 'add'
            elif 'query' in op:
                return 'query'
            elif 'delete' in op:
                return 'delete'
            return 'general'
        except Exception:
            return 'general'

    async def _handle_add_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        """处理添加用药记录操作"""
        # 解析用药信息
        drug_info = await self._parse_drug_info(user_input, state)
        
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
    
    async def _parse_drug_info(self, user_input: str, state: dict) -> dict:
        memory_messages = state.get('history', [])
        ctx_msgs = memory_messages[-6:] if len(memory_messages) > 6 else memory_messages
        system_prompt = "你是一个医疗信息抽取助手。请从用户的最新回复和上下文中，提取用药记录信息。以JSON格式返回，包含以下字段：1. drug_name 药品名称，如果是补充信息且未提及药名，请从上下文中找到药名并填入。如果仍然找不到，填空字符串。 2. dosage 剂量，如'100mg'，'1片'。3. frequency 频率，如'每天一次'，'早晚各一次'。4. time 用药时间，如'昨天晚上八点'、'今天中午'，不要用现在的系统时间。如果字段没有提及并没有在上下文中，请填空字符串。必须且只输出合法的 JSON，不要输出 Markdown 标记，也不要有任何其他解释内容。"
        user_prompt = f"对话上下文：\n{ctx_msgs}\n\n用户最新输入：{user_input}"
        try:
            import json, re
            from datetime import datetime
            response = await self._call_llm(user_prompt, system_prompt, state)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            json_str = match.group(0) if match else response
            data = json.loads(json_str)
            drug_info = {
                'drug_name': data.get('drug_name', ''),
                'dosage': data.get('dosage', ''),
                'frequency': data.get('frequency', ''),
                'time': data.get('time', '')
            }
            if not drug_info['time']:
                 drug_info['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return drug_info
        except Exception:
            from datetime import datetime
            return {'drug_name': '', 'dosage': '', 'frequency': '', 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    def get_system_prompt(self) -> str:
        return Prompts.get_prompt("DRUG_RECORD_AGENT")
