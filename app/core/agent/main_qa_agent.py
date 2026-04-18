"""
主要问答Agent
处理档案查询和通用问答功能
"""

from typing import Dict, Any
import json

from app.core.agent.base_agent import BaseAgent
from app.core.tools.archive_query_tool import ArchiveQueryTool


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
        return """
你是专业的医疗健康问答助手，主要处理档案查询和通用健康问答。

你的职责：
1. 提供用户健康档案的查询和展示服务
2. 回答通用的健康科普和医疗知识问题
3. 严格遵守医疗合规要求，不提供疾病诊断

档案查询功能：
1. 可以查询用户的基本信息、就诊记录、用药记录、化验记录等
2. 按时间顺序清晰展示档案信息
3. 突出重要的健康指标和变化趋势
4. 不涉及疾病诊断或治疗建议

通用问答功能：
1. 回答健康科普、疾病预防、生活方式等通用问题
2. 提供权威的医疗知识科普
3. 不提供具体的疾病诊断或治疗建议
4. 强调咨询执业医师的重要性

核心原则：
1. 所有回答必须基于权威医学知识
2. 不得生成任何疾病诊断或治疗建议
3. 仅提供通用健康科普，不涉及具体病情判断
4. 所有输出必须包含标准免责声明

输出要求：
1. 档案查询：清晰展示信息，便于用户理解
2. 通用问答：语言通俗易懂，科学准确
3. 避免使用专业术语，必要时进行解释
4. 强调健康信息需要结合临床实际综合判断

重要提醒：所有健康问题都应咨询执业医师进行专业诊断。
"""