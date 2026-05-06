from __future__ import annotations

from typing import Dict, Any

from app.core.agent.agent_card import AgentCard
from app.core.agent.base_agent import BaseAgent
from app.core.agent.llm_decision_service import LLMDecisionService
from app.core.tools.drug_entity_extractor import DrugEntityExtractor
from app.core.tools.drug_record_tool import DrugRecordTool
from app.core.prompts import Prompts


class DrugRecordAgent(BaseAgent):
    """用药记录Agent"""

    def __init__(self):
        super().__init__("drug_record_agent")
        self.drug_record_tool = DrugRecordTool()
        self.llm_decision = LLMDecisionService()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")

        operation_type = await self._parse_operation_type(user_input, state)

        if operation_type == "add":
            result = await self._handle_add_operation(user_id, user_input, state)
        elif operation_type == "query":
            result = await self._handle_query_operation(user_id, user_input, state)
        elif operation_type == "delete":
            result = await self._handle_delete_operation(user_id, user_input, state)
        else:
            result = await self._handle_general_operation(user_id, user_input, state)

        state["final_response"] = result

        self._log_agent_call(user_id, self.agent_name, user_input, result)

        return state

    async def _parse_operation_type(self, user_input: str, state: dict) -> str:
        llm_result = await self.llm_decision.classify_operation_type(
            user_input, history=state.get("history", [])
        )
        if llm_result:
            return llm_result

        t = (user_input or "").strip()
        add_keywords = ["记录", "添加", "我吃了", "我服用", "我用了", "昨天", "今天", "两片", "一片", "mg", "毫克"]
        query_keywords = ["查询", "查看", "历史", "最近", "用药记录", "吃过什么药"]
        delete_keywords = ["删除", "移除", "清空"]

        if any(k in t for k in delete_keywords):
            return "delete"
        if any(k in t for k in query_keywords):
            return "query"
        if any(k in t for k in add_keywords):
            return "add"

        return "general"

    async def _handle_add_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        drug_info = await self._parse_drug_info(user_input, state)

        if not drug_info["drug_name"]:
            return "请提供药品名称，例如：我吃了布洛芬"

        try:
            out = await self.drug_record_tool.add_record(
                user_id=user_id,
                drug_name=drug_info["drug_name"],
                dosage=drug_info.get("dosage", ""),
                frequency=drug_info.get("frequency", ""),
                time_text=drug_info.get("time", ""),
            )
            if not out.get("ok"):
                return f"保存用药记录失败：{out.get('message', '未知错误')}"
            if not out.get("created", False):
                return out.get("message", "用药记录已存在，无需重复添加。")
            return (
                f"已为您记录用药信息：{drug_info['drug_name']}，{drug_info.get('frequency','未指定')}，{drug_info.get('dosage','未指定')}。"
                "如需修改或添加其他用药信息，请随时告诉我。"
            )
        except Exception as e:
            return f"保存用药记录时出现错误：{str(e)}"

    async def _handle_query_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        try:
            records = await self.drug_record_tool.list_recent(user_id=user_id, limit=10)
            if not records:
                return "您最近7天内没有用药记录。"

            lines = ["您最近的用药记录如下："]
            for r in records:
                lines.append(
                    f"- {r.get('drug_name')} | 频次：{r.get('frequency') or '未指定'} | 剂量：{r.get('dosage') or '未指定'} | 时间：{r.get('start_date') or '未记录'}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"查询用药记录时出现错误：{str(e)}"

    async def _handle_delete_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        drugs = DrugEntityExtractor.extract_drug_candidates(user_input, max_items=3)
        if not drugs:
            return "请告诉我需要删除哪种药的记录，例如：删除我最近一条布洛芬记录。"

        out = await self.drug_record_tool.soft_delete_latest_by_name(user_id=user_id, drug_name=drugs[0])
        if out.get("ok"):
            return out.get("message", "已删除记录。")
        return out.get("message", "删除失败，请稍后重试。")

    async def _handle_general_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
用户进行用药记录相关操作：
用户输入：{user_input}

请提供用药记录管理的相关帮助信息。
"""

        response = await self._call_llm(user_prompt, system_prompt, state)
        return response

    async def _parse_drug_info(self, user_input: str, state: dict) -> dict:
        llm_result = await self.llm_decision.extract_drug_info(
            user_input, history=state.get("history", [])
        )
        if llm_result and llm_result.get("drug_name"):
            from datetime import datetime
            drug_info = {
                "drug_name": llm_result["drug_name"],
                "dosage": llm_result.get("dosage", ""),
                "frequency": llm_result.get("frequency", ""),
                "time": llm_result.get("start_date_text", ""),
            }
            if not drug_info["time"]:
                drug_info["time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return drug_info

        import re
        candidates = re.findall(r"(阿司匹林|布洛芬|感康|对乙酰氨基酚|头孢|青霉素|奥美拉唑|氯雷他定|二甲双胍|缬沙坦)", user_input)
        rule_drug_name = candidates[0] if candidates else ""

        memory_messages = state.get('history', [])
        ctx_msgs = memory_messages[-6:] if len(memory_messages) > 6 else memory_messages
        system_prompt = "你是一个医疗信息抽取助手。请从用户的最新回复和上下文中，提取用药记录信息。以JSON格式返回，包含以下字段：1. drug_name 药品名称，如果是补充信息且未提及药名，请从上下文中找到药名并填入。如果仍然找不到，填空字符串。 2. dosage 剂量，如'100mg'，'1片'。3. frequency 频率，如'每天一次'，'早晚各一次'。4. time 用药时间，如'昨天晚上八点'、'今天中午'，不要用现在的系统时间。如果字段没有提及并没有在上下文中，请填空字符串。必须且只输出合法的 JSON，不要输出 Markdown 标记，也不要有任何其他解释内容。"
        user_prompt = f"对话上下文：\n{ctx_msgs}\n\n用户最新输入：{user_input}"
        try:
            import json
            from datetime import datetime
            response = await self._call_llm(user_prompt, system_prompt, state)
            match = re.search(r'\{.*\}', response, re.DOTALL)
            json_str = match.group(0) if match else response
            data = json.loads(json_str)
            drug_info = {
                'drug_name': data.get('drug_name', '') or rule_drug_name,
                'dosage': data.get('dosage', ''),
                'frequency': data.get('frequency', ''),
                'time': data.get('time', '')
            }
            if not drug_info['time']:
                 drug_info['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return drug_info
        except Exception:
            from datetime import datetime
            return {'drug_name': rule_drug_name, 'dosage': '', 'frequency': '', 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    def get_system_prompt(self) -> str:
        return Prompts.get_prompt("DRUG_RECORD_AGENT")

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.agent_name,
            description="用药记录增删查能力",
            capabilities=["drug_record_add", "drug_record_query", "drug_record_delete"],
            keywords=["用药记录", "添加用药", "删除记录", "服用"],
            visible_state_keys=["memory_summary", "history_text", "shared_facts", "long_memory_items"],
            priority=2,
        )
