from __future__ import annotations

import uuid
from typing import Dict, Any

from app.common.logger import get_logger
from app.core.agent.agent_card import AgentCard
from app.core.agent.base_agent import BaseAgent
from app.core.agent.llm_decision_service import LLMDecisionService
from app.core.skills.drug_write_confirmation import (
    PENDING_TYPE,
    build_card_message,
    build_options,
    preferred_drug_name,
)
from app.core.tools.drug_entity_extractor import DrugEntityExtractor
from app.core.tools.drug_record_tool import DrugRecordTool
from app.core.prompts import Prompts

logger = get_logger(__name__)


class DrugRecordAgent(BaseAgent):
    """用药记录Agent"""

    def __init__(self):
        super().__init__("drug_record_agent")
        self.drug_record_tool = DrugRecordTool()
        self.llm_decision = LLMDecisionService()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")

        # 用户正在回答上一轮的写操作确认卡：intent_node 已用规则解析好，直接执行，
        # 不再重新判操作类型（"A"/"不记录" 这类回答本身判不出 add/update/delete）。
        resolution = state.get("confirmation_resolution") or {}
        if resolution:
            result = await self._handle_confirmation_answer(user_id, state, resolution)
        else:
            operation_type = await self._parse_operation_type(user_input, state)

            if operation_type == "add":
                result = await self._handle_add_operation(user_id, user_input, state)
            elif operation_type == "update":
                result = await self._handle_update_operation(user_id, user_input, state)
            elif operation_type == "query":
                result = await self._handle_query_operation(user_id, user_input, state)
            elif operation_type == "delete":
                result = await self._handle_delete_operation(user_id, user_input, state)
            else:
                result = await self._handle_general_operation(user_id, user_input, state)

        # 确认卡路径返回 None：本轮不产出回答文本，交给 llm_generate 的 confirmation 分支渲染。
        # 这里**绝不能**写成 state["final_response"] = ""——空串会让 memory_update
        # 往对话历史里塞一条空的助手消息，污染下一轮的决策上下文。
        if result is not None:
            state["final_response"] = result

        self._log_agent_call(user_id, self.agent_name, user_input, result or "<confirmation_pending>")

        return state

    # ---- 写操作二次确认 --------------------------------------------------

    @staticmethod
    def _raw_user_words(state: Dict[str, Any], fallback: str) -> str:
        """用户原话。

        `sub_state["user_input"]` 是 planner 给的步骤 query，可能被改写或加上
        "[背景信息：…]" 前缀；而歧义检测（多药同句、别名命中）必须看用户**原话**，
        否则改写后的文本会凭空多出或少掉药名。`execute_node` 在建子状态前已经存好
        `original_user_input`，这里优先用它。
        """
        return state.get("original_user_input") or fallback

    def _raise_confirmation(
        self,
        *,
        state: Dict[str, Any],
        operation: str,
        draft: dict,
        raw_input: str,
    ) -> bool:
        """写操作门控：需要用户确认时挂起 pending 并返回 True（调用方**不得落库**）。

        返回 False 表示无歧义且字段齐全，调用方按原逻辑直接落库。
        """
        raw_input = self._raw_user_words(state, raw_input)
        options = build_options(draft=draft, raw_input=raw_input, operation=operation)
        if not options:
            return False

        state["needs_confirmation"] = True
        state["confirmation_message"] = build_card_message(
            draft=draft, options=options, operation=operation
        )
        state["pending_confirmation"] = {
            "id": uuid.uuid4().hex[:12],
            "type": PENDING_TYPE,
            "operation": operation,
            "draft": draft,
            "options": options,
            "attempts": 0,
            "raw_user_input": raw_input,
        }
        logger.info(
            "drug write requires confirmation: op=%s options=%s",
            operation,
            [o["label"] for o in options],
        )
        return True

    async def _handle_confirmation_answer(
        self, user_id: str, state: Dict[str, Any], resolution: dict
    ) -> str:
        """执行用户对确认卡的回答。无论确认还是取消，都必须清掉 pending。"""
        pending = resolution.get("pending") or {}
        draft = pending.get("draft") or {}
        operation = pending.get("operation") or "add"
        action = resolution.get("action")
        payload = resolution.get("payload") or {}

        # 清空 pending：假值会触发 memory_update 的 pop 分支，无需额外接口
        state["pending_confirmation"] = {}
        state["needs_confirmation"] = False
        state["confirmation_message"] = ""

        if action == "deny":
            return "好的，这次不写入用药档案。需要时随时告诉我。"

        # select 带具体药名时以用户的更正为准，其余情况沿用 draft
        draft = dict(draft)
        if payload.get("kind") == "drug" and payload.get("value"):
            draft["drug_name"] = payload["value"]

        drug_name = (draft.get("drug_name") or "").strip()
        if not drug_name:
            return "没有拿到药品名称，请重新告诉我，例如：我吃了布洛芬。"

        try:
            if operation == "delete":
                out = await self.drug_record_tool.soft_delete_latest_by_name(
                    user_id=user_id, drug_name=drug_name
                )
                return out.get("message", "已删除记录。") if out.get("ok") else f"删除失败：{out.get('message', '未知错误')}"

            if operation == "update":
                out = await self.drug_record_tool.update_by_name(
                    user_id=user_id,
                    drug_name=drug_name,
                    dosage=draft.get("dosage", ""),
                    frequency=draft.get("frequency", ""),
                    time_text=draft.get("time", ""),
                )
                return out.get("message", "已更新。") if out.get("ok") else f"更新失败：{out.get('message', '未知错误')}"

            out = await self.drug_record_tool.add_record(
                user_id=user_id,
                drug_name=drug_name,
                dosage=draft.get("dosage", ""),
                frequency=draft.get("frequency", ""),
                time_text=draft.get("time", ""),
            )
            if not out.get("ok"):
                return f"保存用药记录失败：{out.get('message', '未知错误')}"
            if not out.get("created", False):
                return out.get("message", "用药记录已存在，无需重复添加。")
            return f"已为您记录用药信息：{drug_name}。"
        except Exception as e:
            return f"保存用药记录时出现错误：{str(e)}"

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
        update_keywords = ["补一下", "补充", "更新", "改成", "改为", "改一下", "修改", "加上", "补上"]

        if any(k in t for k in delete_keywords):
            return "delete"
        # update 优先于 add："改成/补一下/更新" 是修改已有记录，而非记录新药
        if any(k in t for k in update_keywords):
            return "update"
        if any(k in t for k in query_keywords):
            return "query"
        if any(k in t for k in add_keywords):
            return "add"

        return "general"

    async def _handle_add_operation(
        self, user_id: str, user_input: str, state: Dict[str, Any]
    ) -> str | None:
        drug_info = await self._parse_drug_info(user_input, state)

        if not drug_info["drug_name"]:
            return "请提供药品名称，例如：我吃了布洛芬"

        # 门控：药名有歧义 / 字段缺失时**不落库**，先让用户选（返回 None 表示本轮不产出文本）
        if self._raise_confirmation(
            state=state, operation="add", draft=drug_info, raw_input=user_input
        ):
            return None

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
            # 只在有值时才展示字段，避免"布洛芬，，"的空段标点
            info = []
            if drug_info.get("dosage"):
                info.append(f"剂量：{drug_info['dosage']}")
            if drug_info.get("frequency"):
                info.append(f"频次：{drug_info['frequency']}")
            body = f"已为您记录用药信息：{drug_info['drug_name']}"
            if info:
                body += "（" + "、".join(info) + "）"
            body += "。"

            # 缺省字段提醒：补不全也能记录，但告知用户可后续补充
            missing = []
            if not drug_info.get("dosage"):
                missing.append("剂量")
            if not drug_info.get("frequency"):
                missing.append("频次")
            if missing:
                body += f"（注：{'、'.join(missing)}未指定，需要补充可随时告诉我，例如'布洛芬每天两次'。）"
            return body
        except Exception as e:
            return f"保存用药记录时出现错误：{str(e)}"

    async def _handle_query_operation(self, user_id: str, user_input: str, state: Dict[str, Any]) -> str:
        try:
            records = await self.drug_record_tool.list_recent(user_id=user_id, limit=10)
            if not records:
                return "您最近7天内没有用药记录。"

            lines = ["您最近的用药记录如下："]
            for r in records:
                # start_time 优先（含时刻），其次 start_date
                time_display = r.get("start_time") or r.get("start_date") or "未记录"
                lines.append(
                    f"- {r.get('drug_name')} | 频次：{r.get('frequency') or '未指定'} | 剂量：{r.get('dosage') or '未指定'} | 时间：{time_display}"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"查询用药记录时出现错误：{str(e)}"

    async def _handle_delete_operation(
        self, user_id: str, user_input: str, state: Dict[str, Any]
    ) -> str | None:
        drugs = DrugEntityExtractor.extract_drug_candidates(user_input, max_items=3)
        if not drugs:
            return "请告诉我需要删除哪种药的记录，例如：删除我最近一条布洛芬记录。"

        # 规则抽取器会把"删除布洛芬的记录"整句当候选名（见 preferred_drug_name 的说明），
        # 词典能解析出药名时以词典为准，否则按名字去删必然落空。
        drug_name = preferred_drug_name(
            raw_input=self._raw_user_words(state, user_input), fallback=drugs[0]
        )

        # 门控：一次只能删一条，多候选时必须让用户指定——旧实现直接取 drugs[0] 会删错药。
        # 删除不可逆（仅软删），这里宁可多问一句。
        if self._raise_confirmation(
            state=state,
            operation="delete",
            draft={"drug_name": drug_name},
            raw_input=user_input,
        ):
            return None

        out = await self.drug_record_tool.soft_delete_latest_by_name(user_id=user_id, drug_name=drug_name)
        if out.get("ok"):
            return out.get("message", "已删除记录。")
        return out.get("message", "删除失败，请稍后重试。")

    # 用户输入里出现这些词才算"提到了时间"，否则不更新时间字段
    _TIME_KEYWORDS = ["今天", "昨天", "前天", "凌晨", "早上", "上午", "中午", "下午", "晚上", "半夜", "点", "时", "月", "日"]

    async def _handle_update_operation(
        self, user_id: str, user_input: str, state: Dict[str, Any]
    ) -> str | None:
        drug_info = await self._parse_drug_info(user_input, state)
        drug_name = drug_info.get("drug_name", "")
        if not drug_name:
            return "请告诉我需要更新哪种药的记录，例如：布洛芬的剂量改成100mg"

        # 只更新用户明确提到的字段：_parse_drug_info 会把缺失时间默认成"现在"，这里过滤掉
        time_text = ""
        if any(kw in user_input for kw in self._TIME_KEYWORDS):
            time_text = drug_info.get("time", "")

        # 门控：药名有歧义时先确认。update 不吃"字段缺失"这条（update_by_name 本来就
        # 只改点到的字段），见 drug_write_confirmation.build_options 的 D 分支。
        draft = {
            "drug_name": drug_name,
            "dosage": drug_info.get("dosage", ""),
            "frequency": drug_info.get("frequency", ""),
            "time": time_text,
        }
        if self._raise_confirmation(
            state=state, operation="update", draft=draft, raw_input=user_input
        ):
            return None

        out = await self.drug_record_tool.update_by_name(
            user_id=user_id,
            drug_name=drug_name,
            dosage=drug_info.get("dosage", ""),
            frequency=drug_info.get("frequency", ""),
            time_text=time_text,
        )
        if out.get("ok"):
            return out.get("message", "已更新。")
        return out.get("message", "更新失败，请稍后重试。")

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
        from datetime import datetime
        candidates = re.findall(r"(阿司匹林|布洛芬|感康|对乙酰氨基酚|头孢|青霉素|奥美拉唑|氯雷他定|二甲双胍|缬沙坦)", user_input)
        rule_drug_name = candidates[0] if candidates else ""

        dosage = ""
        freq = ""
        dosage_match = re.search(r"(\d+\.?\d*)\s*(mg|毫克|g|克|ml|毫升|片|粒|胶囊|支|瓶|袋)", user_input)
        if dosage_match:
            dosage = dosage_match.group(0)
        freq_match = re.search(r"(一天|每日|每天)\s*(\d+)\s*次|(\d+)\s*次\s*(一天|每日|每天)", user_input)
        if freq_match:
            freq = freq_match.group(0)

        return {
            "drug_name": rule_drug_name,
            "dosage": dosage,
            "frequency": freq,
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

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
