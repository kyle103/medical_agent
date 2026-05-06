from __future__ import annotations

import re

from app.common.logger import get_logger
from app.core.tools.drug_entity_extractor import DrugEntityExtractor
from app.core.tools.drug_interaction_tool import DrugInteractionTool
from app.core.tools.lab_report_tool import LabReportTool

logger = get_logger(__name__)


class ToolExecutor:
    """统一工具执行器：根据工具名称和状态调用对应工具。

    将原 DrugConflictAgent / LabReportAgent 的核心逻辑下沉为 Tool 直接调用，
    LLM 格式化由 Workflow 的 llm_generate 节点统一处理。
    """

    async def execute(self, tool_name: str, state: dict) -> dict:
        dispatch = {
            "drug_interaction": self._execute_drug_interaction,
            "lab_report": self._execute_lab_report,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            logger.error("ToolExecutor unknown tool=%s", tool_name)
            return {"error_msg": f"未知工具: {tool_name}", "final_desc": ""}

        try:
            result = await handler(state)
            logger.info("ToolExecutor tool=%s success", tool_name)
            return result
        except Exception as e:
            logger.error("ToolExecutor tool=%s failed: %s", tool_name, e)
            return {"error_msg": str(e), "final_desc": ""}

    async def _execute_drug_interaction(self, state: dict) -> dict:
        tool = DrugInteractionTool()
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")

        entities = state.get("extract_entities") or {}
        drug_names = entities.get("drug_name_list") if isinstance(entities, dict) else []
        if not drug_names:
            drug_names = DrugEntityExtractor.extract_drug_candidates(user_input, max_items=10)

        if not drug_names:
            return {
                "tool_result": {
                    "drug_list": [],
                    "interaction_result": [],
                    "final_desc": "未识别到有效的药品名称。请提供药品的通用名，例如：布洛芬、阿司匹林等。",
                },
                "intent_type": "drug_conflict",
            }

        tool_result = await tool.check_interactions(
            user_id=user_id,
            drug_name_list=drug_names,
            sync_to_archive=False,
        )
        return {
            "tool_result": tool_result,
            "extract_entities": {"drug_name_list": drug_names},
            "intent_type": "drug_conflict",
        }

    async def _execute_lab_report(self, state: dict) -> dict:
        tool = LabReportTool()
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")

        lab_items = self._extract_lab_items(user_input)
        if not lab_items:
            return {
                "tool_result": {
                    "item_list": [],
                    "final_desc": "未识别到有效的检验指标。请提供指标名称和数值，例如：血糖6.5，血压120/80。",
                },
                "intent_type": "lab_report",
            }

        tool_result = await tool.interpret(
            user_id=user_id,
            lab_item_list=lab_items,
            sync_to_archive=False,
        )
        return {
            "tool_result": tool_result,
            "intent_type": "lab_report",
        }

    def _extract_lab_items(self, user_input: str) -> list[dict]:
        lab_items: list[dict] = []
        common_items = [
            "血糖", "血压", "血脂", "胆固醇", "肝功能", "肾功能",
            "白细胞", "红细胞", "血小板", "尿酸", "转氨酶",
        ]

        lines = user_input.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            separators = [":", "：", "=", "是", "为"]
            for sep in separators:
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        item_name = parts[0].strip()
                        test_value = parts[1].strip()
                        if any(keyword in item_name for keyword in common_items):
                            lab_items.append({
                                "item_name": item_name,
                                "test_value": test_value,
                                "unit": self._infer_unit(test_value),
                            })
                        break

        if not lab_items:
            for item in common_items:
                if item in user_input:
                    numbers = re.findall(r'\d+\.?\d*', user_input)
                    if numbers:
                        lab_items.append({
                            "item_name": item,
                            "test_value": numbers[0],
                            "unit": self._infer_unit(numbers[0]),
                        })

        return lab_items

    @staticmethod
    def _infer_unit(test_value: str) -> str:
        if "/" in test_value:
            return "mmHg"
        try:
            v = float(test_value)
            if "." in test_value and v < 20:
                return "mmol/L"
        except (ValueError, TypeError):
            pass
        return ""
