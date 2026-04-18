"""
化验单解读Agent
专门处理化验单指标解读和检验结果分析
"""

from typing import Dict, Any
import json

from app.core.agent.base_agent import BaseAgent
from app.core.tools.lab_report_tool import LabReportTool


class LabReportAgent(BaseAgent):
    """化验单解读Agent"""
    
    def __init__(self):
        super().__init__("lab_report_agent")
        self.lab_tool = LabReportTool()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理化验单解读请求"""
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")
        
        # 提取检验指标
        lab_items = self._extract_lab_items(user_input)
        
        if not lab_items:
            state["final_response"] = self._add_disclaimer(
                "未识别到有效的检验指标。请提供指标名称和数值，例如：血糖6.5，血压120/80。"
            )
            return state
        
        # 调用化验单解读工具
        try:
            tool_result = await self.lab_tool.interpret(
                user_id=user_id,
                lab_item_list=lab_items,
                sync_to_archive=False,
            )
        except Exception as e:
            state["final_response"] = self._add_disclaimer(
                f"化验单解读服务暂时不可用，请稍后重试。错误信息：{str(e)}"
            )
            return state
        
        # 生成专业化的化验单解读
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
用户请求化验单解读：
用户输入：{user_input}
识别出的检验指标：{lab_items}

解读结果：
{json.dumps(tool_result, ensure_ascii=False, indent=2)}

请基于解读结果生成专业的化验单科普解读，要求：
1. 清晰说明各项指标的异常情况
2. 提供通用的临床意义科普知识
3. 不涉及疾病诊断或治疗建议
4. 语言通俗易懂，适合普通用户理解
"""
        
        response = await self._call_llm(user_prompt, system_prompt, state)
        
        # 合规校验
        ok, msg = self._check_compliance(response)
        if not ok:
            state["error_msg"] = msg
            state["final_response"] = self._add_disclaimer(
                "抱歉，由于合规要求，无法提供相关检验指标解读。"
            )
            return state
        
        state["final_response"] = self._add_disclaimer(response)
        
        # 记录Agent调用
        self._log_agent_call(user_id, self.agent_name, user_input, response)
        
        return state
    
    def _extract_lab_items(self, user_input: str) -> list:
        """从用户输入中提取检验指标"""
        lab_items = []
        
        # 常见检验指标关键词
        common_items = ["血糖", "血压", "血脂", "胆固醇", "肝功能", "肾功能", 
                       "白细胞", "红细胞", "血小板", "尿酸", "转氨酶"]
        
        # 简单的数值提取
        lines = user_input.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 尝试多种分隔符
            separators = [":", "：", "=", "是", "为"]
            for sep in separators:
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        item_name = parts[0].strip()
                        test_value = parts[1].strip()
                        
                        # 检查是否是常见指标
                        if any(keyword in item_name for keyword in common_items):
                            lab_items.append({
                                "item_name": item_name,
                                "test_value": test_value,
                                "unit": self._infer_unit(test_value)
                            })
                        break
        
        # 如果没有提取到结构化数据，尝试关键词匹配
        if not lab_items:
            for item in common_items:
                if item in user_input:
                    # 尝试提取数值
                    import re
                    # 匹配数字（包括小数）
                    numbers = re.findall(r'\d+\.?\d*', user_input)
                    if numbers:
                        lab_items.append({
                            "item_name": item,
                            "test_value": numbers[0],
                            "unit": self._infer_unit(numbers[0])
                        })
        
        return lab_items
    
    def _infer_unit(self, test_value: str) -> str:
        """推断单位"""
        # 简单的单位推断
        if "/" in test_value:
            return "mmHg"  # 血压
        elif "." in test_value and float(test_value) < 20:
            return "mmol/L"  # 血糖等
        else:
            return ""
    
    def get_system_prompt(self) -> str:
        return """
你是专业的化验单解读助手，专门处理临床检验指标的解读和分析。

你的职责：
1. 基于检验指标参考库提供准确的指标解读
2. 生成专业、易懂的检验指标科普解读
3. 严格遵守医疗合规要求，不提供疾病诊断

核心原则：
1. 所有指标解读必须基于权威检验参考库
2. 不得生成任何疾病诊断或治疗建议
3. 仅提供通用临床意义科普，不涉及具体病情判断
4. 所有输出必须包含标准免责声明

解读要求：
1. 清晰说明各项指标的异常情况（升高/降低/正常）
2. 提供通用的临床意义科普知识
3. 强调异常指标的通用注意事项
4. 提醒用户咨询执业医师的必要性

输出要求：
1. 按指标逐项说明异常情况和通用意义
2. 语言通俗易懂，适合普通用户理解
3. 避免使用专业术语，必要时进行解释
4. 强调检验结果需要结合临床症状综合判断

示例输出格式：
[指标名称]：[数值] [单位]
- 参考范围：[参考范围]
- 异常情况：[升高/降低/正常]
- 通用意义：[通用科普内容]

重要提醒：检验结果需要结合临床症状由执业医师综合判断。
"""