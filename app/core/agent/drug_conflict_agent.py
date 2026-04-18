"""
药物冲突查询Agent
专门处理药物相互作用查询和冲突检测
"""

from typing import Dict, Any
import json

from app.core.agent.base_agent import BaseAgent
from app.core.tools.drug_interaction_tool import DrugInteractionTool


class DrugConflictAgent(BaseAgent):
    """药物冲突查询Agent"""
    
    def __init__(self):
        super().__init__("drug_conflict_agent")
        self.drug_tool = DrugInteractionTool()
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理药物冲突查询请求"""
        user_id = state.get("user_id", "")
        user_input = state.get("user_input", "")
        
        # 提取药品名称
        drug_names = self._extract_drug_names(user_input)
        
        if not drug_names:
            state["final_response"] = self._add_disclaimer(
                "未识别到有效的药品名称。请提供药品的通用名，例如：布洛芬、阿司匹林等。"
            )
            return state
        
        # 调用药物冲突检测工具
        try:
            tool_result = await self.drug_tool.check_interactions(
                user_id=user_id,
                drug_name_list=drug_names,
                sync_to_archive=False,
            )
        except Exception as e:
            state["final_response"] = self._add_disclaimer(
                f"药物冲突查询服务暂时不可用，请稍后重试。错误信息：{str(e)}"
            )
            return state
        
        # 生成专业化的药物冲突解读
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
用户查询药物冲突信息：
查询内容：{user_input}
识别出的药品：{drug_names}

检测结果：
{json.dumps(tool_result, ensure_ascii=False, indent=2)}

请基于检测结果生成专业的药物冲突科普解读，要求：
1. 清晰说明药物之间的相互作用类型
2. 提供通用的安全用药科普知识  
3. 不涉及具体剂量调整或用药指导
4. 语言通俗易懂，适合普通用户理解
"""
        
        response = await self._call_llm(user_prompt, system_prompt, state)
        
        # 合规校验
        ok, msg = self._check_compliance(response)
        if not ok:
            state["error_msg"] = msg
            state["final_response"] = self._add_disclaimer(
                "抱歉，由于合规要求，无法提供相关药物信息。"
            )
            return state
        
        state["final_response"] = self._add_disclaimer(response)
        
        # 记录Agent调用
        self._log_agent_call(user_id, self.agent_name, user_input, response)
        
        return state
    
    def _extract_drug_names(self, user_input: str) -> list:
        """从用户输入中提取药品名称"""
        # 简单的关键词提取，实际项目中可以使用更复杂的NLP方法
        drug_keywords = ["布洛芬", "阿司匹林", "青霉素", "头孢", "降压药", "降糖药", 
                        "抗生素", "止痛药", "感冒药", "消炎药"]
        
        found_drugs = []
        for drug in drug_keywords:
            if drug in user_input:
                found_drugs.append(drug)
        
        # 如果没有找到已知药品，返回用户输入中的关键词
        if not found_drugs:
            # 简单的分割提取
            words = user_input.replace("、", ",").replace("和", ",").split(",")
            found_drugs = [word.strip() for word in words if len(word.strip()) > 1]
            
            # 限制返回数量
            found_drugs = found_drugs[:5]
        
        return found_drugs
    
    def get_system_prompt(self) -> str:
        return """
你是专业的药物冲突查询助手，专门处理药物相互作用和配伍禁忌查询。

你的职责：
1. 基于药品知识库提供准确的药物冲突信息
2. 生成专业、易懂的药物相互作用科普解读
3. 严格遵守医疗合规要求，不提供用药建议

核心原则：
1. 所有药物冲突信息必须基于权威药品知识库
2. 不得生成任何无来源的药物相互作用结论
3. 不得提供具体的用药剂量、用药时间等指导性建议
4. 所有输出必须包含标准免责声明

输出要求：
1. 清晰说明药物之间的相互作用类型（如：增强药效、增加副作用风险等）
2. 提供通用的安全用药科普知识
3. 语言通俗易懂，适合普通用户理解
4. 强调咨询执业药师或医生的必要性

示例输出格式：
根据药品知识库信息，[药品A]和[药品B]之间存在[相互作用类型]。

通用安全提示：[通用科普内容]

重要提醒：具体用药请咨询执业医师或药师，遵医嘱用药。
"""