"""
药物冲突查询Agent
专门处理药物相互作用查询和冲突检测
"""

from typing import Dict, Any
import json

from app.core.agent.base_agent import BaseAgent
from app.core.tools.drug_interaction_tool import DrugInteractionTool
from app.core.prompts import Prompts


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
        drug_names = await self._extract_drug_names(user_input)
        
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
    
    async def _extract_drug_names(self, user_input: str) -> list:
        """从用户输入中提取药品名称"""
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
            
            # 处理"和"、"与"、"或"等连接词
            if not candidate_names:
                # 简单的分割提取
                words = user_input.replace("、", ",").replace("和", ",").replace("与", ",").split(",")
                candidate_names = [word.strip() for word in words if len(word.strip()) > 1]
            
            # 使用药品知识库进行匹配
            if candidate_names:
                svc = DrugKnowledgeService()
                matched_drugs = await svc.match_drugs(candidate_names)
                
                # 收集匹配成功的药品名
                found_drugs = []
                for result in matched_drugs:
                    if result.get("match"):
                        found_drugs.append(result["match"]["drug_name"])
                
                # 如果没有匹配到，使用候选名称
                if not found_drugs:
                    found_drugs = candidate_names[:5]  # 限制返回数量
            else:
                found_drugs = []
        except Exception as e:
            # 如果DrugKnowledgeService失败，使用简单的关键词提取
            drug_keywords = ["布洛芬", "阿司匹林", "青霉素", "头孢", "降压药", "降糖药", 
                            "抗生素", "止痛药", "感冒药", "消炎药", "消食片"]
            
            found_drugs = []
            for drug in drug_keywords:
                if drug in user_input:
                    found_drugs.append(drug)
        
        return found_drugs
    
    def get_system_prompt(self) -> str:
        return Prompts.get_prompt("DRUG_CONFLICT_AGENT")