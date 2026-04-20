"""
智能Agent路由器
基于LLM的意图识别和动态路由，采用多Agent协作架构
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.core.agent.drug_conflict_agent import DrugConflictAgent
from app.core.agent.drug_record_agent import DrugRecordAgent
from app.core.agent.lab_report_agent import LabReportAgent
from app.core.agent.main_qa_agent import MainQAAgent
from app.core.agent.orchestrator import QueryOrchestrator
from app.core.llm.llm_service import LLMService
from app.core.session.agent_state_store import AgentStateStore
from app.core.tools.drug_entity_extractor import DrugEntityExtractor
logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Agent能力描述"""
    name: str
    description: str
    keywords: List[str]
    priority: int = 1


class SmartAgentRouter:
    """智能Agent路由器，基于LLM的意图识别和动态路由"""

    STANDARD_NOTICE = "重要提醒：以上内容仅供健康科普信息参考，不构成诊断或具体用药方案；如需用药或治疗决策，请咨询执业医师或药师。"
    NOTICE_KEYWORDS = [
        "重要提醒",
        "仅供参考",
        "仅供健康科普",
        "不构成",
        "请咨询执业医师",
        "请咨询执业药师",
        "咨询执业医师",
        "咨询执业药师",
        "咨询医生",
        "咨询药师",
        "遵医嘱",
        "不能替代",
        "注：以上信息仅供参考",
    ]
    
    def __init__(self):
        # 初始化各个Agent实例
        self.agents = {
            "drug_conflict_agent": DrugConflictAgent(),
            "drug_record_agent": DrugRecordAgent(),
            "lab_report_agent": LabReportAgent(),
            "main_qa_agent": MainQAAgent(),
        }
        
        # 定义Agent能力描述
        self.capabilities = [
            AgentCapability(
                name="drug_conflict_agent",
                description="处理药物相互作用查询，检查两种或多种药物能否同时服用",
                keywords=["相互作用", "一起吃", "同服", "配伍", "冲突", "禁忌", "能不能一起", "同时吃"],
                priority=2
            ),
            AgentCapability(
                name="drug_record_agent",
                description="处理用药记录操作，包括添加、查询、修改用药记录",
                keywords=["吃了", "服用", "用了", "用药记录", "添加用药", "记录用药", "当前用药"],
                priority=2
            ),
            AgentCapability(
                name="lab_report_agent",
                description="解读化验单指标，提供通用参考范围和健康建议",
                keywords=["化验", "检验", "指标", "参考范围", "正常值", "血常规", "尿常规", "mmol", "mg/L"],
                priority=2
            ),
            AgentCapability(
                name="main_qa_agent",
                description="处理通用健康问答、档案查询、科普知识等",
                keywords=["档案", "记录", "就诊", "病历", "体检", "报告", "健康", "科普"],
                priority=1
            )
        ]
        
        self.llm = LLMService()

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        t = (text or "").strip().lower()
        return any(k in t for k in ["是", "是的", "确认", "确定", "好的", "可以", "同意", "yes", "y", "ok"])

    @staticmethod
    def _is_negative(text: str) -> bool:
        t = (text or "").strip().lower()
        return any(k in t for k in ["不", "取消", "不要", "不用", "否", "no", "n"])

    @classmethod
    def _normalize_notice_once(cls, text: str) -> str:
        """收敛重复的警示/免责声明：正文去重后仅在末尾保留一次标准提醒。"""
        raw = (text or "").strip()
        if not raw:
            return raw

        lines = [ln.rstrip() for ln in raw.splitlines()]
        cleaned_lines: List[str] = []
        notice_found = False

        for line in lines:
            if not line.strip():
                cleaned_lines.append("")
                continue
            if any(kw in line for kw in cls.NOTICE_KEYWORDS):
                notice_found = True
                continue
            cleaned_lines.append(line)

        # 压缩连续空行
        compact: List[str] = []
        prev_blank = False
        for line in cleaned_lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            compact.append(line)
            prev_blank = is_blank

        body = "\n".join(compact).strip()
        if not notice_found:
            return body
        if not body:
            return cls.STANDARD_NOTICE
        return f"{body}\n\n{cls.STANDARD_NOTICE}"

    @staticmethod
    def _split_user_queries(text: str) -> List[str]:
        """将一段包含多个问题/指令的输入拆成子查询。"""
        raw = (text or "").strip()
        if not raw:
            return []

        # 先按中文/英文句末符号切分
        parts = re.split(r"[。！？!?；;]+", raw)
        parts = [p.strip(" ，,") for p in parts if p and p.strip(" ，,")]

        # 如果用户没写句号，但使用了明显的并列动作词，再次细分
        out: List[str] = []
        for p in parts:
            sub = re.split(r"(?=帮我|请帮我|另外|还有|并且|同时|我是否|我有|顺便)", p)
            for s in sub:
                s = s.strip(" ，,")
                if s:
                    out.append(s)

        # 保序去重，防止重复切分造成重复执行
        dedup: List[str] = []
        seen = set()
        for q in out:
            if q in seen:
                continue
            seen.add(q)
            dedup.append(q)
        return dedup

    @staticmethod
    def _extract_drug_candidates(text: str) -> List[str]:
        return DrugEntityExtractor.extract_drug_candidates(text, max_items=10)

    def _route_from_state_intent(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        intent = (state.get("intent") or "").strip().lower()
        text = (state.get("user_input") or "").strip()
        if not intent:
            return None

        if intent == "lab":
            return {
                "target_agent": "lab_report_agent",
                "confidence": float(state.get("intent_confidence") or 0.6),
                "reason": "follow intent_classifier: lab",
                "intent_type": "lab_report",
                "needs_confirmation": False,
            }

        if intent == "archive":
            return {
                "target_agent": "main_qa_agent",
                "confidence": float(state.get("intent_confidence") or 0.6),
                "reason": "follow intent_classifier: archive",
                "intent_type": "archive",
                "needs_confirmation": False,
            }

        if intent == "general":
            return {
                "target_agent": "main_qa_agent",
                "confidence": float(state.get("intent_confidence") or 0.6),
                "reason": "follow intent_classifier: general",
                "intent_type": "general",
                "needs_confirmation": False,
            }

        if intent == "drug":
            conflict_keywords = ["相互作用", "一起吃", "同服", "配伍", "冲突", "禁忌", "能不能一起", "可以一起"]
            record_keywords = ["记录", "添加", "我吃了", "我服用", "我用了", "用药记录", "剂量", "频次", "每天", "每次"]
            delete_keywords = ["删除", "移除", "清空"]

            is_conflict = any(k in text for k in conflict_keywords) or ("药" in text and "一起" in text)
            is_record = any(k in text for k in record_keywords)
            is_delete = any(k in text for k in delete_keywords)

            # 优先把“两个药能不能一起”这类问句路由到冲突 Agent
            if is_conflict and not is_record and not is_delete:
                return {
                    "target_agent": "drug_conflict_agent",
                    "confidence": float(state.get("intent_confidence") or 0.7),
                    "reason": "drug sub-intent: interaction check",
                    "intent_type": "drug_conflict",
                    "needs_confirmation": False,
                }

            if (is_record or is_delete) and not is_conflict:
                return {
                    "target_agent": "drug_record_agent",
                    "confidence": float(state.get("intent_confidence") or 0.7),
                    "reason": "drug sub-intent: record operation",
                    "intent_type": "drug_record",
                    "needs_confirmation": False,
                }

            # 两者都不明显时，用候选药名数量辅助判断
            candidates = self._extract_drug_candidates(text)
            if len(candidates) >= 2:
                state.setdefault("extract_entities", {})["drug_name_list"] = candidates[:6]
                return {
                    "target_agent": "drug_conflict_agent",
                    "confidence": 0.7,
                    "reason": "drug sub-intent by multi-drug candidates",
                    "intent_type": "drug_conflict",
                    "needs_confirmation": False,
                }

            return {
                "target_agent": "drug_record_agent",
                "confidence": 0.65,
                "reason": "drug sub-intent default to record",
                "intent_type": "drug_record",
                "needs_confirmation": False,
            }

        return None

    async def _route_and_execute_single(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        conversation_history = state.get("history", [])

        # 1) 优先复用上游 intent_classifier 结果，稳定主链路路由
        intent_result = self._route_from_state_intent(state)

        # 2) 仅在上游结果缺失/不确定时才调用 LLM 做辅助路由
        if not intent_result:
            intent_result = await self.analyze_intent_with_llm(user_input, conversation_history)
        elif float(intent_result.get("confidence", 0.0)) < 0.6:
            llm_intent = await self.analyze_intent_with_llm(user_input, conversation_history)
            if float(llm_intent.get("confidence", 0.0)) >= float(intent_result.get("confidence", 0.0)):
                intent_result = llm_intent

        # 记录意图分析结果
        state["intent_analysis"] = intent_result
        state["target_agent"] = intent_result["target_agent"]
        state["intent_type"] = intent_result["intent_type"]

        logger.info(f"意图分析结果: {intent_result}")

        # 3. 验证目标Agent是否存在
        target_agent = intent_result["target_agent"]
        if target_agent not in self.agents:
            logger.warning(f"未知的Agent: {target_agent}，回退到main_qa_agent")
            target_agent = "main_qa_agent"
            state["target_agent"] = target_agent

        # 4. 如果需要确认，先返回确认信息
        if intent_result.get("needs_confirmation", False):
            state["needs_confirmation"] = True
            state["confirmation_message"] = self._build_confirmation_message(intent_result, user_input)
            state["pending_confirmation"] = {
                "type": "agent_confirmation",
                "payload": {
                    "target_agent": state.get("target_agent"),
                    "original_query": user_input,
                    "intent_type": intent_result.get("intent_type", "general"),
                },
                "expires_at": int(time.time()) + 15 * 60,
            }
            return state

        # 5. 调用目标Agent处理
        try:
            agent = self.agents[target_agent]
            result_state = await agent.process(state)
            state.update(result_state)
        except Exception as e:
            logger.error(f"Agent执行失败: {e}")
            state["error_msg"] = f"处理请求时出现错误: {str(e)}"

            # 尝试回退到主问答Agent
            if target_agent != "main_qa_agent":
                try:
                    fallback_agent = self.agents["main_qa_agent"]
                    fallback_result = await fallback_agent.process(state)
                    state.update(fallback_result)
                    state["fallback_used"] = True
                except Exception as fallback_e:
                    state["error_msg"] = f"主Agent也执行失败: {str(fallback_e)}"

        # 单任务输出也做一次免责声明收敛
        if state.get("final_response"):
            state["final_response"] = self._normalize_notice_once(str(state.get("final_response")))

        return state

    async def _predict_intent_for_query(self, query: str) -> dict | None:
        try:
            from app.core.agent.intent_classifier import IntentClassifier

            clf = IntentClassifier()
            sub_intent = await clf.predict(text=query, stream=False)
            return {
                "intent": sub_intent.intent,
                "confidence": sub_intent.confidence,
                "reason": sub_intent.reason,
            }
        except Exception:
            return None
    
    async def analyze_intent_with_llm(self, user_input: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """使用LLM进行意图分析，返回详细的意图识别结果"""
        
        # 构建系统提示词
        system_prompt = """你是一个医疗助手意图分析专家。请分析用户的输入，判断其意图并选择最合适的处理Agent。

可用的Agent：
- main_qa_agent: 通用健康问答、档案查询、历史记录查询、记忆相关查询
- drug_conflict_agent: 药物相互作用检查
- drug_record_agent: 当前用药记录管理（添加、查询、删除）
- lab_report_agent: 化验单解读

请返回JSON格式：
{
    "target_agent": "agent_name",
    "confidence": 0.9,
    "reason": "分析理由",
    "intent_type": "intent_category",
    "needs_confirmation": true/false
}

intent_type可以是：general, drug_conflict, drug_record, lab_report, archive, memory
needs_confirmation: 当意图不明确或需要用户确认时设为true

特别注意：
1. 如果用户询问"还记得我是谁"、"我的信息"等记忆相关问题时，选择main_qa_agent
2. 如果用户只是简单确认（如"是的是的"、"好的"等），需要结合对话历史判断是否为确认响应
3. 对于用药记录，如果用户明确提到当前用药，需要确认；如果是历史查询，不需要确认
4. 当用户使用"帮我"、"记录"、"我吃了"、"我服用了"等表达时，应识别为drug_record_agent
5. 对于包含多种药品的用药记录请求，也应路由到drug_record_agent"""

        # 构建用户提示词
        history_text = ""
        if conversation_history:
            history_text = "\n对话历史：\n"
            for msg in conversation_history[-5:]:  # 最近5条消息
                role = "用户" if msg.get("role") == "user" else "助手"
                history_text += f"{role}: {msg.get('content', '')}\n"
        
        user_prompt = f"用户输入：{user_input}{history_text}\n\n请分析意图并选择最合适的Agent。"
        
        try:
            response = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=8.0,
                max_tokens=300,
            )
            
            # 解析JSON响应
            if response:
                data = json.loads(response.strip())
                return {
                    "target_agent": data.get("target_agent", "main_qa_agent"),
                    "confidence": float(data.get("confidence", 0.5)),
                    "reason": data.get("reason", "默认选择"),
                    "intent_type": data.get("intent_type", "general"),
                    "needs_confirmation": data.get("needs_confirmation", False)
                }
        except Exception as e:
            logger.warning(f"LLM意图分析失败: {e}")
        
        # 回退到基于规则的分析
        return await self.fallback_intent_analysis(user_input, conversation_history)
    
    async def fallback_intent_analysis(self, user_input: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """基于规则的回退意图分析"""
        
        text = (user_input or "").strip().lower()
        
        # 检查是否为确认响应
        if conversation_history:
            last_assistant_msg = None
            for msg in reversed(conversation_history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg.get("content", "")
                    break
            
            # 如果上一条助手消息是确认消息，且用户输入是简单确认
            if last_assistant_msg and ("确认" in last_assistant_msg or "您想" in last_assistant_msg):
                confirmation_keywords = ["是", "是的", "对", "没错", "确认", "同意", "好的", "ok", "yes", "y"]
                if any(kw in text for kw in confirmation_keywords):
                    # 从历史中提取目标Agent
                    target_agent = "main_qa_agent"
                    if "药物冲突" in last_assistant_msg:
                        target_agent = "drug_conflict_agent"
                    elif "药物记录" in last_assistant_msg:
                        target_agent = "drug_record_agent"
                    elif "化验单" in last_assistant_msg:
                        target_agent = "lab_report_agent"
                    
                    return {
                        "target_agent": target_agent,
                        "confidence": 0.9,
                        "reason": "用户确认了之前的操作意图",
                        "intent_type": "confirmation",
                        "needs_confirmation": False
                    }
        
        # 自我介绍
        if "我是" in text and len(text) < 20:
            return {
                "target_agent": "main_qa_agent",
                "confidence": 0.8,
                "reason": "检测到自我介绍",
                "intent_type": "profile",
                "needs_confirmation": False
            }
        
        # 记忆相关查询
        if any(kw in text for kw in ["还记得", "记忆", "我是谁", "我的信息", "个人信息", "身份"]):
            return {
                "target_agent": "main_qa_agent",
                "confidence": 0.8,
                "reason": "检测到记忆相关查询",
                "intent_type": "memory",
                "needs_confirmation": False
            }
        
        # 药物相互作用
        if any(kw in text for kw in ["相互作用", "一起吃", "同服", "配伍", "冲突", "禁忌", "能不能一起吃", "可以一起吃吗"]):
            return {
                "target_agent": "drug_conflict_agent",
                "confidence": 0.8,
                "reason": "检测到药物相互作用关键词",
                "intent_type": "drug_conflict",
                "needs_confirmation": False
            }
        
        # 用药记录操作
        if any(kw in text for kw in ["吃了", "服用", "用了", "用药记录", "添加用药", "记录用药", "我想记录", "每天", "每次", "mg", "剂量"]):
            # 区分历史查询和当前记录
            if any(time_kw in text for time_kw in ["昨天", "上次", "之前", "以前", "曾经", "还记得"]):
                return {
                    "target_agent": "main_qa_agent",
                    "confidence": 0.7,
                    "reason": "历史用药查询，由主问答Agent处理",
                    "intent_type": "archive",
                    "needs_confirmation": False
                }
            else:
                return {
                    "target_agent": "drug_record_agent",
                    "confidence": 0.8,
                    "reason": "当前用药记录操作",
                    "intent_type": "drug_record",
                    "needs_confirmation": True
                }
        
        # 化验单解读
        if any(kw in text for kw in ["化验", "检验", "指标", "血常规", "尿常规", "mmol", "mg/l"]):
            return {
                "target_agent": "lab_report_agent",
                "confidence": 0.8,
                "reason": "检测到化验单相关关键词",
                "intent_type": "lab_report",
                "needs_confirmation": False
            }
        
        # 档案查询
        if any(kw in text for kw in ["档案", "记录", "就诊", "病历", "体检", "报告"]):
            return {
                "target_agent": "main_qa_agent",
                "confidence": 0.7,
                "reason": "检测到档案查询关键词",
                "intent_type": "archive",
                "needs_confirmation": False
            }
        
        # 默认使用主问答Agent
        return {
            "target_agent": "main_qa_agent",
            "confidence": 0.5,
            "reason": "通用健康问答",
            "intent_type": "general",
            "needs_confirmation": False
        }
    
    async def route_and_execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """智能路由并执行请求"""
        user_input = state.get("user_input", "")
        user_id = state.get("user_id")
        session_id = state.get("session_id")

        # 1) 处理持久化确认态：允许跨轮/重启恢复确认任务
        pending = state.get("pending_confirmation")
        if not pending and user_id and session_id:
            try:
                rt = await AgentStateStore().get_state(user_id=user_id, session_id=session_id)
                pending = rt.get("pending_confirmation") if isinstance(rt, dict) else None
            except Exception:
                pending = None

        if isinstance(pending, dict) and pending:
            expires_at = int(pending.get("expires_at") or 0)
            if expires_at and int(time.time()) > expires_at:
                state["pending_confirmation"] = {}
                pending = None

        if isinstance(pending, dict) and pending:
            payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else pending
            if self._is_affirmative(user_input):
                target = payload.get("target_agent") or "main_qa_agent"
                original_query = payload.get("original_query") or user_input
                resumed = dict(state)
                resumed["user_input"] = original_query
                resumed["target_agent"] = target
                resumed["needs_confirmation"] = False
                resumed.pop("pending_confirmation", None)
                out = await self._route_and_execute_single(resumed)
                out.pop("pending_confirmation", None)
                state["pending_confirmation"] = {}
                return out
            if self._is_negative(user_input):
                state["pending_confirmation"] = {}
                state["final_response"] = "好的，已取消这次待确认操作。"
                state["intent"] = "cancelled"
                state["intent_type"] = "cancelled"
                return state

        # 2) 统一走中心编排器
        orchestrator = QueryOrchestrator(
            split_fn=self._split_user_queries,
            run_single=self._route_and_execute_single,
            predict_intent=self._predict_intent_for_query,
        )
        out = await orchestrator.execute(state)
        if out.get("final_response"):
            out["final_response"] = self._normalize_notice_once(str(out.get("final_response")))
        return out
    
    def _build_confirmation_message(self, intent_result: Dict[str, Any], user_input: str) -> str:
        """构建确认消息"""
        target_agent = intent_result.get("target_agent", "")
        reason = intent_result.get("reason", "")
        
        if target_agent == "drug_conflict_agent":
            return f"您想进行药物冲突检查吗？{reason}"
        elif target_agent == "drug_record_agent":
            return f"您想进行药物记录管理吗？{reason}"
        elif target_agent == "lab_report_agent":
            return f"您想进行化验单解读吗？{reason}"
        else:
            return f"您想进行通用健康问答吗？{reason}"
    
    def _is_confirmation_response(self, user_input: str, state: Dict[str, Any]) -> bool:
        """判断是否为确认响应"""
        # 检查是否处于确认状态
        if not state.get("needs_confirmation", False):
            return False
        
        # 获取对话历史
        conversation_history = state.get("history", [])
        
        # 检查上一条助手消息是否是确认消息
        if conversation_history:
            last_assistant_msg = None
            for msg in reversed(conversation_history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg.get("content", "")
                    break
            
            # 如果上一条助手消息包含确认询问，且用户输入是确认响应
            if last_assistant_msg and ("确认" in last_assistant_msg or "您想" in last_assistant_msg):
                confirmation_keywords = ["是", "是的", "对", "没错", "确认", "同意", "好的", "ok", "yes", "y", "嗯", "行", "可以"]
                
                user_input_lower = user_input.lower().strip()
                
                # 简单的关键词匹配
                for keyword in confirmation_keywords:
                    if keyword in user_input_lower:
                        return True
                
                # 检查否定关键词
                denial_keywords = ["不", "不是", "不对", "取消", "不要", "no", "n"]
                for keyword in denial_keywords:
                    if keyword in user_input_lower:
                        return False
        
        return False
    
    async def _handle_confirmation_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理确认响应"""
        user_input = state.get("user_input", "")
        target_agent = state.get("target_agent", "main_qa_agent")
        
        # 清除确认状态
        state["needs_confirmation"] = False
        state["confirmation_message"] = None
        
        # 如果用户确认，调用目标Agent处理
        if self._is_confirmation_response(user_input, state):
            try:
                agent = self.agents[target_agent]
                result_state = await agent.process(state)
                state.update(result_state)
            except Exception as e:
                logger.error(f"确认后Agent执行失败: {e}")
                state["error_msg"] = f"处理请求时出现错误: {str(e)}"
        else:
            # 用户取消或拒绝确认
            state["final_response"] = "好的，已取消操作。请问您还有其他问题吗？"
            state["intent"] = "cancelled"
        
        return state
    
    def get_available_agents(self) -> List[str]:
        """获取可用的Agent列表"""
        return list(self.agents.keys())
    
    def get_agent_capabilities(self) -> List[Dict[str, Any]]:
        """获取所有Agent的能力描述"""
        return [
            {
                "name": cap.name,
                "description": cap.description,
                "keywords": cap.keywords,
                "priority": cap.priority
            }
            for cap in self.capabilities
        ]
