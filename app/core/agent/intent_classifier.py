from __future__ import annotations

import json
from dataclasses import dataclass

from app.core.llm.llm_service import LLMService
from app.config.settings import settings
from app.core.prompts import Prompts


@dataclass
class IntentResult:
    intent: str
    confidence: float
    reason: str = ""


class IntentClassifier:
    """应用级意图识别。

    设计目标：
    - 优先规则（稳定、低成本）
    - LLM 可选增强（当用户配置了 LLM 时才启用）
    - 输出结构化（intent + confidence + reason）

    注意：意图识别仅做"分类"，不生成任何医疗结论，符合合规边界。
    """

    INTENTS = ("archive", "drug", "lab", "general")

    def __init__(self) -> None:
        self.llm = LLMService()

    @staticmethod
    def _llm_enabled() -> bool:
        def _ok(v: str) -> bool:
            v = (v or '').strip()
            return bool(v) and not (v.startswith('{{') and v.endswith('}}'))

        return _ok(settings.LLM_API_BASE) and _ok(settings.LLM_API_KEY) and _ok(settings.LLM_MODEL_NAME)


    @staticmethod
    def _rule_predict(text: str) -> IntentResult:
        t = (text or "").strip()
        if not t:
            return IntentResult(intent="general", confidence=0.5, reason="empty")

        # archive：档案增删改查/查询我的记录
        archive_keywords = [
            "档案",
            "就诊",
            "病历",
            "处方",
            "用药记录",
            "化验单",
            "体检",
            "报告",
            "历史",
            "我昨天",
            "上次",
            "之前",
        ]

        # drug：相互作用/能否同服/配伍禁忌/两个药
        drug_interaction_keywords = ["相互作用", "一起吃", "同服", "配伍", "冲突", "禁忌", "能不能一起", "同时吃"]
        
        # drug：用药记录添加/当前用药陈述
        drug_record_keywords = ["记录", "吃了", "服用", "用了", "吃", "服用", "使用", "用", "剂量", "频次", "一天", "一次", "每次"]

        # lab：血常规/指标/数值/单位等
        lab_keywords = ["化验", "检验", "指标", "参考范围", "正常值", "血常规", "尿常规", "mmol", "mg/L"]

        if any(k in t for k in lab_keywords):
            return IntentResult(intent="lab", confidence=0.8, reason="rule:lab")

        # 首先检查是否是删除用药记录操作
        delete_keywords = ["删除", "移除", "清空"]
        is_delete = any(k in t for k in delete_keywords)
        if is_delete and "药" in t:
            return IntentResult(intent="drug", confidence=0.8, reason="rule:drug-record-delete")

        # "我昨天吃的什么药/我之前吃了哪些药" 属于档案查询而不是药物相互作用
        if any(k in t for k in archive_keywords) and "药" in t and not any(k in t for k in drug_interaction_keywords):
            return IntentResult(intent="archive", confidence=0.75, reason="rule:archive-drug-history")

        if any(k in t for k in archive_keywords):
            return IntentResult(intent="archive", confidence=0.7, reason="rule:archive")

        # 药物相互作用意图
        if any(k in t for k in drug_interaction_keywords) or ("药" in t and "一起" in t):
            return IntentResult(intent="drug", confidence=0.7, reason="rule:drug-interaction")
        
        # 用药记录添加意图（当前用药陈述）
        # "记录" 是强信号，无需额外要求"药"字
        if "记录" in t and not any(k in t for k in drug_interaction_keywords):
            time_keywords = ["昨天", "上次", "之前", "以前", "曾经", "哪些", "什么药"]
            if any(time_k in t for time_k in time_keywords):
                return IntentResult(intent="archive", confidence=0.7, reason="rule:archive-drug-history-query")
            return IntentResult(intent="drug", confidence=0.8, reason="rule:drug-record-add")

        if any(k in t for k in drug_record_keywords) and "药" in t:
            time_keywords = ["昨天", "上次", "之前", "以前", "曾经", "过", "哪些", "什么药"]
            if any(time_k in t for time_k in time_keywords):
                return IntentResult(intent="archive", confidence=0.7, reason="rule:archive-drug-history-query")
            else:
                return IntentResult(intent="drug", confidence=0.8, reason="rule:drug-record-add")

        strong_drug_action_keywords = ["吃了", "服用", "用了"]
        if any(k in t for k in strong_drug_action_keywords):
            return IntentResult(intent="drug", confidence=0.75, reason="rule:drug-action-without-drug-word")

        # "药"单独出现不再强制判 drug，避免把"我昨天吃的什么药"误判
        if "药" in t:
            return IntentResult(intent="general", confidence=0.55, reason="rule:contains-drug-but-ambiguous")

        return IntentResult(intent="general", confidence=0.6, reason="rule:default")

    async def predict(self, *, text: str, stream: bool = False) -> IntentResult:
        # 1) 规则优先
        rule = self._rule_predict(text)
        if rule.confidence >= 0.75:
            return rule

        # 1.5) 未启用 LLM 时，直接返回规则结果（避免外部请求导致接口变慢）
        if (not settings.INTENT_LLM_ENABLED) or (not self._llm_enabled()):
            return rule

        # 2) LLM 增强：尝试输出结构化 JSON
        system_prompt = Prompts.get_prompt("INTENT_CLASSIFIER")
        user_prompt = ("用户输入：" + (text or "") + "\n" +
                     "请返回 JSON，例如：{\"intent\":\"general\",\"confidence\":0.6,\"reason\":\"...\"}")

        # 说明：意图增强只需要极短输出，强制限制 max_tokens + 超时，避免拖慢主链路
        try:
            raw = await self.llm.chat_completion(
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                timeout_s=float(settings.INTENT_LLM_TIMEOUT_S),
                max_tokens=int(settings.INTENT_LLM_MAX_TOKENS),
            )
            data = json.loads(raw.strip()) if raw else {}
            intent = str(data.get("intent", ""))
            conf = float(data.get("confidence", 0.0) or 0.0)
            reason = str(data.get("reason", ""))
            if intent not in self.INTENTS:
                return rule
            # 取规则与LLM中更“确定”的那个
            if conf >= rule.confidence:
                return IntentResult(intent=intent, confidence=max(min(conf, 1.0), 0.0), reason=reason)
            return rule
        except Exception:
            # LLM 未配置/调用失败/JSON 解析失败：回退规则
            return rule


