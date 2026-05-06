from __future__ import annotations

import json
from dataclasses import dataclass

from app.common.logger import get_logger
from app.core.agent.llm_decision_service import LLMDecisionService
from app.config.settings import settings
from app.core.llm.llm_service import LLMService
from app.core.prompts import Prompts

logger = get_logger(__name__)


@dataclass
class IntentResult:
    intent: str
    confidence: float
    reason: str = ""


class IntentClassifier:
    """意图识别：LLM 优先 + 正则兜底。

    决策流程：
    1. LLM 决策（带工具/Agent 描述暴露，由 LLM 统一判断意图和路由）
    2. 正则/关键词兜底（LLM 不可用或失败时）
    """

    INTENTS = ("archive", "drug", "lab", "general")

    def __init__(self) -> None:
        self.llm = LLMService()
        self.llm_decision = LLMDecisionService()

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

        archive_keywords = [
            "档案", "就诊", "病历", "处方", "用药记录",
            "化验单", "体检", "报告", "历史", "我昨天", "上次", "之前",
        ]
        drug_interaction_keywords = ["相互作用", "一起吃", "同服", "配伍", "冲突", "禁忌", "能不能一起", "同时吃"]
        drug_record_keywords = ["记录", "吃了", "服用", "用了", "吃", "服用", "使用", "用", "剂量", "频次", "一天", "一次", "每次"]
        lab_keywords = ["化验", "检验", "指标", "参考范围", "正常值", "血常规", "尿常规", "mmol", "mg/L"]
        delete_keywords = ["删除", "移除", "清空"]

        if any(k in t for k in lab_keywords):
            return IntentResult(intent="lab", confidence=0.8, reason="rule:lab")

        is_delete = any(k in t for k in delete_keywords)
        if is_delete and "药" in t:
            return IntentResult(intent="drug", confidence=0.8, reason="rule:drug-record-delete")

        if any(k in t for k in archive_keywords) and "药" in t and not any(k in t for k in drug_interaction_keywords):
            return IntentResult(intent="archive", confidence=0.75, reason="rule:archive-drug-history")

        if any(k in t for k in archive_keywords):
            return IntentResult(intent="archive", confidence=0.7, reason="rule:archive")

        if any(k in t for k in drug_interaction_keywords) or ("药" in t and "一起" in t):
            return IntentResult(intent="drug", confidence=0.7, reason="rule:drug-interaction")

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

        if "药" in t:
            return IntentResult(intent="general", confidence=0.55, reason="rule:contains-drug-but-ambiguous")

        return IntentResult(intent="general", confidence=0.6, reason="rule:default")

    async def predict(self, *, text: str, stream: bool = False) -> IntentResult:
        if not self._llm_enabled():
            return self._rule_predict(text)

        if not settings.INTENT_LLM_ENABLED:
            return self._rule_predict(text)

        llm_result = await self.llm_decision.classify_intent_and_route(text)
        if llm_result and llm_result.get("confidence", 0) >= 0.5:
            return IntentResult(
                intent=llm_result["intent"],
                confidence=llm_result["confidence"],
                reason=llm_result.get("reason", "llm"),
            )

        rule = self._rule_predict(text)

        system_prompt = Prompts.get_prompt("INTENT_CLASSIFIER")
        user_prompt = ("用户输入：" + (text or "") + "\n" +
                       "请返回 JSON，例如：{\"intent\":\"general\",\"confidence\":0.6,\"reason\":\"...\"}")

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
            if conf >= rule.confidence:
                return IntentResult(intent=intent, confidence=max(min(conf, 1.0), 0.0), reason=reason)
            return rule
        except Exception:
            return rule
