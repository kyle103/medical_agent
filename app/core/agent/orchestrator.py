from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List


SingleRunner = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
IntentPredictor = Callable[[str], Awaitable[dict | None]]


@dataclass
class SubTaskResult:
    query: str
    content: str
    intent_type: str


class QueryOrchestrator:
    """最小中心编排器：负责多问题拆分、子任务执行与聚合。"""

    def __init__(self, *, split_fn: Callable[[str], List[str]], run_single: SingleRunner, predict_intent: IntentPredictor):
        self.split_fn = split_fn
        self.run_single = run_single
        self.predict_intent = predict_intent

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        sub_queries = self.split_fn(user_input)
        if len(sub_queries) <= 1:
            return await self.run_single(state)

        merged_state = state
        sections: List[SubTaskResult] = []

        for q in sub_queries:
            sub_state = dict(state)
            sub_state["user_input"] = q
            sub_state.pop("final_response", None)
            sub_state.pop("error_msg", None)
            sub_state.pop("target_agent", None)
            sub_state.pop("intent_analysis", None)
            sub_state.pop("extract_entities", None)

            pred = await self.predict_intent(q)
            if pred:
                sub_state["intent"] = pred.get("intent")
                sub_state["intent_confidence"] = pred.get("confidence")
                sub_state["intent_reason"] = pred.get("reason")

            sub_result = await self.run_single(sub_state)
            merged_state.update(sub_result)

            content = (sub_result.get("final_response") or sub_result.get("confirmation_message") or "").strip()
            if content:
                sections.append(
                    SubTaskResult(
                        query=q,
                        content=content,
                        intent_type=str(sub_result.get("intent_type") or sub_result.get("intent") or "general"),
                    )
                )

        if sections:
            merged_state["final_response"] = "我分条为你处理如下：\n\n" + "\n\n".join(
                [f"{i}. {s.query}\n{s.content}" for i, s in enumerate(sections, start=1)]
            )
            merged_state["intent"] = "multi"
            merged_state["intent_type"] = "multi"
            merged_state["intent_analysis"] = {
                "target_agent": "multi",
                "confidence": 0.9,
                "reason": "multi-intent decomposition",
                "intent_type": "multi",
                "needs_confirmation": False,
                "sub_intents": [s.intent_type for s in sections],
            }
        return merged_state

