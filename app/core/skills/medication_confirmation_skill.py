from __future__ import annotations

from typing import Any


class MedicationConfirmationSkill:
    """候选用药事件确认流程（轻量状态机）。"""

    AFFIRMATIVE = {"是", "是的", "确认", "确定", "好", "好的", "y", "yes", "同意", "添加", "保存"}

    def build_confirmation_message(self, candidate_events: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for event in candidate_events:
            drug_name = event.get("drug_name", "未知药品")
            full_text = event.get("full_text", "")
            lines.append(f"我识别到你可能在记录用药：{drug_name}（来自：{full_text}）。是否帮你加入用药档案？（是/否）")
        return "\n\n".join(lines)

    def is_affirmative(self, text: str) -> bool:
        return (text or "").strip().lower() in self.AFFIRMATIVE

