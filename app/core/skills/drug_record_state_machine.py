from __future__ import annotations

import time
from enum import Enum
from typing import Any


class DrugRecordPhase(str, Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class DrugRecordStateMachine:
    FIELD_ORDER = ["dosage", "frequency", "start_date_text", "purpose"]
    FIELD_LABELS = {
        "dosage": "剂量",
        "frequency": "频次",
        "start_date_text": "开始时间",
        "purpose": "用途",
    }
    FIELD_QUESTIONS = {
        "dosage": "请问您服用{drug}的剂量是多少？（例如：一次1片，或50毫克）",
        "frequency": "请问您服用{drug}的频率是怎样的？（例如：一天3次，或早晚各一次）",
        "start_date_text": "请问您是从什么时候开始服用{drug}的？（例如：今天，或2024年1月1日）",
        "purpose": "请问您服用{drug}主要是用于治疗什么？（例如：头痛、发烧、高血压等）",
    }
    TIMEOUT_SECONDS = 15 * 60
    MAX_IRRELEVANT_COUNT = 2

    TRANSITIONS = {
        DrugRecordPhase.IDLE: {DrugRecordPhase.COLLECTING},
        DrugRecordPhase.COLLECTING: {
            DrugRecordPhase.CONFIRMING,
            DrugRecordPhase.CANCELLED,
            DrugRecordPhase.EXPIRED,
        },
        DrugRecordPhase.CONFIRMING: {
            DrugRecordPhase.COMMITTED,
            DrugRecordPhase.CANCELLED,
            DrugRecordPhase.EXPIRED,
            DrugRecordPhase.COLLECTING,
        },
        DrugRecordPhase.COMMITTED: set(),
        DrugRecordPhase.CANCELLED: {DrugRecordPhase.IDLE},
        DrugRecordPhase.EXPIRED: {DrugRecordPhase.IDLE},
    }

    def __init__(self, drug_name: str, initial_info: dict | None = None):
        self.drug_name = drug_name
        self.phase = DrugRecordPhase.IDLE
        self.collected_info: dict[str, Any] = dict(initial_info or {})
        self.current_field: str | None = None
        self.started_at: float | None = None
        self.last_active_at: float | None = None
        self.irrelevant_count: int = 0
        self.confirmation_summary: str | None = None

    def transition(self, new_phase: DrugRecordPhase) -> None:
        if new_phase not in self.TRANSITIONS.get(self.phase, set()):
            raise ValueError(f"Invalid transition: {self.phase} -> {new_phase}")
        self.phase = new_phase
        if new_phase == DrugRecordPhase.COLLECTING and self.started_at is None:
            self.started_at = time.time()
        self.last_active_at = time.time()

    def is_expired(self) -> bool:
        if self.last_active_at is None:
            return False
        return time.time() - self.last_active_at > self.TIMEOUT_SECONDS

    def is_active(self) -> bool:
        return self.phase in {DrugRecordPhase.COLLECTING, DrugRecordPhase.CONFIRMING}

    def get_missing_fields(self) -> list[str]:
        missing = []
        for field in self.FIELD_ORDER:
            val = str(self.collected_info.get(field, "")).strip()
            if not val or val == "未指定":
                missing.append(field)
        return missing

    def next_question(self) -> str | None:
        missing = self.get_missing_fields()
        if not missing:
            return None
        self.current_field = missing[0]
        return self.FIELD_QUESTIONS[self.current_field].format(drug=self.drug_name)

    def collect_answer(self, answer: str) -> dict:
        if self.current_field and self.current_field in self.FIELD_ORDER:
            self.collected_info[self.current_field] = answer
            self.last_active_at = time.time()
            self.irrelevant_count = 0

        missing = self.get_missing_fields()
        if not missing:
            self.transition(DrugRecordPhase.CONFIRMING)
            self.confirmation_summary = self._build_summary()
            return {"phase": "confirming", "summary": self.confirmation_summary}

        self.current_field = missing[0]
        question = self.FIELD_QUESTIONS[self.current_field].format(drug=self.drug_name)
        return {"phase": "collecting", "question": question, "field": self.current_field}

    def handle_irrelevant_input(self, user_input: str) -> dict:
        self.irrelevant_count += 1
        if self.irrelevant_count >= self.MAX_IRRELEVANT_COUNT:
            self.transition(DrugRecordPhase.CANCELLED)
            return {
                "phase": "cancelled",
                "message": "看起来您可能想聊其他话题，用药记录收集已取消。如需重新记录，请随时告诉我。",
            }
        current_q = self.FIELD_QUESTIONS.get(self.current_field, "").format(drug=self.drug_name)
        return {
            "phase": "collecting",
            "message": f"我正在收集您的用药信息，请先回答：{current_q}",
            "original_input": user_input,
        }

    def _build_summary(self) -> str:
        lines = [f"请确认以下用药信息：", f"• 药品名称：{self.drug_name}"]
        for field in self.FIELD_ORDER:
            label = self.FIELD_LABELS[field]
            value = self.collected_info.get(field, "未指定")
            lines.append(f"• {label}：{value}")
        lines.append("\n确认无误请回复'确认'，如需修改请直接告诉我需要修改的内容，取消请回复'取消'。")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "drug_name": self.drug_name,
            "phase": self.phase.value,
            "collected_info": self.collected_info,
            "current_field": self.current_field,
            "started_at": self.started_at,
            "last_active_at": self.last_active_at,
            "irrelevant_count": self.irrelevant_count,
            "confirmation_summary": self.confirmation_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DrugRecordStateMachine:
        sm = cls(drug_name=data["drug_name"], initial_info=data.get("collected_info"))
        sm.phase = DrugRecordPhase(data.get("phase", "idle"))
        sm.current_field = data.get("current_field")
        sm.started_at = data.get("started_at")
        sm.last_active_at = data.get("last_active_at")
        sm.irrelevant_count = data.get("irrelevant_count", 0)
        sm.confirmation_summary = data.get("confirmation_summary")
        return sm
